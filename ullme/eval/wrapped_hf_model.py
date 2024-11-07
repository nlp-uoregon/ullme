from datetime import date
from typing import Dict, List, Union
import mteb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel, FlagLLMModel


class WrappedHFModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            num_gpus: int=0,
            ):
        super().__init__()

        self.mteb_model_meta = mteb.ModelMeta(
            name=model_name_or_path,
            revision='public',
            release_date=date.today().strftime("%Y-%m-%d"),
            languages=None,
        )
        self.model_name_or_path = model_name_or_path
        print(f"Loading model: {model_name_or_path}")
        if model_name_or_path == 'izhx/udever-bloom-7b1':
            from transformers import BloomModel
            self.model = BloomModel.from_pretrained('izhx/udever-bloom-7b1', trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained('izhx/udever-bloom-7b1', trust_remote_code=True)
            self.boq, self.eoq, self.bod, self.eod = '[BOQ]', '[EOQ]', '[BOD]', '[EOD]'
            self.eoq_id, self.eod_id = self.tokenizer.convert_tokens_to_ids([self.eoq, self.eod])

            if self.tokenizer.padding_side != 'left':
                self.tokenizer.padding_side = 'left'
        elif model_name_or_path == 'BAAI/bge-m3':
            self.model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        elif model_name_or_path == 'BAAI/bge-multilingual-gemma2':
            self.model = FlagLLMModel(
                'BAAI/bge-multilingual-gemma2', 
                query_instruction_for_retrieval="Given a web search query, retrieve relevant passages that answer the query.",
                use_fp16=True
                ) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        elif model_name_or_path == 'jinaai/jina-embeddings-v3':
            self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, torch_dtype=torch.float16)
        else:
            # check if gpu is support bf16
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    print("GPU supports bfloat16, using it for faster and more memory efficient computation.")
                    torch_dtype = torch.bfloat16
                else:
                    print("GPU does not support bfloat16, using float16 for faster and more memory efficient computation.")
                    torch_dtype = torch.float16
            # Check whether gpu is ampere or newer
            if torch.cuda.is_available():
                if model_name_or_path == 'nvidia/NV-Embed-v2':
                    attn_implementation = None
                elif torch.cuda.get_device_properties(0).major >= 8 and model_name_or_path not in ['nvidia/NV-Embed-v2']:
                    print("GPU is ampere or newer, using flash_attention_2 for faster and more memory efficient computation.")
                    attn_implementation = 'flash_attention_2'
                else:
                    print("GPU is older than ampere, using sdpa")
                    attn_implementation = 'sdpa'
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True, 
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
            if self.tokenizer.pad_token_id is None:
                print("Tokenizer does not have a pad token. We will use the bos token as pad token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            if self.tokenizer.padding_side != 'right':
                self.tokenizer.padding_side = 'right'

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        if self.model_name_or_path not in ['BAAI/bge-m3', 'BAAI/bge-multilingual-gemma2']:
            self.model.to(self.device)
            if self.num_gpus > 1 and self.model_name_or_path not in ['nvidia/NV-Embed-v2']:
                self.model = nn.DataParallel(self.model)
            self.model.eval()

    @staticmethod
    def average_pool(
            last_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor
            ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    @staticmethod
    def last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
        ) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def pooling(
            self,
            last_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            ):
        if self.model_name_or_path in ['intfloat/multilingual-e5-large']:
            return self.average_pool(last_hidden_states, attention_mask)
        elif self.model_name_or_path in ['BAAI/bge-multilingual-gemma2', 'Alibaba-NLP/gte-multilingual-base', 'Salesforce/SFR-Embedding-2_R']:
            return self.last_token_pool(last_hidden_states, attention_mask)
        else:
            return self.average_pool(last_hidden_states, attention_mask)
    
    @staticmethod
    def me5_format_data(
            text: str,
            is_query: bool = True,
            **kwargs
            ):
        return f"query: {text}" if is_query else f"passage: {text}"
    
    def format_data(
            self, 
            text: str,
            is_query: bool = True,
            **kwargs
            ):
        if self.model_name_or_path in ['intfloat/multilingual-e5-large']:
            return self.me5_format_data(text, is_query)
        else:
            return text
    
    @torch.no_grad()
    def udever_encode(
        self, 
        sentences: List[str], 
        is_query: bool = True, 
        max_length=300
        ):
        bos = self.boq if is_query else self.bod
        eos_id = self.eoq_id if is_query else self.eod_id
        texts = [bos + t for t in sentences]
        encoding = self.tokenizer(
            texts, truncation=True, max_length=max_length - 1, padding=True
        )
        for ids, mask in zip(encoding['input_ids'], encoding['attention_mask']):
            ids.append(eos_id)
            mask.append(1)
        inputs = self.tokenizer.pad(encoding, return_tensors='pt')
        with torch.inference_mode():
            outputs = self.model(**inputs)
            embeds = outputs.last_hidden_state[:, -1]
        return embeds
    
    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        is_query: bool = True,
        batch_size: int = 256,
        max_length: int = 512,
        instruction: str = "",
        **kwargs,
    ):  
        is_single_sentence = False
        if isinstance(sentences, str):
            sentences = [sentences]
            is_single_sentence = True
        
        if self.model_name_or_path=='izhx/udever-bloom-7b1':
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
                batch = sentences[start_index:start_index+batch_size]
                embeddings = self.udever_encode(batch, is_query, max_length)
                all_embeddings.append(embeddings.cpu().float().numpy())
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        elif self.model_name_or_path=='BAAI/bge-m3':
            all_embeddings = self.model.encode(sentences, batch_size=batch_size, max_length=max_length)['dense_vecs']
        elif self.model_name_or_path=='BAAI/bge-multilingual-gemma2':
            all_embeddings = self.model.encode(sentences, batch_size=batch_size, max_length=max_length)
        elif self.model_name_or_path=='nvidia/NV-Embed-v2':
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
                batch = sentences[start_index:start_index+batch_size]
                outputs = self.model.encode(batch, instruction=instruction, max_length=max_length)
                embeddings = F.normalize(outputs, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().float().numpy())
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        elif self.model_name_or_path=='jinaai/jina-embeddings-v3':
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
                batch = sentences[start_index:start_index+batch_size]
                input_texts = [self.format_data(text, is_query) for text in batch]
                embeddings = self.model.encode(input_texts, task="text-matching")
                all_embeddings.append(embeddings)
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
                batch = sentences[start_index:start_index+batch_size]
                input_texts = [self.format_data(text, is_query) for text in batch]
                batch_dict = self.tokenizer(input_texts, max_length=512, padding='longest', truncation=True, return_tensors='pt')
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                outputs = self.model(**batch_dict)
                embeddings = self.pooling(outputs.last_hidden_state, batch_dict['attention_mask'])
                all_embeddings.append(embeddings.cpu().float().numpy())
            all_embeddings = np.concatenate(all_embeddings, axis=0)

        if is_single_sentence:
            return all_embeddings[0]
        return all_embeddings

    @torch.no_grad()
    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        if self.model_name_or_path=='BAAI/bge-multilingual-gemma2':
            return self.model.encode_queries(queries)
        else:
            return self.encode(queries, is_query=True, **kwargs)
    
    @torch.no_grad()
    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        if self.model_name_or_path=='BAAI/bge-multilingual-gemma2':
            return self.model.encode_corpus(corpus)
        else:
            return self.encode(corpus, is_query=False, **kwargs)
       