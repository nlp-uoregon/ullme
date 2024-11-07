from contextlib import nullcontext
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import date
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from tqdm import tqdm
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
)
from transformers.models.mt5.modeling_mt5 import MT5EncoderModel
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import mteb

from ullme.model.bidirectional_modelings.modeling_bidirectional_mistral import BidirectionalMistralForCausalLM
from ullme.model.bidirectional_modelings.modeling_bidirectional_llama import BidirectionalLlamaForCausalLM
from ullme.model.bidirectional_modelings.modeling_bidirectional_phi3 import BidirectionalPhi3ForCausalLM
from ullme.model.bidirectional_modelings.modeling_bidirectional_phi import BidirectionalPhiForCausalLM
from ullme.model.bidirectional_modelings.modeling_bidirectional_qwen2 import BidirectionalQwen2ForCausalLM
from ullme.model.bidirectional_modelings.modeling_bidirectional_gemma2 import BidirectionalGemma2ForCausalLM
from ullme.model.bidirectional_modelings.modeling_nv_embed import LatentAttentionModel
from ullme.model.bidirectional_modelings.config_nvembed import LatentAttentionConfig
from ullme.model.utils import find_all_linear_names
from ullme.special_tokens import SPECIAL_TOKENS


class ULLME(nn.Module):
    def __init__(
            self,
            encoder_name_or_path: str,
            encoder_backbone_type: str = 'mistral',
            pooling_method: str='mean',
            encoder_lora_name: str = 'encoder_lora',
            encoder_lora_target_modules: Union[str, List[str]] = "all",
            loar_r: int = 16,
            lora_alpha: int = 32,
            dropout: float = 0.1,
            attn_implementation: str = 'flash_attention_2',
            model_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        
        super().__init__()
        self.hprams = {
            'encoder_name_or_path': encoder_name_or_path,
            'encoder_backbone_type': encoder_backbone_type,
            'pooling_method': pooling_method,
            'encoder_lora_name': encoder_lora_name,
            'encoder_lora_target_modules': encoder_lora_target_modules,
            'loar_r': loar_r,
            'lora_alpha': lora_alpha,
            'dropout': dropout,
            'attn_implementation': attn_implementation,
            'model_dtype': model_dtype,
        }
        if attn_implementation == "flash_attention_2":
            model_dtype = torch.bfloat16
        self.model_dtype = model_dtype
        self.pooling_method = pooling_method
        self.mteb_model_meta = mteb.ModelMeta(
            name='Lusifer',
            revision='dev',
            release_date=date.today().strftime("%Y-%m-%d"),
            languages=None,
        )
        # Encoder
        self.encoder_tokenizer = self.create_tokenizer(encoder_name_or_path, encoder_backbone_type)
        self.encoder = self.create_transformer(
            model_name_or_path=encoder_name_or_path,
            is_llm_bidirectional=True,
            backbone_type=encoder_backbone_type,
            use_lora=True if encoder_lora_name else False,
            lora_r=loar_r,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            adapter_name=encoder_lora_name,
            attn_implementation=attn_implementation,
            model_dtype=model_dtype,
            target_modules=encoder_lora_target_modules,
        )
        self.use_lora = encoder_lora_name is not None
        if encoder_backbone_type == 'nvidia/NV-Embed-v2':
            print("Loading latent attention model of NV-Embed-v2")
            self.laten_attention_model, loading_info = LatentAttentionModel.from_pretrained('Hieuman/nvembed-v2-latent-attention', output_loading_info=True)
            print(f"Latent attention model loading info: {loading_info}")
            self.laten_attention_model.requires_grad_(False)
        self.encoder_dim = self.encoder.config.hidden_size
        self.encoder_backbone_type = encoder_backbone_type
        
        # Projector
        self.output_projection = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_dim, self.encoder_dim),
        )
    
    def create_tokenizer(self, model_name_or_path: str, backbone_type: str):
        # Load tokenizer
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="right", # Has to be right so masking of instruction tokens works correctly
            trust_remote_code=True,
        )
        pad_token = SPECIAL_TOKENS.get(backbone_type, {}).get("pad", tokenizer.eos_token)
        mask_token = SPECIAL_TOKENS.get(backbone_type, {}).get("mask", tokenizer.unk_token)
        if tokenizer.pad_token_id is None:
            print(f"Tokenizer does not have a pad token. We will use {pad_token} as the pad token.")
            tokenizer.pad_token = pad_token
            assert tokenizer.pad_token_id is not None, "Tokenizer does not have a pad token id"
        if tokenizer.mask_token_id is None:
            print(f"Tokenizer does not have a mask token. We will use {mask_token} as the mask token.")
            tokenizer.mask_token = mask_token
            assert tokenizer.mask_token_id is not None, "Tokenizer does not have a mask token id"
        return tokenizer

    def create_transformer(
            self,
            model_name_or_path: str,
            backbone_type: str = 'mistral',
            is_llm_bidirectional: bool = False,
            use_lora: bool = False,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.1,
            target_modules: Union[str, List[str]] = "all",
            adapter_name: str = 'default',
            quantization: bool = False,
            attn_implementation: str = None,
            model_dtype: torch.dtype = torch.bfloat16,
    ):  
        print(f"Loading model from {model_name_or_path}")
        if use_lora:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                pretraining_tp=1,  # Fix mat1 and mat2 shapes cannot be multiplied  error with LLaMA-2
                # See https://github.com/huggingface/transformers/pull/24906
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False
            )

        if quantization:
            # Prompt warning if quantization is enabled 
            print("Quantization is enabled. This may affect the performance of the model. And currently, quantization is only supported for inference or multi-gpu training WITH DPP.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
        
        kwargs = {
            'pretrained_model_name_or_path': model_name_or_path,
            'config': config,
            'quantization_config': bnb_config,
            'torch_dtype': torch.bfloat16 if attn_implementation == "flash_attention_2" else model_dtype,
            'attn_implementation': attn_implementation,
            'trust_remote_code': True,
            'output_loading_info': True,
        }
        model_class = AutoModel
        if not is_llm_bidirectional:
            if 'mt5' in model_name_or_path:
                model_class = MT5EncoderModel
                kwargs = {
                    'pretrained_model_name_or_path': model_name_or_path, 
                    'config': config,
                    'torch_dtype': torch.bfloat16 if attn_implementation == "flash_attention_2" else model_dtype,
                    'output_loading_info': True,
                    }
            kwargs.pop('attn_implementation')
        else:
            if backbone_type in ["mistral", "nvidia/NV-Embed-v2"]:
                model_class = BidirectionalMistralForCausalLM
            elif backbone_type == "llama":
                model_class = BidirectionalLlamaForCausalLM
            elif backbone_type == "phi3":
                model_class = BidirectionalPhi3ForCausalLM
            elif backbone_type == "phi":
                model_class = BidirectionalPhiForCausalLM
            elif backbone_type == "qwen2":
                model_class = BidirectionalQwen2ForCausalLM
            elif backbone_type == 'gemma2':
                model_class = BidirectionalGemma2ForCausalLM
            else:
                model_class = AutoModel
        
        print(f"Using model class: {model_class}")
        transformer, loading_info = model_class.from_pretrained(**kwargs)
        print(f"Model loading info: {loading_info}")

        if use_lora:
            if target_modules == "all":
                target_modules = find_all_linear_names(transformer, quantization)
            assert isinstance(target_modules, list) or target_modules == 'all-linear', "target_modules must be a list or 'all-linear'"
            task_type = TaskType.FEATURE_EXTRACTION
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=task_type,
                target_modules=target_modules,
            )
            if adapter_name is None:
                adapter_name = 'default'
            transformer: PeftModel = get_peft_model(transformer, lora_config, adapter_name=adapter_name)
        
        return transformer 

    def pooling(
            self,
            hidden_state: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            prompt_length: Optional[torch.Tensor] = None,
    ):  
        if attention_mask is None:
            attention_mask = torch.ones(hidden_state.size(0), hidden_state.size(1), device=hidden_state.device)
        # Pool the hidden states
        # Mask the prompt tokens
        if prompt_length is not None:
            attention_mask = attention_mask.clone()
            for i, l in enumerate(prompt_length):
                attention_mask[i, :l] = 0
                # Make sure not all zeros - If this happens it is a bug
                assert attention_mask[i].sum() > 0, "You have all zeros in the attention mask!"

        # In case the model is distributed across multiple devices; hidden_state may end up on diff device
        hidden_state = hidden_state.to(attention_mask.device)
        if self.pooling_method == 'cls':
            embedding = hidden_state[:, 0]
        elif self.pooling_method == 'lasttoken':
            b, n, d = hidden_state.size()
            # Get the last `1` in the attention mask of each item
            # Often it is just `gather_indices = torch.argmin(attention_mask, 1, keepdim=False) - 1`
            # except when 1) There's all 1's 2) There's 0's before the 1's
            reversed_mask = torch.flip(attention_mask, dims=(1,))
            argmax_reverse = torch.argmax(reversed_mask, dim=1, keepdim=False)
            gather_indices = attention_mask.size(1) - argmax_reverse - 1
            # If there are empty sequences, where the index would become -1 it will crash so set them to 0
            gather_indices = torch.clamp(gather_indices, min=0)
            # Turn indices from shape [b] -> [b, 1, d]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, d)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (b, 1, d)
            # Gather along the seq len: [b, n, d] -> [b, d]
            # Actually no need for the attention mask as we gather the last token where attn_mask=1 but
            # as some indices (which shouldn't be attended to) may be 0 due to clamp, use mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand((b, n, d)).float()
            embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        elif self.pooling_method in ['mean', 'weightedmean']:
            if self.pooling_method == 'weightedmean':
                attention_mask *= attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            embedding = s / d
        else: raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")
        return embedding.contiguous().to(hidden_state.dtype)
    
    def forward(
            self,
            input_ids: torch.Tensor, # (batch_size, seq_len)
            attention_mask: torch.Tensor, # (batch_size, seq_len)
            labels: Optional[torch.Tensor] = None, # (batch_size, seq_len)
            is_encode: bool = True,
            ):
        if is_encode:
            outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    is_causal=False,
                    output_hidden_states=True
                )

            input_reps = outputs.hidden_states[-1] # (bs, seq_len, hidden_size)
            if self.encoder_backbone_type == 'nvidia/NV-Embed-v2':
                pool_mask = attention_mask.clone()
                with torch.autocast(device_type=input_reps.device.type, dtype=self.model_dtype):
                    sentence_representation = self.laten_attention_model(input_reps, pool_mask)
                projected_representation = sentence_representation
            else:
                
                sentence_representation = self.pooling(
                    hidden_state=input_reps,
                    attention_mask=attention_mask,
                    prompt_length=None, 
                )
                with torch.autocast(device_type=sentence_representation.device.type, dtype=self.model_dtype):
                    projected_representation = self.output_projection(sentence_representation) # (batch_size, hidden_size)

            projected_representation = torch.nn.functional.normalize(projected_representation, p=2, dim=-1)
            sentence_representation = torch.nn.functional.normalize(sentence_representation, p=2, dim=-1)
            return {
                'reps': sentence_representation,
                'projected_reps': projected_representation,
            }
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                is_causal=True,
            )
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
            }
    
    def tokenize_example(
            self, 
            example: Tuple[str, str],
            tokenizer: PreTrainedTokenizer,
            is_query: bool = True,
            max_length: int = 512,
    ) -> BatchEncoding:
        query_format = "{instruction}\n{example}"
        candidate_format = "{instruction}\nCandidate:\n{example}"
        if is_query:
            emb_example = query_format.format(instruction=example[0], example=example[1])
        else:
            emb_example = candidate_format.format(instruction=example[0], example=example[1])
        model_inputs = tokenizer(
            text=emb_example,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return model_inputs
    
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
        
        sentences = [(instruction, s) for s in sentences]
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            batch = sentences[start_index:start_index+batch_size]
            inputs = [self.tokenize_example(example, tokenizer=self.encoder_tokenizer, is_query=is_query, max_length=max_length) for example in batch]
            inputs = self.encoder_tokenizer.pad(inputs, return_tensors='pt', pad_to_multiple_of=8)
            inputs = {
                'input_ids': inputs['input_ids'].to(self.device),
                'attention_mask': inputs['attention_mask'].to(self.device),
            }
            with torch.no_grad():
                reps = self(**inputs)['reps']
            all_embeddings.append(reps.cpu().to(torch.float32).numpy())
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if is_single_sentence:
            return all_embeddings[0]
        return all_embeddings

    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, is_query=True, **kwargs)
    
    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, is_query=False, **kwargs)

    def set_model_revision(self, revision: str):
        self.mteb_model_meta.revision = revision


class WrappedULLME(nn.Module):
    def __init__(
            self,
            encoder_name_or_path: str,
            encoder_backbone_type: str = 'mistral',
            pooling_method: str='mean',
            encoder_lora_name: str = 'encoder_lora',
            encoder_lora_target_modules: Union[str, List[str]] = "all",
            loar_r: int = 16,
            lora_alpha: int = 32,
            dropout: float = 0.1,
            attn_implementation: str = 'flash_attention_2',
            model_dtype: torch.dtype = torch.bfloat16,
            model_revision: str = 'dev',
            model_checkpoint: Optional[str] = None,
            num_gpus: int = 8,
    ) -> None:
        super().__init__()

        self.mteb_model_meta = mteb.ModelMeta(
            name='Lusifer',
            revision=model_revision,
            release_date=date.today().strftime("%Y-%m-%d"),
            languages=None,
        )

        self.model = ULLME(
            encoder_name_or_path=encoder_name_or_path,
            encoder_backbone_type=encoder_backbone_type,
            pooling_method=pooling_method,
            encoder_lora_name=encoder_lora_name,
            encoder_lora_target_modules=encoder_lora_target_modules,
            loar_r=loar_r,
            lora_alpha=lora_alpha,
            dropout=dropout,
            attn_implementation=attn_implementation,
            model_dtype=model_dtype,
        )

        if model_checkpoint is not None and os.path.exists(model_checkpoint):
            print(f"Loading model from checkpoint: {model_checkpoint}")
            state_dict = torch.load(model_checkpoint, map_location='cpu', weights_only=False)
            self.model.load_state_dict(state_dict['model'], strict=False)

        self.encoder_tokenizer = self.model.encoder_tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_gpus = min(torch.cuda.device_count(), num_gpus)
        print(f"Using {self.num_gpus} GPUs")
        self.model.to(self.device)
        if self.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
        self.model.eval()

    def tokenize_example(
            self, 
            example: Tuple[str, str],
            tokenizer: PreTrainedTokenizer,
            is_query: bool = True,
            max_length: int = 512,
    ) -> BatchEncoding:
        query_format = "{instruction}\n{example}"
        candidate_format = "{instruction}\nCandidate:\n{example}"
        if is_query:
            emb_example = query_format.format(instruction=example[0], example=example[1])
        else:
            emb_example = candidate_format.format(instruction=example[0], example=example[1])
        model_inputs = tokenizer(
            text=emb_example,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return model_inputs
    
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
        
        sentences = [(instruction, s) for s in sentences]
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            batch = sentences[start_index:start_index+batch_size]
            inputs = [self.tokenize_example(example, tokenizer=self.encoder_tokenizer, is_query=is_query, max_length=max_length) for example in batch]
            inputs = self.encoder_tokenizer.pad(inputs, return_tensors='pt', pad_to_multiple_of=8)
            inputs = {
                'input_ids': inputs['input_ids'].to(self.device),
                'attention_mask': inputs['attention_mask'].to(self.device),
            }
            reps = self.model(**inputs)['reps']
            all_embeddings.append(reps.cpu().to(torch.float32).numpy())
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if is_single_sentence:
            return all_embeddings[0]
        return all_embeddings

    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, is_query=True, **kwargs)
    
    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, is_query=False, **kwargs)



