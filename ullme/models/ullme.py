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
from transformers.models.t5.modeling_t5 import T5EncoderModel
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import mteb

from ullme.models.modeling_bidirectional_mistral import BidirectionalMistralForCausalLM
from ullme.models.modeling_bidirectional_llama import BidirectionalLlamaForCausalLM
from ullme.models.modeling_bidirectional_phi3 import BidirectionalPhi3ForCausalLM
from ullme.models.modeling_bidirectional_phi import BidirectionalPhiForCausalLM
from ullme.models.modeling_bidirectional_qwen2 import BidirectionalQwen2ForCausalLM
from ullme.models.modeling_bidirectional_cohere import BidirectionalCohereForCausalLM
from ullme.models.modeling_bidirectional_gemma import BidirectionalGemmaForCausalLM
from ullme.models.modeling_bidirectional_gemma2 import BidirectionalGemma2ForCausalLM
from ullme.special_tokens import SPECIAL_TOKENS
from ullme.models.utils import find_all_linear_names


class ULLME(nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            model_backbone_type: str,
            pooling_method: str='mean',
            lora_name: str = None,
            loar_r: int = 16,
            lora_alpha: int = 32,
            dropout: float = 0.1,
            attn_implementation: str = 'flash_attention_2',
            model_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hprams = {
            'model_name_or_path': model_name_or_path,
            'model_backbone_type': model_backbone_type,
            'pooling_method': pooling_method,
            'lora_name': lora_name,
            'loar_r': loar_r,
            'lora_alpha': lora_alpha,
            'dropout': dropout,
            'attn_implementation': attn_implementation,
            'model_dtype': model_dtype,
        }

        self.use_lora = True if lora_name else False
        self.adapter_name = lora_name

        self.tokenizer = self.create_tokenizer(model_name_or_path)

        if attn_implementation == "flash_attention_2":
            model_dtype = torch.bfloat16

        self.model = self.create_transformer(
            model_name_or_path=model_name_or_path,
            is_llm_bidirectional=True,
            backbone_type=model_backbone_type,
            use_lora=True if lora_name else False,
            lora_r=loar_r,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            adapter_name=lora_name,
            attn_implementation=attn_implementation,
            model_dtype=model_dtype,
        )
        if model_backbone_type is None:
            self.is_lm_with_head = False
        else:
            self.is_lm_with_head = True

        self.encoder_dim = self.model.config.hidden_size

        self.pooling_method = pooling_method

        self.output_projection = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim, dtype=model_dtype),
            nn.ReLU(),
            nn.Linear(self.encoder_dim, self.encoder_dim, dtype=model_dtype),
        )
    
    def create_tokenizer(self, model_name_or_path: str):
        # Load tokenizer
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="right", # Has to be right so masking of instruction tokens works correctly
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            print("Tokenizer does not have a pad token. We will use the bos token as pad token.")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    
    def create_transformer(
            self,
            model_name_or_path: str,
            backbone_type: str,
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
        }
        if not is_llm_bidirectional or backbone_type is None:
            if 't5' in model_name_or_path:
                model_class = T5EncoderModel
                kwargs = {
                    'pretrained_model_name_or_path': model_name_or_path, 
                    'config': config,
                    'torch_dtype': torch.bfloat16 if attn_implementation == "flash_attention_2" else model_dtype,
                    }
            else:
                model_class = AutoModel
        else:
            if backbone_type == "mistral":
                model_class = BidirectionalMistralForCausalLM
            elif backbone_type == "llama":
                model_class = BidirectionalLlamaForCausalLM
            elif backbone_type == "phi3":
                model_class = BidirectionalPhi3ForCausalLM
            elif backbone_type == "phi":
                model_class = BidirectionalPhiForCausalLM
            elif backbone_type == "qwen2":
                model_class = BidirectionalQwen2ForCausalLM
            elif backbone_type == "cohere":
                model_class = BidirectionalCohereForCausalLM
            elif backbone_type == "gemma":
                model_class = BidirectionalGemmaForCausalLM
            elif backbone_type == 'gemma2':
                model_class = BidirectionalGemma2ForCausalLM
            else:
                raise NotImplementedError(f"Backbone type {backbone_type} not implemented")
            
        transformer: PreTrainedModel = model_class.from_pretrained(**kwargs)

        if use_lora:
            if target_modules == "all":
                target_modules = find_all_linear_names(transformer, quantization)
            assert isinstance(target_modules, list) or target_modules == 'all-linear', "target_modules must be a list or 'all-linear'"
            task_type = TaskType.CAUSAL_LM
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=task_type,
                target_modules=target_modules,
            )
            transformer: PeftModel = get_peft_model(transformer, lora_config, adapter_name=adapter_name)
        
        return transformer 

    def pooling(
            self,
            hidden_state: torch.Tensor,
            attention_mask: torch.Tensor = None,
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
            prompt_length: Optional[torch.Tensor] = None, # (batch_size)
            is_generate: bool = False,
            disable_lora: bool = False,
    ):  
        if self.is_lm_with_head == False and is_generate:
            raise ValueError("Model does not have a head for generation")
        
        if is_generate:
            labels = input_ids.clone().contiguous()
            # labels[i][j] = -100 if attention_mask[i][j] == 0 else labels[i][j]
            labels[attention_mask == 0] = -100
            if self.use_lora and disable_lora:
                with self.model.disable_adapter():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        is_causal=True,
                        labels = input_ids.clone().contiguous(),
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    is_causal=True,
                    labels = input_ids.clone().contiguous(),
                )

            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
            }
        else:
            if self.is_lm_with_head:
                encoder_representation = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                ).hidden_states[-1] # (batch_size, seq_len, hidden_size)
            else:
                encoder_representation = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                ).hidden_states[-1] # (batch_size, seq_len, hidden_size)

            sentence_representation = self.pooling(
                hidden_state=encoder_representation,
                attention_mask=attention_mask,
                prompt_length=prompt_length,
            ) # (batch_size, hidden_size)

            projected_representation = self.output_projection(sentence_representation) # (batch_size, hidden_size)

            return {
                'reps': sentence_representation,
                'projection': projected_representation
            }


class WrappedULLME(nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            model_backbone_type: str=None,
            pooling_method: str='mean',
            lora_name: str = None,
            loar_r: int = 16,
            lora_alpha: int = 32,
            dropout: float = 0.1,
            attn_implementation: str = 'flash_attention_2',
            model_dtype: torch.dtype = torch.bfloat16,
            model_revision: str = 'dev',
            model_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.mteb_model_meta = mteb.ModelMeta(
            name='Lusifer',
            revision=model_revision,
            release_date=date.today().strftime("%Y-%m-%d"),
            languages=None,
        )

        self.model = ULLME(
            model_name_or_path=model_name_or_path,
            model_backbone_type=model_backbone_type,
            pooling_method=pooling_method,
            lora_name=lora_name,
            loar_r=loar_r,
            lora_alpha=lora_alpha,
            dropout=dropout,
            attn_implementation=attn_implementation,
            model_dtype=model_dtype,
        )

        if model_checkpoint is not None and os.path.exists(model_checkpoint):
            print(f"Loading model from checkpoint: {model_checkpoint}")
            state_dict = torch.load(model_checkpoint, map_location='cpu')
            self.model.load_state_dict(state_dict['model'], strict=False)

        self.special_tokens = SPECIAL_TOKENS.get(model_backbone_type, {})
        self.tokenizer = self.model.tokenizer

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_gpus = torch.cuda.device_count()
        self.model.to(self.device)
        if self.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
        self.model.eval()

    def tokenize_example(
            self, 
            example: Tuple[str, str],
            is_query: bool = True,
            max_length: int = 512,
    ) -> BatchEncoding:
        bos = self.special_tokens.get("bos", "")
        user_bos = self.special_tokens.get("user_bos", "")
        eos = self.special_tokens.get("eos", "")
        eot = self.special_tokens.get("eot", "")
        query_prompt = bos + user_bos + "{instruction}."
        query_format = bos + user_bos + "{instruction}." + "\n{example}" + eot + eos
        candidate_prompt = bos + user_bos + "{instruction}. Candidate:" + "\n"
        candidate_format = bos + user_bos + "{instruction}. Candidate:" + "\n" + "{example}" + eot + eos
        if is_query:
            emb_prompt = query_prompt.format(instruction=example[0])
            emb_example = query_format.format(instruction=example[0], example=example[1])
        else:
            emb_prompt = candidate_prompt.format(instruction=example[0])
            emb_example = candidate_format.format(instruction=example[0], example=example[1])
        model_inputs = self.tokenizer(
            text=emb_example,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False if self.special_tokens!={} else True, # already added
        )
        prompt_length = len(self.tokenizer.tokenize(emb_prompt))
        model_inputs['prompt_length'] = prompt_length
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
            inputs = [self.tokenize_example(example, is_query=is_query, max_length=max_length) for example in batch]
            inputs = self.tokenizer.pad(inputs, return_tensors='pt', pad_to_multiple_of=8)
            inputs = {
                'input_ids': inputs['input_ids'].to(self.device),
                'attention_mask': inputs['attention_mask'].to(self.device),
                'prompt_length': inputs['prompt_length'].to(self.device),
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


