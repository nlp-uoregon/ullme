# Modified from NV-Embed-v2
from typing import List, Union, Dict, Mapping, Optional, Tuple, TypedDict
import torch
import os
import json
import numpy as np
from functools import partial
from contextlib import nullcontext
from transformers import AutoModel, PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoTokenizer
from transformers.models.mistral.modeling_mistral import MISTRAL_INPUTS_DOCSTRING
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_attention_mask_for_sdpa
from transformers import MistralModel, MistralConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
)
from einops import rearrange, repeat
from tqdm.auto import tqdm
from torch.nn.attention import SDPBackend
from datasets import Dataset
from torch.utils.data import DataLoader

from ullme.model.bidirectional_modelings.config_nvembed import NVEmbedConfig, LatentAttentionConfig, BidirectionalMistralConfig
from ullme.model.bidirectional_modelings.modeling_bidirectional_mistral import BidirectionalMistral

logger = logging.get_logger(__name__)

class NVEmbedFeatures(TypedDict):
    input_dict: torch.Tensor
    attention_mask: torch.Tensor
    pool_mask: torch.Tensor
   
def _move_to_device(maybe_tensor, device: torch.device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device, non_blocking=device.type == "cuda")
    elif isinstance(maybe_tensor, dict):
        return {key: _move_to_device(value, device) for key, value in maybe_tensor.items()}
    elif isinstance(maybe_tensor, list):
        return [_move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([_move_to_device(x, device) for x in maybe_tensor])
    elif isinstance(maybe_tensor, Mapping):
        return type(maybe_tensor)({k: _move_to_device(v, device) for k, v in maybe_tensor.items()})
    else:
        return maybe_tensor

def move_to_device(sample, device: torch.device):
    if device.type == "cpu":
        return sample
    
    if len(sample) == 0:
        return {}
    return _move_to_device(sample, device)


def input_transform_func(
    tokenizer: PreTrainedTokenizerFast,
    examples: Dict[str, List],
    always_add_eos: bool,
    max_length: int,
    instruction: str,
) -> BatchEncoding:
    if always_add_eos:
        examples['input_texts'] = [instruction + input_example + tokenizer.eos_token for input_example in examples['input_texts']]
    batch_dict = tokenizer(
        examples['input_texts'],
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        return_tensors="pt",
        truncation=True)
    return batch_dict


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = torch.nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)

class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * torch.nn.functional.gelu(gates)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class Attention(torch.nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = torch.nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias = False)

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)


class LatentAttentionModel(PreTrainedModel):
    config_class = LatentAttentionConfig

    def __init__(self, config: LatentAttentionConfig):
        super().__init__(config)
        ## cross-attention block
        num_latents, latent_dim, cross_heads, cross_dim_head = config.num_latents_value, config.latent_dim, config.num_cross_heads, config.cross_dim_head
        dim = config.hidden_dim
        # init latent_attention and latents
        self.cross_attend_blocks = torch.nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head),
                    context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim)),
        ])
        self.output_normalize = config.output_normalize
        self.latents = torch.nn.Parameter(torch.randn(num_latents, latent_dim))

    def forward(self, hiddens, attention_mask: torch.Tensor=None):
        ## cross-attention block
        cross_attn, cross_ff = self.cross_attend_blocks
        b, *_, device = *hiddens.shape, hiddens.device
        x = repeat(self.latents, 'n d -> b n d', b = b)
        hiddens = cross_attn(hiddens, context = x, mask = None) + hiddens
        hiddens = cross_ff(hiddens) + hiddens
        if attention_mask !=None:
            s = torch.sum(hiddens * attention_mask.unsqueeze(-1).float(), dim=1) 
            d = attention_mask.sum(dim=1, keepdim=True).float()
            hiddens = s / d
            if self.output_normalize:
                hiddens = torch.nn.functional.normalize(hiddens, p=2, dim=-1)
        return hiddens # 
    
    
class NVEmbedModel(PreTrainedModel):
    config_class = NVEmbedConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["MistralDecoderLayer", "LatentAttentionModel"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True
    
    def __init__(self, config: NVEmbedConfig):
        super().__init__(config)
        self.latent_attention_model = LatentAttentionModel(config.latent_attention_config)
        self.embedding_model = BidirectionalMistral(
            config.text_config,
        ) if config.text_config is not None else None
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_config._name_or_path) if config.text_config is not None else None
        self.padding_side = config.padding_side
        self.is_mask_instruction = config.is_mask_instruction
        self.add_eos = config.add_eos
        self.mask_type = config.mask_type
        if config.add_pad_token and self.tokenizer is not None:
            self.add_pad_token()

    def add_pad_token(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = self.padding_side

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pool_mask: Optional[torch.Tensor]=None, return_dict: bool=True):
        autocast_ctx = torch.autocast if torch.cuda.is_available() else nullcontext
        with autocast_ctx("cuda"):
            ## decoder only layer
            outputs = self.embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ## latent attention layer
            embeds = self.latent_attention_model(
                outputs.last_hidden_state,
                pool_mask,
            )
        if not return_dict:
            return (embeds,)
        return {"sentence_embeddings": embeds}
        