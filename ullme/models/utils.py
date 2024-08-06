import functools
from typing import List, Optional, Set
import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from transformers import PreTrainedModel, Conv1D
from peft.tuners.lora import LoraLayer


def find_all_linear_names(model: nn.Module, quantization: Optional[bool] = False):
    if not isinstance(model, PreTrainedModel):
        raise ValueError("Model must be an instance of `transformers.PreTrainedModel`")
    
    if quantization:
        from bitsandbytes.nn import Linear4bit

        cls = (Linear4bit, Conv1D)
    else:
        cls = (torch.nn.Linear, Conv1D)

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.rsplit(".", 1)[-1]  # get the base name
            lora_module_names.add(names)
            
    if "lm_head" in lora_module_names:  
        lora_module_names.remove("lm_head")

    # ignore the last classification head for text generation models
    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]
        lora_module_names -= {last_module_name}
        
    return list(lora_module_names)


def get_wrapping_policy(transformer_layers: Set[nn.Module]):
    """
    A wrapping policy for Lusifer models that wraps:
    1. all leaf modules with requires_grad=True.
    2. all trainable sequential modules.
    3. all LoraLayer modules.
    4. all transformer layers with a specific transformer_layer_cls.
    """
    def lambda_policy_fn(module):
        # All leaf modules with requires_grad=True
        is_atomic_trainable_layer = (len(list(module.named_children())) == 0) and (getattr(module, "weight", None) is not None) and (module.weight.requires_grad)
        is_trainable_seqential = isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module if hasattr(m, "weight"))
        is_lora_layer = isinstance(module, LoraLayer)
        return is_atomic_trainable_layer or is_trainable_seqential or is_lora_layer
    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(transformer_layers),
    )

    policies=[lambda_policy, transformer_wrap_policy]
    return functools.partial(_or_policy, policies=policies)


def get_activation_checkpointing_policy(transformer_layers: Set[nn.Module]):
    """
    A activation checkpointing policy for Lusifer models that wraps:
    1. all trainable sequential modules.
    2. all transformer layers with a specific transformer_layer_cls.
    """
    def lambda_policy_fn(module):
        is_trainable_seqential = isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module if hasattr(m, "weight"))
        return is_trainable_seqential 
    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(transformer_layers),
    )

    policies=[lambda_policy, transformer_wrap_policy]
    return functools.partial(_or_policy, policies=policies)
