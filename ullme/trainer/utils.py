from collections import UserDict
import functools
from itertools import repeat
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger


def split_input(model_input, chunk_size: int) -> List:
    """
    Split model input into chunks.
    :param model_input: model input
    :param chunk_size: chunk size
    :return: list of input chunks with same format as model_input
    """
    if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, torch.Tensor) for x in model_input.values()):
        keys = list(model_input.keys())
        chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
        return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    elif isinstance(model_input, list) and all(isinstance(x, torch.Tensor) for x in model_input):
        chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
        return [list(s) for s in zip(*chunked_x)]

    elif isinstance(model_input, torch.Tensor):
        return list(model_input.split(chunk_size, dim=0))

    elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
        args_chunks = split_input(model_input[0], chunk_size)
        kwargs_chunks = split_input(model_input[1], chunk_size)
        return list(zip(args_chunks, kwargs_chunks))
    
    elif isinstance(model_input, tuple) and list(map(type, model_input)) == [dict, dict]:
        args_chunks = split_input(model_input[0], chunk_size) # list of dicts
        global_kwargs = model_input[1]
        for args_chunk in args_chunks:
            args_chunk.update(global_kwargs)
        return args_chunks
    
    else:
        raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')
    

# Wrap the model using LoRA policy from llama-recipes or custom policy:
# This checks for lora layers (has weight and requires_grad)
def get_wrapping_policy(transformer_layers: List[nn.Module]):
    """
    A generic wrapping policy that wraps:
    1. all leaf modules with requires_grad=True.
    2. all transformer layers with a specific transformer_layer_cls.
    """
    def lambda_policy_fn(module):
        # All leaf modules with requires_grad=True
        return (len(list(module.named_children())) == 0) and (getattr(module, "weight", None) is not None) and (module.weight.requires_grad)
    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

    all_transformer_wrap_policies = []
    for transformer_layer in transformer_layers:
        transformer_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer},
        )
        all_transformer_wrap_policies.append(transformer_wrap_policy)

    policies=[lambda_policy] + all_transformer_wrap_policies
    return functools.partial(_or_policy, policies=policies)


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
    loss_weight_mask: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
    labels = labels[:, 1:].clone()
    loss_weight_mask = loss_weight_mask[..., 1:].clone().contiguous() if loss_weight_mask is not None else None
    loss_mask = labels != label_pad_token_id
    loss_weight_mask = loss_weight_mask * loss_mask if loss_weight_mask is not None else loss_mask
    logits = logits[:, :-1, :].clone() # (batch_size, seq_len, vocab_size)
    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2) # (batch_size, seq_len)
    average_logs = (per_token_logps * loss_weight_mask).sum(-1) / loss_weight_mask.sum(-1)
    if average_log_prob:
        return (per_token_logps * loss_weight_mask).sum(-1) / loss_weight_mask.sum(-1), average_logs # (batch_size,)
    else:
        return (per_token_logps * loss_weight_mask).sum(-1), average_logs # (batch_size,)


def get_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Prints the number of trainable parameters in the model.

    Args:
        model (`PreTrainedModel`):
            The model to print the number of trainable parameters for.

    Returns:
        `Tuple[int, int, float]`:
            The number of trainable parameters, the total number of parameters and the
            percentage of trainable parameters.
    """
    trainable_params = 0
    all_param = 0
    trainable_layers = []
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            trainable_layers.append(name)

    return trainable_params, all_param, 100 * trainable_params / all_param, trainable_layers


def get_cosine_schedule_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        min_reduce_rate: float = 0.0,
    ) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over num_warmup_steps, then decreases to min_reduce_rate*lr on a cosine schedule over
    the remaining num_training_steps-num_warmup_steps (assuming num_cycles = 0.5).

    Args:
        optimizer (`torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for which to increase the learning rate.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, optional, defaults to `0.5`):
            The number of waves in the cosine schedule (the default is to just decrease from the max
            learning rate to the min learning rate).
        last_epoch (`int`, optional, defaults to `-1`):
            The index of the last epoch when resuming training.
        min_reduce_rate (`float`, optional, defaults to `0.0`):
            The minimum percentage of the learning rate to retain at the end of the schedule.
    Returns:
        `torch.optim.lr_scheduler.LambdaLR`:
            The learning rate scheduler.
    """

    def lr_lambda(current_step):
        # Linearly increase learning rate from min_reduce_rate to 1.0 over num_warmup_steps
        if current_step < num_warmup_steps:
            return  min_reduce_rate + (1.0 - min_reduce_rate) * current_step / max(1, num_warmup_steps)
        # Cosin schedule learning rate from 1.0 to min_reduce_rate
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine_lr_multiple = 0.5 * (
            1.0 + min_reduce_rate + math.cos(math.pi * progress * num_cycles * 2.0) * (1.0 - min_reduce_rate)
        )
        return max(min_reduce_rate, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def choose_logger(
    logger_name: Literal["csv", "tensorboard", "wandb"],
    out_dir: Path,
    name: str,
    log_interval: int = 1,
    resume: Optional[bool] = None,
    **kwargs: Any,
):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", flush_logs_every_n_steps=log_interval, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", **kwargs)
    if logger_name == "wandb":
        return WandbLogger(project=name, resume=resume, **kwargs)
    raise ValueError(f"`--logger_name={logger_name}` is not a valid option. Choose from 'csv', 'tensorboard', 'wandb'.")


def trainable_filter(key: str, value: Any, trainable_layers: List[str]=[]) -> bool:
    if any([layer in key for layer in trainable_layers]):
        # print("Layer to save: ", key)
        return True
    else:
        return False

