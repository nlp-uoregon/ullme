from dataclasses import asdict
import datetime
import json
import os
import yaml
from pathlib import Path
from typing import Any, List
from functools import partial
import torch
from torch.distributed.fsdp.api import ShardingStrategy
import lightning as L
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from lightning import seed_everything
from transformers import PreTrainedTokenizer, HfArgumentParser
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from transformers.models.phi.modeling_phi import PhiDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.cohere.modeling_cohere import CohereDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

from ullme.data_modules.rep_learning_datamodule import RepLearningDataModule
from ullme.models.ullme import ULLME
from ullme.models.utils import get_wrapping_policy, get_activation_checkpointing_policy
from ullme.trainer.gradcache_trainer import GradCacheTrainer
from ullme.args import DataArguments, ModelArguments, TrainingArguments
from ullme.trainer.utils import choose_logger, get_cosine_schedule_with_warmup, get_trainable_parameters, trainable_filter


backbone_to_layer_type = {
    'mistral': MistralDecoderLayer,
    'llama': LlamaDecoderLayer,
    'phi3': Phi3DecoderLayer,
    'phi': PhiDecoderLayer,
    'qwen2': Qwen2DecoderLayer,
    'cohere': CohereDecoderLayer,
    'gemma': GemmaDecoderLayer,
    'gemma2': Gemma2DecoderLayer,
}

def get_dataloaders(
        fabric: L.Fabric, 
        data_module: RepLearningDataModule,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments, 
        model_args: ModelArguments,
        training_args: TrainingArguments,
        epoch: int = 0,
        ):
    data_module.connect(
        world_size=fabric.world_size,
        global_rank=fabric.global_rank,
        tokenizer=tokenizer, 
        special_tokens_set=model_args.model_backbone_type,
        global_batch_size=training_args.global_batch_size,
        max_seq_length=data_args.max_seq_length,
        number_training_samples=data_args.number_training_samples,
        neg_per_sample=data_args.neg_per_sample,
        pos_per_sample=data_args.pos_per_sample,
    )
    data_module.set_epoch(epoch)
    with fabric.rank_zero_first():
        data_module.setup()
        train_dataloader = data_module.train_dataloader()
        train_dataloader = fabric.setup_dataloaders(
            train_dataloader,
            use_distributed_sampler=False,
            move_to_device=True
        )
    return train_dataloader


def main(
        fabric: L.Fabric,
        train_data: RepLearningDataModule,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        ):
    fabric.seed_everything(training_args.seed)

    # Initialize model
    with fabric.rank_zero_first():
        model = ULLME(
            model_name_or_path=model_args.model_name_or_path,
            model_backbone_type=model_args.model_backbone_type,
            pooling_method=model_args.pooling_method,
            lora_name=model_args.lora_name,
            loar_r=model_args.loar_r,
            lora_alpha=model_args.lora_alpha,
            dropout=model_args.dropout,
            attn_implementation=model_args.attn_implementation,
        )
    tokenizer = model.tokenizer
    trainable_params, all_param, trainable_params_percentage, trainable_layers = get_trainable_parameters(model)
    filter_fn = partial(trainable_filter, trainable_layers=trainable_layers) if trainable_params_percentage < 100 else None
    fabric.print(f"Number of trainable parameters: {trainable_params/1e6:.2f}M")
    fabric.print(f"Total number of parameters: {all_param/1e6:.2f}M")
    fabric.print(f"Percentage of trainable parameters: {trainable_params_percentage:.2f}%")
    model = fabric.setup_module(model)
    fabric.print("Model after wrapping")
    fabric.print(model)

    # Prepare the dataloaders
    train_dataloader = get_dataloaders(
        fabric=fabric,
        data_module=train_data,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
    )
    fabric.barrier()

    # Setup the optimizer and scheduler
    step_per_epoch = len(train_dataloader)
    lr_max_steps = min(training_args.max_steps, step_per_epoch * training_args.max_epochs)
    warmup_steps = min(training_args.warmpup_proportion * lr_max_steps, 500)
    lr = training_args.learning_rate
    min_lr = training_args.min_learning_rate
    min_reduce_rate = min_lr / lr
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.999),
    )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=lr_max_steps,
        min_reduce_rate=min_reduce_rate,
    )

    # Load the checkpoint if needed
    stage = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iter_num": 0,
        "epoch_num": 0,
    }
    if training_args.checkpoint_file is not None:
        checkpoint_path = os.path.join(training_args.checkpoint_dir, training_args.checkpoint_file)
        if os.path.exists(checkpoint_path):
            fabric.print(f"Load checkpoint from {checkpoint_path}")
            fabric.load(checkpoint_path, stage, strict=False)
    model = stage.pop("model")

    # Initialize the trainer
    trainer = GradCacheTrainer(
        fabric=fabric,
        con_loss_type=training_args.con_loss_type,
        gen_loss_type=training_args.gen_loss_type,
        use_kl_loss=training_args.use_kl_loss,
        reference_free=training_args.preference_free,
        label_smoothing=training_args.label_smoothing,
        beta=training_args.beta,
        temperature=training_args.temperature,
        is_distance=training_args.is_distance,
        use_miner=training_args.use_miner,
        chunk_size=training_args.gc_chunk_size,
    )

    current_epoch_num = stage.get("epoch_num", 0)
    fabric.print(f"Start training from epoch {current_epoch_num}")
    for epoch in range(current_epoch_num, training_args.max_epochs):
        train_dataloader = get_dataloaders(
            fabric=fabric,
            data_module=train_data,
            tokenizer=tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            epoch=epoch,
        )
        fabric.barrier()
        checkpoint_path = trainer.fit_epoch(
            model=model,
            train_loader=train_dataloader,
            stage=stage,
            lr_max_steps=lr_max_steps,
            grad_norm_clip=training_args.grad_norm_clip,
            log_interval=training_args.log_interval,
            checkpoint_iterval=training_args.checkpoint_interval,
            checkpoint_dir=training_args.checkpoint_dir,
            checkpoint_filter=filter_fn,
            model_revision=training_args.model_revision,
            eval_batch_size=training_args.eval_batch_size,
        )
        fabric.barrier()
        # Reload the model from the checkpoint 
        torch.cuda.empty_cache()
        stage['model'] = model
        fabric.load(checkpoint_path, stage, strict=False)
        model = stage.pop("model")
        if stage["iter_num"] > lr_max_steps:
            break
    
    fabric.print("Training is finished")


def setup(data_args: DataArguments, model_args: ModelArguments, training_args: TrainingArguments):
    seed_everything(training_args.seed)

    training_metadata_path = data_args.training_metadata_path
    # Load the list of dict from jsonl file
    with open(training_metadata_path, "r") as f:
        metadata = [json.loads(line) for line in f]
    train_data = RepLearningDataModule(
        metadata=metadata,
        num_workers=data_args.num_workers,
        seed=training_args.seed,
    )

    strategy = training_args.strategy
    if training_args.nodes > 1 or training_args.devices > 1:
        if training_args.strategy == 'fsdp':
            # Config sharding strategy
            if training_args.sharding_strategy == "full_shard":
                sharding_strategy = ShardingStrategy.FULL_SHARD
            elif training_args.sharding_strategy == "shard_grad_op":
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            elif training_args.sharding_strategy == "ddp":
                sharding_strategy = ShardingStrategy.NO_SHARD
            elif training_args.sharding_strategy == "hybrid_full_shard":
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            elif training_args.sharding_strategy == "hybrid_shard_grad_op":
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
            else:
                raise ValueError("Invalid sharding strategy")

            backbone_layer_types = {backbone_to_layer_type[model_args.model_backbone_type]}
            wrapping_policy = get_wrapping_policy(backbone_layer_types) if model_args.lora_name is not None else backbone_layer_types
            activation_checkpointing_policy = get_activation_checkpointing_policy(backbone_layer_types) if model_args.lora_name is not None else backbone_layer_types
            
            strategy = FSDPStrategy(
                auto_wrap_policy=wrapping_policy,
                activation_checkpointing_policy=activation_checkpointing_policy if training_args.activation_checkpointing else None,
                sharding_strategy=sharding_strategy,
                limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
                state_dict_type="full",
                cpu_offload=training_args.use_cpu_offload,
                timeout=datetime.timedelta(hours=5), # makeing large timeout for model evaluation
            )
        elif training_args.strategy == 'ddp':
            strategy = DDPStrategy(
                find_unused_parameters=True, 
                timeout=datetime.timedelta(hours=5), # makeing large timeout for model evaluation
            )
    else:
        strategy = "auto"

    logger_dir = os.path.join(training_args.checkpoint_dir, f"logs_{training_args.logger_type}")
    os.makedirs(logger_dir, exist_ok=True)
    logger = choose_logger(
        logger_name=training_args.logger_type,
        out_dir=Path(logger_dir),
        name=training_args.logger_name,
        log_interval=training_args.log_interval,
    )

     # check whether gpu is support bf16
    if not torch.cuda.is_bf16_supported():
        training_args.precision = '16-mixed'

    fabric = L.Fabric(
        accelerator='gpu',
        strategy=strategy,
        devices=training_args.devices,
        num_nodes=training_args.nodes,
        precision=training_args.precision,
        loggers=logger,
    )
    fabric.launch(
        main,
        train_data=train_data,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
    )


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_DATASETS_TRUST_REMOTE_CODE']='1'
    
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    torch.set_float32_matmul_precision('high')
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to the yaml config file",
    )
    parser.add_argument(
        "--model_revision", type=str, default=None, help="Model revision"
    )
    parser.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes"
    )
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices"
    )
    parser.add_argument(
        "--gc_chunk_size", type=int, default=8, help="Gradient cache chunk size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--min_learning_rate", type=float, default=0.0, help="Minimum learning rate"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--checkpoint_file", type=str, default=None, help="Checkpoint file to resume training"
    )

    args = parser.parse_args()
    config_file = args.config_file

    hf_parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    print(f"Loading yaml config {config_file}")
    data_args, model_args, training_args = hf_parser.parse_yaml_file(yaml_file=config_file)
    # Add read-only arguments
    if args.model_revision is not None:
        training_args.model_revision = args.model_revision
    training_args.nodes = args.nodes
    training_args.devices = args.devices
    training_args.gc_chunk_size = args.gc_chunk_size
    training_args.learning_rate = args.learning_rate
    training_args.min_learning_rate = args.min_learning_rate
    training_args.checkpoint_dir = args.checkpoint_dir
    training_args.checkpoint_file = args.checkpoint_file

    config_file_path = Path(training_args.checkpoint_dir) / "config.yaml"
    config_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file_path, "w") as f:
        yaml.dump(asdict(data_args), f)
        yaml.dump(asdict(model_args), f)
        yaml.dump(asdict(training_args), f)

    setup(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
    )

