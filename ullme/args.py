from dataclasses import dataclass, field
from typing import List


@dataclass
class DataArguments:
    training_metadata_path: str = field(
        metadata={"help": "Path to the training metadata file that contains the metadata for the training data in the form of {'name': 'dataset_name', 'path': 'path_to_dataset', 'instruction': 'instruction_for_dataset', 'enable_cross_batch_negative_sampling': True/False}"},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum sequence length."}
    )
    number_training_samples: int = field(
        default=1_000_000,
        metadata={"help": "The number of training samples per dataset."}
    )
    neg_per_sample: int = field(
        default=1,
        metadata={"help": "The number of negative samples per sample."}
    )
    pos_per_sample: int = field(
        default=1,
        metadata={"help": "The number of positive samples per sample."}
    )
    num_workers: int = field(
        default=0,
        metadata={"help": "Number of workers to use for data loading"}
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The name or path of the model to use"}
    )
    model_backbone_type: str = field(
        metadata={"help": "The type of the model backbone"},
    )
    pooling_method: str = field(
        default='mean',
        metadata={"help": "Pooling method to use. Can be mean/weightedmean/cls/lasttoken"}
    )
    lora_name: str = field(
        default='lora',
        metadata={"help": "Lora name. If None then no lora is used."}
    )
    loar_r: int = field(
        default=16,
        metadata={"help": "LoRA r parameter."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter."}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability."}
    )
    attn_implementation: str = field(
        default='flash_attention_2',
        metadata={"help": "The attention implementation which can be eager/sdpa/flash_attention_2"}
    )

@dataclass
class TrainingArguments:
    seed: int = field(
        default=777,
        metadata={"help": "Seed for reproducibility"}
    )

    model_revision: str = field(
        default='dev.v0',
        metadata={"help": "Model revision"}
    )
    
    nodes: int = field(
        default=1,
        metadata={"help": "Number of nodes to use for training"}
    )
    devices: int = field(
        default=1,
        metadata={"help": "Number of devices per node to use for training"}
    )
    precision: str = field(
        default='bf16-true',
        metadata={"help": "Precision to use. Can be bf16-true/bf16-mixed/16-mixed/32"}
    )
    strategy: str = field(
        default='fsdp',
        metadata={"help": "Strategy to use. Currently only supports dpp and fsdp"}
    )
    use_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to use CPU offload or not"}
    )
    sharding_strategy: str = field(
        default='full_shard',
        metadata={"help": "Sharding strategy to use. Can be full_shard/shard_grad_op/ddp/hybrid_full_shard/hybrid_shard_grad_op"}
    )
    quantization: bool = field(
        default=False,
        metadata={"help": "Whether to use quantization pretrained model. Note that, this is not supported in FSDP multi-GPU training yet."}
    )
    activation_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use activation checkpointing or not"}
    )

    con_loss_type: str = field(
        default='NTXentLoss',
        metadata={"help": "The Contrastive Loss function to use. Can be 'NTXentLoss' or 'SupConLoss'"}
    )
    use_miner: bool = field(
        default=True,
        metadata={"help": "Whether to use miner or not. The MultiSimilarityMiner will be used."}
    )
    is_distance: bool = field(
        default=True,
        metadata={"help": "Whether to use distance metric or not. If True, LpDistance will be used, otherwise CosineSimilarity."}
    )
    gen_loss_type: str = field(
        default=None,
        metadata={"help": "The Generation Loss function to use. Can be 'sft' for supervised fine-tuning, sigmoid/hinge/ipo/kto_pair for preference fine-tuning or None for no generation loss"}
    )
    use_kl_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use KL loss to align between reps scores and gen scores"}
    )
    preference_free: bool = field(
        default=False,
        metadata={"help": "Whether to use preference model in preference fine-tuning"}
    )
    label_smoothing: float = field(
        default=0.,
        metadata={"help": "The label smoothing value"}
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta value for the preference loss"}
    )
    temperature: float = field(
        default=0.1,
        metadata={"help": "The temperature for softmax"}
    )

    global_batch_size: int = field(
        default=32,
        metadata={"help": "The global batch size."}
    )
    gc_chunk_size: int = field(
        default=1,
        metadata={"help": "GradCache chunk size. If None, not use GradCache. In the case of OOM, try to reduce this value."}
    )
    eval_batch_size: int = field(
        default=32,
        metadata={"help": "Evaluation batch size"}
    )
    max_epochs: int = field(
        default=10,
        metadata={"help": "Maximum number of epochs to train"}
    )
    max_steps: int = field(
        default=float("inf"),
        metadata={"help": "Maximum number of steps to train"}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate"}
    )
    min_learning_rate: float = field(
        default=0.0,
        metadata={"help": "Minimum learning rate"}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay to apply."},
        )
    warmpup_proportion: float = field(
        default=0.1,
        metadata={"help": "Proportion of training steps to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."}
    )
    grad_norm_clip: float = field(
        default=None,
        metadata={"help": "Gradient norm clipping value. In the case of nan gradients, try to use this."}
    )

    checkpoint_dir: str = field(
        default=None,
        metadata={"help": "Directory to save checkpoints"}
    )
    checkpoint_file: str = field(
        default=None,
        metadata={"help": "File to save checkpoints"}
    )
    checkpoint_interval: int = field(
        default=1000,
        metadata={"help": "Interval to save the checkpoint"}
    )
    logger_type: str = field(
        default='wandb',
        metadata={"help": "Name of the logger to use. Can be wandb/tensorboard"}
    )
    logger_name: str = field(
        default='default',
        metadata={"help": "Name of the logger"}
    )
    log_interval: int = field(
        default=1,
        metadata={"help": "Interval to log the training progress"}
    )

