# *ULLME: A Unified Framework for Large Language Model Embeddings with Generation-Augmented Learning*

<!-- [![EMNLP](https://img.shields.io/badge/EMNLP-2024-b31b1b.svg)](https://arxiv.org/) -->

[![License](https://img.shields.io/badge/License-Apache2.0-FFD4E.svg)](https://huggingface.co/Hieuman/GenC-LlaMa)
[![HF Link](https://img.shields.io/badge/HF%20Models-ULLME-FFD21E.svg)](https://huggingface.co/Hieuman/GenC-LlaMa)

ULLME is a flexible, plug-and-play implementation that enables bidirectional attention across various LLMs and supports a range of fine-tuning strategies to learn passage embeddings.

## Installation
ULLME can be easily installed via one of the following methods:

### Using pip
Coming soon!

### From source
```bash
git clone https://github.com/nlp-uoregon/ullme.git
cd ullme
pip install -e .
# if you using flash-attention-2 (this is the default for ullme)
pip install flash-attn --no-build-isolation
```

## Usage
ULLME offers follwing main features: 

### Enabling Bidirectional Attention
ULLME can support enhancing HuggingFace models by adding support for bidirectional processing in decoder-only Large Language Models (LLMs), as well as sequence encoding and pooling operations. 
```python
from ullme.models import ULLME

model = ULLME(
    model_name_or_path="mistralai/Mistral-7B-v0.1",
    model_backbone_type="mistral",
    lora_name="ullme-mistral",
    loar_r=16,
    lora_alpha=32,
    )
input_sentence = "This a example sentence."
model_inputs = model.tokenizer(
    [input_sentence], 
    return_tensors='pt'
    )
model_output = model(
    input_ids=model_inputs['input_ids'],
    attention_mask=model_inputs['attention_mask'],
    is_generate=False
    )
```
The ULLME's returned model is a PyTorch object, providing users with the flexibility to integrate it into various frameworks or pipelines. By default the ULLME model uses the `mean` pooling strategy. The ```is_generate``` parameter plays a crucial role in controlling the attention mechanism: when set to ```False```, the model employs bidirectional attention, optimizing it for dense retrieval tasks, while ```True``` reverts the model to causal attention, mimicking the standard Hugging Face Transformer model output.

### Fine-tuning Strategies
Our ULLME framework supports multiple fine-tuning strategies
```python
from ullme.trainer import GradCacheTrainer

trainer = GradCacheTrainer(
    con_loss_type='NTXentLoss',
    gen_loss_type='sigmoid', # 'sft'
    use_kl_loss=True
)
trainer.fit_epoch(
    model=model,
    train_loader=train_dataloader,
)
```
#### Contrastive Learning (CL)
ULLME enables efficient and effective CL. It comes equipped with a range of advanced features designed to enhance the CL process and optimize performance, such as GradCache, cross-devices contrastive loss computation, miners, ... Note that, ULLME enables CL by default. 

#### Generative manner Fine-tuning
ULLME not only supports Contrastive Learning (CL) but also enables Supervised Fine-Tuning (SFT) and provides a range of preference loss functions to further enhance model performance. The loss functions that can be easily selected through the `gen_loss_type` argument, currently support `sft`, `sigmoid`(i.e., DPO), `kto`, `ipo`.

#### Alignment between Generation-based score and Representation-based score.
In ULLME, we also introduce a novel fine-tuning strategy, GRL, that explicitly aligns the model's understanding of relevance in both embedding and generation spaces through a Kullback-Leibler (KL) divergence loss. You can enbale this by set `use_kl_loss=True`. 

### Evaluation on MTEB
```python
from ullme.models import WrappedULLME
from ullme.eval import eval_mteb_dataset

model = WrappedULLME(
    model_name_or_path="mistralai/Mistral-7B-v0.1",
    model_backbone_type="mistral",
    lora_name="ullme-mistral",
    loar_r=16,
    lora_alpha=32,
    model_checkpoint="path/to/your/checkpoint"
    )
eval_result = eval_mteb_dataset(
    model=model,
    dataset_name='MSMARCO',
    langs=['eng'],
    )
>> {'eng': 35.8}
```
ULLME streamlines the evaluation process by integrating direct support for evaluating LLM-based text embedding models over MTEB. ULLME allows users to select specific datasets and language subsets for evaluation through parameters `dataset_name` and `langs`.


## Model List

We publish three fine-tuned model using GRL on three popular LLMs: [Meta-Llama-3-8B](https://huggingface.co/Hieuman/GenC-LlaMa); [Mistral-2-7B](https://huggingface.co/Hieuman/GenC-Mistral); [Phi-1.5B](https://huggingface.co/Hieuman/GenC-Phi1.5)


## Finetuning CLI 

To finetune the Meta-Llama-3-8B model, run the following command:

```bash
python -m genc.main \
    --config_file scripts/configs/llama.yaml \
    --nodes 4 \
    --devices 1 \
    --gc_chunk_size 4 \
    --output_dir output/ullme-grl-llam3
```


## Evaluation CLI
To evaluate the model on the MTEB benchmark, run the following command:
```bash
python -m eval.eval_mteb \
    --config_file scripts/configs/llama.yaml \
    --output_dir output/ullme-grl-llam3/checkpoint.ckpt
```

## Bugs or questions?
If you have any questions about the code, feel free to open an issue on the GitHub repository.

