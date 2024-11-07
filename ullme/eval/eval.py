import os
import json
from functools import partial
from pathlib import Path
from typing import List, Union
import datasets.config
from datasets import disable_caching
import numpy as np
import torch
import torch.nn as nn
import mteb
from tqdm import tqdm

from ullme.eval.constants import MTEB_DS_TO_PROMPT, QUICK_EVAL, LANG_TO_CODES, MULTILINGUAL_DS_TO_PROMPT
from ullme.eval.wrapped_hf_model import WrappedHFModel
from ullme.model.ullme import WrappedULLME

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


def eval_mteb_dataset(
        dataset_name: str,
        instruction: str,
        langs: List[str],
        model: nn.Module,
        output_folder: str='results',
        batch_size: int=32,
        max_length: int=512,
        aggregation=np.mean,
):  
    task = mteb.get_task(task_name=dataset_name, languages=langs)
    task_name = task.metadata.name
    langs = task.languages

    if 'test' in task.metadata.eval_splits:
        eval_splits = ['test']
    elif 'test.full' in task.metadata.eval_splits:
        eval_splits = ['test.full']
    elif 'test_expert' in task.metadata.eval_splits:
        eval_splits = ['test_expert']
    elif 'validation' in task.metadata.eval_splits:
        eval_splits = ['validation']
    elif 'dev' in task.metadata.eval_splits:
        eval_splits = ['dev']
    elif 'train' in task.metadata.eval_splits:
        eval_splits = ['train']
    else:
        raise ValueError(f"Could not find any of the evaluation splits in the task {task_name}")

    evaluator = mteb.MTEB(tasks=[task],)
    mteb_model_meta = model.mteb_model_meta
    model_path_name = mteb_model_meta.model_name_as_path()
    model_revision = mteb_model_meta.revision
    results_path = Path(output_folder) / model_path_name / model_revision / f'{task_name}.json'
    if results_path.exists():
        print(f"Loading results from {results_path}")
        results = mteb.MTEBResults.from_disk(results_path)
    else:
        print(f"Running evaluation for {task_name}")
        results = evaluator.run(
            model=model,
            verbosity=2,
            output_folder=output_folder,
            eval_splits=eval_splits,
            overwrite_results=True,
            trust_remote_code=True,
            encode_kwargs={'instruction': instruction, 'batch_size': batch_size, 'max_length': max_length},
        ) 
        results = results[0]
    all_results = {}
    for lang in langs:
        try:
            test_result_lang = results.get_score(
                languages=[lang],
                aggregation=aggregation,
            )
        except Exception as e:
            print(f"Error in getting the score for {dataset_name} in {lang}: {e}")
            breakpoint()
        all_results[lang] = test_result_lang

    print(f"Results for {dataset_name}: {all_results}")
    return all_results


def eval_mteb(
        model: nn.Module,
        output_folder: str='results',
        batch_size: int=32,
        max_length: int=512,
        is_quick_run: bool=False,
):  
    langs = ['eng']
    results = {
        'avg': {}
    }
    for task in MTEB_DS_TO_PROMPT.keys():
        dataset_in_task = MTEB_DS_TO_PROMPT[task]
        all_task_results = []
        for dataset in dataset_in_task.keys():
            if is_quick_run and dataset not in QUICK_EVAL:
                continue
            main_metric = eval_mteb_dataset(
                dataset_name=dataset,
                instruction=dataset_in_task[dataset],
                langs=langs,
                model=model,
                output_folder=f"{output_folder}/{langs[0]}",
                batch_size=batch_size,
                max_length=max_length,
            )
            results[dataset] = main_metric['eng']
            all_task_results.append(main_metric['eng'])
        
        if len(all_task_results) != 0:
            avg_task_result = sum(all_task_results) / len(all_task_results)
            results['avg'][task] = avg_task_result
    
    all_results = [r for r in results.values() if isinstance(r, float)]
    avg_result = sum(all_results) / len(all_results)
    results['avg']['all_tasks'] = avg_result
    print(f"Results for all tasks: {results}")

    with open(os.path.join(output_folder, 'mteb_all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    return results


def eval_multilingual(
        model: nn.Module,
        langs: List[str] = 'all',
        output_folder: str='results',
        batch_size: int=32,
        max_length: int=512,
        is_quick_run: bool=False,
):  
    if isinstance(langs, list):
        name = '_'.join(langs)
    if langs == 'all':
        langs = LANG_TO_CODES.keys()
        name = 'all'
    
    all_results = {
        'avg': {},
    }
    _all_results = []
    for lang in langs:
        datasets_in_lang = MULTILINGUAL_DS_TO_PROMPT[lang]
        lang_codes = LANG_TO_CODES[lang]
        all_results[lang] = {}
        for dataset in datasets_in_lang.keys():
            if is_quick_run and dataset not in QUICK_EVAL:
                continue
            main_metric = eval_mteb_dataset(
                dataset_name=dataset,
                instruction=datasets_in_lang[dataset],
                langs=lang_codes,
                model=model,
                output_folder=f"{output_folder}/{lang_codes[0]}",
                batch_size=batch_size,
                max_length=max_length,
            )
            lang_results = [main_metric[lang_code] for lang_code in lang_codes if lang_code in main_metric]
            assert len(lang_results) > 0, f"No results for {dataset} in {lang_codes}"
            lang_results = sum(lang_results) / len(lang_results)
            all_results[lang][dataset] = lang_results
            _all_results.append(lang_results)
        if len(all_results[lang]) != 0:
            lang_avg_result = sum([x for x in all_results[lang].values()]) / len(all_results[lang])
            all_results['avg'][lang] = lang_avg_result
    
    all_results['avg']['all_tasks'] = sum(_all_results) / len(_all_results)
    print(f"Results for all languages: {all_results}")

    with open(os.path.join(output_folder, f'{name}_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    return all_results


if __name__=='__main__':
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    import argparse
    from transformers import HfArgumentParser
    from ullme.args import DataArguments, ModelArguments, TrainingArguments

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--model_checkpoint", type=str, default=None, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--output_folder", type=str, default='results', help="Path to the output folder"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max length of the input sequence",
    )
    parser.add_argument(
        "--is_quick_run", action='store_true', help="Whether to run the quick evaluation",
    )
    parser.add_argument(
        "--langs", type=str, nargs='+', default='all', help="Languages to evaluate",
    )
    parser.add_argument(
        "--dataname", type=str, default=None, help="Name of the dataset to evaluate",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=8, help="Number of GPUs to use",
    )

    args = parser.parse_args()
    
    print(f"Evaluating model {args.model_name_or_path} on {args.langs} languages of {args.dataname} dataset")
    if args.model_checkpoint is not None:
        config_file = args.model_name_or_path
        model_checkpoint = args.model_checkpoint
        hf_parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
        print(f"Loading yaml config {config_file}")
        data_args, model_args, training_args = hf_parser.parse_yaml_file(yaml_file=config_file)
        model = WrappedULLME(
            encoder_name_or_path=model_args.encoder_name_or_path,
            encoder_backbone_type=model_args.encoder_backbone_type,
            pooling_method='mean',
            encoder_lora_name=model_args.encoder_lora_name,
            encoder_lora_target_modules=model_args.encoder_lora_target_modules,
            loar_r=model_args.loar_r,
            lora_alpha=model_args.lora_alpha,
            dropout=model_args.dropout,
            attn_implementation=model_args.attn_implementation,
            model_dtype=torch.bfloat16,
            model_revision=training_args.model_revision,
            model_checkpoint=model_checkpoint,
            num_gpus=args.num_gpus,
        )
    else:
        model = WrappedHFModel(
            model_name_or_path=args.model_name_or_path,
            num_gpus=8,
        )
    batch_size = args.batch_size * model.num_gpus if model.num_gpus > 0 else args.batch_size

    if args.dataname is None and (args.langs=='all' or 'all' in args.langs):
        print("Evaluating on all MTEB datasets")
        multilingual_results = eval_multilingual(
            model=model,
            output_folder=args.output_folder,
            batch_size=batch_size,
            max_length=args.max_length,
            is_quick_run=args.is_quick_run
        )
        results = eval_mteb(
            model=model,
            output_folder=args.output_folder,
            batch_size=batch_size,
            max_length=args.max_length,
            is_quick_run=args.is_quick_run,
        )
    elif args.dataname=='all_mteb_en':
        print("Evaluating on all english MTEB datasets")
        results = eval_mteb(
            model=model,
            output_folder=args.output_folder,
            batch_size=batch_size,
            max_length=args.max_length,
            is_quick_run=args.is_quick_run,
        )
    elif args.dataname=='exclude_mteb':
        print("Evaluating on all MTEB datasets excluding the english ones")
        multilingual_results = eval_multilingual(
            model=model,
            output_folder=args.output_folder,
            batch_size=batch_size,
            max_length=args.max_length,
            is_quick_run=args.is_quick_run
        )
    else:
        langs = args.langs
        dataname = args.dataname
        for lang in langs:
            instruction = None
            if lang=='en':
                try:
                    prompt_dicts = {}
                    for task in MTEB_DS_TO_PROMPT.keys():
                        prompt_dicts.update(MTEB_DS_TO_PROMPT[task])
                    instruction = prompt_dicts[dataname]
                except:
                    instruction = MULTILINGUAL_DS_TO_PROMPT.get(lang, {}).get(dataname, None)
            else:
                instruction = MULTILINGUAL_DS_TO_PROMPT.get(lang, {}).get(dataname, None)
            if instruction is None:
                print(f"Could not find the instruction for {dataname} in {lang}")
                continue
            langs = LANG_TO_CODES[lang]
            eval_mteb_dataset(
                dataset_name=dataname,
                instruction=instruction,
                langs=langs,
                model=model,
                output_folder=f"{args.output_folder}/{langs[0]}",
                batch_size=batch_size,
                max_length=args.max_length
            )

        
            
        
