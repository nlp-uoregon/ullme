import bisect
from collections import defaultdict
import math
import os
import random
from typing import Dict, Tuple
import einops
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset
import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding
import datasets

from ullme.data_modules.constants import DATA, PRETRAINING_RECONSTRUCT, PRETRAINING_PASSAGE2QUERY, PRETRAINING_QUERY2PASSAGE


class ULLMEDataset(Dataset):
    def __init__(
            self, 
            data_name: str,
            number_training_samples: int=1_000_000,
            neg_per_sample: int=1,
            pos_per_sample: int=1,
            seed: int=777,
            ):
        super().__init__()
        self.data_name = data_name
        self.data_path = DATA[data_name]['data_path']
        self.instruction = DATA[data_name]['instruction']
        self.enable_cross_batch_negative_sampling = DATA[data_name].get('enable_cross_batch_negative_sampling', True)
        self.number_training_samples = number_training_samples
        self.neg_per_sample = neg_per_sample
        self.pos_per_sample = pos_per_sample
        self.seed = seed
        print(f"Seed: {self.seed}")
        self.rng = random.Random(self.seed)

        self.data, self.cluster = self.get_data()

    def get_data(self):
        print(f"Loading data from {self.data_name}")
        number_data = self.number_training_samples
        dataset = datasets.load_dataset(self.data_name, split='train')
        max_num_worker_suggest = 1
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            print("Failed to get number of CPU cores, using default value 1")

        if len(dataset) > number_data:
            cluster = set(dataset['cluster'])
            example_per_cluster = math.ceil(number_data / len(cluster))
            cluster_with_id = dataset.map(lambda example, idx: {'id': idx, 'cluster': example['cluster']}, with_indices=True, num_proc=max_num_worker_suggest, remove_columns=dataset.column_names, load_from_cache_file=False)
            cluster_with_id = cluster_with_id.to_pandas()
            # group by cluster
            cluster_with_id = cluster_with_id.groupby('cluster')['id'].apply(list).reset_index()
            cluster_with_id = cluster_with_id.to_dict(orient='records')
            # sort by the number of examples in the cluster
            cluster_with_id.sort(key=lambda x: len(x['id']))

            # get the examples
            selected_index = []
            for clus in cluster_with_id:
                in_cluster_index = clus['id']
                in_cluster_index.sort()
                in_cluster_index = self.rng.sample(in_cluster_index, min(len(in_cluster_index), example_per_cluster))
                selected_index.extend(in_cluster_index)

            if len(selected_index) < number_data:
                all_data_index = list(range(len(dataset)))
                self.rng.shuffle(all_data_index)
                for idx in all_data_index:
                    if idx not in selected_index:
                        selected_index.append(idx)
                    if len(selected_index) >= number_data:
                        break
            selected_index.sort()
            dataset = dataset.select(selected_index)

        print(f"Assigning cluster to each example for the dataset {self.data_name} of size {len(dataset)}...")
        cluster = dataset.map(lambda example, idx: {'cluster': example['cluster'], 'id': idx}, with_indices=True, 
                                      num_proc=max_num_worker_suggest, remove_columns=dataset.column_names, load_from_cache_file=False)
        # group by cluster
        cluster = cluster.to_pandas()
        cluster = cluster.groupby('cluster')['id'].apply(list).reset_index()
        cluster = cluster.to_dict(orient='records')
        cluster.sort(key=lambda x: x['cluster'])
        cluster = {clus['cluster']: sorted(clus['id']) for clus in cluster}
            
        return dataset, cluster

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.data[index]
        pos = self.rng.sample(example['positive'], min(len(example['positive']), self.pos_per_sample))
        neg = self.rng.sample(example['negative'], min(len(example['negative']), self.neg_per_sample))
        assert len(pos) > 0, "At least one positive example per sample, got {} in idx {}. Please check the data {}".format(example, index, self.data_name)
        assert len(neg) > 0, "At least one negative example per sample, got {} in idx {}. Please check the data {}".format(example, index, self.data_name)

        if self.data_name in PRETRAINING_PASSAGE2QUERY:
            alignment_instruction = "Please write a query that can be used to retrieve above passage."
            is_passage2query = True
        elif self.data_name in PRETRAINING_QUERY2PASSAGE:
            alignment_instruction = "Please write a passage that can be used to answer above query."
            is_passage2query = False
        else:
            alignment_instruction = "Please write a query that can be used to retrieve above passage."
            is_passage2query = True

        return {
            'query_label': index,
            'query': example['query'], # str
            'positive': pos, # list of str
            'negative': neg, # list of str
            'instruction': self.instruction,
            'alignment_instruction': alignment_instruction,
            'enable_cross_batch_negative_sampling': self.enable_cross_batch_negative_sampling,
        }


class ConcatULLMEDataset(ConcatDataset):
    """
    An extension of ConcatDataset that guarantees that each example has a unique query_label.
    """
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        example =  self.datasets[dataset_idx][sample_idx]

        # Update the query_label to be unique across all datasets
        example['query_label'] = idx
        return example


class ULLMECollator:
    def __init__(
            self, 
            max_length: int, 
            tokenizer: PreTrainedTokenizer, 
            ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.formater = "{instruction}\n{example}"
        self.candidate_formater = "{instruction}\nCandidate:\n{example}"
        self.gen_example_format = "{instruction}. Query:" + "\n{query}\n" + "\n{response}\n"

    def tokenize_example(
            self,
            example: str,
            is_gen: bool=False,
            is_query: bool=False,
            instruction: str="",
            ) -> BatchEncoding:
        if len(example) == 0:
            print('example:', example)
        if is_gen:
            example = example
        elif is_query:
            example = self.formater.format(instruction=instruction, example=example)
        else:
            example = self.formater.format(instruction=instruction, example=example)
        model_inputs = self.tokenizer(
            example,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        return model_inputs

    def __call__(self, batch):
        batch_size = len(batch)

        query_labels = [example['query_label'] for example in batch]
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        enable_cross_batch_negative_sampling = all([example['enable_cross_batch_negative_sampling'] for example in batch])

        min_pos_per_sample = min([len(example['positive']) for example in batch])
        min_neg_per_sample = min([len(example['negative']) for example in batch])
        assert min_pos_per_sample > 0, "At least one positive example per sample"
        assert min_neg_per_sample > 0, "At least one negative example per sample"

        batch_query = []
        batch_pos = []
        batch_neg = []
        batch_choice = []
        batch_reject = []
        for example in batch:
            q = example['query']
            pos = example['positive']
            neg = example['negative']
            instruction = example['instruction']
            alignment_instruction = example['alignment_instruction']
            
            _q = self.tokenize_example(q, is_query=True, instruction=instruction)
            batch_query.append(_q)
            neg = random.sample(neg, min_neg_per_sample)
            for example in neg:
                reject = self.gen_example_format.format(instruction=alignment_instruction, query=q, response=example)
                n = self.tokenize_example(example, is_query=False, instruction=instruction)
                reject = self.tokenize_example(reject, is_gen=True)
                batch_reject.append(reject)
                batch_neg.append(n)
            pos = random.sample(pos, min_pos_per_sample)
            for example in pos:
                choice = self.gen_example_format.format(instruction=alignment_instruction, query=q, response=example)
                choice = self.tokenize_example(choice, is_gen=True)
                p = self.tokenize_example(example, is_query=False, instruction=instruction)
                batch_choice.append(choice)
                batch_pos.append(p)
        
        cons_batch = batch_query + batch_pos + batch_neg
        in_batch = self.tokenizer.pad(cons_batch, return_tensors='pt')
        cons_ids = einops.rearrange(in_batch['input_ids'], '(b n) l -> b n l', b=batch_size) # (batch_size, 1 + #p + #n, in_seq_len)
        cons_attn_mask = einops.rearrange(in_batch['attention_mask'], '(b n) l -> b n l', b=batch_size) # (batch_size, 1 + #p + #n, in_seq_len)

        gen_batch = batch_choice + batch_reject
        gen_batch = self.tokenizer.pad(gen_batch, return_tensors='pt') 
        gen_batch_ids = einops.rearrange(gen_batch['input_ids'], '(b n) l -> b n l', b=batch_size) # (batch_size, #p + #n, gen_seq_len)
        gen_batch_attn_mask = einops.rearrange(gen_batch['attention_mask'], '(b n) l -> b n l', b=batch_size)
        gen_batch_labels = gen_batch_ids.clone()
        gen_batch_labels[gen_batch_attn_mask == 0] = -100

        return {
            'enable_cross_batch_negative_sampling': enable_cross_batch_negative_sampling,
            'query_labels': query_labels, # (batch_size,)
            'min_pos_per_sample': min_pos_per_sample,
            'cons_input_ids': cons_ids, # (batch_size, 1 + #p + #n, in_seq_len)
            'cons_attention_mask': cons_attn_mask, # (batch_size, 1 + #p + #n, in_seq_len)
            'gen_input_ids': gen_batch_ids, # (batch_size, #p + #n, gen_seq_len)
            'gen_attention_mask': gen_batch_attn_mask, # (batch_size, #p + #n, gen_seq_len)
            'gen_labels': gen_batch_labels, # (batch_size, #p + #n, gen_seq_len)
        }
            
