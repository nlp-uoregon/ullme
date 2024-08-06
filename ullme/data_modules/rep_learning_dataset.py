import bisect
import random
from typing import Dict, Tuple
import einops
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset
from transformers import PreTrainedTokenizer, BatchEncoding
import datasets


class RepLearningDataset(Dataset):
    def __init__(
            self,
            data_name: str,
            data_path: str,
            instruction: str="",
            enable_cross_batch_negative_sampling: bool=True,
            number_training_samples: int=1_000_000,
            neg_per_sample: int=1,
            pos_per_sample: int=1,
            seed: int=777,
            ):
        super().__init__()
        self.data_name = data_name
        self.instruction = instruction
        self.enable_cross_batch_negative_sampling = enable_cross_batch_negative_sampling
        self.number_training_samples = number_training_samples
        self.neg_per_sample = neg_per_sample
        self.pos_per_sample = pos_per_sample
        self.seed = seed
        try:
            data = datasets.load_dataset(data_name, split='train')
        except:
            assert data_path is not None, "Please provide the path to the dataset to load the data locally."
            data = datasets.load_dataset('json', data_files=data_path, split='train')
        if len(data) > number_training_samples:
            data = data.train_test_split(train_size=number_training_samples, seed=seed, shuffle=True)['train']
        self.data = data

        self.rng = random.Random(self.seed)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        pos = self.rng.sample(example['positive'], min(len(example['positive']), self.pos_per_sample))
        neg = self.rng.sample(example['negative'], min(len(example['negative']), self.neg_per_sample))
        assert len(pos) > 0, "At least one positive example per sample. Please check the data {}".format(self.data_name)
        assert len(neg) > 0, "At least one negative example per sample. Please check the data {}".format(self.data_name)
        return {
            'query_label': idx,
            'query': example['query'], # str
            'positive': pos, # list of str
            'negative': neg, # list of str
            'instruction': self.instruction,
            'enable_cross_batch_negative_sampling': self.enable_cross_batch_negative_sampling,
        }
    

class ConcatRepLearningDataset(ConcatDataset):
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


class RepLearningCollator:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            special_tokens: Dict[str, str],
            joint_training: bool=False,
            max_seq_length: int=512,
            label_pad_token_id: int=-100,
            ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_pad_token_id = label_pad_token_id
        self.joint_training = joint_training

        bos = special_tokens.get("bos", "")
        user_bos = special_tokens.get("user_bos", "")
        assistant_bos = special_tokens.get("assistant_bos", "")
        eos = special_tokens.get("eos", "")
        eot = special_tokens.get("eot", "")
        self.query_prompt = bos + user_bos + "{instruction}." 
        self.query_format = bos + user_bos + "{instruction}." + "\n{example}" + eot + eos
        self.candidate_prompt = bos + user_bos + "{instruction}. Candidate:" + "\n"
        self.candidate_format = bos + user_bos + "{instruction}. Candidate:" + "\n" + "{example}" + eot + eos
        self.gen_example_format = bos + user_bos + "{instruction}. Query:" + "\n{query}\n" + eot + assistant_bos + "\n{response}\n" + eot + eos
        
    def tokenize_example(
            self,
            example: str,
            is_query: bool,
            instruction: str="",
            ) -> BatchEncoding:
        if is_query:
            prompt = self.query_prompt.format(instruction=instruction)
            example = self.query_format.format(instruction=instruction, example=example)
        else:
            prompt = self.candidate_prompt.format(instruction=instruction)
            example = self.candidate_format.format(instruction=instruction, example=example)
        model_inputs = self.tokenizer(
            example,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors=None,
            add_special_tokens=False, # already added in the format
        )
        # find the prompt length
        prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        try:
            assert len(model_inputs['input_ids']) > prompt_length, f"Input length is less than prompt length: {len(model_inputs['input_ids'])} <= {prompt_length}."
            model_inputs['prompt_length'] = prompt_length
        except:
            print('model_inputs:', model_inputs)
            print('example:', example)
            print('prompt:', prompt)

        return model_inputs
    
    def tokenize_gen_example(
            self,
            query: str,
            response: str,
            instruction: str="",
            ) -> BatchEncoding:
        choice_example = self.gen_example_format.format(instruction=instruction, query=query, response=response)
        model_inputs = self.tokenizer(
            choice_example,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors=None,
            add_special_tokens=False, # already added in the format
        )
        return model_inputs
    
    def __call__(self, batch):
        batch_size = len(batch)
        
        query_labels = [example['query_label'] for example in batch]
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        # if all the examples in the batch have enable_cross_batch_negative_sampling==True, then the batch has enable_cross_batch_negative_sampling==True
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
        for i, example in enumerate(batch):
            q = example['query']
            pos = example['positive']
            neg = example['negative']
            instruction = example['instruction']
            q = [instruction, q]
            if len(pos) > min_pos_per_sample:
                pos = random.sample(pos, min_pos_per_sample) 
                pos = [[instruction, p] for p in pos]
            if len(neg) > min_neg_per_sample:
                neg = random.sample(neg, min_neg_per_sample)
                neg = [[instruction, n] for n in neg]
            batch_query.append(q)
            batch_pos.extend(pos)
            batch_neg.extend(neg)
            batch_choice.extend([self.tokenize_gen_example(query=q[1], instruction=q[0], response=p) for p in pos])
            batch_reject.extend([self.tokenize_gen_example(query=q[1], instruction=q[0], response=n) for n in neg])
        
        batch_query = [self.tokenize_example(example=x[1], is_query=True, instruction=x[0]) for x in batch_query]
        batch_pos = [self.tokenize_example(example=x[1], is_query=False, instruction=x[0]) for x in batch_pos]
        batch_neg = [self.tokenize_example(example=x[1], is_query=False, instruction=x[0]) for x in batch_neg]
        len_q = len(batch_query)
        len_p = len(batch_pos)
        len_n = len(batch_neg)
        assert len_p == len(batch_choice), "The number of positive examples and choices should be the same."
        assert len_n == len(batch_reject), "The number of negative examples and rejects should be the same."

        batch = batch_query + batch_pos + batch_neg
        batch = self.tokenizer.pad(batch, return_tensors='pt', pad_to_multiple_of=8) 
        gen_batch = batch_choice + batch_reject
        gen_batch = self.tokenizer.pad(gen_batch, return_tensors='pt', pad_to_multiple_of=8)

        query_input_ids = batch['input_ids'][:len_q]
        query_attention_mask = batch['attention_mask'][:len_q]
        query_prompt_length = batch['prompt_length'][:len_q]

        pos_input_ids = batch['input_ids'][len_q:len_q+len_p]
        pos_input_ids = einops.rearrange(pos_input_ids, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)
        pos_attention_mask = batch['attention_mask'][len_q:len_q+len_p]
        pos_attention_mask = einops.rearrange(pos_attention_mask, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)
        pos_prompt_length = batch['prompt_length'][len_q:len_q+len_p]
        pos_prompt_length = einops.rearrange(pos_prompt_length, '(b n) -> b n', b=batch_size, n=min_pos_per_sample)

        neg_input_ids = batch['input_ids'][len_q+len_p:]
        neg_input_ids = einops.rearrange(neg_input_ids, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)
        neg_attention_mask = batch['attention_mask'][len_q+len_p:]
        neg_attention_mask = einops.rearrange(neg_attention_mask, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)
        neg_prompt_length = batch['prompt_length'][len_q+len_p:]
        neg_prompt_length = einops.rearrange(neg_prompt_length, '(b n) -> b n', b=batch_size, n=min_neg_per_sample)

        choice_input_ids = gen_batch['input_ids'][:len_p]
        choice_input_ids = einops.rearrange(choice_input_ids, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)
        choice_attention_mask = gen_batch['attention_mask'][:len_p]
        choice_attention_mask = einops.rearrange(choice_attention_mask, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)

        reject_input_ids = gen_batch['input_ids'][len_p:]
        reject_input_ids = einops.rearrange(reject_input_ids, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)
        reject_attention_mask = gen_batch['attention_mask'][len_p:]
        reject_attention_mask = einops.rearrange(reject_attention_mask, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)

        return {
            'enable_cross_batch_negative_sampling': enable_cross_batch_negative_sampling,
            'query_labels': query_labels, # (batch_size,)
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'query_prompt_length': query_prompt_length,
            'pos_input_ids': pos_input_ids,
            'pos_attention_mask': pos_attention_mask,
            'pos_prompt_length': pos_prompt_length,
            'neg_input_ids': neg_input_ids,
            'neg_attention_mask': neg_attention_mask,
            'neg_prompt_length': neg_prompt_length,
            'choice_input_ids': choice_input_ids,
            'choice_attention_mask': choice_attention_mask,
            'reject_input_ids': reject_input_ids,
            'reject_attention_mask': reject_attention_mask,
        }

            
