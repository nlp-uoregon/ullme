import os
from typing import Dict, List
import torch
from torch.utils.data import DataLoader, Sampler
import lightning as L
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ullme.data_modules.rep_learning_dataset import RepLearningDataset, ConcatRepLearningDataset, RepLearningCollator
from ullme.special_tokens import SPECIAL_TOKENS


class ConcatedDataSampler(Sampler):
    """
    A sampler for ConcatDataset that gurantees that each batch will comes from same dataset.
    """ 
    def __init__(
            self,
            each_data_sizes: List[int],
            global_batch_size: int,
            shuffle: bool = True,
            num_replicas: int = 1,
            rank: int = 0,
            seed: int = 777,
            drop_last: bool = False,
            ):
        """
        :param each_data_sizes: list of sizes of each dataset
        :param global_batch_size: global batch size
        :param shuffle: whether to shuffle the indices
        :param num_replicas: number of replicas i.e. number of gpus
        :param rank: rank of the current gpu
        :param seed: seed for random number generator
        :param drop_last: whether to drop the last batch if it is incomplete
        """
        self.each_data_sizes = each_data_sizes
        self.batch_size = global_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.indices = self.set_indices()
        self.num_samples = len(self.indices) // self.num_replicas

    def set_indices(self):
        """
        Set the indices for the sampler based on the each_data_sizes and global_batch_size to guarantee that each batch comes from the same dataset.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            indices = [torch.randperm(n, generator=g).tolist() for n in self.each_data_sizes]
        else:
            indices = [list(range(n)) for n in self.each_data_sizes]

        # increase the indices by the offset
        for i in range(len(self.each_data_sizes)):
            indices[i] = [idx + sum(self.each_data_sizes[:i]) for idx in indices[i]]
        batched_indices = []
        for data_indices in indices:
            _batched_indices = list(torch.split(torch.tensor(data_indices), self.batch_size))
            batched_indices.append(_batched_indices)
        
        # Create separate batches from the remaining samples
        incomplete_indices = []
        for b in batched_indices:
            if len(b[-1]) < self.batch_size:
                incomplete_indices.append(b.pop())
        
        if self.drop_last is False and len(incomplete_indices) != 0:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=g).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(incomplete_indices, self.batch_size))
            if len(mixed_batches[-1]) < self.batch_size:
                mixed_batches.pop()
            batched_indices = sum(batched_indices, []) + mixed_batches
        else:
            batched_indices = sum(batched_indices, [])

        if self.shuffle:
            # Shuffle the batches 
            order = torch.randperm(len(batched_indices), generator=g).tolist()
        else:
            order = list(range(len(batched_indices)))
                         
        indices = []
        for batch_idx in order:
            indices.extend([int(i) for i in batched_indices[batch_idx]])
        return indices

    def __iter__(self):
        # subsample
        indices = self.indices[self.rank:len(self.indices):self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class RepLearningDataModule(L.LightningDataModule):
    def __init__(
            self, 
            metadata: List[Dict[str, str]], 
            num_workers: int = 4,
            seed: int = 777
            ):
        """
        :param metadata: list of metadata dictionaries for each dataset in the form of 
        {'name': 'dataset_name', 'path': 'path_to_dataset', 'instruction': 'instruction_for_dataset', 'enable_cross_batch_negative_sampling': True/False}
        """
        super().__init__()
        self.metadata = metadata
        # check if the metadata is valid
        for meta in self.metadata:
            assert 'name' in meta, f"Metadata must contain 'name' key. Got: {meta}"
        self.metadata.sort(key=lambda x: x['name'])
        self.num_workers = num_workers
        self.seed = seed
    
    def connect(
            self,
            world_size: int = 1,
            global_rank: int = 0,
            tokenizer: PreTrainedTokenizer = None, 
            special_tokens_set: str = 't5',
            global_batch_size: int = 32,
            max_seq_length: int = 512,
            number_training_samples: int = 1_000_000,
            neg_per_sample: int = 1,
            pos_per_sample: int = 1,
            ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.special_tokens_set = SPECIAL_TOKENS[special_tokens_set]
        self.global_batch_size = global_batch_size
        self.max_seq_length = max_seq_length
        self.number_training_samples = number_training_samples
        self.neg_per_sample = neg_per_sample
        self.pos_per_sample = pos_per_sample
        self.batch_size = self.global_batch_size // self.world_size
        assert self.global_batch_size % self.world_size == 0, "Global batch size must be divisible by world size."
        assert self.batch_size > 0, "Batch size must be greater than 0. i.e. world_size must be less than or equal to global_batch_size"
    
    def set_epoch(self, epoch: int) -> None:
        self.seed = self.seed + epoch

    def setup(self, stage: str='') -> None:
        train_datasets = []
        for meta in self.metadata:
            ds = RepLearningDataset(
                data_name=meta['name'],
                data_path=meta.get('path', None),
                instruction=meta.get('instruction', ""),
                enable_cross_batch_negative_sampling=meta.get('enable_cross_batch_negative_sampling', False),
                number_training_samples=self.number_training_samples,
                neg_per_sample=self.neg_per_sample,
                pos_per_sample=self.pos_per_sample,
                seed=self.seed,
            )
            if len(ds) > 0:
                train_datasets.append(ds)
                if self.global_rank == 0:
                    print(f"Loaded {meta['name']} dataset with {len(ds)} samples.")
            else:
                print(f"Skipping {meta['name']} dataset as it has no samples.")
        assert len(train_datasets) > 0, f"No datasets loaded. Please check the data: {self.metadata}"
        self.train_ds = ConcatRepLearningDataset(train_datasets)
    
    def train_dataloader(self) -> DataLoader:
        max_num_worker_suggest = 1
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
        num_workers = min(self.num_workers, max_num_worker_suggest)
        collator = RepLearningCollator(
            tokenizer=self.tokenizer,
            special_tokens=self.special_tokens_set,
            max_seq_length=self.max_seq_length,
            label_pad_token_id=-100
        )
        each_data_sizes = [len(dataset) for dataset in self.train_ds.datasets]
        sampler = ConcatedDataSampler(
            each_data_sizes=each_data_sizes,
            global_batch_size=self.global_batch_size,
            shuffle=True,
            num_replicas=self.world_size,
            rank=self.global_rank,
            seed=self.seed,
            drop_last=False,
        )

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collator,
        )


