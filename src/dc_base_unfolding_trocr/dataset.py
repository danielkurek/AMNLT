import torch
import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from lightning import LightningDataModule

from AMNLT.utils.dc_base_unfolding_utils.dataset import CTCDataset
from AMNLT.utils.dc_base_unfolding_utils.data_preprocessing import ctc_batch_preparation
from ..ExperimentCommonConfig import ExperimentCommonConfig
from ..utils.concat_cycle_sampler import ConcatDatasetCycleSampler

from pathlib import Path
import json

class MergedDatasets(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
    
    def get_mx_hw(self):
        hw_list = [x.get_mx_hw() for x in self.datasets]
        return max([x[0] for x in hw_list]), max([x[1] for x in hw_list])

class CTCDatasetMerged(LightningDataModule):
    def __init__(self, config: ExperimentCommonConfig, model_name, allow_downsample = True):
        super().__init__()

        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        
        self.train_sets = []
        self.val_sets = []
        self.test_sets = []

        self.samples_per_batch = {}

        self.has_downsample_dataset = False
        max_normal_dataset_train_size = -1

        for dataset_config in config.data:
            img_fixed_height = None
            if isinstance(dataset_config.resize, (list,tuple)) \
                and len(dataset_config.resize) == 2 \
                and dataset_config.resize[1] is None:
                img_fixed_height = dataset_config.resize[0]
            elif dataset_config.resize is not None:
                raise ValueError("Resize can be either None (null) or have fixed height, other types are not supported")

            self.train_sets.append(CTCDataset(dataset_config.dataset_name, "train", model_name, img_fixed_height=img_fixed_height, encoding_type=dataset_config.transcript_format))
            if dataset_config.use_for_validation:
                self.val_sets.append(CTCDataset(dataset_config.dataset_name, "validation", model_name, train=False, img_fixed_height=img_fixed_height, encoding_type=dataset_config.transcript_format))
            self.test_sets.append(CTCDataset(dataset_config.dataset_name, "test", model_name, train=False, img_fixed_height=img_fixed_height, encoding_type=dataset_config.transcript_format))
            
            if allow_downsample and dataset_config.downsample:
                self.has_downsample_dataset = True
            else:
                max_normal_dataset_train_size = max(max_normal_dataset_train_size, len(self.train_sets[-1]))
        
        if self.has_downsample_dataset:
            assert config.downsample_size is not None or max_normal_dataset_train_size > 0, "Cannot infer the downsampling size for the datasets"
            downsample_size = config.downsample_size if config.downsample_size is not None else max_normal_dataset_train_size
            for i, dataset_config in enumerate(config.data):
                if not dataset_config.downsample:
                    continue
                assert self.train_sets[i].name == dataset_config.dataset_name
                self.samples_per_batch[i] = downsample_size
        
        # merge vocabs
        tokens = set()
        for dataset in self.train_sets:
            tokens.update(dataset.w2i.keys())
        tokens.remove("<PAD>")
        vocab = sorted(tokens)

        self.w2i = {}
        self.i2w = {}
        for i, w in enumerate(vocab):
            self.w2i[w] = i + 1
            self.i2w[i + 1] = w
        self.w2i["<PAD>"] = 0
        self.i2w[0] = "<PAD>"

        # save vocab
        w2i_path = Path("data") / config.name / f"w2i_{config.vocab_name}.json"
        with open(w2i_path, "w") as file:
            json.dump(self.w2i, file)

        for datasets in [self.train_sets, self.val_sets, self.test_sets]:
            for dataset in datasets:
                dataset.w2i = self.w2i
                dataset.i2w = self.i2w
                dataset.w2i_path = w2i_path
        

        self.train_set = MergedDatasets(self.train_sets)
        self.val_set = MergedDatasets(self.val_sets)
        self.test_set = MergedDatasets(self.test_sets)

    def train_dataloader(self):
        if self.has_downsample_dataset:
            sampler = ConcatDatasetCycleSampler(self.train_set, self.samples_per_batch)
            return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, collate_fn=ctc_batch_preparation)
        else:
            return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=ctc_batch_preparation)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, 1, num_workers=self.num_workers, collate_fn=ctc_batch_preparation)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, 1, num_workers=self.num_workers, collate_fn=ctc_batch_preparation)
