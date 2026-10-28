import torch
import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from lightning import LightningDataModule

from AMNLT.utils.smt_dan_utils.utils import check_and_retrieveVocabulary
from AMNLT.scripts.smt_dan.data_amnlt import AMNLTSingleSystem, batch_preparation_img2seq
from ..ExperimentCommonConfig import ExperimentCommonConfig
from ..utils.concat_cycle_sampler import ConcatDatasetCycleSampler

class MergedDatasets(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        for ds in self.datasets:
            ds.set_dictionaries(w2i, i2w)
    
    def get_max_hw(self):
        hw_list = [x.get_max_hw() for x in self.datasets]
        return max([x[0] for x in hw_list]), max([x[1] for x in hw_list])
    
    def get_max_seqlen(self):
        return max([x.get_max_seqlen() for x in self.datasets])

class AMNLTDatasetMerged(LightningDataModule):
    def __init__(self, config: ExperimentCommonConfig):
        super().__init__()

        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        
        self.train_sets = []
        self.val_sets = []
        self.test_sets = []

        for dataset_config in config.data:
            self.train_sets.append(AMNLTSingleSystem(dataset_config.dataset_name, "train", dataset_config.transcript_format, dataset_config.resize, augment=True))
            if dataset_config.use_for_validation:
                self.val_sets.append(AMNLTSingleSystem(dataset_config.dataset_name, "validation", dataset_config.transcript_format, dataset_config.resize))
            self.test_sets.append(AMNLTSingleSystem(dataset_config.dataset_name, "test", dataset_config.transcript_format, dataset_config.resize))

        self.train_set = MergedDatasets(self.train_sets)
        self.val_set = MergedDatasets(self.val_sets)
        self.test_set = MergedDatasets(self.test_sets)
        
        gts = [x.get_gt() for x in self.train_sets] + [x.get_gt() for x in self.val_sets] + [x.get_gt() for x in self.test_sets]
        w2i, i2w = check_and_retrieveVocabulary(gts, "vocab", config.vocab_name)
        
        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

class AMNLTDatasetMergedDownsampling(LightningDataModule):
    def __init__(self, config: ExperimentCommonConfig, seed: int|None = None):
        super().__init__()

        self.seed = seed
        self.train_generator = None
        if self.seed is not None:
            self.train_generator = torch.Generator().manual_seed(self.seed)
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        
        self.train_sets = []
        self.val_sets = []
        self.test_sets = []
        self.samples_per_batch = {}
        self.has_downsample_dataset = False
        max_normal_dataset_train_size = -1
        for dataset_config in config.data:
            self.train_sets.append(AMNLTSingleSystem(dataset_config.dataset_name, "train", dataset_config.transcript_format, dataset_config.resize, augment=True))
            if dataset_config.use_for_validation:
                self.val_sets.append(AMNLTSingleSystem(dataset_config.dataset_name, "validation", dataset_config.transcript_format, dataset_config.resize))
            self.test_sets.append(AMNLTSingleSystem(dataset_config.dataset_name, "test", dataset_config.transcript_format, dataset_config.resize))
            if dataset_config.downsample:
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
        self.train_set = MergedDatasets(self.train_sets)
        self.val_set = MergedDatasets(self.val_sets)
        self.test_set = MergedDatasets(self.test_sets)
        
        gts = [x.get_gt() for x in self.train_sets] + [x.get_gt() for x in self.val_sets] + [x.get_gt() for x in self.test_sets]
        w2i, i2w = check_and_retrieveVocabulary(gts, "vocab", config.vocab_name)
        
        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)

    def train_dataloader(self):
        sampler = ConcatDatasetCycleSampler(self.train_set, self.samples_per_batch, generator=self.train_generator, pad_last=True)
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, collate_fn=batch_preparation_img2seq)
    
    def val_dataloader(self):
        # TODO: weight each dataset according to its size
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)