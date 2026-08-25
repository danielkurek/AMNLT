import torch
import numpy as np
from lightning import LightningDataModule

from AMNLT.utils.smt_dan_utils.utils import check_and_retrieveVocabulary
from AMNLT.scripts.smt_dan.data_amnlt import AMNLTSingleSystem, batch_preparation_img2seq
from ..ExperimentCommonConfig import ExperimentCommonConfig

class MergeDatasets:
    def __init__(self, datasets, modifiers: None|list[int]=None):
        self.datasets = datasets
        self.modifiers = [int(x) for x in modifiers] if modifiers is not None else [1] * len(datasets)
    
    def __len__(self):
        return sum(len(x[0]) * x[1] for x in zip(self.datasets, self.modifiers))
    
    def __getitem__(self, index):
        for dataset,modifier in zip(self.datasets, self.modifiers):
            if index < len(dataset)*modifier:
                return dataset[index%len(dataset)]
            index -= len(dataset)*modifier
        raise IndexError("Index is too large or there is a mistake")
    
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

class DownsampleDataset:
    def __init__(self, dataset, length):
        self.dataset = dataset
        if length > len(self.dataset):
            raise ValueError("Downsampled dataset length cannot be larger than the dataset length")
        self.length = length
        self.resample_indices()
        
    
    def __len__(self):
        return self.length
    
    def resample_indices(self):
        self.indices = np.random.randint(0, len(self.dataset), size=self.length)
    
    def __getitem__(self, index):
        if index > len(self.indices):
            raise IndexError()
        return self.dataset[self.indices[index]]
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.dataset.set_dictionaries(w2i, i2w)
    
    def get_max_hw(self):
        return self.dataset.get_max_hw()
    
    def get_max_seqlen(self):
        return self.dataset.get_max_seqlen()
    
    def get_gt(self):
        return self.dataset.get_gt()

class AMNLTDatasetMergedDownsampling(LightningDataModule):
    def __init__(self, config: ExperimentCommonConfig):
        super().__init__()

        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        
        self.train_sets = []
        self.val_sets = []
        self.test_sets = []
        self.has_downsample_dataset = False
        max_normal_dataset_train_size = -1
        for dataset_config in config.data:
            self.train_sets.append(AMNLTSingleSystem(dataset_config.dataset_name, "train", dataset_config.transcript_format, dataset_config.reduce_ratio, augment=True))
            if dataset_config.use_for_validation:
                self.val_sets.append(AMNLTSingleSystem(dataset_config.dataset_name, "validation", dataset_config.transcript_format, dataset_config.reduce_ratio))
            self.test_sets.append(AMNLTSingleSystem(dataset_config.dataset_name, "test", dataset_config.transcript_format, dataset_config.reduce_ratio))
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
                self.train_sets[i] = DownsampleDataset(self.train_sets[i], downsample_size)
        self.train_set = MergeDatasets(self.train_sets)
        self.val_set = MergeDatasets(self.val_sets)
        self.test_set = MergeDatasets(self.test_sets)
        
        gts = [x.get_gt() for x in self.train_sets] + [x.get_gt() for x in self.val_sets] + [x.get_gt() for x in self.test_sets]
        w2i, i2w = check_and_retrieveVocabulary(gts, "vocab", config.vocab_name)
        
        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)

    def train_dataloader(self):
        self.train_sets[0].resample_indices()
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)