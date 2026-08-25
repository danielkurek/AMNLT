import fire
import json
import torch
from .data_amnlt_merged import AMNLTDatasetMerged
from AMNLT.scripts.smt_dan.dan_trainer import DAN_Trainer
import os

from ..ExperimentCommonConfig import experiment_config_from_dict
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main(config_path, checkpoint_path, threads=2, dataset_index=None):
    if threads is not None and threads > 0:
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != threads:
            torch.set_num_interop_threads(threads)
    # Check if checkpoint path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")
    
    with open(config_path, "r") as f:
        config = experiment_config_from_dict(json.load(f))
    
    datamodule = AMNLTDatasetMerged(config)
    dataset_name = config.name

    model = DAN_Trainer.load_from_checkpoint(checkpoint_path, weights_only=False)
    model.freeze()
    
    experiment_name = f"DAN_{dataset_name}_test"
    loggers = [
        TensorBoardLogger(
            save_dir="logs/",
            name=experiment_name
        ),
        WandbLogger(
            project='DAN_AMNLT-Test',
            group=dataset_name,
            name=experiment_name,
            log_model=False)
    ]
    
    trainer = Trainer(
                      logger=loggers,
                      precision="16-mixed")

    if dataset_index is not None:
        dataset_index = int(dataset_index)
        datamodule.test_set = datamodule.test_sets[dataset_index]
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    fire.Fire(main)