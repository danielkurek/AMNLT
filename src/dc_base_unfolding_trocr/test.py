import gc
import os
import random
import json

import fire
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from AMNLT.utils.dc_base_unfolding_utils.data_preprocessing import IMG_HEIGHT
from AMNLT.models.dc_base_unfolding_trocr_models.model import CTCTrainedCRNN, LightningE2EModelUnfolding

from src.ExperimentCommonConfig import experiment_config_from_dict
from src.dc_base_unfolding_trocr.dataset import CTCDatasetMerged

from pathlib import Path

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def test(
    config_path,
    checkpoint_path,
    model_name,
    dataset_index=None,
    project="AMNLT",
    group="Baseline-CharacterLevel",
    entity="el_iseo",
    ctc="greedy",
    threads=2
):
    gc.collect()
    torch.cuda.empty_cache()
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

    # Experiment info
    print(f"Running experiment: {project} - {group}")
    print(f"\tTest dataset: {config.name}")
    print(f"\tCheckpoint path ({config.name}): {checkpoint_path}")


    # Dataset
    datamodule = CTCDatasetMerged(config, model_name, allow_downsample=False)
    
    if model_name == "cnnt2d":
        train_max_h, train_max_w = datamodule.train_set.get_mx_hw()
        val_max_h, val_max_w = datamodule.val_set.get_mx_hw()
        test_max_h, test_max_w = datamodule.test_set.get_mx_hw()

        # Calcular los máximos globales de altura y anchura
        max_h = max(train_max_h, val_max_h, test_max_h)
        max_w = max(train_max_w, val_max_w, test_max_w)
    else:
        max_h, max_w = 0, 0

    # Model
    if(model_name == "crnn"):
        model = CTCTrainedCRNN.load_from_checkpoint(
            checkpoint_path, ctc=ctc, ytest_i2w=datamodule.i2w, ds_name=config.name
        )
    elif(model_name == "fcn" or model_name == "crnnunfolding" or model_name == "cnnt2d" or model_name == "van"):
        model = LightningE2EModelUnfolding.load_from_checkpoint(
            checkpoint_path, ytest_i2w=datamodule.i2w, model=model_name, mh=max_h, mw=max_w
        )
    model.freeze()

    # Test: automatically auto-loads the best weights from the previous run
    run_name = f"{model_name.upper()}-Test-{config.name}{f"[{dataset_index}]" if dataset_index is not None else ""}"
    loggers = [
        TensorBoardLogger(
            save_dir="logs/",
            name=run_name
        ),
        WandbLogger(
            project=project,
            group=group,
            name=run_name,
            log_model=False,
            entity=entity,
        )
    ]
    trainer = Trainer(
        logger=loggers,
        precision="16-mixed",
    )

    if dataset_index is not None:
        dataset_index = int(dataset_index)
        datamodule.test_set = datamodule.test_sets[dataset_index]

    trainer.test(model, dataloaders=datamodule)


if __name__ == "__main__":
    fire.Fire(test)