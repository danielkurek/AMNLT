import gc
import random
import json

import fire
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from AMNLT.models.dc_base_unfolding_trocr_models.model import CTCTrainedCRNN, LightningE2EModelUnfolding #, CTCTrainedVAN # <- appears to be missing
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

def train(
    config_path,
    model_name,
    allow_downsample=True,
    epochs=1000,
    patience=5,
    use_augmentations=True,
    metric_to_monitor="val_cer",
    project="AMNLT",
    group="Baseline-CharacterLevel",
    entity="el_iseo",
    fine_tune=False,
    pretrain_model_path=None,
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

    # Experiment info
    print(f"Running experiment: {project} - {group} - {model_name}")
    print(f"\tDataset(s): {config_path}")
    print(f"\tAllow downsample: {allow_downsample}")
    print(f"\tAugmentations: {use_augmentations}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tMetric to monitor: {metric_to_monitor}")

    with open(config_path, "r") as f:
        config = experiment_config_from_dict(json.load(f))

    # Get datasets
    datamodule = CTCDatasetMerged(config, model_name, allow_downsample=allow_downsample)
    
    if model_name == "cnnt2d":
        # TODO: speed up
        train_max_h, train_max_w = datamodule.train_set.get_mx_hw()
        val_max_h, val_max_w = datamodule.val_set.get_mx_hw()
        test_max_h, test_max_w = datamodule.test_set.get_mx_hw()

        # Calcular los máximos globales de altura y anchura
        max_h = max(train_max_h, val_max_h, test_max_h)
        max_w = max(train_max_w, val_max_w, test_max_w)
    else:
        max_h, max_w = 0, 0

    # Model
    if(fine_tune):
        if(model_name == "crnn"):
            model = CTCTrainedCRNN.load_from_checkpoint(pretrain_model_path, strict=False)
        elif(model_name == "fcn"):
            model = LightningE2EModelUnfolding.load_from_checkpoint(pretrain_model_path, strict=False)
    else:
        if(model_name == "crnn"):
            model = CTCTrainedCRNN(
                w2i=datamodule.w2i, i2w=datamodule.i2w, ctc=ctc, use_augmentations=use_augmentations, ds_name=config.name,
                img_height=datamodule.max_fixed_height
            )
        elif(model_name == "fcn" or model_name == "crnnunfolding" or model_name == "cnnt2d" or model_name == "van"):
            model = LightningE2EModelUnfolding(
                w2i=datamodule.w2i, i2w=datamodule.i2w, model=model_name, use_augmentations=use_augmentations, mh=max_h, mw=max_w,
                img_height=datamodule.max_fixed_height
            )
        elif(model_name == "ctc_van"):
            model = CTCTrainedVAN(
                w2i=datamodule.w2i, i2w=datamodule.i2w, ctc=ctc, use_augmentations=use_augmentations, ds_name=config.name
            )
    if(model_name == "crnn"):
        for ds in datamodule.train_sets:
            ds.crnn_width_reduction = model.model.cnn.width_reduction

    # Train and validate
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{group}",
            filename=f"{config.name}_{model_name}_{ctc}",
            monitor=metric_to_monitor,
            verbose=True,
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor=metric_to_monitor,
            min_delta=0.1,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.00,
            check_on_train_epoch_end=False,
        ),
    ]
    experiment_name = f"{model_name.upper()}-Train-{config.name}"
    loggers = [
        TensorBoardLogger(
            save_dir="logs/",
            name=experiment_name
        ),
        WandbLogger(
            project=project,
            group=group,
            name=experiment_name,
            log_model=False,
            entity=entity,
        )
    ]
    trainer = Trainer(
        accelerator="cuda",
        logger=loggers,
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        deterministic=False,  # If True, raises error saying that CTC loss does not have this behaviour
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
        fast_dev_run=False,  # Set to True to check if everything is working
    )
    trainer.fit(model, datamodule=datamodule)

    # Test
    if(model_name == "crnn"):
        model = CTCTrainedCRNN.load_from_checkpoint(
            callbacks[0].best_model_path, ytest_i2w=datamodule.i2w
        )
    elif(model_name == "fcn" or model_name == "crnnunfolding" or model_name == "cnnt2d" or model_name == "van"):
        model = LightningE2EModelUnfolding.load_from_checkpoint(
            callbacks[0].best_model_path, ytest_i2w=datamodule.i2w
        )
    elif(model_name == "ctc_van"):
        model = CTCTrainedVAN.load_from_checkpoint(
            callbacks[0].best_model_path, ytest_i2w=datamodule.i2w
        )
    model.freeze()
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train)