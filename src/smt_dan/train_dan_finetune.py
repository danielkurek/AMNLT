import fire
import json
import torch
from AMNLT.scripts.smt_dan.data_amnlt import AMNLTDataset
from AMNLT.scripts.smt_dan.dan_trainer import DAN_Trainer
import os

from AMNLT.configs.smt_dan_config.ExperimentConfig import experiment_config_from_dict
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from pathlib import Path

torch.set_float32_matmul_precision('high')

def main(config_path, checkpoint_path, vocab_name, patience=5, threads=2, gradient_accumulation=1):
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
    
    config.data.vocab_name = vocab_name
    datamodule = AMNLTDataset(config=config.data)
    dataset_name = Path(config_path).stem + "_finetune"

    model = DAN_Trainer.load_from_checkpoint(checkpoint_path, weights_only=False)

    # Set lower learning rate
    def configure_optimizers_new(self):
        return torch.optim.Adam(list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()), lr=1e-5, amsgrad=False)
    import types
    model.configure_optimizers = types.MethodType(configure_optimizers_new, model)
    
    experiment_name = f"DAN_{dataset_name}_finetune"
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

    checkpointer = ModelCheckpoint(dirpath=f"weights/{dataset_name}/", filename=f"{dataset_name}_DAN", 
                                   monitor="val_CER", mode='min',
                                   save_top_k=1, verbose=True)
    early_stopper = EarlyStopping(
            monitor="val_CER",
            min_delta=0.1,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            check_on_train_epoch_end=False,
        )
    
    trainer = Trainer(max_epochs=10000, 
                      check_val_every_n_epoch=1, 
                      logger=loggers, callbacks=[checkpointer, early_stopper],
                      precision="16-mixed",
                      reload_dataloaders_every_n_epochs=1,
                      accumulate_grad_batches=gradient_accumulation)

    trainer.fit(model, datamodule=datamodule)

    model = DAN_Trainer.load_from_checkpoint(checkpointer.best_model_path, weights_only=False)

    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    fire.Fire(main)