import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from npgpt import ClmDataModule, SmilesGptModel, SmilesGptTrainingConfig, get_tokenizer


def train(
    config: SmilesGptTrainingConfig,
    tokenizer_filename: str,
    dataset_file: str,
    logging: bool = True,
    checkpoint_dir: str = "checkpoints",
):
    tokenizer = get_tokenizer(config, tokenizer_filename)
    model = SmilesGptModel(config, tokenizer)
    data_module = ClmDataModule(
        dataset_file,
        tokenizer,
        train_ratio=config.train_ratio,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_length=config.max_length,
    )

    if logging:
        logger = WandbLogger(project=config.project_name, name=config.run_name)
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        logger=logger,
    )

    trainer.fit(model, data_module)
    trainer.save_checkpoint(f"{checkpoint_dir}/model.ckpt")


if __name__ == "__main__":
    config = SmilesGptTrainingConfig(batch_size=128)
    tokenizer_filename = "externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json"
    dataset_file = "data/coconut.smi"

    train(config, tokenizer_filename, dataset_file, logging=False)
