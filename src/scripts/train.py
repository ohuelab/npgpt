import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from npgpt import ClmDataModule, SmilesGptModel, SmilesGptTrainingConfig, get_tokenizer


def train(
    config: SmilesGptTrainingConfig,
    tokenizer_filename: str,
    dataset_file: str,
    from_hf: bool = False,
    logging: bool = True,
    checkpoint_dir: str = "checkpoints",
    pretrained_model_path: str | None = None,
):
    tokenizer = get_tokenizer(config, tokenizer_filename, from_hf)


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

    trainer.fit(model, data_module, ckpt_path=pretrained_model_path)
    trainer.save_checkpoint(f"{checkpoint_dir}/model.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SmilesGptModel.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/coconut.smi",
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json",
        help="Path to the tokenizer file.",
    )
    parser.add_argument(
        "--use_hf_tokenizer",
        type=bool,
        default=False,
        help="Whether to use a HuggingFace tokenizer.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Path to the checkpoint directory.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to a pretrained model checkpoint to continue training from.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train for.",
    )

    args = parser.parse_args()

    config = SmilesGptTrainingConfig(
        batch_size=128,
        max_epochs=args.epochs,
    )

    train(
        config,
        args.tokenizer,
        args.dataset,
        from_hf=args.use_hf_tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        logging=True,
        pretrained_model_path=args.pretrained_model_path,
    )
