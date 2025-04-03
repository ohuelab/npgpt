from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast


class ClmDataset(Dataset):
    def __init__(
        self, file_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int = 512
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r") as f:
            self.data = f.read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.data[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoded.input_ids


class ClmDataModule(LightningDataModule):
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerFast,
        train_ratio: float = 0.8,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 512,
    ):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def setup(self, stage=None):
        dataset = ClmDataset(self.file_path, self.tokenizer, self.max_length)
        self.train_dataset, self.val_dataset = random_split(
            dataset, [self.train_ratio, 1 - self.train_ratio]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
