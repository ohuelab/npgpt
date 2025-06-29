import os
from pathlib import Path

from pytorch_lightning import LightningDataModule
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast


class ClmDataset(Dataset):
    def __init__(
        self, file_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int = 512
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r") as f:
            lines = f.read().splitlines()
            self.data = []
            for line in lines:
                parts = line.split()
                if len(parts) == 0:
                    continue
                self.data.append(parts[0])

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
        canonical: bool = False,
    ):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.canonical = canonical
        self.collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def _get_canonical_file_path(self) -> str:
        path = Path(self.file_path)
        canonical_path = path.parent / f"{path.stem}_canonical{path.suffix}"
        return str(canonical_path)

    def _create_canonical_dataset(self) -> str:
        canonical_path = self._get_canonical_file_path()

        if os.path.exists(canonical_path):
            print(f"Canonical file already exists: {canonical_path}")
            return canonical_path

        print(f"Creating canonical SMILES file: {canonical_path}")

        with open(self.file_path, "r") as f:
            total_lines = sum(1 for line in f if line.strip())

        with open(self.file_path, "r") as f_in, open(canonical_path, "w") as f_out:
            for line in tqdm(f_in, total=total_lines, desc="Processing SMILES"):
                parts = line.strip().split()
                if len(parts) == 0:
                    continue

                smiles = parts[0]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                        f_out.write(canonical_smiles + "\n")
                    else:
                        print(f"Warning: Invalid SMILES skipped: {smiles}")
                except Exception as e:
                    print(f"Warning: Error processing SMILES {smiles}: {e}")

        return canonical_path

    def setup(self, stage=None):
        if self.canonical:
            dataset_path = self._create_canonical_dataset()
        else:
            dataset_path = self.file_path

        dataset = ClmDataset(dataset_path, self.tokenizer, self.max_length)
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
