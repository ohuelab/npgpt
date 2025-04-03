import sys

sys.path.append("externals/smiles-gpt")

from smiles_gpt.tokenization import SMILESAlphabet, SMILESBPETokenizer
from transformers import PreTrainedTokenizerFast

from npgpt.config import SmilesGptConfig


def train_tokenizer(
    config: SmilesGptConfig,
    tokenizer_train_files: str | list[str],
    checkpoint_path: str = "checkpoints",
    tokenizer_filename: str = "tokenizer.json",
) -> PreTrainedTokenizerFast:
    tokenizer = SMILESBPETokenizer(dropout=None)

    alphabet = SMILESAlphabet()
    alphabet_list = list(alphabet.get_alphabet())
    tokenizer.train(
        files=tokenizer_train_files,
        vocab_size=config.vocab_size + len(alphabet_list),
        min_frequency=config.min_frequency,
        initial_alphabet=alphabet_list,
    )
    tokenizer.save_model(checkpoint_path)
    tokenizer.save(tokenizer_filename)

    tokenizer = tokenizer.get_hf_tokenizer(
        tokenizer_filename, model_max_length=config.max_length
    )

    return tokenizer


def get_tokenizer(
    config: SmilesGptConfig,
    tokenizer_filename: str = "tokenizer.json",
) -> PreTrainedTokenizerFast:
    return SMILESBPETokenizer.get_hf_tokenizer(
        tokenizer_filename, model_max_length=config.max_length
    )
