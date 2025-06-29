import argparse
import math
import os

import torch
from rdkit import Chem
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from npgpt import SmilesGptModel, SmilesGptTrainingConfig, get_tokenizer
from npgpt.config import SmilesGptGenerationConfig


def generate_smiles(
    model: SmilesGptModel,
    tokenizer: PreTrainedTokenizerFast,
    config: SmilesGptGenerationConfig,
    initial_smiles: str | None = None,
    batch_size: int = 1000,
    canonical: bool = False,
) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    generated_smiles = []
    with torch.no_grad():
        num_batches = math.ceil(config.num_samples / batch_size)
        for batch_idx in tqdm(range(num_batches), desc="Generating SMILES"):
            current_batch_size = min(
                batch_size, config.num_samples - batch_idx * batch_size
            )
            if initial_smiles is not None:
                smiles_to_encode = initial_smiles
                if canonical:
                    try:
                        mol = Chem.MolFromSmiles(initial_smiles)
                        if mol is not None:
                            smiles_to_encode = Chem.MolToSmiles(mol, canonical=True)
                    except Exception:
                        pass

                initial_tokens = tokenizer.encode(
                    smiles_to_encode, add_special_tokens=False
                )
                input_ids = torch.tensor(
                    [[tokenizer.bos_token_id] + initial_tokens] * current_batch_size
                ).to(device)
            else:
                input_ids = torch.tensor(
                    [[tokenizer.bos_token_id]] * current_batch_size
                ).to(device)

            outputs = model.model.generate(
                input_ids,
                max_length=config.max_length,
                do_sample=config.do_sample,
                top_p=config.top_p,
                temperature=config.temperature,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            smiles_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            if canonical:
                canonical_smiles = []
                for smiles in smiles_list:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            canonical_smiles.append(
                                Chem.MolToSmiles(mol, canonical=True)
                            )
                        else:
                            canonical_smiles.append(smiles)
                    except Exception:
                        canonical_smiles.append(smiles)
                generated_smiles.extend(canonical_smiles)
            else:
                generated_smiles.extend(smiles_list)

    return generated_smiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SMILES strings using SmilesGptModel."
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
        "--checkpoint",
        type=str,
        default="checkpoints/smiles-gpt/model.ckpt",
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--initial_smiles",
        type=str,
        default=None,
        help="Initial SMILES string to start generation from.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of SMILES strings to generate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for generating SMILES strings.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_smiles.smi",
        help="Path to save generated SMILES (as .smi file).",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Convert generated SMILES to canonical form.",
    )

    args = parser.parse_args()

    training_config = SmilesGptTrainingConfig()
    generation_config = SmilesGptGenerationConfig(
        num_samples=args.num_samples,
    )

    tokenizer = get_tokenizer(training_config, args.tokenizer, args.use_hf_tokenizer)

    model = SmilesGptModel.load_from_checkpoint(
        args.checkpoint,
        config=training_config,
        tokenizer=tokenizer,
        strict=False,
    )

    if args.use_hf_tokenizer and "SmilesTokenizer_PubChem_1M" in args.tokenizer:
        tokenizer.bos_token_id = 12
        tokenizer.eos_token_id = 13
        tokenizer.pad_token_id = 0

    smiles_list = generate_smiles(
        model,
        tokenizer,
        generation_config,
        initial_smiles=args.initial_smiles,
        batch_size=args.batch_size,
        canonical=args.canonical,
    )

    output_path = args.output
    if not output_path.endswith(".smi"):
        output_path += ".smi"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(smiles_list))

    print(f"{args.num_samples} SMILES saved to {output_path}")
