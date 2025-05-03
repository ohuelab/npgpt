import argparse
import math
import os

import torch
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
                initial_tokens = tokenizer.encode(
                    initial_smiles, add_special_tokens=False
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

    args = parser.parse_args()

    training_config = SmilesGptTrainingConfig()
    generation_config = SmilesGptGenerationConfig(
        num_samples=args.num_samples,
    )

    tokenizer = get_tokenizer(training_config, args.tokenizer)
    model = SmilesGptModel.load_from_checkpoint(
        args.checkpoint,
        config=training_config,
        tokenizer=tokenizer,
        strict=False,
    )

    smiles_list = generate_smiles(
        model,
        tokenizer,
        generation_config,
        initial_smiles=args.initial_smiles,
        batch_size=args.batch_size,
    )

    output_path = args.output
    if not output_path.endswith(".smi"):
        output_path += ".smi"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(smiles_list))

    print(f"{args.num_samples} SMILES saved to {output_path}")
