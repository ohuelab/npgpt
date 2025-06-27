import argparse
import csv
import os
from typing import List, Tuple

import torch
from tqdm import tqdm

from npgpt import SmilesGptModel, SmilesGptTrainingConfig, get_tokenizer


def calculate_perplexity(
    model: SmilesGptModel,
    tokenizer,
    smiles: str,
    device: torch.device,
) -> float:
    """Calculate perplexity for a single SMILES string."""
    tokens = tokenizer.encode(smiles, add_special_tokens=True)
    input_ids = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        outputs = model({"input_ids": input_ids, "labels": input_ids})
        loss = outputs.loss

    perplexity = torch.exp(loss).item()
    return perplexity


def evaluate_np_likeness(
    model: SmilesGptModel,
    tokenizer,
    smiles_list: List[str],
    batch_size: int = 32,
) -> List[Tuple[str, float]]:
    """Evaluate naturalness of multiple SMILES strings using perplexity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []

    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Evaluating SMILES"):
        batch = smiles_list[i : i + batch_size]

        for smiles in batch:
            try:
                perplexity = calculate_perplexity(model, tokenizer, smiles, device)
                results.append((smiles, perplexity))
            except Exception as e:
                print(f"Error processing SMILES '{smiles}': {e}")
                results.append((smiles, float("inf")))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate naturalness of SMILES using perplexity."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="kohbanye/SmilesTokenizer_PubChem_1M",
        help="Path to the tokenizer file or HuggingFace model name.",
    )
    parser.add_argument(
        "--use_hf_tokenizer",
        type=bool,
        default=True,
        help="Whether to use a HuggingFace tokenizer.",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="Single SMILES string to evaluate.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to .smi file containing SMILES strings.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="naturalness_scores.csv",
        help="Path to save evaluation results (CSV format).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )

    args = parser.parse_args()

    if args.smiles is None and args.input_file is None:
        parser.error("Either --smiles or --input_file must be provided.")

    # Load model and tokenizer
    training_config = SmilesGptTrainingConfig()
    tokenizer = get_tokenizer(training_config, args.tokenizer, args.use_hf_tokenizer)

    model = SmilesGptModel.load_from_checkpoint(
        args.checkpoint,
        config=training_config,
        tokenizer=tokenizer,
        strict=False,
    )

    # Prepare SMILES list
    if args.smiles:
        smiles_list = [args.smiles]
    else:
        with open(args.input_file, "r") as f:
            smiles_list = [line.strip() for line in f if line.strip()]

    # Evaluate naturalness
    results = evaluate_np_likeness(model, tokenizer, smiles_list, args.batch_size)

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "perplexity"])
        writer.writerows(results)

    print(f"Results saved to {args.output}")

    # Print summary statistics
    perplexities = [p for _, p in results if p != float("inf")]
    if perplexities:
        print("\nSummary:")
        print(f"  Total SMILES: {len(results)}")
        print(f"  Valid evaluations: {len(perplexities)}")
        print(f"  Average perplexity: {sum(perplexities) / len(perplexities):.2f}")
        print(f"  Min perplexity: {min(perplexities):.2f}")
        print(f"  Max perplexity: {max(perplexities):.2f}")


if __name__ == "__main__":
    main()
