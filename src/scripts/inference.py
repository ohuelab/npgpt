import torch

from npgpt import SmilesGptModel, SmilesGptTrainingConfig, get_tokenizer
from npgpt.config import SmilesGptGenerationConfig


def generate_smiles(
    model: SmilesGptModel,
    tokenizer,
    config: SmilesGptGenerationConfig,
) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    generated_smiles = []
    with torch.no_grad():
        for _ in range(config.num_samples):
            # Start with BOS token
            input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)

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

            smiles = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_smiles.append(smiles)

    return generated_smiles


if __name__ == "__main__":
    training_config = SmilesGptTrainingConfig()
    generation_config = SmilesGptGenerationConfig()
    tokenizer_filename = "externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json"
    checkpoint_path = "checkpoints/smiles-gpt/model.ckpt"

    tokenizer = get_tokenizer(training_config, tokenizer_filename)
    model = SmilesGptModel.load_from_checkpoint(
        checkpoint_path,
        config=training_config,
        tokenizer=tokenizer,
        strict=False,
    )

    smiles_list = generate_smiles(model, tokenizer, generation_config)

    print("Generated SMILES:")
    for i, smiles in enumerate(smiles_list, 1):
        print(f"{i}. {smiles}")
