# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NPGPT is a research implementation of "Natural Product-Like Compound Generation with GPT-based Chemical Language Models". It uses GPT-based models to generate SMILES strings representing natural product-like chemical compounds, trained on the COCONUT natural product library.

## Development Commands

### Environment Setup

```bash
# Install dependencies (requires uv package manager)
uv sync

# Initialize git submodules (required for externals/smiles-gpt)
git submodule update --init --recursive
```

### Training

```bash
# Basic training with default parameters
uv run python src/scripts/train.py

# Training with HPC job script (takes dataset, checkpoint_dir, pretrained_model_path as args)
./train.sh data/coconut.smi checkpoints/run_name path/to/pretrained.ckpt

# Training with specific parameters
uv run python src/scripts/train.py \
  --dataset data/coconut.smi \
  --tokenizer kohbanye/SmilesTokenizer_PubChem_1M \
  --use_hf_tokenizer True \
  --checkpoint_dir checkpoints/my_run \
  --epochs 60 \
  --pretrained_model_path checkpoints/pretrained.ckpt
```

### Inference

```bash
# Generate SMILES strings using trained model
uv run python src/scripts/inference.py

# Run inference notebook
uv run jupyter notebook src/notebooks/inference.ipynb
```

### Code Quality

```bash
# Type checking with mypy
uv run mypy src/

# Linting with ruff
uv run ruff check src/
uv run ruff format src/
```

## Architecture Overview

### Core Components

1. **Model Architecture** (`src/npgpt/model.py`)

   - `SmilesGptModel`: PyTorch Lightning module wrapping GPT for SMILES generation
   - Integrates with HuggingFace transformers for tokenization
   - Supports both custom tokenizers and HuggingFace tokenizers

2. **Data Pipeline** (`src/npgpt/data.py`)

   - `ClmDataModule`: PyTorch Lightning data module for causal language modeling
   - Handles SMILES string tokenization and batching
   - Automatic train/validation split

3. **Configuration** (`src/npgpt/config.py`)

   - `SmilesGptTrainingConfig`: Training hyperparameters and model configuration
   - `SmilesGptGenerationConfig`: Generation parameters for inference
   - Uses Pydantic for validation and type safety

4. **Tokenization** (`src/npgpt/tokenizer.py`)
   - Supports both custom SMILES tokenizers and HuggingFace tokenizers
   - Default: `kohbanye/SmilesTokenizer_PubChem_1M` from HuggingFace

### Directory Structure

- `src/npgpt/`: Core package with model implementation
- `src/scripts/`: Training and inference scripts
- `src/notebooks/`: Jupyter notebooks for interactive work
- `externals/smiles-gpt/`: Git submodule with base SMILES-GPT implementation
- `checkpoints/`: Model checkpoint storage
- `data/`: Dataset storage (gitignored)
- `wandb/`: Weights & Biases experiment tracking logs
- `lightning_logs/`: PyTorch Lightning training logs

### Key Dependencies

- PyTorch Lightning for training orchestration
- HuggingFace Transformers for model and tokenizer
- RDKit for chemical structure validation
- Weights & Biases for experiment tracking
- Pydantic for configuration management

### Training Workflow

1. Data is loaded from SMILES files (one SMILES string per line)
2. Tokenizer converts SMILES to token sequences
3. Model trains using causal language modeling objective
4. Checkpoints saved to specified directory
5. Training tracked via Weights & Biases (optional)

### Model Variants

The project supports two pre-trained base models:

- smiles-gpt: GPT-2 model trained on SMILES strings
- ChemGPT: GPT Neo model trained on SELFIES strings

Both can be fine-tuned on the COCONUT natural product dataset.
