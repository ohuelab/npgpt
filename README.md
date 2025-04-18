# NPGPT

This is the implementation of the paper "NPGPT: Natural Product-Like Compound Generation with GPT-based Chemical Language Models" by Koh Sakano, Kairi Furui, and Masahito Ohue.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already. Then, clone the repository and install the dependencies.

```shell
git clone https://github.com/ohuelab/npgpt.git
cd npgpt
git submodule update --init --recursive
uv sync
```

## Training

First, download the training dataset from the following link and place it in the `data/` directory:

- [Training Dataset (coconut.smi)](https://drive.google.com/file/d/1UUnBz_WdLnVvx40Fe4Sf-yAQ_1vTJc0x/view?usp=sharing)

This dataset contains molecules from the [COCONUT](https://coconut.naturalproducts.net/) natural product library converted to SMILES format.

To train the model, run the following command:

```shell
uv run python src/scripts/train.py
```

## Fine-tuned Models

Fine-tuned models trained on the [COCONUT](https://coconut.naturalproducts.net/) dataset are available. We provide two models fine-tuned from two different pre-trained models: [smiles-gpt](https://doi.org/10.33774/chemrxiv-2021-5fwjd) and [ChemGPT](https://doi.org/10.1038/s42256-023-00740-3).

You can download the models from the following link:

- [NPGPT (smiles-gpt)](https://drive.google.com/drive/folders/1olCPouDkaJ2OBdNaM-G7IU8T6fBpvPMy?usp=drive_link)
- [NPGPT (ChemGPT)](https://drive.google.com/drive/folders/1P7g4x62PDBWQn5GoIbIJBCBHPE84kIOu?usp=drive_link)


## Inference

To generate SMILES strings using the trained model, first ensure that you have placed the model checkpoint files in the `checkpoints/smiles-gpt/` directory for the smiles-gpt model. Then run the following command:

```shell
uv run python src/scripts/inference.py
```

## Google Colab

We also provide a Google Colab notebook for inference without local installation:

- [NPGPT Inference Notebook](https://colab.research.google.com/drive/1it66xbMEc_T2J2DnDprPMWzI-uRtQVHb?usp=sharing)

This notebook allows you to generate SMILES strings using our pre-trained models directly in your browser.

