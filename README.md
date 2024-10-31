# NPGPT

This is the implementation of the paper "NPGPT: Natural Product-Like Compound Generation with GPT-based Chemical Language Models" by Koh Sakano, Kairi Furui, and Masahito Ohue.

## Installation

Install [Rye](https://rye.astral.sh/guide/installation/) if you haven't already. Then, clone the repository and install the dependencies.

```shell
git clone https://github.com/ohuelab/npgpt.git
git submodule update --init --recursive
cd npgpt
rye sync
```

## Fine-tuned Models

Fine-tuned models trained on the [COCONUT](https://coconut.naturalproducts.net/) dataset are available. We provide two models fine-tuned from two different pre-trained models: [smiles-gpt](https://doi.org/10.33774/chemrxiv-2021-5fwjd) and [ChemGPT](https://doi.org/10.1038/s42256-023-00740-3).

You can download the models from the following link:

- [NPGPT (smiles-gpt)](https://drive.google.com/drive/folders/1olCPouDkaJ2OBdNaM-G7IU8T6fBpvPMy?usp=drive_link)
- [NPGPT (ChemGPT)](https://drive.google.com/drive/folders/1P7g4x62PDBWQn5GoIbIJBCBHPE84kIOu?usp=drive_link)
