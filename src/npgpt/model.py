import pytorch_lightning as pl
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from npgpt.config import SmilesGptConfig


class SmilesGptModel(pl.LightningModule):
    def __init__(
        self,
        config: SmilesGptConfig,
        tokenizer: PreTrainedTokenizerFast,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        gpt2_config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            bos_token_id=self.tokenizer.bos_token_id or 1,
            eos_token_id=self.tokenizer.eos_token_id or 2,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            n_positions=config.max_length,
            n_ctx=config.max_length,
        )
        self.model = GPT2LMHeadModel(gpt2_config)

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> CausalLMOutputWithCrossAttentions:
        input_ids = data["input_ids"]
        labels = data["labels"]
        return self.model(input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch
        outputs = self.forward(input_ids)
        return outputs

    def validation_step(self, batch, batch_idx):
        input_ids = batch
        outputs = self.forward(input_ids)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(  # type: ignore
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_eps,
            betas=self.config.adam_betas,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.config.scheduler_T_max,
            eta_min=self.config.final_learning_rate,  # type: ignore
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
