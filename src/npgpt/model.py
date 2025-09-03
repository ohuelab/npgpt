import pytorch_lightning as pl
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from npgpt.config import SmilesGptTrainingConfig
from npgpt.chiral_utils import get_chiral_token_groups
from npgpt.loss import create_chiral_aware_loss


class SmilesGptModel(pl.LightningModule):
    def __init__(
        self,
        config: SmilesGptTrainingConfig,
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
        
        # Initialize chiral token groups and loss module
        if config.enable_chiral_unlikelihood or config.chiral_loss_weight != 1.0:
            self.single_at_tokens, self.double_at_tokens = get_chiral_token_groups(tokenizer)
        else:
            self.single_at_tokens, self.double_at_tokens = set(), set()
        
        # Create chiral-aware loss module
        self.chiral_loss_fn = create_chiral_aware_loss(
            single_at_tokens=self.single_at_tokens,
            double_at_tokens=self.double_at_tokens,
            chiral_loss_weight=config.chiral_loss_weight,
            enable_chiral_unlikelihood=config.enable_chiral_unlikelihood,
            chiral_unlikelihood_weight=config.chiral_unlikelihood_weight,
        )

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> CausalLMOutputWithCrossAttentions:
        input_ids = data["input_ids"]
        labels = data["labels"]
        return self.model(input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch
        labels = input_ids.get("labels", input_ids.get("input_ids"))
        if labels is None:
            labels = input_ids["input_ids"]
        
        # Forward pass to get logits
        outputs = self.forward(input_ids)
        
        # Compute chiral-aware loss
        loss_dict = self.chiral_loss_fn(outputs.logits, labels, return_dict=True)
        
        # Log all losses
        self.log("train_loss", loss_dict["loss"])
        self.log("train_causal_lm_loss", loss_dict["causal_lm_loss"])
        
        if self.config.enable_chiral_unlikelihood:
            self.log("train_chiral_loss", loss_dict["chiral_loss"])
            self.log("train_weighted_chiral_loss", loss_dict["weighted_chiral_loss"])
        
        return {"loss": loss_dict["loss"]}

    def validation_step(self, batch, batch_idx):
        input_ids = batch
        labels = input_ids.get("labels", input_ids.get("input_ids"))
        if labels is None:
            labels = input_ids["input_ids"]
        
        # Forward pass to get logits
        outputs = self.forward(input_ids)
        
        # Compute chiral-aware loss
        loss_dict = self.chiral_loss_fn(outputs.logits, labels, return_dict=True)
        
        # Log all losses
        self.log("val_loss", loss_dict["loss"], sync_dist=True)
        self.log("val_causal_lm_loss", loss_dict["causal_lm_loss"], sync_dist=True)
        
        if self.config.enable_chiral_unlikelihood:
            self.log("val_chiral_loss", loss_dict["chiral_loss"], sync_dist=True)
            self.log("val_weighted_chiral_loss", loss_dict["weighted_chiral_loss"], sync_dist=True)
        
        return {"loss": loss_dict["loss"]}

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
