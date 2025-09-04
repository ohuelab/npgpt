"""Custom loss modules for SMILES generation training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from .chiral_utils import (
    compute_weighted_chiral_loss,
    compute_chiral_unlikelihood_loss,
)


class ChiralAwareLoss(nn.Module):
    """Loss module that combines standard causal LM loss with chiral-specific losses."""

    def __init__(
        self,
        single_at_tokens: set[int],
        double_at_tokens: set[int],
        chiral_loss_weight: float = 1.0,
        enable_chiral_unlikelihood: bool = False,
        chiral_unlikelihood_weight: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.single_at_tokens = single_at_tokens
        self.double_at_tokens = double_at_tokens
        self.chiral_tokens = single_at_tokens.union(double_at_tokens)

        self.chiral_loss_weight = chiral_loss_weight
        self.enable_chiral_unlikelihood = enable_chiral_unlikelihood
        self.chiral_unlikelihood_weight = chiral_unlikelihood_weight
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, return_dict: bool = True
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute chiral-aware loss.

        Args:
            logits: Model logits of shape (batch_size, seq_len, vocab_size)
            labels: Target token IDs of shape (batch_size, seq_len)
            return_dict: If True, return dict with loss components

        Returns:
            Total loss tensor or dict with loss components
        """
        # Shift for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute base cross-entropy loss per token
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction="none",
        ).view(shift_labels.shape)

        # Apply chiral token weighting if enabled
        if self.chiral_loss_weight != 1.0 and self.chiral_tokens:
            weighted_loss_per_token = compute_weighted_chiral_loss(
                loss_per_token,
                shift_labels,
                self.single_at_tokens,
                self.double_at_tokens,
                chiral_weight=self.chiral_loss_weight,
                ignore_index=self.ignore_index,
            )
        else:
            weighted_loss_per_token = loss_per_token

        # Compute weighted causal LM loss
        valid_mask = (shift_labels != self.ignore_index).float()
        causal_lm_loss = (weighted_loss_per_token * valid_mask).sum() / valid_mask.sum()

        total_loss = causal_lm_loss

        # Add chiral unlikelihood loss if enabled
        chiral_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if self.enable_chiral_unlikelihood and self.chiral_tokens:
            chiral_loss = compute_chiral_unlikelihood_loss(
                shift_logits,
                shift_labels,
                self.single_at_tokens,
                self.double_at_tokens,
                ignore_index=self.ignore_index,
            )
            total_loss = total_loss + self.chiral_unlikelihood_weight * chiral_loss

        if return_dict:
            return {
                "loss": total_loss,
                "causal_lm_loss": causal_lm_loss,
                "chiral_loss": chiral_loss,
                "weighted_chiral_loss": self.chiral_unlikelihood_weight * chiral_loss,
            }
        else:
            return total_loss


def create_chiral_aware_loss(
    single_at_tokens: set[int],
    double_at_tokens: set[int],
    chiral_loss_weight: float = 1.0,
    enable_chiral_unlikelihood: bool = False,
    chiral_unlikelihood_weight: float = 0.1,
    ignore_index: int = -100,
) -> ChiralAwareLoss:
    """Factory function to create ChiralAwareLoss module."""
    return ChiralAwareLoss(
        single_at_tokens=single_at_tokens,
        double_at_tokens=double_at_tokens,
        chiral_loss_weight=chiral_loss_weight,
        enable_chiral_unlikelihood=enable_chiral_unlikelihood,
        chiral_unlikelihood_weight=chiral_unlikelihood_weight,
        ignore_index=ignore_index,
    )
