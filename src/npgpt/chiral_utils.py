from transformers import PreTrainedTokenizerFast
import torch


def get_chiral_token_groups(
    tokenizer: PreTrainedTokenizerFast,
) -> tuple[set[int], set[int]]:
    """Get chiral token groups separated by directionality.

    Args:
        tokenizer: The tokenizer to analyze

    Returns:
        Tuple of (single_at_tokens, double_at_tokens) where:
        - single_at_tokens: Set of token IDs containing single '@'
        - double_at_tokens: Set of token IDs containing '@@'
    """
    vocab = tokenizer.get_vocab()

    single_at_tokens = set()
    double_at_tokens = set()

    for token, token_id in vocab.items():
        if "@@" in token:
            double_at_tokens.add(token_id)
        elif "@" in token:
            single_at_tokens.add(token_id)

    return single_at_tokens, double_at_tokens


def get_opposite_chiral_tokens(
    token_id: int, single_at_tokens: set[int], double_at_tokens: set[int]
) -> set[int]:
    """Get the set of opposite chiral tokens for a given token.

    Args:
        token_id: The target token ID
        single_at_tokens: Set of single '@' token IDs
        double_at_tokens: Set of '@@' token IDs

    Returns:
        Set of token IDs that are opposites of the target token.
        Returns empty set if token_id is not a chiral token.
    """
    if token_id in single_at_tokens:
        return double_at_tokens
    elif token_id in double_at_tokens:
        return single_at_tokens
    else:
        return set()


def create_chiral_opposition_mask(
    labels: torch.Tensor,
    vocab_size: int,
    single_at_tokens: set[int],
    double_at_tokens: set[int],
    device: torch.device,
) -> torch.Tensor:
    """Create a mask for chiral opposition tokens.

    Args:
        labels: Target token IDs tensor of shape (batch_size, seq_len)
        vocab_size: Size of the vocabulary
        single_at_tokens: Set of single '@' token IDs
        double_at_tokens: Set of '@@' token IDs
        device: Device to place the mask tensor

    Returns:
        Boolean mask tensor of shape (batch_size, seq_len, vocab_size) where
        True indicates positions of opposite chiral tokens to suppress.
    """
    batch_size, seq_len = labels.shape
    mask = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.bool, device=device)

    for b in range(batch_size):
        for s in range(seq_len):
            label_id = int(labels[b, s].item())
            if label_id == -100:  # Ignore padding tokens
                continue

            opposite_tokens = get_opposite_chiral_tokens(
                label_id, single_at_tokens, double_at_tokens
            )

            if opposite_tokens:  # If current token is chiral
                for opposite_id in opposite_tokens:
                    mask[b, s, opposite_id] = True

    return mask


def compute_weighted_chiral_loss(
    loss: torch.Tensor,
    labels: torch.Tensor,
    single_at_tokens: set[int],
    double_at_tokens: set[int],
    chiral_weight: float = 1.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Apply weight multiplier to losses at chiral token positions.

    Args:
        loss: Cross-entropy loss tensor of shape (batch_size, seq_len)
        labels: Target token IDs of shape (batch_size, seq_len)
        single_at_tokens: Set of single '@' token IDs
        double_at_tokens: Set of '@@' token IDs
        chiral_weight: Weight multiplier for chiral positions
        ignore_index: Token ID to ignore in loss computation

    Returns:
        Weighted loss tensor with same shape as input loss
    """
    if chiral_weight == 1.0:
        return loss

    device = loss.device
    chiral_tokens = single_at_tokens.union(double_at_tokens)

    # Create weight mask
    weight_mask = torch.ones_like(loss, device=device)

    # Apply chiral weight to positions with chiral tokens using vectorized operations
    chiral_token_list = list(chiral_tokens)
    if chiral_token_list:
        # Create boolean mask for chiral positions
        chiral_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
        for token_id in chiral_token_list:
            chiral_mask |= labels == token_id

        # Exclude ignore_index positions
        valid_mask = labels != ignore_index
        chiral_mask &= valid_mask

        # Apply weight to chiral positions
        weight_mask[chiral_mask] = chiral_weight

    return loss * weight_mask


def compute_chiral_unlikelihood_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    single_at_tokens: set[int],
    double_at_tokens: set[int],
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute the chiral unlikelihood loss.

    Args:
        logits: Model logits of shape (batch_size, seq_len, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_len)
        single_at_tokens: Set of single '@' token IDs
        double_at_tokens: Set of '@@' token IDs
        ignore_index: Token ID to ignore in loss computation

    Returns:
        Scalar tensor representing the chiral unlikelihood loss
    """
    # Handle various logits shapes
    if len(logits.shape) == 2:
        # Shape: (seq_len, vocab_size)
        _seq_len, vocab_size = logits.shape
        _batch_size = 1
        logits = logits.unsqueeze(0)  # Add batch dimension -> (1, seq_len, vocab_size)
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(0)  # Add batch dimension -> (1, seq_len)
    elif len(logits.shape) == 3:
        # Shape: (batch_size, seq_len, vocab_size)
        _batch_size, _seq_len, vocab_size = logits.shape
    else:
        # Unexpected shape, return zero loss
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    device = logits.device

    # Create opposition mask
    opposition_mask = create_chiral_opposition_mask(
        labels, vocab_size, single_at_tokens, double_at_tokens, device
    )

    # Get probabilities
    probs = torch.softmax(logits, dim=-1)

    # Apply mask to get probabilities of opposite tokens
    opposite_probs = probs * opposition_mask.float()

    # Compute unlikelihood loss: -log(1 - p) for opposite tokens
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    unlikelihood_loss = -torch.log(1 - opposite_probs + epsilon)

    # Only consider positions where we have chiral tokens in labels
    valid_positions = labels != ignore_index

    # Sum over vocabulary dimension to get loss per position
    loss_per_position = unlikelihood_loss.sum(dim=-1)

    # Apply valid positions mask and compute mean
    valid_loss = loss_per_position * valid_positions.float()
    total_loss = valid_loss.sum()
    valid_count = valid_positions.sum()

    if valid_count > 0:
        return total_loss / valid_count
    return torch.tensor(0.0, device=device)


def analyze_chiral_tokens(tokenizer: PreTrainedTokenizerFast) -> dict:
    """Analyze chiral tokens in the tokenizer vocabulary.

    Args:
        tokenizer: The tokenizer to analyze

    Returns:
        Dictionary containing analysis results
    """
    single_at, double_at = get_chiral_token_groups(tokenizer)
    vocab = tokenizer.get_vocab()

    # Get token strings for analysis
    single_at_tokens = []
    double_at_tokens = []

    for token, token_id in vocab.items():
        if token_id in single_at:
            single_at_tokens.append((token, token_id))
        elif token_id in double_at:
            double_at_tokens.append((token, token_id))

    return {
        "single_at_count": len(single_at),
        "double_at_count": len(double_at),
        "total_chiral_count": len(single_at) + len(double_at),
        "single_at_tokens": sorted(single_at_tokens, key=lambda x: x[1]),
        "double_at_tokens": sorted(double_at_tokens, key=lambda x: x[1]),
        "opposition_pairs": len(single_at) * len(double_at),
    }
