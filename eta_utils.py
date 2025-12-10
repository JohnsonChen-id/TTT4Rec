"""
ETA (Efficient Test-Time Adaptation) utilities for TTT4Rec
Based on EATA: Efficient Test-Time Model Adaptation without Forgetting (ICML 2022)

IMPROVED VERSION: Sparsity-aware metrics for recommendation systems
- Uses top-k entropy instead of full entropy (better for sparse distributions)
- Uses top-k cosine similarity instead of full cosine similarity
- Better handling of extreme sparsity in recommendation systems
"""

import torch
import torch.nn.functional as F
import math


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temperature = 1.0
    x = x / temperature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def topk_entropy(logits: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Compute entropy of top-k items only (sparsity-aware).
    
    This is more meaningful for recommendation systems where distributions
    are extremely sparse (>99% items have near-zero probability).
    
    Args:
        logits: Model logits [batch_size, num_items]
        k: Number of top items to consider (default: 10)
    
    Returns:
        Entropy values [batch_size] computed on top-k items only
    """
    probs = logits.softmax(1)
    # Get top-k probabilities
    topk_probs, _ = torch.topk(probs, min(k, probs.size(1)), dim=1)
    # Renormalize to form a valid distribution
    topk_probs = topk_probs / (topk_probs.sum(1, keepdim=True) + 1e-8)
    # Compute entropy on top-k distribution
    entropy = -(topk_probs * torch.log(topk_probs + 1e-8)).sum(1)
    return entropy


def check_item_frequency(predicted_items: torch.Tensor, recent_items: torch.Tensor, max_frequency: float = 0.3) -> torch.Tensor:
    """
    Check if predicted items have been predicted too frequently recently (Jaccard-ish overlap).
    
    This replaces cosine similarity for redundancy filtering. Instead of comparing probability
    vectors, we check if the predicted item ID has been predicted too often in recent steps.
    This prevents the model from collapsing into recommending the same popular item repeatedly.
    
    Args:
        predicted_items: Predicted item IDs [batch_size] (argmax of logits)
        recent_items: Recent predicted item IDs [N] (sliding window of recent predictions)
        max_frequency: Maximum allowed frequency (default: 0.3 = 30% of recent predictions)
    
    Returns:
        is_diverse: Boolean tensor [batch_size] indicating if item is diverse (not too frequent)
    """
    if recent_items is None or recent_items.size(0) == 0:
        # No recent items, all are diverse
        return torch.ones(predicted_items.size(0), dtype=torch.bool, device=predicted_items.device)
    
    # Ensure same device
    recent_items = recent_items.to(predicted_items.device)
    
    # Count frequency of each predicted item in recent items
    batch_size = predicted_items.size(0)
    is_diverse = torch.ones(batch_size, dtype=torch.bool, device=predicted_items.device)
    
    # For each predicted item, check its frequency in recent items
    for i in range(batch_size):
        item_id = predicted_items[i].item()
        frequency = (recent_items == item_id).float().mean().item()
        # Item is diverse if frequency is below threshold
        is_diverse[i] = frequency < max_frequency
    
    return is_diverse


def update_recent_items(recent_items, new_predicted_items: torch.Tensor, max_history: int = 100):
    """
    Update sliding window of recently predicted item IDs for redundancy filtering.
    
    Args:
        recent_items: Current history of recent item IDs [N] or None
        new_predicted_items: New predicted item IDs [batch_size] (argmax of logits)
        max_history: Maximum number of recent items to keep (default: 100)
    
    Returns:
        Updated history [min(N + batch_size, max_history)]
    """
    if recent_items is None:
        recent_items = torch.tensor([], dtype=torch.long, device=new_predicted_items.device)
    else:
        # Ensure same device
        recent_items = recent_items.to(new_predicted_items.device)
    
    # Concatenate new items
    updated = torch.cat([recent_items, new_predicted_items])
    
    # Keep only last max_history items
    if updated.size(0) > max_history:
        updated = updated[-max_history:]
    
    return updated


def filter_samples_eta(
    logits, 
    recent_items, 
    e_margin, 
    max_frequency,
    use_topk: bool = True,
    top_k: int = 10,
    min_samples_ratio: float = 0.1
):
    """
    Filter samples using ETA's two-stage filtering with sparsity-aware metrics.
    1. Filter unreliable samples (high entropy)
    2. Filter redundant samples (items predicted too frequently recently)
    
    Args:
        logits: Model logits [batch_size, num_items] or [batch_size, 2] for BPR
        recent_items: Recent predicted item IDs [N] or None (sliding window)
        e_margin: Entropy threshold for filtering unreliable samples
        max_frequency: Maximum allowed frequency for redundancy filtering (default: 0.3)
        use_topk: Whether to use top-k metrics (default: True, recommended for sparse distributions)
        top_k: Number of top items to consider for top-k metrics (default: 10)
        min_samples_ratio: Minimum fraction of batch to keep (default: 0.1)
    
    Returns:
        filter_ids_1: Indices of reliable samples (after entropy filtering)
        filter_ids_2: Indices of reliable+non-redundant samples (after both filters)
        entropys: Entropy values for filtered samples
        predicted_items: Predicted item IDs [batch_size] (argmax of logits)
    """
    batch_size = logits.size(0)
    min_samples = max(1, int(batch_size * min_samples_ratio))
    
    # Get predicted items (argmax)
    predicted_items = logits.argmax(dim=1)  # [batch_size]
    
    # Compute entropy (use top-k for sparse distributions)
    if use_topk and logits.size(1) > top_k:
        entropys = topk_entropy(logits, k=top_k)
    else:
        entropys = softmax_entropy(logits)
    
    # Stage 1: Filter unreliable samples (low entropy = high confidence)
    filter_ids_1 = torch.where(entropys < e_margin)[0]
    
    # Ensure minimum samples
    if filter_ids_1.size(0) < min_samples:
        # If too few samples pass, take the min_samples with lowest entropy
        _, top_indices = torch.topk(entropys, min_samples, largest=False)
        filter_ids_1 = top_indices
    
    if filter_ids_1.size(0) == 0:
        # Fallback: use all samples if none pass
        filter_ids_1 = torch.arange(batch_size, device=logits.device)
    
    # Get predicted items for reliable samples
    predicted_items_filtered = predicted_items[filter_ids_1]
    entropys_filtered = entropys[filter_ids_1]
    
    # Stage 2: Filter redundant samples (frequency-based)
    if recent_items is not None and recent_items.size(0) > 0 and filter_ids_1.size(0) > min_samples:
        # Check frequency of predicted items in recent history
        is_diverse = check_item_frequency(
            predicted_items_filtered, 
            recent_items, 
            max_frequency=max_frequency
        )
        filter_ids_2 = torch.where(is_diverse)[0]
        
        # Ensure minimum samples
        if filter_ids_2.size(0) < min_samples:
            # If too few samples pass, keep all reliable samples
            filter_ids_2 = torch.arange(filter_ids_1.size(0), device=filter_ids_1.device)
        
        if filter_ids_2.size(0) == 0:
            # Fallback: use all reliable samples
            filter_ids_2 = torch.arange(filter_ids_1.size(0), device=filter_ids_1.device)
    else:
        # No recent items or too few samples, use all reliable samples
        filter_ids_2 = torch.arange(filter_ids_1.size(0), device=filter_ids_1.device)
    
    # Map filter_ids_2 back to original batch indices
    final_filter_ids = filter_ids_1[filter_ids_2]
    
    return filter_ids_1, final_filter_ids, entropys_filtered[filter_ids_2], predicted_items[final_filter_ids]


def compute_eta_loss_coefficients(entropys, e_margin, normalize: bool = True):
    """
    Compute reweighting coefficients for ETA loss.
    Samples with lower entropy (higher confidence) get higher weights.
    
    Args:
        entropys: Entropy values [num_selected]
        e_margin: Entropy margin
        normalize: Whether to normalize coefficients (default: True)
    
    Returns:
        coeff: Reweighting coefficients [num_selected]
    """
    # Coefficient: 1 / exp(entropy - e_margin)
    # Lower entropy -> higher coefficient
    entropy_diff = entropys.clone().detach() - e_margin
    entropy_diff = torch.clamp(entropy_diff, min=-2.0, max=2.0)  # Limit range
    coeff = 1.0 / (torch.exp(entropy_diff) + 1e-8)
    
    if normalize:
        # Normalize to prevent extreme gradients
        coeff = coeff / (coeff.mean() + 1e-8)
    
    return coeff
