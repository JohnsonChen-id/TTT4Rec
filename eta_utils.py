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


def topk_cosine_similarity(probs_1: torch.Tensor, probs_2: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Compute cosine similarity of top-k items only (sparsity-aware).
    
    This is more meaningful for recommendation systems where most items
    have near-zero probability and similarity should focus on top items.
    
    Args:
        probs_1: Probability vector [num_items] or [batch_size, num_items]
        probs_2: Probability vector [num_items] or [batch_size, num_items]
        k: Number of top items to consider (default: 10)
    
    Returns:
        Cosine similarity scalar or [batch_size]
    """
    # Handle both 1D and 2D cases
    if probs_1.dim() == 1:
        probs_1 = probs_1.unsqueeze(0)
        probs_2 = probs_2.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Get top-k items and their indices
    topk_1, topk_indices_1 = torch.topk(probs_1, min(k, probs_1.size(1)), dim=1)
    topk_2, topk_indices_2 = torch.topk(probs_2, min(k, probs_2.size(1)), dim=1)
    
    # Create sparse vectors with only top-k items
    batch_size = probs_1.size(0)
    num_items = probs_1.size(1)
    
    sparse_1 = torch.zeros_like(probs_1)
    sparse_2 = torch.zeros_like(probs_2)
    
    for i in range(batch_size):
        sparse_1[i].scatter_(0, topk_indices_1[i], topk_1[i])
        sparse_2[i].scatter_(0, topk_indices_2[i], topk_2[i])
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(sparse_1, sparse_2, dim=1)
    
    if squeeze_output:
        similarity = similarity.squeeze(0)
    
    return similarity


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


def update_model_probs(current_model_probs, new_probs, top_k: int = 10, alpha: float = 0.1):
    """
    Update moving average of model probabilities for redundancy filtering (cosine similarity mode).
    Uses top-k averaging for better handling of sparse distributions.
    
    Args:
        current_model_probs: Current moving average [num_items] or None
        new_probs: New probability vectors [batch_size, num_items]
        top_k: Number of top items to focus on (default: 10)
        alpha: Exponential moving average factor (default: 0.1)
    
    Returns:
        Updated moving average [num_items]
    """
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                # For initial average, use top-k weighted average
                batch_mean = new_probs.mean(0)
                return batch_mean
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                # Exponential moving average
                batch_mean = new_probs.mean(0)
                return (1 - alpha) * current_model_probs + alpha * batch_mean


def filter_samples_eta(
    logits, 
    redundancy_data,  # Can be recent_items (frequency) or current_model_probs (cosine)
    e_margin, 
    redundancy_threshold,  # Can be max_frequency (frequency) or d_margin (cosine)
    redundancy_filter_type: str = 'frequency',  # 'frequency' or 'cosine'
    use_topk: bool = True,
    top_k: int = 10,
    min_samples_ratio: float = 0.1
):
    """
    Filter samples using ETA's two-stage filtering with sparsity-aware metrics.
    1. Filter unreliable samples (high entropy)
    2. Filter redundant samples (using frequency or cosine similarity)
    
    Args:
        logits: Model logits [batch_size, num_items] or [batch_size, 2] for BPR
        redundancy_data: For 'frequency': recent_items [N] or None
                        For 'cosine': current_model_probs [num_items] or None
        e_margin: Entropy threshold for filtering unreliable samples
        redundancy_threshold: For 'frequency': max_frequency (default: 0.3)
                             For 'cosine': d_margin (default: 0.5)
        redundancy_filter_type: 'frequency' or 'cosine' (default: 'frequency')
        use_topk: Whether to use top-k metrics (default: True, recommended for sparse distributions)
        top_k: Number of top items to consider for top-k metrics (default: 10)
        min_samples_ratio: Minimum fraction of batch to keep (default: 0.1)
    
    Returns:
        filter_ids_1: Indices of reliable samples (after entropy filtering)
        filter_ids_2: Indices of reliable+non-redundant samples (after both filters)
        entropys: Entropy values for filtered samples
        predicted_items_or_probs: For 'frequency': predicted_items [batch_size]
                                  For 'cosine': probs [batch_size, num_items]
    """
    batch_size = logits.size(0)
    min_samples = max(1, int(batch_size * min_samples_ratio))
    
    # Get predicted items (argmax) - needed for frequency filtering
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
    
    # Get data for reliable samples
    predicted_items_filtered = predicted_items[filter_ids_1]
    entropys_filtered = entropys[filter_ids_1]
    probs_filtered = logits[filter_ids_1].softmax(1)  # For cosine similarity
    
    # Stage 2: Filter redundant samples
    if redundancy_filter_type == 'frequency':
        # Frequency-based filtering (current implementation)
        recent_items = redundancy_data
        if recent_items is not None and recent_items.size(0) > 0 and filter_ids_1.size(0) > min_samples:
            # Check frequency of predicted items in recent history
            is_diverse = check_item_frequency(
                predicted_items_filtered, 
                recent_items, 
                max_frequency=redundancy_threshold
            )
            filter_ids_2 = torch.where(is_diverse)[0]
            
            # Ensure minimum samples
            if filter_ids_2.size(0) < min_samples:
                filter_ids_2 = torch.arange(filter_ids_1.size(0), device=filter_ids_1.device)
            
            if filter_ids_2.size(0) == 0:
                filter_ids_2 = torch.arange(filter_ids_1.size(0), device=filter_ids_1.device)
        else:
            filter_ids_2 = torch.arange(filter_ids_1.size(0), device=filter_ids_1.device)
        
        # Return predicted items for frequency mode
        return_data = predicted_items[filter_ids_1[filter_ids_2]]
        
    else:  # redundancy_filter_type == 'cosine'
        # Cosine similarity-based filtering
        current_model_probs = redundancy_data
        if current_model_probs is not None and filter_ids_1.size(0) > min_samples:
            # Compute cosine similarity with moving average
            if use_topk and probs_filtered.size(1) > top_k:
                # Use top-k cosine similarity for sparse distributions
                cosine_similarities = torch.zeros(probs_filtered.size(0), device=probs_filtered.device)
                for i in range(probs_filtered.size(0)):
                    cosine_similarities[i] = topk_cosine_similarity(
                        current_model_probs, probs_filtered[i], k=top_k
                    )
            else:
                # Use full cosine similarity
                cosine_similarities = F.cosine_similarity(
                    current_model_probs.unsqueeze(0),  # [1, num_items]
                    probs_filtered,  # [num_reliable, num_items]
                    dim=1  # [num_reliable]
                )
            
            # Filter samples with low cosine similarity (diverse samples)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < redundancy_threshold)[0]
            
            # Ensure minimum samples
            if filter_ids_2.size(0) < min_samples:
                # If too few samples pass, take the min_samples with lowest similarity
                _, top_indices = torch.topk(torch.abs(cosine_similarities), min_samples, largest=False)
                filter_ids_2 = top_indices
            
            if filter_ids_2.size(0) == 0:
                filter_ids_2 = torch.arange(filter_ids_1.size(0), device=filter_ids_1.device)
        else:
            filter_ids_2 = torch.arange(filter_ids_1.size(0), device=filter_ids_1.device)
        
        # Return probabilities for cosine mode (needed for updating moving average)
        return_data = probs_filtered[filter_ids_2]
    
    # Map filter_ids_2 back to original batch indices
    final_filter_ids = filter_ids_1[filter_ids_2]
    
    return filter_ids_1, final_filter_ids, entropys_filtered[filter_ids_2], return_data


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
