# ETA Implementation in TTT4Rec (Improved Version)

## Overview

This document describes the **improved implementation** of **ETA (Efficient Test-Time Adaptation)** in TTT4Rec, based on the EATA paper (ICML 2022). This version addresses the fundamental challenges of applying ETA to recommendation systems with extreme sparsity.

## Key Improvements

### 1. **Sparsity-Aware Metrics**

The original ETA implementation used full entropy and full cosine similarity, which are unreliable for recommendation systems with >99% sparsity. The improved version uses:

- **Top-k Entropy**: Computes entropy on top-k items only (default: k=10)
- **Top-k Cosine Similarity**: Computes similarity on top-k items only
- **Better Thresholds**: Auto-sets thresholds based on top-k entropy scale

### 2. **Optional Training-Time Filtering**

- **`eta_apply_during_training=False`** (default, recommended): ETA is only applied during evaluation
- **`eta_apply_during_training=True`**: ETA is applied during training (use with caution)

**Why this matters**: Filtering during training removes valuable learning signals, especially early in training when the model is uncertain. The default setting applies ETA only during evaluation, which is more appropriate for recommendation systems.

### 3. **Better Defaults for Recommendation Systems**

- **`eta_use_topk=True`**: Uses top-k metrics (recommended for sparse distributions)
- **`eta_top_k=10`**: Focuses on top-10 items (matches typical recommendation evaluation)
- **Auto e_margin**: Automatically sets threshold based on top-k entropy scale

## What is ETA?

ETA is a test-time adaptation method that:
1. **Filters unreliable samples**: Only updates on samples with low entropy (high confidence)
2. **Filters redundant samples**: Skips samples similar to previously seen ones
3. **Reweights losses**: Gives higher weight to more confident predictions

## Implementation Details

### Files

1. **`eta_utils.py`**: Contains ETA utility functions
   - `topk_entropy()`: Computes entropy of top-k items (sparsity-aware)
   - `topk_cosine_similarity()`: Computes cosine similarity of top-k items
   - `filter_samples_eta()`: Two-stage filtering with sparsity-aware metrics
   - `update_model_probs()`: Maintains moving average of probabilities
   - `compute_eta_loss_coefficients()`: Computes reweighting coefficients

2. **`ttt.py`**: Added ETA parameters to `TTTConfig`
   - `use_eta`: Enable/disable ETA (default: False)
   - `e_margin`: Entropy threshold (None = auto)
   - `d_margin`: Cosine similarity threshold (default: 0.5)
   - `eta_top_k`: Number of top items for top-k metrics (default: 10)
   - `eta_use_topk`: Use top-k metrics (default: True)
   - `eta_apply_during_training`: Apply during training (default: False)

3. **`main.py`**: Integrated ETA into `TTT4Rec` model
   - Uses sparsity-aware metrics
   - Optional training-time filtering
   - Supports both BPR and CE loss types

### How It Works

#### For Cross-Entropy Loss (CE):

1. Compute logits: `[batch_size, n_items]`
2. Calculate **top-k entropy** for each sample (if `eta_use_topk=True`)
3. **Stage 1 Filter**: Keep samples with `entropy < e_margin` (reliable samples)
4. **Stage 2 Filter**: From reliable samples, keep those with `|cosine_similarity| < d_margin` (non-redundant)
5. Compute loss only on filtered samples (if `eta_apply_during_training=True`)
6. Apply reweighting: `coeff = 1 / exp(entropy - e_margin)`
7. Update moving average of probabilities

#### For BPR Loss:

1. Convert pos/neg scores to 2-class logits: `[batch_size, 2]`
2. Apply same filtering process as CE
3. Compute BPR loss on filtered samples with reweighting

### Configuration

#### Recommended Configuration (Evaluation Only):

```python
ttt_config = TTTConfig(
    # ... other parameters ...
    use_eta=True,                      # Enable ETA
    e_margin=None,                     # Auto: log(top_k) * 0.80
    d_margin=0.5,                      # Cosine similarity margin
    eta_top_k=10,                     # Top-k items for metrics
    eta_use_topk=True,                 # Use top-k metrics (recommended)
    eta_apply_during_training=False    # Only apply during evaluation (recommended)
)
```

#### Experimental Configuration (Training + Evaluation):

```python
ttt_config = TTTConfig(
    # ... other parameters ...
    use_eta=True,
    e_margin=None,
    d_margin=0.5,
    eta_top_k=10,
    eta_use_topk=True,
    eta_apply_during_training=True    # Apply during training (experimental)
)
```

### Parameters

- **`e_margin`**: Entropy threshold for filtering unreliable samples
  - Auto-set to `log(top_k) * 0.80` if `eta_use_topk=True`
  - Auto-set to `log(n_items) * 0.90` if `eta_use_topk=False`
  - Lower = more selective (fewer samples updated)

- **`d_margin`**: Cosine similarity threshold for redundancy filtering
  - Default: 0.5
  - Lower = more selective (fewer redundant samples)
  - Typical range: 0.3 - 0.7

- **`eta_top_k`**: Number of top items to consider
  - Default: 10 (matches typical recommendation evaluation)
  - Should match your evaluation top-k (e.g., NDCG@10)

- **`eta_use_topk`**: Whether to use top-k metrics
  - Default: True (recommended for sparse distributions)
  - Set to False to use full entropy/similarity (not recommended)

- **`eta_apply_during_training`**: Whether to apply ETA during training
  - Default: False (recommended)
  - Set to True to apply during training (experimental, may hurt performance)

### Why Top-K Metrics?

Recommendation systems have extreme sparsity (>99%):
- Most items have near-zero probability
- Full entropy is misleading (even confident predictions have relatively high entropy)
- Full cosine similarity is noisy on sparse vectors
- Top-k metrics focus on relevant items (top-10, top-50)

**Example**:
```python
# Full entropy on sparse distribution
probs = [0.4, 0.3, 0.2, 0.05, 0.03, 0.01, 0.005, ...]  # 10,000+ items
full_entropy = 2.5  # Misleading - doesn't capture confidence well

# Top-k entropy (k=10)
topk_probs = [0.4, 0.3, 0.2, 0.05, 0.03, 0.01, 0.005, 0.002, 0.001, 0.0005]
topk_entropy = 1.8  # More meaningful - focuses on relevant items
```

### Statistics

The model tracks:
- `num_samples_update_1`: Number of reliable samples (after entropy filtering)
- `num_samples_update_2`: Number of reliable + non-redundant samples (after both filters)

Access via: `model.num_samples_update_1`, `model.num_samples_update_2`

### Expected Benefits

1. **Better Metrics**: Top-k entropy and similarity are more meaningful for sparse distributions
2. **Flexible Application**: Can apply only during evaluation (recommended) or during training
3. **Better Defaults**: Auto-sets thresholds based on top-k entropy scale
4. **Sparsity-Aware**: Handles extreme sparsity (>99%) in recommendation systems

### Comparison with Original ETA

| Aspect | Original ETA | Improved ETA |
|--------|-------------|--------------|
| **Entropy** | Full entropy | Top-k entropy (sparsity-aware) |
| **Similarity** | Full cosine | Top-k cosine (sparsity-aware) |
| **Threshold** | Manual | Auto-set based on top-k scale |
| **Training** | Always applied | Optional (default: evaluation only) |
| **Sparsity** | Assumes dense | Handles >99% sparsity |

### Usage Example

```python
# Recommended: Evaluation only
ttt_config = TTTConfig(
    use_eta=True,
    eta_use_topk=True,
    eta_top_k=10,
    eta_apply_during_training=False  # Only during evaluation
)

model = TTT4Rec(ttt_config, rec_config, dataset)

# ETA will be applied during evaluation (predict, full_sort_predict)
# Training uses all samples (no filtering)
```

### Notes

- **Default behavior**: ETA is disabled (`use_eta=False`)
- **Recommended**: Use `eta_apply_during_training=False` (evaluation only)
- **Sparsity**: Top-k metrics are essential for >99% sparsity
- **Top-k**: Should match your evaluation metric (e.g., NDCG@10 → `eta_top_k=10`)
- **Performance**: Evaluation-only ETA avoids removing learning signals during training

### Troubleshooting

**Problem**: ETA filters out too many samples
- **Solution**: Increase `e_margin` or `d_margin`
- **Solution**: Use `eta_apply_during_training=False` (evaluation only)

**Problem**: ETA doesn't help performance
- **Solution**: Check if dataset has distribution shift (minimal shift = ETA less beneficial)
- **Solution**: Ensure `eta_use_topk=True` for sparse distributions
- **Solution**: Use `eta_apply_during_training=False` (training-time filtering may hurt)

**Problem**: Entropy values seem wrong
- **Solution**: Ensure `eta_use_topk=True` (top-k entropy has different scale than full entropy)
- **Solution**: Check auto-set `e_margin` is appropriate for top-k entropy scale
