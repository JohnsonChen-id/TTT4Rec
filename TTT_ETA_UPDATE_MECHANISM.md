# TTT and ETA Update Mechanism in TTT4Rec

## Overview

This document explains **where** and **how** TTT (Test-Time Training) and ETA (Efficient Test-Time Adaptation) updates occur in the TTT4Rec architecture, and the **order of operations** when they work together.

## Architecture Context

Based on the TTT4Rec architecture diagram:

```
Inputs → Embedding Layer → Position Encoding
                           ↓
                    [Combined]
                           ↓
              ┌─────────────────────────────┐
              │   Residual Block (N×)      │
              │  ┌────────────────────────┐ │
              │  │ Layer Norm             │ │
              │  │ Sequence Modeling Block│ │ ← TTT Layer (here!)
              │  │ Add & Norm             │ │
              │  └────────────────────────┘ │
              │  ┌────────────────────────┐ │
              │  │ Layer Norm             │ │
              │  │ Feed Forward           │ │
              │  │ Add & Norm             │ │
              │  └────────────────────────┘ │
              └─────────────────────────────┘
                           ↓
                  Prediction Layer
                           ↓
              Output Probabilities
```

## Two Types of Updates

### 1. TTT Internal Updates (During Forward Pass)

**Location**: Inside the **Sequence Modeling Block** (`seq_modeling_block`)

**What Gets Updated**: TTT-specific parameters (`W1`, `b1`, `W2`, `b2`) that are **per-sample** and **temporary**

**When**: During every forward pass, as the sequence is processed in mini-batches

**How**: Through the `scan()` mechanism that processes mini-batches sequentially

#### Detailed Flow:

```python
# In ttt.py - TTTBase.forward() and TTTLinear.ttt() / TTTMLP.ttt()

1. Input sequence is split into mini-batches (e.g., mini_batch_size=8)
2. For each mini-batch:
   a. Initialize TTT params (W1, b1, W2, b2) from previous mini-batch OR base params
   b. Forward pass through TTT layer
   c. Compute reconstruction target: XV - XK
   d. Compute gradients via dual form or primal form
   e. Update TTT params: W1_new, b1_new, W2_new, b2_new
   f. Use updated params for next mini-batch in same sequence
3. Output final hidden states
```

**Key Points**:
- These updates are **temporary** and **per-sequence**
- They don't persist across different input sequences
- They're part of the forward pass computation
- They adapt to the **current sequence** being processed

**Code Location**: `ttt.py` lines 872-1100 (TTTLinear) and 1112-1299 (TTTMLP)

### 2. ETA Updates (During Evaluation, After Forward Pass)

**Location**: Updates the **projection layers** (`q_proj`, `k_proj`, `v_proj`, `o_proj`) at the **input** of the Sequence Modeling Block

**What Gets Updated**: Standard PyTorch `nn.Linear` layers that produce Q, K, V from input embeddings

**When**: During evaluation (`full_sort_predict()`), **after** the initial forward pass

**How**: Through gradient descent using an SGD optimizer

#### Detailed Flow:

```python
# In main.py - TTT4Rec.full_sort_predict()

1. Initial Forward Pass (with TTT internal updates)
   ├─ TTT processes sequence with mini-batch scan
   ├─ TTT updates its internal W1, b1, W2, b2 per mini-batch
   └─ Output: logits [batch_size, n_items]

2. ETA Filtering
   ├─ Compute entropy for each sample
   ├─ Stage 1: Filter unreliable samples (entropy < e_margin)
   └─ Stage 2: Filter redundant samples (cosine similarity < d_margin)

3. ETA Adaptation (if filtered samples exist)
   ├─ Get filtered samples: item_seq_filtered
   ├─ Forward pass on filtered samples (with gradients enabled)
   │  └─ This forward pass ALSO includes TTT internal updates
   ├─ Compute entropy loss on filtered samples
   ├─ Reweight loss: coeff = 1 / exp(entropy - e_margin)
   ├─ Backward pass: compute gradients for q_proj, k_proj, v_proj, o_proj
   └─ Optimizer step: update projection layer weights

4. Recompute Predictions
   ├─ Forward pass on ALL samples (with updated projection layers)
   ├─ TTT internal updates happen again (with new projection outputs)
   └─ Output: final logits [batch_size, n_items]
```

**Key Points**:
- These updates **persist** across batches during evaluation
- They modify the **base model parameters** (projection layers)
- They use **gradient descent** (SGD optimizer)
- They adapt to the **test distribution** over multiple batches

**Code Location**: `main.py` lines 344-381

## Order of Operations: How TTT and ETA Work Together

### During Evaluation (with ETA enabled):

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Initial Forward Pass                                 │
│ ──────────────────────────────────────────────────────────── │
│                                                               │
│  Input: item_seq [B, L]                                      │
│    ↓                                                          │
│  Embedding Layer                                              │
│    ↓                                                          │
│  TTT Layers (Sequence Modeling Block)                         │
│    │                                                          │
│    ├─ q_proj, k_proj, v_proj (current weights)              │
│    │   ↓                                                      │
│    ├─ TTT Internal Processing:                               │
│    │   ├─ Mini-batch 1: Update W1, b1, W2, b2              │
│    │   ├─ Mini-batch 2: Update W1, b1, W2, b2 (from prev)   │
│    │   └─ ... (sequential updates per mini-batch)            │
│    │                                                          │
│    └─ o_proj (current weights)                               │
│    ↓                                                          │
│  Output: seq_output [B, hidden_size]                        │
│    ↓                                                          │
│  Logits: [B, n_items]                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 2: ETA Filtering                                       │
│ ──────────────────────────────────────────────────────────── │
│                                                               │
│  Compute entropy for each sample                             │
│    ↓                                                          │
│  Stage 1: Filter unreliable (entropy < e_margin)             │
│    ↓                                                          │
│  Stage 2: Filter redundant (similarity < d_margin)          │
│    ↓                                                          │
│  Result: filter_ids_2 (indices of reliable+non-redundant)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 3: ETA Adaptation (if filtered samples exist)          │
│ ──────────────────────────────────────────────────────────── │
│                                                               │
│  Get filtered samples: item_seq[filter_ids_2]               │
│    ↓                                                          │
│  Forward Pass on Filtered Samples (with gradients)          │
│    │                                                          │
│    ├─ TTT Layers                                             │
│    │   ├─ q_proj, k_proj, v_proj (current weights)         │
│    │   │   └─ Gradients computed for these!                 │
│    │   │                                                      │
│    │   ├─ TTT Internal Processing:                          │
│    │   │   └─ Updates W1, b1, W2, b2 (per mini-batch)       │
│    │   │      (These are temporary, not updated by ETA)      │
│    │   │                                                      │
│    │   └─ o_proj (current weights)                          │
│    │       └─ Gradients computed for this!                   │
│    │                                                          │
│    └─ Output: logits_filtered                                │
│        ↓                                                      │
│  Compute entropy loss                                        │
│    ↓                                                          │
│  Reweight: loss = (entropy * coeff).mean()                  │
│    ↓                                                          │
│  Backward: compute gradients for q_proj, k_proj, v_proj, o_proj
│    ↓                                                          │
│  Optimizer.step(): UPDATE q_proj, k_proj, v_proj, o_proj   │
│    └─ These weights are now MODIFIED                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 4: Recompute Predictions                                │
│ ──────────────────────────────────────────────────────────── │
│                                                               │
│  Forward Pass on ALL Samples (no gradients)                │
│    │                                                          │
│    ├─ TTT Layers                                             │
│    │   ├─ q_proj, k_proj, v_proj (UPDATED weights!)         │
│    │   │   └─ These now produce different Q, K, V          │
│    │   │                                                      │
│    │   ├─ TTT Internal Processing:                          │
│    │   │   └─ Updates W1, b1, W2, b2 (per mini-batch)       │
│    │   │      (Uses new Q, K, V from updated projections)   │
│    │   │                                                      │
│    │   └─ o_proj (UPDATED weights!)                          │
│    │                                                          │
│    └─ Output: final logits [B, n_items]                     │
│        └─ These reflect both TTT internal updates AND       │
│           ETA updates to projection layers                    │
└─────────────────────────────────────────────────────────────┘
```

## Key Insights

### 1. **TTT Updates First (During Forward Pass)**

TTT updates its internal parameters (`W1`, `b1`, `W2`, `b2`) **during** the forward pass, as it processes each mini-batch sequentially. These updates are:
- **Temporary**: Only for the current sequence
- **Sequential**: Each mini-batch uses updated params from previous mini-batch
- **Internal**: Part of the forward computation, not gradient-based

### 2. **ETA Updates After (Using Gradients)**

ETA updates the projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) **after** the forward pass, using:
- **Gradient descent**: Standard backpropagation
- **Filtered samples**: Only reliable and non-redundant samples
- **Reweighted loss**: Higher weight for more confident predictions
- **Persistent**: Updates persist across batches

### 3. **They Work Together**

1. **First forward pass**: TTT updates internally, produces initial predictions
2. **ETA filters**: Identifies reliable samples
3. **ETA adapts**: Updates projection layers using filtered samples
   - During this adaptation forward pass, TTT **also** updates internally
   - But ETA only updates the projection layers (not TTT internal params)
4. **Final forward pass**: Uses updated projection layers
   - TTT updates internally again, but now with **better Q, K, V** from updated projections

### 4. **What ETA Updates vs. What TTT Updates**

| Component | Updated By | Type | Persistence |
|-----------|-----------|------|-------------|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | **ETA** | Gradient descent | Persistent across batches |
| `W1`, `b1`, `W2`, `b2` (TTT internal) | **TTT** | Sequential scan | Temporary, per-sequence |
| `learnable_ttt_lr_weight`, `learnable_ttt_lr_bias` | **Training** | Gradient descent | Fixed after training |
| `ttt_norm_weight`, `ttt_norm_bias` | **Training** | Gradient descent | Fixed after training |

## Code References

### TTT Internal Updates:
- **Location**: `ttt.py`
- **Method**: `TTTLinear.ttt()` (lines 956-1100) or `TTTMLP.ttt()` (lines 1112-1299)
- **Mechanism**: `scan()` function (lines 462-489) processes mini-batches sequentially
- **Called from**: `TTTBase.forward()` (lines 881-946)

### ETA Updates:
- **Location**: `main.py`
- **Method**: `TTT4Rec.full_sort_predict()` (lines 270-395)
- **Optimizer initialization**: `_init_eta_optimizer()` (lines 155-177)
- **Update step**: Lines 374-377

## Example: Single Batch Processing

```python
# Batch arrives: item_seq [32, 50] (32 samples, 50 sequence length)

# Step 1: Initial forward pass
seq_output = self.forward(item_seq, item_seq_len)
# Inside forward():
#   - TTT processes 50 tokens in mini-batches of 8
#   - Mini-batch 1-6: TTT updates W1, b1, W2, b2 sequentially
#   - Output: seq_output [32, hidden_size]

logits = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
# logits: [32, n_items]

# Step 2: ETA filtering
filter_ids_1, filter_ids_2, entropys, _ = filter_samples_eta(...)
# filter_ids_2: [12] (12 reliable+non-redundant samples)

# Step 3: ETA adaptation
if filter_ids_2.size(0) > 0:
    item_seq_filtered = item_seq[filter_ids_2]  # [12, 50]
    
    # Forward on filtered samples (with gradients)
    seq_output_filtered = self.forward(item_seq_filtered, ...)
    # Inside forward():
    #   - TTT processes 50 tokens in mini-batches of 8
    #   - TTT updates W1, b1, W2, b2 (temporary, per sequence)
    #   - BUT: q_proj, k_proj, v_proj, o_proj have gradients enabled
    
    logits_filtered = torch.matmul(seq_output_filtered, ...)
    loss = compute_entropy_loss(logits_filtered, ...)
    
    # Backward: gradients flow to q_proj, k_proj, v_proj, o_proj
    loss.backward()
    self.eta_optimizer.step()
    # q_proj, k_proj, v_proj, o_proj weights are NOW UPDATED

# Step 4: Recompute all predictions
seq_output = self.forward(item_seq, item_seq_len)
# Inside forward():
#   - TTT processes with UPDATED q_proj, k_proj, v_proj, o_proj
#   - TTT updates W1, b1, W2, b2 (temporary, per sequence)
#   - Output reflects both updates

logits = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
# Final logits: [32, n_items] with improved predictions
```

## Summary

1. **TTT updates happen FIRST** (during forward pass, per mini-batch)
   - Updates: `W1`, `b1`, `W2`, `b2`
   - Mechanism: Sequential scan
   - Scope: Temporary, per-sequence

2. **ETA updates happen AFTER** (during evaluation, after filtering)
   - Updates: `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - Mechanism: Gradient descent
   - Scope: Persistent, across batches

3. **They complement each other**:
   - TTT adapts to the **current sequence** (temporary adaptation)
   - ETA adapts to the **test distribution** (persistent adaptation)
   - ETA improves the **input** to TTT (better Q, K, V projections)
   - TTT then processes these improved inputs with its internal adaptation

4. **Order matters**:
   - Initial forward → Filter → Adapt projections → Recompute
   - The final recomputation uses both: updated projections (from ETA) + TTT internal adaptation

