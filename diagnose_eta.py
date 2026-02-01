"""
Diagnostic script to analyze why ETA might be underperforming.

This script helps verify:
1. Distribution shift between train/test
2. Entropy distribution during training
3. How many samples ETA would filter
"""

import torch
import numpy as np
from collections import Counter
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataset import SequentialDataset
from main import CustomSequentialDataset, TTT4Rec
from ttt import TTTConfig
import matplotlib.pyplot as plt

def analyze_distribution_shift(config_path='config.yaml'):
    """Analyze distribution shift between train and test sets."""
    print("=" * 60)
    print("ANALYZING DISTRIBUTION SHIFT")
    print("=" * 60)
    
    rec_config = Config(model=TTT4Rec, config_file_list=[config_path])
    dataset = CustomSequentialDataset(rec_config)
    train_data, valid_data, test_data = data_preparation(rec_config, dataset)
    
    # Get item frequencies
    train_items = train_data.dataset.inter_feat['item_id'].numpy()
    test_items = test_data.dataset.inter_feat['item_id'].numpy()
    
    train_item_freq = Counter(train_items)
    test_item_freq = Counter(test_items)
    
    # Calculate overlap
    train_items_set = set(train_items)
    test_items_set = set(test_items)
    overlap = len(train_items_set & test_items_set)
    total_test_items = len(test_items_set)
    
    print(f"\nItem Vocabulary:")
    print(f"  Train unique items: {len(train_items_set)}")
    print(f"  Test unique items: {len(test_items_set)}")
    print(f"  Overlap: {overlap}/{total_test_items} ({100*overlap/total_test_items:.1f}%)")
    
    # Calculate popularity distribution similarity (KL divergence)
    all_items = train_items_set | test_items_set
    train_probs = np.array([train_item_freq.get(item, 0) for item in all_items])
    test_probs = np.array([test_item_freq.get(item, 0) for item in all_items])
    
    train_probs = train_probs / train_probs.sum()
    test_probs = test_probs / test_probs.sum()
    
    # KL divergence
    kl_div = np.sum(test_probs * np.log((test_probs + 1e-10) / (train_probs + 1e-10)))
    print(f"\nPopularity Distribution:")
    print(f"  KL Divergence: {kl_div:.4f} (lower = more similar)")
    print(f"  Interpretation: {'Minimal shift' if kl_div < 0.1 else 'Moderate shift' if kl_div < 1.0 else 'Large shift'}")
    
    # Temporal analysis if timestamps exist
    if 'timestamp' in train_data.dataset.inter_feat.columns:
        train_times = train_data.dataset.inter_feat['timestamp'].numpy()
        test_times = test_data.dataset.inter_feat['timestamp'].numpy()
        
        print(f"\nTemporal Analysis:")
        print(f"  Train time range: {train_times.min()} to {train_times.max()}")
        print(f"  Test time range: {test_times.min()} to {test_times.max()}")
        print(f"  Time gap: {test_times.min() - train_times.max()}")
        print(f"  Interpretation: {'Temporal split' if test_times.min() > train_times.max() else 'Random split'}")
    
    return {
        'item_overlap': overlap / total_test_items,
        'kl_divergence': kl_div,
        'has_temporal_split': test_times.min() > train_times.max() if 'timestamp' in train_data.dataset.inter_feat.columns else None
    }

def analyze_entropy_distribution(model, data_loader, device='cpu', num_batches=10):
    """Analyze entropy distribution during training/inference."""
    print("\n" + "=" * 60)
    print("ANALYZING ENTROPY DISTRIBUTION")
    print("=" * 60)
    
    model.eval()
    entropies = []
    filter_stats = {'total': 0, 'filtered_1': 0, 'filtered_2': 0}
    
    with torch.no_grad():
        for batch_idx, interaction in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            interaction = interaction.to(device)
            item_seq = interaction[model.ITEM_SEQ]
            item_seq_len = interaction[model.ITEM_SEQ_LEN]
            
            # Forward pass
            seq_output = model.forward(item_seq, item_seq_len)
            
            # Compute logits
            if model.loss_type == 'CE':
                test_item_emb = model.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            else:  # BPR
                pos_items = interaction[model.POS_ITEM_ID]
                neg_items = interaction[model.NEG_ITEM_ID]
                pos_items_emb = model.item_embedding(pos_items)
                neg_items_emb = model.item_embedding(neg_items)
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
                logits = torch.stack([pos_score, neg_score], dim=1)
            
            # Compute entropy
            from eta_utils import softmax_entropy
            batch_entropies = softmax_entropy(logits)
            entropies.extend(batch_entropies.cpu().numpy())
            
            # Simulate ETA filtering
            if model.use_eta:
                from eta_utils import filter_samples_eta
                filter_ids_1, filter_ids_2, _, _ = filter_samples_eta(
                    logits, model.current_model_probs, model.e_margin, model.d_margin
                )
                filter_stats['total'] += logits.size(0)
                filter_stats['filtered_1'] += filter_ids_1.size(0)
                filter_stats['filtered_2'] += filter_ids_2.size(0)
    
    entropies = np.array(entropies)
    
    print(f"\nEntropy Statistics:")
    print(f"  Mean: {entropies.mean():.4f}")
    print(f"  Std: {entropies.std():.4f}")
    print(f"  Min: {entropies.min():.4f}")
    print(f"  Max: {entropies.max():.4f}")
    print(f"  Median: {np.median(entropies):.4f}")
    
    if model.use_eta:
        print(f"\nETA Filtering (with e_margin={model.e_margin:.4f}):")
        print(f"  Total samples: {filter_stats['total']}")
        print(f"  After entropy filter: {filter_stats['filtered_1']} ({100*filter_stats['filtered_1']/filter_stats['total']:.1f}%)")
        print(f"  After redundancy filter: {filter_stats['filtered_2']} ({100*filter_stats['filtered_2']/filter_stats['total']:.1f}%)")
        print(f"  Filtered out: {filter_stats['total'] - filter_stats['filtered_2']} ({100*(filter_stats['total'] - filter_stats['filtered_2'])/filter_stats['total']:.1f}%)")
    
    # Compare to e_margin
    if model.use_eta:
        below_threshold = (entropies < model.e_margin).sum()
        print(f"\nSamples below e_margin ({model.e_margin:.4f}): {below_threshold}/{len(entropies)} ({100*below_threshold/len(entropies):.1f}%)")
        print(f"  Interpretation: {'Too aggressive' if below_threshold < 0.1 * len(entropies) else 'Moderate' if below_threshold < 0.5 * len(entropies) else 'Lenient'}")
    
    return entropies

def analyze_sparsity(model, data_loader, device='cpu', num_batches=10):
    """Analyze sparsity in probability distributions."""
    print("\n" + "=" * 60)
    print("ANALYZING SPARSITY IN PROBABILITY DISTRIBUTIONS")
    print("=" * 60)
    
    model.eval()
    sparsity_stats = {
        'total_items': model.n_items,
        'avg_topk_mass': [],
        'avg_effective_items': [],  # Items with prob > 1/n_items
        'sparsity_ratios': []
    }
    
    with torch.no_grad():
        for batch_idx, interaction in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            interaction = interaction.to(device)
            item_seq = interaction[model.ITEM_SEQ]
            item_seq_len = interaction[model.ITEM_SEQ_LEN]
            
            # Forward pass
            seq_output = model.forward(item_seq, item_seq_len)
            
            # Compute logits and probabilities
            if model.loss_type == 'CE':
                test_item_emb = model.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            else:  # BPR
                pos_items = interaction[model.POS_ITEM_ID]
                neg_items = interaction[model.NEG_ITEM_ID]
                pos_items_emb = model.item_embedding(pos_items)
                neg_items_emb = model.item_embedding(neg_items)
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
                logits = torch.stack([pos_score, neg_score], dim=1)
            
            probs = logits.softmax(1)
            
            # Analyze sparsity
            for i in range(probs.size(0)):
                prob_vec = probs[i]
                
                # Top-k mass (k=10)
                top10_mass = torch.topk(prob_vec, min(10, prob_vec.size(0)))[0].sum().item()
                sparsity_stats['avg_topk_mass'].append(top10_mass)
                
                # Effective items (prob > 1/n_items)
                threshold = 1.0 / model.n_items
                effective_items = (prob_vec > threshold).sum().item()
                sparsity_stats['avg_effective_items'].append(effective_items)
                
                # Sparsity ratio (items with prob < threshold)
                sparsity_ratio = 1.0 - (effective_items / model.n_items)
                sparsity_stats['sparsity_ratios'].append(sparsity_ratio)
    
    print(f"\nSparsity Statistics:")
    print(f"  Total items: {sparsity_stats['total_items']}")
    print(f"  Average top-10 mass: {np.mean(sparsity_stats['avg_topk_mass']):.4f} ({100*np.mean(sparsity_stats['avg_topk_mass']):.1f}%)")
    print(f"  Average effective items (prob > 1/n): {np.mean(sparsity_stats['avg_effective_items']):.1f}")
    print(f"  Average sparsity ratio: {np.mean(sparsity_stats['sparsity_ratios']):.4f} ({100*np.mean(sparsity_stats['sparsity_ratios']):.1f}%)")
    print(f"  Interpretation: {'Extremely sparse' if np.mean(sparsity_stats['sparsity_ratios']) > 0.99 else 'Very sparse' if np.mean(sparsity_stats['sparsity_ratios']) > 0.95 else 'Moderately sparse'}")
    
    # Compare to ETA's assumptions
    print(f"\nETA Assumptions vs Reality:")
    print(f"  ETA assumes: Dense distributions (all items possible)")
    print(f"  Reality: {100*np.mean(sparsity_stats['sparsity_ratios']):.1f}% sparsity (most items have near-zero probability)")
    print(f"  Top-10 items contain: {100*np.mean(sparsity_stats['avg_topk_mass']):.1f}% of probability mass")
    print(f"  → ETA's entropy and cosine similarity metrics are unreliable on such sparse distributions")
    
    return sparsity_stats

def main():
    """Run all diagnostics."""
    print("\n" + "=" * 60)
    print("ETA DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    # 1. Analyze distribution shift
    shift_results = analyze_distribution_shift()
    
    # 2. Load model and analyze entropy
    rec_config = Config(model=TTT4Rec, config_file_list=['config.yaml'])
    dataset = CustomSequentialDataset(rec_config)
    train_data, valid_data, test_data = data_preparation(rec_config, dataset)
    
    ttt_config = TTTConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=50,
        rope_theta=1000.0,
        ttt_layer_type="mlp",
        ttt_base_lr=1.0,
        mini_batch_size=8,
        use_gate=False,
        pre_conv=False,
        share_qk=False,
        use_eta=True,  # Enable to see filtering
        e_margin=None,
        d_margin=0.5
    )
    
    model = TTT4Rec(ttt_config, rec_config, train_data.dataset)
    
    # Analyze sparsity
    sparsity_results = analyze_sparsity(model, test_data, device=rec_config['device'])
    
    # Analyze on test set
    print("\nAnalyzing on TEST set:")
    test_entropies = analyze_entropy_distribution(model, test_data, device=rec_config['device'])
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    if shift_results['kl_divergence'] < 0.1:
        print("✓ Minimal distribution shift detected")
        print("  → ETA may not be beneficial")
    elif shift_results['kl_divergence'] < 1.0:
        print("⚠ Moderate distribution shift detected")
        print("  → ETA might help, but filtering during training is problematic")
    else:
        print("⚠ Large distribution shift detected")
        print("  → ETA could help, but apply only during evaluation, not training")
    
    if sparsity_results['sparsity_ratios']:
        avg_sparsity = np.mean(sparsity_results['sparsity_ratios'])
        if avg_sparsity > 0.99:
            print(f"\n⚠⚠⚠ EXTREME SPARSITY DETECTED: {100*avg_sparsity:.1f}%")
            print("  → This is a FUNDAMENTAL problem for ETA")
            print("  → Entropy and cosine similarity are unreliable on sparse distributions")
            print("  → ETA's assumptions don't hold in recommendation systems")
            print("  → STRONG RECOMMENDATION: Disable ETA")
    
    if model.use_eta:
        filter_ratio = (len(test_entropies) - (test_entropies < model.e_margin).sum()) / len(test_entropies)
        if filter_ratio > 0.5:
            print(f"\n⚠ ETA filters out {100*filter_ratio:.1f}% of samples")
            print("  → This removes valuable learning signals")
            print("  → Recommendation: Disable ETA or apply only during evaluation")

if __name__ == '__main__':
    main()

