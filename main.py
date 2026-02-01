"""
   Modified from https://github.com/chengkai-liu/Mamba4Rec
"""

# %cd /gdrive/My Drive/TTT4Rec

import sys
import logging
import os
from logging import getLogger

import torch
import torch.nn.functional as F
from torch import nn

# Monkey-patch torch.load to set weights_only=False by default
# This fixes RecBole checkpoint loading with PyTorch 2.6+
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.data.dataset import SequentialDataset
from recbole.model.loss import BPRLoss

from ttt import TTTModel, TTTConfig
from eta_utils import filter_samples_eta, update_recent_items, update_model_probs, compute_eta_loss_coefficients, topk_entropy, softmax_entropy


class CustomSequentialDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

    def leave_one_out(self, group_by, leave_one_num=1, leave_one_mode='valid_and_test'):
        """
        rewrite function "leave_one_out" to deal with specific train/valid/test split ratios
        """
        split_ratios = self.config['split_ratio']
        self.logger.info(f'Split ratio for train/valid/test: {split_ratios}')

        train_data, valid_data, test_data = self.split_by_ratio(ratios=split_ratios, group_by=group_by)

        return train_data, valid_data, test_data

class TTT4Rec(SequentialRecommender):
    def __init__(self, ttt_config, rec_config, dataset):
        super(TTT4Rec, self).__init__(rec_config, dataset)

        self.hidden_size = rec_config["hidden_size"]
        self.loss_type = rec_config["loss_type"]
        self.dropout_prob = rec_config["dropout_prob"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.ttt_layers = TTTModel(ttt_config)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # ETA parameters
        self.use_eta = ttt_config.use_eta
        # Flag to disable TTT adaptation (for testing saved models without adaptation)
        self.disable_ttt_adaptation = getattr(ttt_config, 'disable_ttt_adaptation', False)
        if self.use_eta:
            # e_margin will be set from first test batch (percentile-based)
            # If manually specified, use that value
            self.e_margin = ttt_config.e_margin  # None = auto from first test batch
            self.redundancy_threshold = ttt_config.d_margin  # For frequency: max_frequency, for cosine: similarity threshold
            self.redundancy_filter_type = ttt_config.redundancy_filter_type  # 'frequency' or 'cosine'
            self.eta_top_k = ttt_config.eta_top_k
            self.eta_use_topk = ttt_config.eta_use_topk
            # ETA is ONLY applied during evaluation (predict/full_sort_predict)
            # Redundancy filtering data: depends on filter type
            if self.redundancy_filter_type == 'frequency':
                self.recent_items = None  # For frequency-based filtering
                self.current_model_probs = None
            else:  # cosine
                self.recent_items = None
                self.current_model_probs = None  # Moving average of model probabilities
            # Flag to track if e_margin has been calibrated from first test batch
            self.e_margin_calibrated = False
            # Statistics
            self.num_samples_total = 0  # Total samples processed
            self.num_samples_update_1 = 0  # Reliable samples (after entropy filter)
            self.num_samples_update_2 = 0  # Reliable + non-redundant samples (after both filters)
            # ETA optimizer will be initialized after model is fully set up
            self.eta_optimizer = None
            self.eta_optimizer_initialized = False

        self.apply(self._init_weights)
    
    def get_eta_statistics(self):
        """
        Get ETA filtering statistics.
        
        Returns:
            dict: Statistics including total samples, filtered samples, and percentages
        """
        if not self.use_eta or self.num_samples_total == 0:
            return None
        
        filtered_by_entropy = self.num_samples_total - self.num_samples_update_1
        filtered_by_redundancy = self.num_samples_update_1 - self.num_samples_update_2
        filtered_total = self.num_samples_total - self.num_samples_update_2
        
        stats = {
            'total_samples': self.num_samples_total,
            'reliable_samples': self.num_samples_update_1,
            'reliable_and_non_redundant': self.num_samples_update_2,
            'filtered_by_entropy': filtered_by_entropy,
            'filtered_by_redundancy': filtered_by_redundancy,
            'filtered_total': filtered_total,
            'pct_reliable': (self.num_samples_update_1 / self.num_samples_total) * 100,
            'pct_final': (self.num_samples_update_2 / self.num_samples_total) * 100,
            'pct_filtered': (filtered_total / self.num_samples_total) * 100,
            'pct_filtered_entropy': (filtered_by_entropy / self.num_samples_total) * 100,
            'pct_filtered_redundancy': (filtered_by_redundancy / self.num_samples_total) * 100,
        }
        return stats
    
    def reset_eta_statistics(self):
        """Reset ETA statistics (useful between validation and test phases)."""
        if self.use_eta:
            self.num_samples_total = 0
            self.num_samples_update_1 = 0
            self.num_samples_update_2 = 0
            self.e_margin_calibrated = False
            self.recent_items = None
            self.current_model_probs = None
    
    def _init_eta_optimizer(self):
        """Initialize ETA optimizer for test-time adaptation on TTT layer parameters."""
        if self.eta_optimizer_initialized or not self.use_eta:
            return
        
        # Adapt TTT projection layers (q_proj, k_proj, v_proj, o_proj) which are fully differentiable
        # These are standard linear layers that participate in the forward pass
        ttt_params = []
        for layer in self.ttt_layers.layers:
            if hasattr(layer, 'seq_modeling_block'):
                # Get projection layer parameters (these are used directly in forward pass)
                for name, param in layer.seq_modeling_block.named_parameters():
                    if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name:
                        # Ensure parameters require gradients
                        param.requires_grad_(True)
                        ttt_params.append(param)
        
        if len(ttt_params) > 0:
            self.eta_optimizer = torch.optim.SGD(ttt_params, lr=0.001, momentum=0.9)
            self.eta_optimizer_initialized = True
        else:
            self.eta_optimizer = None
            self.eta_optimizer_initialized = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        batch_size, seq_length, hidden_size = item_emb.shape
        # Use the same device as item_emb (works for both CPU and GPU)
        device = item_emb.device
        attention_mask = (torch.arange(seq_length, device=device).expand(batch_size, seq_length)) < item_seq_len.unsqueeze(1)
        attention_mask = attention_mask.long()

        item_emb = self.ttt_layers(inputs_embeds=item_emb,
                                   attention_mask = attention_mask).last_hidden_state

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            
            # ETA is NOT applied during training - only during evaluation
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B, n_items]
            
            # ETA is NOT applied during training - only during evaluation
            loss = self.loss_fct(logits, pos_items)
            return loss

    def _calibrate_e_margin_from_batch(self, logits):
        """
        Calibrate e_margin from first test batch using percentile (60% quantile or mean).
        This replaces theoretical max entropy with actual operating range.
        """
        if self.e_margin_calibrated or self.e_margin is not None:
            return  # Already calibrated or manually set
        
        # Compute entropy for this batch
        if self.eta_use_topk and logits.size(1) > self.eta_top_k:
            entropys = topk_entropy(logits, k=self.eta_top_k)
        else:
            entropys = softmax_entropy(logits)
        
        # Use 60% quantile (or mean) as threshold
        # This calibrates to actual operating range rather than theoretical max
        e_margin_candidate = torch.quantile(entropys, 0.4).item()
        # Alternative: use mean
        # e_margin_candidate = entropys.mean().item()
        
        self.e_margin = e_margin_candidate
        self.e_margin_calibrated = True
        
        # Log calibration (logger is already imported at module level)
        logger = getLogger()
        logger.info(set_color("ETA e_margin calibrated", "green") + 
                   f" from first batch: {self.e_margin:.4f} (60% quantile)")

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        
        # Note: ETA filtering is only applied in full_sort_predict() where we have
        # full logits and can get predicted items. In predict(), we only compute
        # scores for a specific test_item (ground truth), so no ETA filtering here.
        
        return scores

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items. Called per batch during evaluation.
        
        ETA behavior with batched evaluation:
        - First batch: Calibrates e_margin from 60% quantile of batch entropy
        - Subsequent batches: Uses calibrated e_margin and accumulated recent_items
        - Recent items accumulate across all batches in current evaluation phase
        - ETA filters samples and adapts model on filtered samples
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # Initial forward pass to get predictions
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        logits = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        
        # ETA is applied during evaluation
        if self.use_eta:
            # Initialize optimizer if not already done
            self._init_eta_optimizer()
            
            # Calibrate e_margin from first batch if not already done
            # This happens once per evaluation phase (validation or test)
            self._calibrate_e_margin_from_batch(logits)
            
            # Track total samples
            batch_size = logits.size(0)
            self.num_samples_total += batch_size
            
            # Prepare redundancy data based on filter type
            if self.redundancy_filter_type == 'frequency':
                redundancy_data = self.recent_items
            else:  # cosine
                redundancy_data = self.current_model_probs
            
            # Apply ETA filtering
            filter_ids_1, filter_ids_2, entropys, redundancy_output = filter_samples_eta(
                logits,
                redundancy_data,
                self.e_margin,
                self.redundancy_threshold,
                redundancy_filter_type=self.redundancy_filter_type,
                use_topk=self.eta_use_topk and logits.size(1) > self.eta_top_k,
                top_k=self.eta_top_k,
                min_samples_ratio=0.1
            )
            
            # Update statistics
            self.num_samples_update_1 += filter_ids_1.size(0)
            self.num_samples_update_2 += filter_ids_2.size(0)
            
            # Update redundancy filtering data based on filter type
            if self.redundancy_filter_type == 'frequency':
                # For frequency mode: update recent items history
                predicted_items = logits.argmax(dim=1)  # Get predicted items
                self.recent_items = update_recent_items(
                    self.recent_items, 
                    predicted_items,
                    max_history=100
                )
            else:  # cosine
                # For cosine mode: update moving average of probabilities
                probs = logits.softmax(1)  # [batch_size, num_items]
                self.current_model_probs = update_model_probs(
                    self.current_model_probs,
                    probs,
                    top_k=self.eta_top_k,
                    alpha=0.1
                )
            
            # ETA ADAPTATION: Use filtered samples to adapt model
            # Skip adaptation if disabled (for testing saved models without adaptation)
            if filter_ids_2.size(0) > 0 and self.eta_optimizer is not None and not self.disable_ttt_adaptation:
                # Get filtered samples
                item_seq_filtered = item_seq[filter_ids_2]
                item_seq_len_filtered = item_seq_len[filter_ids_2]
                
                # Enable gradients for adaptation
                # Use torch.enable_grad() context to ensure gradients are computed
                with torch.enable_grad():
                    # Temporarily set to train mode for dropout/batch norm
                    was_training = self.training
                    self.train()
                    
                    # Forward pass on filtered samples (with gradients)
                    seq_output_filtered = self.forward(item_seq_filtered, item_seq_len_filtered)
                    logits_filtered = torch.matmul(
                        seq_output_filtered, test_items_emb.transpose(0, 1)
                    )  # [filtered_B, n_items]
                    
                    # Compute entropy loss on filtered samples (like EATA)
                    if self.eta_use_topk and logits_filtered.size(1) > self.eta_top_k:
                        entropys_filtered = topk_entropy(logits_filtered, k=self.eta_top_k)
                    else:
                        entropys_filtered = softmax_entropy(logits_filtered)
                    
                    # Reweight entropy loss
                    coeff = compute_eta_loss_coefficients(entropys_filtered, self.e_margin, normalize=True)
                    loss = (entropys_filtered * coeff).mean()
                    
                    # Update model parameters (only TTT layers)
                    self.eta_optimizer.zero_grad()
                    loss.backward()
                    self.eta_optimizer.step()
                    
                    # Restore original training state
                    if not was_training:
                        self.eval()
                
                # Recompute predictions on all samples with adapted model
                with torch.no_grad():
                    seq_output = self.forward(item_seq, item_seq_len)
                    logits = torch.matmul(
                        seq_output, test_items_emb.transpose(0, 1)
                    )  # [B, n_items]
            
            # Return scores for all samples (RecBole requirement)
            scores = logits
        else:
            scores = logits
        
        return scores

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
        use_eta=True,
        e_margin=None,
        redundancy_filter_type='cosine',
        d_margin=0.3,
        eta_top_k=50,  # Number of top items for top-k metrics (recommended for sparse distributions)
        eta_use_topk=True)  # Use top-k metrics (recommended for recommendation systems with >99% sparsity)

rec_config = Config(model=TTT4Rec, config_file_list=['config.yaml'])

# Initialize dummy distributed process group if not already initialized
# This is needed because RecBole calls torch.distributed.barrier() even in single-process mode
if not torch.distributed.is_initialized():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(
        backend='gloo',
        init_method='env://',
        world_size=1,
        rank=0
    )

init_seed(rec_config['seed'], rec_config['reproducibility'])

# logger initialization
init_logger(rec_config)
logger = getLogger()
# logger.setLevel(logging.INFO)
# logger.info(sys.argv)
# logger.info(rec_config)

# dataset filtering
# dataset = create_dataset(rec_config)
dataset = CustomSequentialDataset(rec_config)
# logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(rec_config, dataset)

# model loading and initialization
# local_rank defaults to 0 in single-process mode (RecBole sets it automatically)
try:
    local_rank = rec_config["local_rank"]
except (KeyError, AttributeError):
    local_rank = 0
init_seed(rec_config["seed"] + local_rank, rec_config["reproducibility"])
model = TTT4Rec(ttt_config, rec_config, train_data.dataset).to(rec_config['device'])
logger.info(model)

# Log ETA configuration if enabled
if model.use_eta:
    e_margin_str = "auto (from first test batch)" if model.e_margin is None else f"{model.e_margin:.4f}"
    logger.info(set_color("ETA enabled", "green") + 
                f": e_margin={e_margin_str}, redundancy_threshold={model.redundancy_threshold:.2f}, "
                f"top_k={model.eta_top_k}, use_topk={model.eta_use_topk}")
else:
    logger.info(set_color("ETA disabled", "yellow") + " (using all samples)")

transform = construct_transform(rec_config)
flops = get_flops(model, dataset, rec_config["device"], logger, transform)
logger.info(set_color("FLOPs", "blue") + f": {flops}")

# trainer loading and initialization
trainer = Trainer(rec_config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(
    train_data, valid_data, show_progress=rec_config["show_progress"]
)

# Log ETA filtering statistics from validation if enabled
if model.use_eta:
    eta_stats = model.get_eta_statistics()
    if eta_stats:
        logger.info(set_color("ETA Filtering Statistics (Validation Phase)", "cyan"))
        logger.info(f"  Total samples processed: {eta_stats['total_samples']:,}")
        logger.info(f"  Reliable samples (after entropy filter): {eta_stats['reliable_samples']:,} ({eta_stats['pct_reliable']:.2f}%)")
        logger.info(f"  Final samples (after both filters): {eta_stats['reliable_and_non_redundant']:,} ({eta_stats['pct_final']:.2f}%)")
        logger.info(f"  Filtered by entropy: {eta_stats['filtered_by_entropy']:,} ({eta_stats['pct_filtered_entropy']:.2f}%)")
        logger.info(f"  Filtered by redundancy: {eta_stats['filtered_by_redundancy']:,} ({eta_stats['pct_filtered_redundancy']:.2f}%)")
        logger.info(f"  Total filtered: {eta_stats['filtered_total']:,} ({eta_stats['pct_filtered']:.2f}%)")
        logger.info(f"  → Model adapted on {eta_stats['pct_final']:.2f}% of samples ({eta_stats['reliable_and_non_redundant']:,}/{eta_stats['total_samples']:,})")
    
    # Reset statistics for test phase
    model.reset_eta_statistics()
    model.e_margin_calibrated = False  # Recalibrate on test set
    # Reset redundancy filtering data (handled by reset_eta_statistics)

# model evaluation
test_result = trainer.evaluate(
    test_data, show_progress=rec_config["show_progress"]
)

# Log ETA filtering statistics if enabled
if model.use_eta:
    eta_stats = model.get_eta_statistics()
    if eta_stats:
        logger.info(set_color("ETA Filtering Statistics (Test Phase)", "cyan"))
        logger.info(f"  Total samples processed: {eta_stats['total_samples']:,}")
        logger.info(f"  Reliable samples (after entropy filter): {eta_stats['reliable_samples']:,} ({eta_stats['pct_reliable']:.2f}%)")
        logger.info(f"  Final samples (after both filters): {eta_stats['reliable_and_non_redundant']:,} ({eta_stats['pct_final']:.2f}%)")
        logger.info(f"  Filtered by entropy: {eta_stats['filtered_by_entropy']:,} ({eta_stats['pct_filtered_entropy']:.2f}%)")
        logger.info(f"  Filtered by redundancy: {eta_stats['filtered_by_redundancy']:,} ({eta_stats['pct_filtered_redundancy']:.2f}%)")
        logger.info(f"  Total filtered: {eta_stats['filtered_total']:,} ({eta_stats['pct_filtered']:.2f}%)")
        logger.info(f"  → Model adapted on {eta_stats['pct_final']:.2f}% of samples ({eta_stats['reliable_and_non_redundant']:,}/{eta_stats['total_samples']:,})")

# environment_tb = get_environment(rec_config)
# logger.info(
#     "The running environment of this training is as follows:\n"
#     + environment_tb.draw()
# )

logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
logger.info(set_color("test result", "yellow") + f": {test_result}")



