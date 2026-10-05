

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import warnings
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx
from models.model import MultiTaskGCN
from data.scripts.adjacancy_matrics import AdjacencyMatrixProcessor
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from data.scripts.features import FeatureBuilder
from data.scripts.losses import AdaptiveHeadLoss
from data.scripts.interpreters import NodeInterpreter
from data.scripts.viz import Viz
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import f1_score
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from scipy import stats
from sklearn.utils import resample
from models.confusion import save_task_confusion   # if trainer.py is imported as part of the models package

__all__ = ["Trainer", "make_overview_radars", "run_stratified_kfold", "utilis"]



def compute_bootstrap_ci(metric_values, n_bootstrap=1000, ci=95):
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        metric_values: Array of metric values (e.g., accuracies per bootstrap sample)
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval percentage (e.g., 95)
    
    Returns:
        (lower_bound, upper_bound, mean, std)
    """
    if len(metric_values) == 0:
        return (0, 0, 0, 0)
    
    alpha = 100 - ci
    lower_percentile = alpha / 2
    upper_percentile = 100 - alpha / 2
    
    lower = np.percentile(metric_values, lower_percentile)
    upper = np.percentile(metric_values, upper_percentile)
    mean = np.mean(metric_values)
    std = np.std(metric_values)
    
    return (lower, upper, mean, std)


def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, ci=95, random_state=42):
    """
    Bootstrap a metric (e.g., accuracy, F1, AUC) with confidence intervals.
    
    Args:
        y_true: True labels
        y_pred: Predictions or probabilities
        metric_func: Function that takes (y_true, y_pred) and returns a scalar metric
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval percentage
        random_state: Random seed for reproducibility
    
    Returns:
        dict with 'mean', 'std', 'ci_lower', 'ci_upper', 'values'
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample with replacement
        indices = resample(np.arange(n_samples), n_samples=n_samples, replace=True)
        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]
        
        try:
            score = metric_func(y_true_bs, y_pred_bs)
            if np.isfinite(score):
                bootstrap_scores.append(score)
        except Exception:
            continue
    
    if len(bootstrap_scores) == 0:
        return {'mean': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0, 'values': []}
    
    lower, upper, mean, std = compute_bootstrap_ci(bootstrap_scores, n_bootstrap, ci)
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': lower,
        'ci_upper': upper,
        'values': bootstrap_scores
    }


def compare_models_statistically(metric_values_model1, metric_values_model2, test='wilcoxon'):
    """
    Perform statistical test to compare two models.
    
    Args:
        metric_values_model1: Array of metric values for model 1 (e.g., bootstrap scores)
        metric_values_model2: Array of metric values for model 2
        test: 'wilcoxon' or 'ttest'
    
    Returns:
        dict with 'statistic', 'p_value', 'significant'
    """
    if test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(metric_values_model1, metric_values_model2)
    elif test == 'ttest':
        statistic, p_value = stats.ttest_rel(metric_values_model1, metric_values_model2)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def compute_all_metrics_with_ci(y_true, y_pred, y_proba=None, n_bootstrap=1000):
    """
    Compute all classification metrics with confidence intervals.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC)
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        dict with metrics and their confidence intervals
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {}
    
    # Accuracy
    acc_result = bootstrap_metric(y_true, y_pred, accuracy_score, n_bootstrap)
    metrics['accuracy'] = acc_result
    
    # Precision (macro)
    precision_result = bootstrap_metric(y_true, y_pred, 
                                         lambda yt, yp: precision_score(yt, yp, average='macro', zero_division=0),
                                         n_bootstrap)
    metrics['precision'] = precision_result
    
    # Recall (macro)
    recall_result = bootstrap_metric(y_true, y_pred,
                                      lambda yt, yp: recall_score(yt, yp, average='macro', zero_division=0),
                                      n_bootstrap)
    metrics['recall'] = recall_result
    
    # F1 (macro)
    f1_result = bootstrap_metric(y_true, y_pred,
                                  lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
                                  n_bootstrap)
    metrics['f1'] = f1_result
    
    # AUC (if probabilities provided)
    if y_proba is not None:
        try:
            auc_result = bootstrap_metric(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba,
                                           lambda yt, yp: roc_auc_score(yt, yp),
                                           n_bootstrap)
            metrics['auc'] = auc_result
        except Exception:
            metrics['auc'] = {'mean': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0, 'values': []}
    
    return metrics


def compute_regression_metrics_with_ci(y_true, y_pred, n_bootstrap=1000):
    """
    Compute regression metrics with confidence intervals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        dict with metrics and their confidence intervals
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {}
    
    # R²
    r2_result = bootstrap_metric(y_true, y_pred, r2_score, n_bootstrap)
    metrics['r2'] = r2_result
    
    # RMSE
    rmse_result = bootstrap_metric(y_true, y_pred, 
                                    lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
                                    n_bootstrap)
    metrics['rmse'] = rmse_result
    
    # MAE
    mae_result = bootstrap_metric(y_true, y_pred, mean_absolute_error, n_bootstrap)
    metrics['mae'] = mae_result
    
    return metrics


# ============================================================================
# FeatureInterpreter (unchanged – kept for completeness)
# ============================================================================
class FeatureInterpreter:
    """
    Permutation importance over *feature groups* based on FeatureBuilder layout,
    plus frequency-band importance derived from RFFT bin groups.
    """
    BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 70)}

    def __init__(self, model, num_nodes=19, device="cpu"):
        self.model = model
        self.num_nodes = num_nodes
        self.device = device

    @staticmethod
    def default_groups(in_dim: int, task: str, with_shapes: bool = True, with_complexity: bool = True):
        import numpy as _np
        groups = []
        if task in ("detection", "classification"):
            i = 0
            groups.append(("Mean Spectrum (100)", _np.arange(i, i+100))); i += 100
            groups.extend([
                ("BandPower Δ..γ (5)", _np.arange(i, i+5)),
                ("BandPower Total (1)", _np.arange(i+5, i+6)),
                ("RelPower Δ..γ (5)", _np.arange(i+6, i+11)),
                ("Ratios α/θ, β/α (2)", _np.arange(i+11, i+13)),
                ("Spectral Entropy (1)", _np.arange(i+13, i+14))
            ]); i += 14
            if with_shapes:
                groups.append(("Spectral Shapes (6)", _np.arange(i, i+6))); i += 6
            if with_complexity:
                groups.append(("Permutation Entropy (1)", _np.arange(i, i+1))); i += 1
        else:
            i = 0
            groups.append(("Mean Spectrum (100)", _np.arange(i, i+100))); i += 100
            groups.append(("Std Spectrum (100)", _np.arange(i, i+100))); i += 100
            groups.extend([
                ("BandPower Δ..γ (5)", _np.arange(i, i+5)),
                ("BandPower Total (1)", _np.arange(i+5, i+6)),
                ("RelPower Δ..γ (5)", _np.arange(i+6, i+11)),
                ("Ratios α/θ, β/α (2)", _np.arange(i+11, i+13)),
                ("Spectral Entropy (1)", _np.arange(i+13, i+14))
            ]); i += 14
            if with_shapes:
                groups.append(("Spectral Shapes (6)", _np.arange(i, i+6))); i += 6
            if with_complexity:
                groups.append(("Permutation Entropy (1)", _np.arange(i, i+1))); i += 1

        used = np.concatenate([g[1] for g in groups]) if groups else np.array([], dtype=int)
        assert used.max() < in_dim, f"group index overflow: max={used.max()} in_dim={in_dim}"
        return groups

    @staticmethod
    def _band_indices_for_bins(rfft_bins=100, band_span=(1,4), start_col=0):
        lo, hi = band_span
        lo = max(1, int(np.floor(lo))); hi = min(rfft_bins, int(np.floor(hi)))
        idx = np.arange(start_col + (lo-1), start_col + hi)
        return idx

    def band_groups(self, task: str, in_dim: int, rfft_bins=100):
        groups = []
        if task in ("detection", "classification"):
            base = 0
            for name, span in self.BANDS.items():
                groups.append((f"{name}", self._band_indices_for_bins(rfft_bins, span, base)))
        else:
            base_mean = 0
            base_std  = 100
            for name, span in self.BANDS.items():
                idx_mean = self._band_indices_for_bins(rfft_bins, span, base_mean)
                idx_std  = self._band_indices_for_bins(rfft_bins, span, base_std)
                idx = np.concatenate([idx_mean, idx_std])
                groups.append((f"{name}", idx))
        for _, idx in groups:
            if len(idx) == 0:
                raise ValueError("Band group produced empty index set; check rfft_bins or feature layout.")
            if idx.max() >= in_dim:
                raise ValueError("Band group index overflow relative to in_dim.")
        return groups

    @torch.no_grad()
    def permutation_importance_by_group(self, loader, groups, task="classification", n_repeats=5):
        from sklearn.metrics import mean_squared_error, accuracy_score
        device = self.device

        def eval_metric():
            preds, labels = [], []
            for batch in loader:
                batch = batch.to(device)
                ew = getattr(batch, "edge_weight", None)
                if task == "detection":
                    out = self.model(batch.x, batch.edge_index, batch, task=task, edge_weight=ew).sigmoid()
                    preds.extend((out > 0.5).cpu().numpy()); labels.extend(batch.y.cpu().numpy())
                elif task in ["classification", "forecast_label", "early_clf"]:
                    head = "classification" if task == "classification" else "forecast_label"
                    out = self.model(batch.x, batch.edge_index, batch, task=head, edge_weight=ew)
                    preds.extend(out.argmax(dim=1).cpu().numpy()); labels.extend(batch.seq_targets.cpu().numpy())
                else:
                    out = self.model(batch.x, batch.edge_index, batch, task="forecast_time", edge_weight=ew)
                    preds.extend(out.cpu().numpy()); labels.extend(batch.seq_targets.cpu().numpy())
            if task in ["early_reg", "forecast_time", "forecast_label"]:
                return -np.sqrt(mean_squared_error(labels, preds))
            else:
                return accuracy_score(labels, preds)

        base = eval_metric()
        names, drops = [], []
        for (name, idxs) in groups:
            scores = []
            for _ in range(n_repeats):
                preds, labels = [], []
                for batch in loader:
                    batch = batch.to(device)
                    ew = getattr(batch, "edge_weight", None)
                    x_perm = batch.x.clone()
                    cols = torch.as_tensor(idxs, device=x_perm.device, dtype=torch.long)
                    perm_rows = torch.randperm(x_perm.size(0), device=x_perm.device)
                    x_perm[:, cols] = x_perm[perm_rows][:, cols]

                    if task == "detection":
                        out = self.model(x_perm, batch.edge_index, batch, task="detection", edge_weight=ew).sigmoid()
                        preds.extend((out > 0.5).cpu().numpy()); labels.extend(batch.y.cpu().numpy())
                    elif task in ["classification", "forecast_label", "early_clf"]:
                        head = "classification" if task == "classification" else "forecast_label"
                        out = self.model(x_perm, batch.edge_index, batch, task=head, edge_weight=ew)
                        preds.extend(out.argmax(dim=1).cpu().numpy()); labels.extend(batch.seq_targets.cpu().numpy())
                    else:
                        out = self.model(x_perm, batch.edge_index, batch, task="forecast_time", edge_weight=ew)
                        preds.extend(out.cpu().numpy()); labels.extend(batch.seq_targets.cpu().numpy())

                    if task in ["early_reg", "forecast_time","forecast_label"]:
                        score = -np.sqrt(mean_squared_error(labels, preds))
                    else:
                        score = accuracy_score(labels, preds)
                    scores.append(score)
            names.append(name)
            drops.append(base - float(np.mean(scores)))
        return names, np.array(drops, dtype=float), float(base)

# ====================================================================
# STATISTICAL FUNCTIONS FOR CONFIDENCE INTERVALS
# ====================================================================

def bootstrap_ci(self, y_true, y_pred, metric_func, n_bootstrap=1000, ci=95, random_state=42):
    """
    Compute bootstrap confidence interval for any metric.
    """
    np.random.seed(random_state)
    n = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    lower = np.percentile(bootstrap_scores, (100 - ci) / 2)
    upper = np.percentile(bootstrap_scores, 100 - (100 - ci) / 2)
    mean = np.mean(bootstrap_scores)
    std = np.std(bootstrap_scores)
    
    return lower, upper, mean, std


def bootstrap_ci_continuous(self, y_true, y_pred, metric_func, n_bootstrap=1000, ci=95):
    """Bootstrap CI for continuous metrics (R², RMSE)."""
    np.random.seed(42)
    n = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    lower = np.percentile(bootstrap_scores, (100 - ci) / 2)
    upper = np.percentile(bootstrap_scores, 100 - (100 - ci) / 2)
    mean = np.mean(bootstrap_scores)
    
    return lower, upper, mean


def test_vs_random(self, scores, chance_level=0.5):
    """One-sample t-test against random baseline."""
    t_stat, p_value = stats.ttest_1samp(scores, chance_level)
    return t_stat, p_value


def paired_test(self, scores_a, scores_b):
    """Paired t-test for model comparison."""
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return t_stat, p_value
# ============================================================================
# Trainer class
# ============================================================================
class Trainer:
    def __init__(
        self,
        num_features: int,
        num_hiddens: int,
        num_classes: int,
        dropout: float,
        num_heads: int,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        pooled_results,
        DC: bool = False,
        RC: bool = False,
        channel_names: List[str] = None,
        out_dir: str = r"D:\PhD Research\Experiments\Gen_EEG\runs\graphs",
        data_directory: str = r"F:\tuh_data\train",
        seed: int = 42,
        seq_len: int = 100,
        graph_method: str = 'pearson',
        graph_params: Dict[str, Any] = None,
        precomputed_edge_index: torch.Tensor = None,
        precomputed_edge_weight: torch.Tensor = None,
    ):
        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Handle channel names
        if channel_names is None:
            channel_names = [
                'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
            ]
        else:
            while isinstance(channel_names, list) and len(channel_names) > 0 and isinstance(channel_names[0], list):
                channel_names = [item for sublist in channel_names for item in sublist]
        
        self.channel_names = channel_names
        self.num_nodes = len(self.channel_names)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_lr = learning_rate
        self.base_wd = 0.0001
        self.seq_len = seq_len
        self.graph_method = graph_method
        self.graph_params = graph_params or {}

        # viz & labels
        self.viz = Viz(out_dir)
        self.real_class_names = ['gnsz', 'fnsz', 'tcsz', 'absz', 'mysz', 'cpsz', 'tnsz']

        self.model = None
        self.num_features_cfg = num_features
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.dropout = dropout

        self.adaptive_loss = AdaptiveHeadLoss(smoothing=0.05, focal_gamma=3.5)

        # ----- Graph topology & weights -----
        base_edge_index = torch.tril_indices(self.num_nodes, self.num_nodes, offset=-1)
        expected_pairs = self.num_nodes * (self.num_nodes - 1) // 2

        # Use precomputed edge weights if provided
        if precomputed_edge_index is not None and precomputed_edge_weight is not None:
            print("Using precomputed edge weights...")
            self.edge_index = precomputed_edge_index.to(self.device)
            self.edge_weight = precomputed_edge_weight.to(self.device)
        elif pooled_results is not None:
            # Compute edge weights using adjacency processor
            adj_proc = AdjacencyMatrixProcessor(
                pooled_results, 
                data_directory=data_directory, 
                channel_names=self.channel_names,
                graph_method=graph_method,
                graph_params=graph_params
            )
            raw = adj_proc.compute_all_edge_weights(DC=not RC, RC=RC)

            def _coerce_to_offdiag_lower_tri_1d(w: torch.Tensor, n: int) -> Union[torch.Tensor, None]:
                if w.ndim == 1:
                    num = w.numel()
                    if num == n * (n - 1) // 2:
                        return w
                    if num == n * (n + 1) // 2:
                        r0, c0 = torch.tril_indices(n, n, offset=0)
                        mask = (r0 != c0)
                        return w[mask]
                    if num == n * n:
                        W = w.view(n, n)
                        r, c = torch.tril_indices(n, n, offset=-1)
                        return W[r, c]
                    return None
                lead_dims = tuple(range(0, w.ndim - 1))
                w1 = w.mean(dim=lead_dims)
                return _coerce_to_offdiag_lower_tri_1d(w1, n)

            if isinstance(raw, torch.Tensor) and raw.numel() > 0:
                weights = _coerce_to_offdiag_lower_tri_1d(raw, self.num_nodes)
                if weights is None:
                    print("⚠️ Could not coerce edge weights; using uniform.")
                    weights = torch.ones(expected_pairs, dtype=torch.float32)
                else:
                    if weights.numel() != expected_pairs:
                        print(f"⚠️ Edge weight length {weights.numel()} != expected {expected_pairs}. Adjusting.")
                        if weights.numel() > expected_pairs:
                            weights = weights[:expected_pairs]
                        else:
                            weights = torch.cat([weights, torch.zeros(expected_pairs - weights.numel(), dtype=weights.dtype)])
            else:
                print("⚠️ No valid edge weights found. Using default uniform weights.")
                weights = torch.ones(expected_pairs, dtype=torch.float32)

            self.edge_index, self.edge_weight = self._make_undirected(base_edge_index, weights)
            self.edge_index = self.edge_index.to(self.device)
            self.edge_weight = self.edge_weight.to(self.device)
        else:
            # No pooled_results and no precomputed weights - use uniform weights
            print("No pooled_results or precomputed weights provided. Using uniform edge weights.")
            weights = torch.ones(expected_pairs, dtype=torch.float32)
            self.edge_index, self.edge_weight = self._make_undirected(base_edge_index, weights)
            self.edge_index = self.edge_index.to(self.device)
            self.edge_weight = self.edge_weight.to(self.device)

        print(f"Edge weights - min: {self.edge_weight.min():.6f}, max: {self.edge_weight.max():.6f}, mean: {self.edge_weight.mean():.6f}")
        print(f"Edge weights - any NaN: {torch.isnan(self.edge_weight).any()}")

        self.feature_scaler = None
        self.regression_scaler = None

    def compute_statistical_analysis(self, y_true, y_pred, y_proba=None, task='classification', n_bootstrap=1000):
        """
        Compute comprehensive statistical analysis including confidence intervals.
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            y_proba: Predicted probabilities (for AUC)
            task: 'classification', 'detection', or 'regression'
            n_bootstrap: Number of bootstrap iterations
        
        Returns:
            dict with metrics, confidence intervals, and statistical tests
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        results = {}
        
        if task in ['classification', 'detection']:
            # Classification/Detection metrics
            metrics = compute_all_metrics_with_ci(y_true, y_pred, y_proba, n_bootstrap)
            
            # Format for printing
            print(f"\n{'='*70}")
            print(f"STATISTICAL ANALYSIS - {task.upper()}")
            print(f"{'='*70}")
            print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'95% CI':<20}")
            print(f"{'-'*70}")
            
            for metric_name, metric_data in metrics.items():
                if metric_data['mean'] != 0 or metric_data['ci_lower'] != 0:
                    print(f"{metric_name:<15} {metric_data['mean']:.4f}     "
                        f"{metric_data['std']:.4f}     "
                        f"[{metric_data['ci_lower']:.4f}, {metric_data['ci_upper']:.4f}]")
            
            results['metrics'] = metrics
            
            # Also compute per-class metrics if classification
            if task == 'classification' and len(np.unique(y_true)) <= 10:
                unique_classes = np.unique(y_true)
                per_class_results = {}
                
                print(f"\n{'='*70}")
                print(f"PER-CLASS METRICS WITH CONFIDENCE INTERVALS")
                print(f"{'='*70}")
                
                for cls in unique_classes:
                    y_true_bin = (y_true == cls).astype(int)
                    y_pred_bin = (y_pred == cls).astype(int)
                    
                    # Precision for this class
                    try:
                        prec_result = bootstrap_metric(y_true_bin, y_pred_bin,
                                                        lambda yt, yp: precision_score(yt, yp, zero_division=0),
                                                        n_bootstrap)
                        rec_result = bootstrap_metric(y_true_bin, y_pred_bin,
                                                    lambda yt, yp: recall_score(yt, yp, zero_division=0),
                                                    n_bootstrap)
                        f1_result = bootstrap_metric(y_true_bin, y_pred_bin,
                                                    lambda yt, yp: f1_score(yt, yp, zero_division=0),
                                                    n_bootstrap)
                        
                        per_class_results[cls] = {
                            'precision': prec_result,
                            'recall': rec_result,
                            'f1': f1_result
                        }
                        
                        class_name = self.real_class_names[cls] if cls < len(self.real_class_names) else f"Class {cls}"
                        print(f"\n{class_name}:")
                        print(f"  Precision: {prec_result['mean']:.4f} ± {prec_result['std']:.4f} "
                            f"CI: [{prec_result['ci_lower']:.4f}, {prec_result['ci_upper']:.4f}]")
                        print(f"  Recall:    {rec_result['mean']:.4f} ± {rec_result['std']:.4f} "
                            f"CI: [{rec_result['ci_lower']:.4f}, {rec_result['ci_upper']:.4f}]")
                        print(f"  F1:        {f1_result['mean']:.4f} ± {f1_result['std']:.4f} "
                            f"CI: [{f1_result['ci_lower']:.4f}, {f1_result['ci_upper']:.4f}]")
                    except Exception as e:
                        print(f"  Could not compute metrics for class {cls}: {e}")
                
                results['per_class'] = per_class_results
        
        elif task == 'regression':
            # Regression metrics
            metrics = compute_regression_metrics_with_ci(y_true, y_pred, n_bootstrap)
            
            print(f"\n{'='*70}")
            print(f"STATISTICAL ANALYSIS - REGRESSION")
            print(f"{'='*70}")
            print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'95% CI':<20}")
            print(f"{'-'*70}")
            
            for metric_name, metric_data in metrics.items():
                print(f"{metric_name:<15} {metric_data['mean']:.4f}     "
                    f"{metric_data['std']:.4f}     "
                    f"[{metric_data['ci_lower']:.4f}, {metric_data['ci_upper']:.4f}]")
            
            results['metrics'] = metrics
        
        return results


    def compare_with_baseline(self, y_true, y_pred_model, y_pred_baseline, task='classification'):
        """
        Compare model performance against a baseline with statistical significance.
        
        Args:
            y_true: True labels
            y_pred_model: Model predictions
            y_pred_baseline: Baseline predictions (e.g., random, mean, or simple model)
            task: 'classification' or 'regression'
        
        Returns:
            dict with comparison results
        """
        from sklearn.metrics import accuracy_score, mean_squared_error
        
        results = {}
        
        if task == 'classification':
            metric_func = accuracy_score
            metric_name = 'Accuracy'
        else:
            metric_func = lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp))
            metric_name = 'Negative RMSE'
        
        # Bootstrap both models
        model_scores = bootstrap_metric(y_true, y_pred_model, metric_func, n_bootstrap=1000)['values']
        baseline_scores = bootstrap_metric(y_true, y_pred_baseline, metric_func, n_bootstrap=1000)['values']
        
        # Statistical comparison
        comparison = compare_models_statistically(model_scores, baseline_scores, test='wilcoxon')
        
        print(f"\n{'='*70}")
        print(f"MODEL VS BASELINE COMPARISON ({metric_name})")
        print(f"{'='*70}")
        print(f"Model Mean {metric_name}: {np.mean(model_scores):.4f}")
        print(f"Baseline Mean {metric_name}: {np.mean(baseline_scores):.4f}")
        print(f"Improvement: {np.mean(model_scores) - np.mean(baseline_scores):.4f}")
        print(f"Wilcoxon p-value: {comparison['p_value']:.6f}")
        print(f"Statistically Significant: {'YES' if comparison['significant'] else 'NO'}")
        
        results['model_mean'] = np.mean(model_scores)
        results['baseline_mean'] = np.mean(baseline_scores)
        results['improvement'] = np.mean(model_scores) - np.mean(baseline_scores)
        results['p_value'] = comparison['p_value']
        results['significant'] = comparison['significant']
        
        return results

    @staticmethod
    def _make_undirected(edge_index, edge_weight=None):
        ei_rev = edge_index[[1, 0], :]
        edge_index_ud = torch.cat([edge_index, ei_rev], dim=1)
        if edge_weight is not None:
            ew_ud = torch.cat([edge_weight, edge_weight], dim=0)
        else:
            ew_ud = None
        return edge_index_ud, ew_ud

    @staticmethod
    def _stratified_split_strict(X, Y, test_size=0.3, random_state=42):
        X = np.asarray(X); Y = np.asarray(Y)
        if Y.dtype.kind in "fc":
            raise ValueError("Strict stratified split called for non-categorical Y.")
        counts = Counter(Y.tolist())
        too_small = [c for c, n in counts.items() if n < 2]
        if len(too_small) > 0:
            val_idx = []
            for cls in sorted(counts):
                idx_cls = np.where(Y == cls)[0]
                if idx_cls.size > 0:
                    val_idx.append(idx_cls[0])
            val_idx = np.array(val_idx, dtype=int)
            desired_val = int(np.ceil(test_size * len(Y)))
            remain_idx = np.setdiff1d(np.arange(len(Y)), val_idx, assume_unique=False)
            if remain_idx.size > 0 and desired_val > val_idx.size:
                rX, rY = X[remain_idx], Y[remain_idx]
                sss = StratifiedShuffleSplit(n_splits=1, test_size=desired_val - val_idx.size,
                                             random_state=random_state)
                r_tr, r_va = next(sss.split(rX, rY))
                val_idx = np.concatenate([val_idx, remain_idx[r_va]], axis=0)
            train_idx = np.setdiff1d(np.arange(len(Y)), val_idx, assume_unique=False)
            return X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr, va = next(sss.split(X, Y))
        for seed in [random_state + k for k in range(1, 8)]:
            val_classes = set(Y[va].tolist())
            if len(val_classes) == len(counts):
                break
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            tr, va = next(sss.split(X, Y))
        return X[tr], X[va], Y[tr], Y[va]

    @staticmethod
    def safe_r2_score(y_true, y_pred, verbose=False):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) == 0:
            if verbose:
                warnings.warn("No valid data after filtering NaNs/infs")
            return 0.0
        if np.var(y_true) == 0:
            if verbose:
                print("Warning: y_true has zero variance. R² is undefined.")
            return 0.0
        return float(r2_score(y_true, y_pred))
    
   
    @staticmethod
    def hybrid_oversample(X_tr, Y_tr, num_nodes, floor=6, smote_cap=0.8, seed=42):
        X_flat = X_tr.reshape(X_tr.shape[0], -1)
        counts = np.bincount(Y_tr)
        counts = counts[counts > 0] if (len(counts) and counts.sum()) else counts
        if len(counts) == 0:
            return X_tr, Y_tr
        class_counts = np.bincount(Y_tr)
        tiny_targets = {cls: floor for cls, c in enumerate(class_counts) if 0 < c < floor}
        if tiny_targets:
            ros = RandomOverSampler(sampling_strategy=tiny_targets, random_state=seed)
            X_flat, Y_tr = ros.fit_resample(X_flat, Y_tr)
            class_counts = np.bincount(Y_tr)
        majority = class_counts.max()
        cap = max(floor, int(smote_cap * majority))
        smote_targets = {cls: cap for cls, c in enumerate(class_counts) if 0 < c < cap}
        if smote_targets:
            min_samples = min(v for v in class_counts if v > 0)
            k_neighbors = max(1, min(5, min_samples - 1))
            sm = SMOTE(random_state=seed, k_neighbors=k_neighbors, sampling_strategy=smote_targets)
            X_flat, Y_tr = sm.fit_resample(X_flat, Y_tr)
        X_tr = X_flat.reshape(-1, num_nodes, X_flat.shape[1] // num_nodes)
        return X_tr, Y_tr

    def create_graph_batches(self, X, Y, task: str = None, shuffle: bool = True):
        data_list = []
        for i in range(len(X)):
            x_np = X[i]
            if x_np.shape[0] != self.num_nodes and x_np.shape[1] == self.num_nodes:
                x_np = x_np.T
            if x_np.shape[0] != self.num_nodes:
                raise ValueError(f"Expected {self.num_nodes} nodes, got {x_np.shape[0]} in X[{i}] for task {task}")
            x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
            y_val = Y[i]
            if task in ("early_reg", "detection"):
                y = torch.as_tensor(y_val, dtype=torch.float16, device=self.device)
            else:
                y = torch.as_tensor(y_val, dtype=torch.long, device=self.device)
            data = Data(x=x, edge_index=self.edge_index, y=y)
            if self.edge_weight is not None:
                data.edge_weight = self.edge_weight
            data_list.append(data)
        return DataLoader(data_list, batch_size=self.batch_size, shuffle=shuffle)

    def _sequence_loader_from_arrays(self, X_seq, Y_seq, task, shuffle):
        from torch.utils.data import Dataset, DataLoader as TorchDataLoader
        from torch_geometric.data import Batch

        if X_seq.ndim == 3:
            n_seq, n_nodes, total_features = X_seq.shape
            if total_features % self.seq_len != 0:
                raise ValueError(f"Total features {total_features} not divisible by seq_len {self.seq_len}")
            n_features = total_features // self.seq_len
            X_seq = X_seq.reshape(n_seq, self.seq_len, n_nodes, n_features)
            print(f"[FIX] Reshaped X_seq from 3D to 4D: {X_seq.shape}")
        elif X_seq.ndim != 4:
            raise ValueError(f"Expected X_seq to be 3D or 4D, got {X_seq.ndim}D")

        seq_len = X_seq.shape[1]
        n_nodes = X_seq.shape[2]
        n_features = X_seq.shape[3]

        # models/trainer.py, right after line ~816 where n_features is computed:
        print(">>> RC per-window node feature dim (B) =", n_features)

        if n_nodes != self.num_nodes:
            raise ValueError(f"Node count mismatch: data has {n_nodes}, model expects {self.num_nodes}")

        class ArrayDataset(Dataset):
            def __init__(self, X_seq, Y_seq, edge_index, edge_weight):
                self.X_seq = X_seq
                self.Y_seq = Y_seq
                self.edge_index = edge_index
                self.edge_weight = edge_weight
                self.seq_len = X_seq.shape[1]

            def __len__(self):
                return len(self.X_seq)

            def __getitem__(self, idx):
                graphs = []
                for t in range(self.seq_len):
                    x_np = self.X_seq[idx, t]
                    x = torch.tensor(x_np, dtype=torch.float32)
                    data = Data(x=x, edge_index=self.edge_index)
                    if self.edge_weight is not None:
                        data.edge_weight = self.edge_weight
                    graphs.append(data)
                return graphs, self.Y_seq[idx]

        def collate_fn(batch):
            all_graphs = []
            all_targets = []
            for graphs, target in batch:
                all_graphs.extend(graphs)
                all_targets.append(target)

            total_windows = len(all_graphs)
            if total_windows % seq_len != 0:
                n_keep = (total_windows // seq_len) * seq_len
                all_graphs = all_graphs[:n_keep]
                n_seq = n_keep // seq_len
                all_targets = all_targets[:n_seq]
                if not hasattr(collate_fn, 'warned'):
                    print(f"Warning: Truncated {total_windows - n_keep} windows to align with seq_len={seq_len}")
                    collate_fn.warned = True

            batch_graph = Batch.from_data_list(all_graphs)
            dtype = torch.float32 if task == 'early_reg' else torch.long
            batch_graph.seq_targets = torch.tensor(all_targets, dtype=dtype)
            return batch_graph

        dataset = ArrayDataset(X_seq, Y_seq, self.edge_index, self.edge_weight)
        loader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )
        return loader, X_seq, Y_seq
    def extract_node_features_for_visualization(self, loader, num_samples=200):
        """Extract node-level features from the model for visualization."""
        if loader is None:
            return None, None
        
        self.model.eval()
        all_node_features = []
        all_targets = []
        sample_count = 0
        
        with torch.no_grad():
            for batch in loader:
                if sample_count >= num_samples:
                    break
                try:
                    batch = batch.to(self.device)
                    ew = getattr(batch, "edge_weight", None)
                    
                    node_feat = self.model._backbone(batch.x, batch.edge_index, edge_weight=ew)
                    batch_idx = self.model._get_batch_index(batch)
                    num_graphs = batch_idx.max().item() + 1
                    num_nodes_per_graph = node_feat.size(0) // num_graphs
                    
                    window_node_feat = node_feat.view(num_graphs, num_nodes_per_graph, -1)
                    all_node_features.append(window_node_feat.cpu().numpy())
                    
                    if hasattr(batch, 'seq_targets'):
                        all_targets.append(batch.seq_targets.cpu().numpy())
                    elif hasattr(batch, 'y'):
                        all_targets.append(batch.y.cpu().numpy())
                    
                    sample_count += num_graphs
                except Exception as e:
                    continue
        
        if all_node_features:
            node_features = np.concatenate(all_node_features, axis=0)
            targets = np.concatenate(all_targets, axis=0) if all_targets else None
            return node_features, targets
        
        return None, None
    def generate_interpretability_visualizations(self, val_loader, task_name):
        """Generate interpretability visualizations after training."""
        try:
            from data.scripts.interpretability_viz import InterpretabilityVisualizer
            
            viz = InterpretabilityVisualizer(output_dir="interpretability")
            
            viz.run_all_visualizations(
                model=self.model,
                loader=val_loader,
                device=self.device,
                channel_names=self.channel_names,
                num_hiddens=self.num_hiddens,
                task_name=task_name,
                seq_len=self.seq_len
            )
        except Exception as e:
            print(f"Interpretability visualizations failed: {e}")
            import traceback
            traceback.print_exc()
    # def extract_node_features_for_visualization(self, loader, num_samples=200):
    #     """Extract node-level features from the model for visualization."""
    #     self.model.eval()
    #     all_node_features = []
    #     all_targets = []
    #     sample_count = 0
        
    #     with torch.no_grad():
    #         for batch in loader:
    #             if sample_count >= num_samples:
    #                 break
                    
    #             batch = batch.to(self.device)
    #             ew = getattr(batch, "edge_weight", None)
                
    #             node_feat = self.model._backbone(batch.x, batch.edge_index, edge_weight=ew)
    #             batch_idx = self.model._get_batch_index(batch)
    #             num_graphs = batch_idx.max().item() + 1
    #             num_nodes_per_graph = node_feat.size(0) // num_graphs
                
    #             window_node_feat = node_feat.view(num_graphs, num_nodes_per_graph, -1)
    #             all_node_features.append(window_node_feat.cpu().numpy())
                
    #             if hasattr(batch, 'seq_targets'):
    #                 all_targets.append(batch.seq_targets.cpu().numpy())
    #             elif hasattr(batch, 'y'):
    #                 all_targets.append(batch.y.cpu().numpy())
                
    #             sample_count += num_graphs
        
    #     if all_node_features:
    #         node_features = np.concatenate(all_node_features, axis=0)
    #         targets = np.concatenate(all_targets, axis=0) if all_targets else None
    #         return node_features, targets
        
    #     return None, None
    # ----------------------------------------------------------------------
    # Main train method
    # ----------------------------------------------------------------------

    # def train(
    #         self,
    #         X,
    #         Y,
    #         file_ids=None,
    #         X_val=None,           # NEW
    #         Y_val=None,           # NEW
    #         detection: bool = False,
    #         classification: bool = False,
    #         early_reg: bool = False,
    #         early_clf: bool = False,
    #         explain_after: bool = False,
    #         explain_path: str | None = None
    #     ):
    
    def train(
        self,
        X,
        Y,
        file_ids=None,
        X_val=None,              # kept only for backward compatibility; ignored in K-fold mode
        Y_val=None,              # kept only for backward compatibility; ignored in K-fold mode
        detection: bool = False,
        classification: bool = False,
        early_reg: bool = False,
        early_clf: bool = False,
        explain_after: bool = False,
        explain_path: str | None = None,
        freeze_backbone: bool = False,
        backbone_ckpt: str | None = None,
        save_backbone_to: str | None = None,
        freeze_gru: bool = False,
        gru_ckpt: str | None = None,
        save_gru_to: str | None = None,
        use_kfold: bool = True,
        n_splits: int = 5,
        random_seed: int = 42,
        kfold_out_dir: str = "kfold_results",
    ):
        """
        Train with leakage-safe PATIENT-WISE K-fold cross-validation.

        Categorical tasks use StratifiedGroupKFold; time-to-seizure
        regression uses GroupKFold.  Every patient is wholly contained in
        either the training or validation partition for a given fold.

        Scaling, oversampling, class weighting and model fitting are performed
        independently inside each fold.  A fresh model is created for every
        fold.  Fold-level sample/class summaries and performance metrics are
        saved as CSV files.
        """
        import json
        from pathlib import Path
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, precision_score,
            recall_score, f1_score, roc_auc_score, average_precision_score,
            confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
        )

        # ------------------------------------------------------------
        # Basic validation / task selection
        # ------------------------------------------------------------
        flags = [detection, classification, early_reg, early_clf]
        if sum(bool(v) for v in flags) != 1:
            raise ValueError(
                "Exactly one task flag must be True: detection, classification, "
                "early_reg or early_clf."
            )

        task = (
            "detection" if detection else
            "classification" if classification else
            "early_reg" if early_reg else
            "early_clf"
        )
        model_task = "forecast_label" if early_clf else task

        X = np.asarray(X)
        Y = np.asarray(Y)
        if file_ids is None:
            raise ValueError(
                "Patient IDs (file_ids) are required for patient-wise K-fold CV. "
                "Pass the patient-id array returned by the data loader."
            )
        if isinstance(file_ids, torch.Tensor):
            file_ids = file_ids.detach().cpu().numpy()
        file_ids = np.asarray(file_ids, dtype=object).reshape(-1)
        Y = Y.reshape(-1)

        if len(X) != len(Y) or len(Y) != len(file_ids):
            raise ValueError(
                f"Length mismatch: X={len(X)}, Y={len(Y)}, patient_ids={len(file_ids)}"
            )

        unique_patients = np.unique(file_ids)
        if len(unique_patients) < 2:
            raise ValueError(
                "Patient-wise K-fold requires at least two unique patients."
            )

        requested_splits = int(n_splits)
        if requested_splits < 2:
            raise ValueError("n_splits must be >= 2")
        actual_splits = min(requested_splits, len(unique_patients))
        if actual_splits != requested_splits:
            print(
                f"[K-FOLD] Requested {requested_splits} folds but only "
                f"{len(unique_patients)} patients are available. "
                f"Using {actual_splits} folds for this run."
            )

        os.makedirs(kfold_out_dir, exist_ok=True)
        task_out_dir = os.path.join(kfold_out_dir, task)
        os.makedirs(task_out_dir, exist_ok=True)
        os.makedirs("models/checkpoints", exist_ok=True)

        # ------------------------------------------------------------
        # Human-readable class names
        # ------------------------------------------------------------
        if detection:
            class_name_map = {0: "bckg/non-seizure", 1: "seizure"}
        else:
            class_name_map = {
                0: "gnsz", 1: "fnsz", 2: "tcsz", 3: "absz",
                4: "mysz", 5: "cpsz", 6: "tnsz",
            }

        def _print_full_dataset_summary():
            print("\n" + "=" * 86)
            print(f"FULL DATASET SUMMARY — {task.upper()}")
            print("=" * 86)
            print(f"Samples:          {len(Y)}")
            print(f"Unique patients:  {len(unique_patients)}")
            counts_per_patient = np.array(
                [np.sum(file_ids == p) for p in unique_patients], dtype=int
            )
            print(
                "Samples/patient:  "
                f"min={counts_per_patient.min()}, max={counts_per_patient.max()}, "
                f"mean={counts_per_patient.mean():.2f}"
            )
            if not early_reg:
                print("\nSamples per class:")
                classes, counts = np.unique(Y.astype(int), return_counts=True)
                for cls, cnt in zip(classes, counts):
                    pct = 100.0 * cnt / len(Y)
                    print(
                        f"  {int(cls):>2} {class_name_map.get(int(cls), str(cls)):<20} "
                        f"{int(cnt):>7} ({pct:6.2f}%)"
                    )
            else:
                yy = Y.astype(float)
                print(
                    "Regression target: "
                    f"min={yy.min():.3f}, max={yy.max():.3f}, "
                    f"mean={yy.mean():.3f}, median={np.median(yy):.3f}, "
                    f"std={yy.std():.3f}"
                )
            print("=" * 86)

        _print_full_dataset_summary()
        if early_reg or early_clf:
            if X.ndim != 4:
                raise ValueError(
                    f"Forecasting expects X=(samples, sequence, nodes, features), got {X.shape}"
                )
            B, L, N, raw_dim = X.shape
            if N != self.num_nodes:
                raise ValueError(f"Expected {self.num_nodes} nodes, got {N}")

            if raw_dim != 121:
                print(f"[features] Building forecasting features from raw_dim={raw_dim} ...")
                fb = FeatureBuilder(
                    fs=200,
                    rfft_bins=100,
                    with_time=False,
                    with_shapes=True,
                    with_complexity=True,
                    with_connectivity=False,   # fold-specific graph weights below
                )
                feature_windows = []
                for b in range(B):
                    for l in range(L):
                        window_batch = X[b, l][np.newaxis, :, :]
                        built = fb.build(window_batch, mode=task)
                        feat = built[0] if isinstance(built, tuple) else built
                        if feat.ndim == 4:
                            feat = feat.squeeze(0).squeeze(0)
                        elif feat.ndim == 3:
                            feat = feat.squeeze(0)
                        feature_windows.append(feat)
                X_feat = np.stack(feature_windows, axis=0)
                in_dim_actual = X_feat.shape[-1]
                X_graph = X_feat.reshape(B, L, N, in_dim_actual)
            else:
                X_graph = X.astype(np.float32, copy=False)
                in_dim_actual = raw_dim
        else:
            fb = FeatureBuilder(
                fs=200,
                rfft_bins=100,
                with_time=False,
                with_shapes=True,
                with_complexity=True,
                with_connectivity=False,
            )
            built = fb.build(X, mode=task)
            X_feat = built[0] if isinstance(built, tuple) else built
            # Expected FeatureBuilder output is (samples, time, nodes, features).
            X_graph = X_feat.mean(axis=1) if X_feat.ndim == 4 else X_feat
            if X_graph.ndim != 3:
                raise ValueError(
                    f"Expected graph features (samples,nodes,features), got {X_graph.shape}"
                )
            in_dim_actual = X_graph.shape[-1]

        print(f"[features] Final X_graph shape={X_graph.shape}; in_dim={in_dim_actual}")

        # ------------------------------------------------------------
        # Folds: group-safe and stratified for categorical tasks
        # ------------------------------------------------------------
        if early_reg:
            splitter = GroupKFold(n_splits=actual_splits)
            split_iter = splitter.split(X_graph, Y, groups=file_ids)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=actual_splits,
                shuffle=True,
                random_state=random_seed,
            )
            split_iter = splitter.split(
                X_graph,
                Y.astype(int),
                groups=file_ids
            )

        folds = list(split_iter)

        # ------------------------------------------------------------
        # Dataset/fold-summary helpers
        # ------------------------------------------------------------
        summary_rows = []
        fold_metric_rows = []
        all_oof_true = []
        all_oof_pred = []
        all_oof_prob = []
        best_global_score = -float("inf")
        best_global_fold = None
        best_global_checkpoint = None

        def _append_fold_summary(fold_no, split_name, idx):
            ids = file_ids[idx]
            yy = Y[idx]
            n_pat = len(np.unique(ids))
            summary_rows.append({
                "fold": fold_no,
                "split": split_name,
                "class_id": "ALL",
                "class_name": "ALL",
                "samples": len(idx),
                "percentage": 100.0,
                "patients": n_pat,
            })
            if early_reg:
                vals = yy.astype(float)
                summary_rows.append({
                    "fold": fold_no,
                    "split": split_name,
                    "class_id": "REGRESSION",
                    "class_name": "target",
                    "samples": len(vals),
                    "percentage": 100.0,
                    "patients": n_pat,
                    "target_min": float(np.min(vals)),
                    "target_max": float(np.max(vals)),
                    "target_mean": float(np.mean(vals)),
                    "target_median": float(np.median(vals)),
                    "target_std": float(np.std(vals)),
                })
            else:
                all_classes = sorted(np.unique(Y.astype(int)).tolist())
                for cls in all_classes:
                    n = int(np.sum(yy.astype(int) == cls))
                    pct = 100.0 * n / len(yy) if len(yy) else 0.0
                    summary_rows.append({
                        "fold": fold_no,
                        "split": split_name,
                        "class_id": int(cls),
                        "class_name": class_name_map.get(int(cls), f"class_{cls}"),
                        "samples": n,
                        "percentage": pct,
                        "patients": n_pat,
                    })

        def _print_fold_summary(fold_no, train_idx, val_idx):
            tr_p = np.unique(file_ids[train_idx])
            va_p = np.unique(file_ids[val_idx])
            overlap = set(tr_p.tolist()).intersection(set(va_p.tolist()))
            if overlap:
                raise RuntimeError(
                    f"Patient leakage detected in fold {fold_no}: {sorted(overlap)}"
                )
            print("\n" + "#" * 86)
            print(f"FOLD {fold_no}/{actual_splits}")
            print("#" * 86)
            print(f"Train samples:       {len(train_idx)}")
            print(f"Validation samples:  {len(val_idx)}")
            print(f"Train patients:      {len(tr_p)}")
            print(f"Validation patients: {len(va_p)}")
            print(f"Patient overlap:     {len(overlap)}")

            if not early_reg:
                print("\nTraining class distribution:")
                for cls in sorted(np.unique(Y.astype(int)).tolist()):
                    n = int(np.sum(Y[train_idx].astype(int) == cls))
                    pct = 100.0 * n / len(train_idx)
                    print(
                        f"  {cls:>2} {class_name_map.get(cls, str(cls)):<20} "
                        f"{n:>7} ({pct:6.2f}%)"
                    )
                print("\nValidation class distribution:")
                for cls in sorted(np.unique(Y.astype(int)).tolist()):
                    n = int(np.sum(Y[val_idx].astype(int) == cls))
                    pct = 100.0 * n / len(val_idx)
                    print(
                        f"  {cls:>2} {class_name_map.get(cls, str(cls)):<20} "
                        f"{n:>7} ({pct:6.2f}%)"
                    )
            else:
                for name, idx in (("Train", train_idx), ("Validation", val_idx)):
                    vals = Y[idx].astype(float)
                    print(
                        f"{name} target: n={len(vals)}, min={vals.min():.3f}, "
                        f"max={vals.max():.3f}, mean={vals.mean():.3f}, "
                        f"median={np.median(vals):.3f}, std={vals.std():.3f}"
                    )

            _append_fold_summary(fold_no, "train", train_idx)
            _append_fold_summary(fold_no, "validation", val_idx)

        def _set_fold_graph_from_training(X_train_scaled):
            """Create fold-local functional edge weights from training data only."""
            if X_train_scaled.ndim == 4:
                # sequences x windows x nodes x features
                node_matrix = X_train_scaled.mean(axis=-1).reshape(-1, self.num_nodes)
            elif X_train_scaled.ndim == 3:
                node_matrix = X_train_scaled.mean(axis=-1)
            else:
                raise ValueError(f"Unsupported graph array shape {X_train_scaled.shape}")

            if node_matrix.shape[0] < 2:
                corr = np.eye(self.num_nodes, dtype=np.float32)
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = np.corrcoef(node_matrix.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            corr = np.abs((corr + corr.T) / 2.0)
            np.fill_diagonal(corr, 0.0)

            r, c = np.tril_indices(self.num_nodes, k=-1)
            weights = corr[r, c].astype(np.float32)
            if np.allclose(weights, 0):
                weights = np.ones_like(weights, dtype=np.float32)

            edge_index = torch.tensor(np.vstack([r, c]), dtype=torch.long)
            edge_weight = torch.tensor(weights, dtype=torch.float32)
            self.edge_index, self.edge_weight = self._make_undirected(edge_index, edge_weight)
            self.edge_index = self.edge_index.to(self.device)
            self.edge_weight = self.edge_weight.to(self.device)

        def _make_model():
            model = MultiTaskGCN(
                hidden_dim=self.num_hiddens,
                in_dim=in_dim_actual,
                num_classes=self.num_classes,
                dropout=self.dropout,
                seq_len=self.seq_len,
                use_uncertainty=False,
            ).to(self.device)

            if backbone_ckpt and os.path.exists(backbone_ckpt):
                n = model.load_backbone(backbone_ckpt, strict=True)
                print(f"[shared] loaded backbone ({n} tensors) from {backbone_ckpt}")
            if freeze_backbone:
                model.freeze_backbone()
                print("[shared] backbone FROZEN")
            if gru_ckpt and os.path.exists(gru_ckpt):
                n = model.load_temporal(gru_ckpt, strict=True)
                print(f"[shared] loaded GRU ({n} tensors) from {gru_ckpt}")
            if freeze_gru:
                model.freeze_temporal()
                print("[shared] GRU FROZEN")
            return model

        # def _evaluate(loader, regression_scaler=None):
        #     self.model.eval()
        #     y_true, y_pred, y_prob = [], [], []
        #     with torch.no_grad():
        #         for batch in loader:
        #             batch = batch.to(self.device)
        #             ew = getattr(batch, "edge_weight", None)
        #             if detection:
        #                 logits = self.model(
        #                     batch.x, batch.edge_index, batch,
        #                     task="detection", edge_weight=ew
        #                 ).squeeze(-1)
        #                 prob = torch.sigmoid(logits)
        #                 pred = (prob >= 0.5).long()
        #                 target = batch.y.long().view(-1)
        #                 y_true.extend(target.detach().cpu().numpy().tolist())
        #                 y_pred.extend(pred.detach().cpu().numpy().tolist())
        #                 y_prob.extend(prob.detach().cpu().numpy().tolist())
        #             elif classification:
        #                 logits = self.model(
        #                     batch.x, batch.edge_index, batch,
        #                     task="classification", edge_weight=ew
        #                 )
        #                 prob = torch.softmax(logits, dim=1)
        #                 pred = prob.argmax(dim=1)
        #                 target = batch.y.long().view(-1)
        #                 y_true.extend(target.detach().cpu().numpy().tolist())
        #                 y_pred.extend(pred.detach().cpu().numpy().tolist())
        #                 y_prob.extend(prob.detach().cpu().numpy().tolist())
        #             elif early_clf:
        #                 logits = self.model(
        #                     batch.x, batch.edge_index, batch,
        #                     task="forecast_label", edge_weight=ew
        #                 )
        #                 prob = torch.softmax(logits, dim=1)
        #                 pred = prob.argmax(dim=1)
        #                 target = batch.seq_targets.long().view(-1)
        #                 y_true.extend(target.detach().cpu().numpy().tolist())
        #                 y_pred.extend(pred.detach().cpu().numpy().tolist())
        #                 y_prob.extend(prob.detach().cpu().numpy().tolist())
        #             else:
        #                 out = self.model(
        #                     batch.x, batch.edge_index, batch,
        #                     task="forecast_time", edge_weight=ew
        #                 ).view(-1)
        #                 target = batch.seq_targets.float().view(-1)
        #                 pred_sc = out.detach().cpu().numpy()
        #                 true_sc = target.detach().cpu().numpy()
        #                 if regression_scaler is not None:
        #                     pred_log = regression_scaler.inverse_transform(
        #                         pred_sc.reshape(-1, 1)
        #                     ).reshape(-1)
        #                     true_log = regression_scaler.inverse_transform(
        #                         true_sc.reshape(-1, 1)
        #                     ).reshape(-1)
        #                     pred_raw = np.expm1(pred_log)
        #                     true_raw = np.expm1(true_log)
        #                 else:
        #                     pred_raw, true_raw = pred_sc, true_sc
        #                 y_true.extend(true_raw.tolist())
        #                 y_pred.extend(pred_raw.tolist())
        #     return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
        
        def _evaluate(loader, regression_scaler=None):
            self.model.eval()

            y_true, y_pred, y_prob = [], [], []

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    ew = getattr(batch, "edge_weight", None)

                    if detection:
                        logits = self.model(
                            batch.x,
                            batch.edge_index,
                            batch,
                            task="detection",
                            edge_weight=ew
                        ).reshape(-1)

                        prob = torch.sigmoid(logits)
                        pred = (prob >= 0.5).long()
                        target = batch.y.long().reshape(-1)

                        y_true.extend(
                            target.detach().cpu().numpy().reshape(-1).tolist()
                        )
                        y_pred.extend(
                            pred.detach().cpu().numpy().reshape(-1).tolist()
                        )
                        y_prob.extend(
                            prob.detach().cpu().numpy().reshape(-1).tolist()
                        )

                    elif classification:
                        logits = self.model(
                            batch.x,
                            batch.edge_index,
                            batch,
                            task="classification",
                            edge_weight=ew
                        ).reshape(-1, self.num_classes)

                        prob = torch.softmax(logits, dim=1)
                        pred = prob.argmax(dim=1)
                        target = batch.y.long().reshape(-1)

                        y_true.extend(
                            target.detach().cpu().numpy().reshape(-1).tolist()
                        )
                        y_pred.extend(
                            pred.detach().cpu().numpy().reshape(-1).tolist()
                        )
                        y_prob.extend(
                            prob.detach().cpu().numpy().tolist()
                        )

                    elif early_clf:
                        logits = self.model(
                            batch.x,
                            batch.edge_index,
                            batch,
                            task="forecast_label",
                            edge_weight=ew
                        ).reshape(-1, self.num_classes)

                        prob = torch.softmax(logits, dim=1)
                        pred = prob.argmax(dim=1)
                        target = batch.seq_targets.long().reshape(-1)

                        y_true.extend(
                            target.detach().cpu().numpy().reshape(-1).tolist()
                        )
                        y_pred.extend(
                            pred.detach().cpu().numpy().reshape(-1).tolist()
                        )
                        y_prob.extend(
                            prob.detach().cpu().numpy().tolist()
                        )

                    else:
                        out = self.model(
                            batch.x,
                            batch.edge_index,
                            batch,
                            task="forecast_time",
                            edge_weight=ew
                        ).reshape(-1)

                        target = batch.seq_targets.float().reshape(-1)

                        pred_sc = out.detach().cpu().numpy().reshape(-1)
                        true_sc = target.detach().cpu().numpy().reshape(-1)

                        if regression_scaler is not None:
                            pred_log = regression_scaler.inverse_transform(
                                pred_sc.reshape(-1, 1)
                            ).reshape(-1)

                            true_log = regression_scaler.inverse_transform(
                                true_sc.reshape(-1, 1)
                            ).reshape(-1)

                            pred_raw = np.expm1(pred_log)
                            true_raw = np.expm1(true_log)

                        else:
                            pred_raw = pred_sc
                            true_raw = true_sc

                        y_true.extend(true_raw.reshape(-1).tolist())
                        y_pred.extend(pred_raw.reshape(-1).tolist())

            return (
                np.asarray(y_true),
                np.asarray(y_pred),
                np.asarray(y_prob)
            )
        def _classification_metrics(y_true, y_pred, y_prob):
            result = {
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                "f1_weighted": f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
            }
            if detection:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    result["specificity"] = tn / (tn + fp) if (tn + fp) else 0.0
                    result["sensitivity"] = tp / (tp + fn) if (tp + fn) else 0.0
                if len(np.unique(y_true)) == 2 and y_prob.size:
                    result["roc_auc"] = roc_auc_score(y_true, y_prob.reshape(-1))
                    result["pr_auc"] = average_precision_score(y_true, y_prob.reshape(-1))
            return result

        def _regression_metrics(y_true, y_pred):
            return {
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": self.safe_r2_score(y_true, y_pred),
            }

        # ------------------------------------------------------------
        # K-fold training
        # ------------------------------------------------------------
        for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
            np.random.seed(random_seed + fold_no)
            torch.manual_seed(random_seed + fold_no)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed + fold_no)

            _print_fold_summary(fold_no, train_idx, val_idx)

            fold_dir = os.path.join(task_out_dir, f"fold_{fold_no}")
            ckpt_dir = os.path.join("models", "checkpoints", task, f"fold_{fold_no}")
            confusion_dir = os.path.join("confusion", task, f"fold_{fold_no}")
            os.makedirs(fold_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(confusion_dir, exist_ok=True)

            X_train = X_graph[train_idx]
            X_fold_val = X_graph[val_idx]
            Y_train_raw = Y[train_idx]
            Y_fold_val_raw = Y[val_idx]

            # --------------------------------------------------------
            # Fold-local feature scaling
            # --------------------------------------------------------
            self.feature_scaler = StandardScaler()
            tr_shape = X_train.shape
            va_shape = X_fold_val.shape
            X_train_sc = self.feature_scaler.fit_transform(
                X_train.reshape(-1, tr_shape[-1])
            ).reshape(tr_shape)
            X_val_sc = self.feature_scaler.transform(
                X_fold_val.reshape(-1, va_shape[-1])
            ).reshape(va_shape)

            # Fold-local graph weights use TRAINING fold only.
            _set_fold_graph_from_training(X_train_sc)

            # --------------------------------------------------------
            # Targets / loaders
            # --------------------------------------------------------
            self.regression_scaler = None
            Y_for_weights = Y_train_raw.copy()

            if early_reg:
                self.regression_scaler = StandardScaler()
                y_tr_log = np.log1p(Y_train_raw.astype(float))
                y_va_log = np.log1p(Y_fold_val_raw.astype(float))
                Y_train_model = self.regression_scaler.fit_transform(
                    y_tr_log.reshape(-1, 1)
                ).reshape(-1)
                Y_val_model = self.regression_scaler.transform(
                    y_va_log.reshape(-1, 1)
                ).reshape(-1)
                train_loader, _, _ = self._sequence_loader_from_arrays(
                    X_train_sc, Y_train_model, task, shuffle=True
                )
                val_loader, _, _ = self._sequence_loader_from_arrays(
                    X_val_sc, Y_val_model, task, shuffle=False
                )
            elif early_clf:
                Y_train_model = Y_train_raw.astype(int)
                Y_val_model = Y_fold_val_raw.astype(int)
                train_loader, _, _ = self._sequence_loader_from_arrays(
                    X_train_sc, Y_train_model, task, shuffle=True
                )
                val_loader, _, _ = self._sequence_loader_from_arrays(
                    X_val_sc, Y_val_model, task, shuffle=False
                )
            else:
                Y_train_model = Y_train_raw.astype(int)
                Y_val_model = Y_fold_val_raw.astype(int)

                # Oversampling is TRAIN-FOLD ONLY.
                if len(np.unique(Y_train_model)) > 1:
                    try:
                        X_train_sc, Y_train_model = self.hybrid_oversample(
                            X_train_sc,
                            Y_train_model,
                            num_nodes=self.num_nodes,
                            floor=10,
                            smote_cap=1.0,
                            seed=random_seed + fold_no,
                        )
                        print(
                            "[oversampling] training fold distribution after balancing:",
                            dict(zip(*np.unique(Y_train_model, return_counts=True)))
                        )
                    except Exception as exc:
                        print(f"[oversampling] skipped for fold {fold_no}: {exc}")

                train_loader = self.create_graph_batches(
                    X_train_sc, Y_train_model, task=task, shuffle=True
                )
                val_loader = self.create_graph_batches(
                    X_val_sc, Y_val_model, task=task, shuffle=False
                )

            # --------------------------------------------------------
            # Fresh model per fold
            # --------------------------------------------------------
            self.model = _make_model()

            # Class weighting is based on ORIGINAL (pre-oversampling) fold labels.
            self.adaptive_loss = AdaptiveHeadLoss(smoothing=0.05, focal_gamma=3.5)
            if classification or early_clf:
                present = np.unique(Y_for_weights.astype(int))
                if len(present) > 1:
                    cw_present = compute_class_weight(
                        class_weight="balanced",
                        classes=present,
                        y=Y_for_weights.astype(int),
                    ).astype(float)
                    weights = np.ones(self.num_classes, dtype=float)
                    weights[present] = cw_present
                    missing = np.setdiff1d(np.arange(self.num_classes), present)
                    if missing.size:
                        weights[missing] = float(cw_present.max())
                    weights_t = torch.tensor(
                        weights, dtype=torch.float32, device=self.device
                    )
                    self.adaptive_loss = AdaptiveHeadLoss(
                        smoothing=0.05,
                        focal_gamma=5.0,
                        class_weights=weights_t,
                    )
                    print("[loss] class weights:", np.round(weights, 3).tolist())

            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError("No trainable parameters remain after freezing.")

            lr = 1e-4 if early_reg else self.base_lr
            wd = 1e-3 if early_reg else self.base_wd
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
            self.scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=max(1, self.num_epochs), eta_min=1e-6
                )
                if early_reg else
                StepLR(self.optimizer, step_size=10, gamma=0.5)
            )

            best_score = -float("inf")
            best_epoch = 0
            patience = 0
            early_stop_patience = 50
            best_path = os.path.join(ckpt_dir, f"{task}_best.pth")
            history_rows = []

            # --------------------------------------------------------
            # Epoch loop
            # --------------------------------------------------------
            for epoch in range(1, self.num_epochs + 1):
                self.model.train()
                total_loss = 0.0
                n_batches = 0

                for batch in train_loader:
                    batch = batch.to(self.device)
                    ew = getattr(batch, "edge_weight", None)
                    self.optimizer.zero_grad()

                    if detection:
                        out = self.model(
                            batch.x, batch.edge_index, batch,
                            task="detection", edge_weight=ew
                        ).squeeze(-1)
                        loss = self.adaptive_loss.detection_loss(out, batch.y.float())
                    elif classification:
                        out = self.model(
                            batch.x, batch.edge_index, batch,
                            task="classification", edge_weight=ew
                        )
                        loss = self.adaptive_loss.classification_loss(
                            out, batch.y.long(), num_classes=self.num_classes
                        )
                    elif early_clf:
                        out = self.model(
                            batch.x, batch.edge_index, batch,
                            task="forecast_label", edge_weight=ew
                        )
                        loss = self.adaptive_loss.classification_loss(
                            out, batch.seq_targets.long(), num_classes=self.num_classes
                        )
                    else:
                        out = self.model(
                            batch.x, batch.edge_index, batch,
                            task="forecast_time", edge_weight=ew
                        ).view(-1)
                        loss = F.smooth_l1_loss(out, batch.seq_targets.float().view(-1))

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    total_loss += float(loss.item())
                    n_batches += 1

                avg_loss = total_loss / max(1, n_batches)
                y_true_ep, y_pred_ep, y_prob_ep = _evaluate(
                    val_loader, regression_scaler=self.regression_scaler
                )

                if early_reg:
                    ep_metrics = _regression_metrics(y_true_ep, y_pred_ep)
                    score = -ep_metrics["rmse"]
                    metric_text = (
                        f"Val MAE={ep_metrics['mae']:.3f}s | "
                        f"RMSE={ep_metrics['rmse']:.3f}s | R2={ep_metrics['r2']:.4f}"
                    )
                else:
                    ep_metrics = _classification_metrics(y_true_ep, y_pred_ep, y_prob_ep)
                    score = ep_metrics["f1_macro"]
                    metric_text = (
                        f"Val Acc={ep_metrics['accuracy']:.4f} | "
                        f"BalAcc={ep_metrics['balanced_accuracy']:.4f} | "
                        f"MacroF1={ep_metrics['f1_macro']:.4f}"
                    )

                history_rows.append({
                    "fold": fold_no,
                    "epoch": epoch,
                    "loss": avg_loss,
                    **ep_metrics,
                })

                print(
                    f"Fold {fold_no} | Epoch {epoch}/{self.num_epochs} | "
                    f"Loss={avg_loss:.5f} | {metric_text}"
                )

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    patience = 0
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "fold": fold_no,
                            "task": task,
                            "score": score,
                            "in_dim": in_dim_actual,
                            "num_nodes": self.num_nodes,
                            "num_classes": self.num_classes,
                            "seq_len": self.seq_len,
                        },
                        best_path,
                    )
                else:
                    patience += 1

                self.scheduler.step()
                if patience >= early_stop_patience:
                    print(
                        f"[early stopping] Fold {fold_no} stopped at epoch {epoch}; "
                        f"best epoch={best_epoch}."
                    )
                    break

            # --------------------------------------------------------
            # Load fold-best model and final fold evaluation
            # --------------------------------------------------------
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            y_true, y_pred, y_prob = _evaluate(
                val_loader, regression_scaler=self.regression_scaler
            )

            if early_reg:
                fold_metrics = _regression_metrics(y_true, y_pred)
                fold_score = -fold_metrics["rmse"]
            else:
                fold_metrics = _classification_metrics(y_true, y_pred, y_prob)
                fold_score = fold_metrics["f1_macro"]

            fold_metric_rows.append({
                "fold": fold_no,
                "best_epoch": best_epoch,
                "train_samples": len(train_idx),
                "validation_samples": len(val_idx),
                "train_patients": len(np.unique(file_ids[train_idx])),
                "validation_patients": len(np.unique(file_ids[val_idx])),
                **fold_metrics,
            })

            pd.DataFrame(history_rows).to_csv(
                os.path.join(fold_dir, "training_history.csv"), index=False
            )

            np.savez_compressed(
                os.path.join(fold_dir, "validation_predictions.npz"),
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                validation_indices=val_idx,
                validation_patient_ids=file_ids[val_idx],
            )

            all_oof_true.extend(y_true.tolist())
            all_oof_pred.extend(y_pred.tolist())
            if y_prob.size:
                all_oof_prob.extend(y_prob.tolist())

            print(f"\n[FOLD {fold_no}] FINAL METRICS")
            for k, v in fold_metrics.items():
                print(f"  {k:<22}: {v:.6f}")

            # Per-fold confusion matrix for categorical tasks.
            if not early_reg:
                try:
                    confusion_task = "forecast_label" if early_clf else task
                    save_task_confusion(
                        self.model,
                        val_loader,
                        confusion_task,
                        self.device,
                        out_dir=confusion_dir,
                    )
                except Exception as exc:
                    print(f"[confusion] fold {fold_no} skipped: {exc}")

            if explain_after:
                try:
                    self.generate_interpretability_visualizations(
                        val_loader, model_task
                    )
                except Exception as exc:
                    print(f"[interpretability] fold {fold_no} skipped: {exc}")

            if fold_score > best_global_score:
                best_global_score = fold_score
                best_global_fold = fold_no
                best_global_checkpoint = best_path

            # Fold-specific shared exports are always preserved when requested.
            if save_backbone_to:
                root, ext = os.path.splitext(save_backbone_to)
                fold_backbone = f"{root}_fold{fold_no}{ext or '.pth'}"
                self.model.save_backbone(fold_backbone)
                print(f"[shared] fold backbone saved -> {fold_backbone}")
            if save_gru_to:
                root, ext = os.path.splitext(save_gru_to)
                fold_gru = f"{root}_fold{fold_no}{ext or '.pth'}"
                self.model.save_temporal(fold_gru)
                print(f"[shared] fold GRU saved -> {fold_gru}")

        # ------------------------------------------------------------
        # Save dataset summary and fold metrics
        # ------------------------------------------------------------
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(task_out_dir, f"{task}_kfold_dataset_summary.csv")
        summary_df.to_csv(summary_csv, index=False)

        metrics_df = pd.DataFrame(fold_metric_rows)
        metrics_csv = os.path.join(task_out_dir, f"{task}_kfold_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)

        # Aggregate numerical metrics across folds.
        aggregate_rows = []
        metric_columns = [
            c for c in metrics_df.columns
            if c not in {
                "fold", "best_epoch", "train_samples", "validation_samples",
                "train_patients", "validation_patients"
            }
            and np.issubdtype(metrics_df[c].dtype, np.number)
        ]
        for col in metric_columns:
            vals = metrics_df[col].astype(float).to_numpy()
            aggregate_rows.append({
                "metric": col,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })
        aggregate_df = pd.DataFrame(aggregate_rows)
        aggregate_csv = os.path.join(task_out_dir, f"{task}_kfold_aggregate.csv")
        aggregate_df.to_csv(aggregate_csv, index=False)

        np.savez_compressed(
            os.path.join(task_out_dir, f"{task}_oof_predictions.npz"),
            y_true=np.asarray(all_oof_true),
            y_pred=np.asarray(all_oof_pred),
            y_prob=np.asarray(all_oof_prob),
        )

        print("\n" + "=" * 86)
        print(f"{actual_splits}-FOLD PATIENT-WISE CV COMPLETE — {task.upper()}")
        print("=" * 86)
        print(f"Dataset summary: {summary_csv}")
        print(f"Fold metrics:    {metrics_csv}")
        print(f"Aggregate:       {aggregate_csv}")
        print(f"Best fold:       {best_global_fold}")
        for _, row in aggregate_df.iterrows():
            print(
                f"  {row['metric']:<22} = {row['mean']:.6f} ± {row['std']:.6f}"
            )
        print("=" * 86)

        # Export the best fold's shared components to legacy requested paths,
        # while fold-specific exports above remain available for rigorous CV.
        if best_global_checkpoint and (save_backbone_to or save_gru_to):
            ckpt = torch.load(best_global_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if save_backbone_to:
                self.model.save_backbone(save_backbone_to)
                print(f"[shared] best-fold backbone -> {save_backbone_to}")
            if save_gru_to:
                self.model.save_temporal(save_gru_to)
                print(f"[shared] best-fold GRU -> {save_gru_to}")

        return {
            "task": task,
            "n_splits": actual_splits,
            "best_fold": best_global_fold,
            "best_checkpoint": best_global_checkpoint,
            "fold_metrics": fold_metric_rows,
            "aggregate_metrics": aggregate_rows,
            "dataset_summary_csv": summary_csv,
            "metrics_csv": metrics_csv,
            "aggregate_csv": aggregate_csv,
        }

