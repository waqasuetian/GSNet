
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import warnings
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import torch
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
from sklearn.model_selection import StratifiedKFold, GroupKFold
from scipy import stats
from sklearn.utils import resample

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

    def train(
            self,
            X,
            Y,
            file_ids=None,
            X_val=None,           # NEW
            Y_val=None,           # NEW
            detection: bool = False,
            classification: bool = False,
            early_reg: bool = False,
            early_clf: bool = False,
            explain_after: bool = False,
            explain_path: str | None = None
        ):
    
        os.makedirs("models/checkpoints", exist_ok=True)
        val_loader = None
        epoch_idx = []
        train_trace = []
        val_trace = []
        all_preds = []
        all_labels = []
        X = np.array(X)
        Y = np.array(Y)

        task = "detection" if detection else "classification" if classification else "early_reg" if early_reg else "early_clf"
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
        from scipy.stats import wilcoxon
# # ====================================================================
# # FORECASTING BRANCH (early_reg / early_clf)
# # ====================================================================
        if early_reg or early_clf:
            if X.ndim != 4:
                raise ValueError(f"Expected X for forecasting to have 4 dims, got {X.shape}")
            B, L, N, raw_dim = X.shape
            if N != self.num_nodes:
                raise ValueError(f"Expected {self.num_nodes} nodes, got {N}")

            # Build features from raw signals if needed
            if raw_dim == 200:
                print(f"Applying FeatureBuilder to raw signals...")
                fb = FeatureBuilder(
                    fs=200, rfft_bins=100,
                    with_time=False,
                    with_shapes=True,
                    with_complexity=True,
                    with_connectivity=True
                )

                # Process each (sample, time) window individually
                # X shape: (B, L, N, raw_dim) -> we iterate over B and L
                all_features = []
                all_conn = []
                for b in range(B):
                    for l in range(L):
                        # Extract one window: shape (N, raw_dim)
                        window = X[b, l]  # (N, raw_dim)
                        # FeatureBuilder expects (batch, nodes, time) -> add batch dim
                        window_batch = window[np.newaxis, :, :]  # (1, N, raw_dim)
                        feat, conn = fb.build(window_batch, mode=task)
                        # feat shape: (1, 1, N, feat_dim) or (1, N, feat_dim)
                        if feat.ndim == 4:
                            feat = feat.squeeze(0).squeeze(0)  # (N, feat_dim)
                        elif feat.ndim == 3:
                            feat = feat.squeeze(0)  # (N, feat_dim)
                        all_features.append(feat)
                        all_conn.append(conn.squeeze(0) if conn is not None else None)

                # Stack into (B*L, N, feat_dim)
                X_feat = np.stack(all_features, axis=0)  # (B*L, N, feat_dim)
                # Average connectivity over windows
                if all_conn[0] is not None:
                    conn_stack = np.stack(all_conn, axis=0)  # (B*L, N, N)
                    conn = conn_stack.mean(axis=0)           # (N, N)
                else:
                    conn = None

                in_dim_actual = X_feat.shape[-1]
                # Reshape to (B, L, N, feat_dim)
                X_graph = X_feat.reshape(B, L, N, in_dim_actual)
                print(f"Final X_graph shape: {X_graph.shape}, feature dim: {in_dim_actual}")
            else:
                print(f"Using pre-computed features (shape: {X.shape})")
                X_graph = X
                in_dim_actual = raw_dim
                conn = None

            # ========== CREATE GRAPH EDGES FROM CONNECTIVITY ==========
            if conn is not None:
                C = conn
                # Ensure symmetric, non-negative, zero diagonal
                C = np.abs(C)
                C = (C + C.T) / 2.0
                np.fill_diagonal(C, 0.0)
                r, c = np.triu_indices(N, k=1)
                edge_weight = torch.tensor(C[r, c], dtype=torch.float32)
                edge_index = torch.tensor(np.vstack([r, c]), dtype=torch.long)
                self.edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(self.device)
                self.edge_weight = torch.cat([edge_weight, edge_weight], dim=0).to(self.device)
                print(f"Graph edges created from conn: {len(self.edge_weight)}")
            else:
                print("WARNING: conn is None, will compute functional connectivity after scaling.")
                self.edge_index = None
                self.edge_weight = None

            # Patient‑wise split (no leakage)
            if file_ids is None:
                raise ValueError("patient_ids must be provided for forecasting tasks")
            if isinstance(file_ids, list):
                file_ids = np.array(file_ids)
            elif isinstance(file_ids, torch.Tensor):
                file_ids = file_ids.numpy()

            gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
            train_idx, val_idx = next(gss.split(X_graph, Y, groups=file_ids))
            X_train_seq = X_graph[train_idx]
            X_val_seq   = X_graph[val_idx]
            Y_train_seq = Y[train_idx]
            Y_val_seq   = Y[val_idx]

            # Ensure 4D shape (B, L, N, F)
            for name, arr in [("X_train_seq", X_train_seq), ("X_val_seq", X_val_seq)]:
                if arr.ndim == 3:
                    n_seq, n_nodes, total_feat = arr.shape
                    if total_feat % self.seq_len != 0:
                        raise ValueError(f"Total features {total_feat} not divisible by seq_len {self.seq_len}")
                    n_feat = total_feat // self.seq_len
                    arr = arr.reshape(n_seq, self.seq_len, n_nodes, n_feat)
                    if name == "X_train_seq":
                        X_train_seq = arr
                    else:
                        X_val_seq = arr

            # Scale features (per‑node per‑feature)
            train_shape = X_train_seq.shape
            val_shape   = X_val_seq.shape
            X_train_flat = X_train_seq.reshape(-1, train_shape[-1])
            X_val_flat   = X_val_seq.reshape(-1, val_shape[-1])
            self.feature_scaler = StandardScaler()
            X_train_sc = self.feature_scaler.fit_transform(X_train_flat).reshape(train_shape)
            X_val_sc   = self.feature_scaler.transform(X_val_flat).reshape(val_shape)

            # ----- Target scaling for regression -----
            if early_reg:
                Y_train_log = np.log1p(Y_train_seq)
                Y_val_log   = np.log1p(Y_val_seq)
                self.regression_scaler = StandardScaler()
                Y_train_sc = self.regression_scaler.fit_transform(Y_train_log.reshape(-1, 1)).flatten()
                Y_val_sc   = self.regression_scaler.transform(Y_val_log.reshape(-1, 1)).flatten()
                print(f"[Target] log1p+standardise: train mean={Y_train_sc.mean():.3f}, std={Y_train_sc.std():.3f}")
            else:  # early_clf
                Y_train_sc = Y_train_seq
                Y_val_sc   = Y_val_seq
                unique, counts = np.unique(Y_train_sc, return_counts=True)
                class_weights = {c: len(Y_train_sc) / (len(unique) * cnt) for c, cnt in zip(unique, counts)}
                self.class_weights_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(self.num_classes)],
                                                        dtype=torch.float, device=self.device)

            # ========== COMPUTE FALLBACK EDGES IF NOT ALREADY SET ==========
            if self.edge_weight is None:
                print("Computing functional connectivity from scaled features...")
                sample = X_train_sc[:min(5000, X_train_sc.shape[0])]  # (B, L, N, F)
                Bs, Ls, Ns, Fs = sample.shape
                node_avg = sample.mean(axis=-1).reshape(-1, Ns)      # (B*L, N)
                corr = np.corrcoef(node_avg.T)                       # (N, N)
                corr = np.abs(corr)
                np.fill_diagonal(corr, 0)
                r, c = np.triu_indices(N, k=1)
                edge_weight = torch.tensor(corr[r, c], dtype=torch.float32)
                edge_index = torch.tensor(np.vstack([r, c]), dtype=torch.long)
                self.edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(self.device)
                self.edge_weight = torch.cat([edge_weight, edge_weight], dim=0).to(self.device)
                print(f"Fallback graph edges created: {len(self.edge_weight)}")

                 # Create data loaders (ensure seq_targets is set)
            train_loader, _, _ = self._sequence_loader_from_arrays(
                X_train_sc, Y_train_sc, task, shuffle=True
            )
            val_loader, _, _ = self._sequence_loader_from_arrays(
                X_val_sc, Y_val_sc, task, shuffle=False
            )

  # ====================================================================
# DETECTION / CLASSIFICATION BRANCH
# ====================================================================
        else:
            # Feature extraction
            fb = FeatureBuilder(
                fs=200, rfft_bins=100,
                with_time=False,
                with_shapes=True,
                with_complexity=True,
                with_connectivity=False
            )
            X_built = fb.build(X, mode=task)
            if isinstance(X_built, tuple):
                X_feat, conn = X_built
            else:
                X_feat, conn = X_built, None

            X_graph = X_feat.mean(axis=1)
            in_dim_actual = X_graph.shape[-1]

            if conn is not None:
                C = conn.mean(axis=0)
                C = (C + C.T) / 2.0
                np.fill_diagonal(C, 0.0)
                r, c = np.tril_indices(C.shape[0], k=-1)
                w = torch.tensor(C[r, c], dtype=torch.float32, device=self.device)
                ei = torch.tril_indices(self.num_nodes, self.num_nodes, offset=-1)
                self.edge_index, self.edge_weight = self._make_undirected(ei, w)
                self.edge_index = self.edge_index.to(self.device)
                self.edge_weight = self.edge_weight.to(self.device)

            # Split
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_graph, Y, test_size=0.2, random_state=42, stratify=Y
            )

            # Define Y_train_sc and Y_val_sc for consistency with forecasting branch
            Y_train_sc = Y_train
            Y_val_sc = Y_val

            # Scale features
            self.feature_scaler = StandardScaler()
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])
            X_val_flat   = X_val.reshape(-1, X_val.shape[-1])
            X_train_sc = self.feature_scaler.fit_transform(X_train_flat).reshape(X_train.shape)
            X_val_sc   = self.feature_scaler.transform(X_val_flat).reshape(X_val.shape)

            # Oversampling (if needed)
            if detection or classification:
                uniq = np.unique(Y_train)
                if len(uniq) > 1:
                    X_train_sc, Y_train = self.hybrid_oversample(
                        X_train_sc, Y_train, num_nodes=self.num_nodes, floor=10, smote_cap=1, seed=42
                    )
                    # Update Y_train_sc after oversampling
                    Y_train_sc = Y_train
                else:
                    print("Warning: single class; skip oversampling and rely on class weights.")

            # Create loaders
            train_loader = self.create_graph_batches(X_train_sc, Y_train, task=task, shuffle=True)
            val_loader   = self.create_graph_batches(X_val_sc,   Y_val,   task=task, shuffle=False)

        # --------------------------------------------------------------------
        # Create model (shared for all tasks)
        # --------------------------------------------------------------------
        self.model = MultiTaskGCN(
            hidden_dim=self.num_hiddens,
            in_dim=in_dim_actual,
            num_classes=self.num_classes,
            dropout=self.dropout,
            seq_len=self.seq_len,
            use_uncertainty=False
        ).to(self.device)

        print(f"=================== {task.capitalize()} ===================")

        # # ---- Optimizer and scheduler ----
        # self.optimizer = optim.Adam(
        #     self.model.parameters(),
        #     lr=1e-4 if early_reg else self.base_lr,
        #     weight_decay=1e-3 if early_reg else self.base_wd
        # )
        # self.scheduler = (
        #     ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-7)
        #     if early_reg else
        #     StepLR(self.optimizer, step_size=10, gamma=0.5)
        # )
         # Lower LR and higher weight decay for forecasting
        lr = 1e-4 if early_reg else 0.005
        wd = 5e-2 if early_reg else 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        # Scheduler: reduce on plateau (monitor MAE for regression, F1 for classification)
        if early_reg:
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=1e-4, weight_decay=5e-2
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs, eta_min=1e-6
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.base_lr, weight_decay=self.base_wd
            )
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        # if early_reg:
        #     self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
        #                                     patience=10, min_lr=1e-7)
        # else:
        #     self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
        #                                     patience=10, min_lr=1e-7)

        # ---- Class weights for classification ----
        if classification or early_clf:
            all_classes = np.arange(len(self.real_class_names))
            present = np.unique(Y_train_sc.astype(int))
            
            if len(present) > 1:
                cw_present = compute_class_weight(
                    class_weight='balanced',
                    classes=present,
                    y=Y_train_sc.astype(int)
                ).astype(float)
                
                class_weights = np.ones(len(all_classes), dtype=float)
                class_weights[present] = cw_present
                if len(cw_present) > 0:
                    class_weights[np.setdiff1d(all_classes, present)] = float(cw_present.max())
                class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
                self.adaptive_loss = AdaptiveHeadLoss(smoothing=0.05, focal_gamma=5, class_weights=class_weights)
                print("[loss] class_weights:", class_weights.detach().cpu().numpy().round(3).tolist())

                # Give extra weight to minority classes (mysz=4, tnsz=6)
                # minority_classes = [4,6]
                # for cls in minority_classes:
                #     if cls < len(class_weights):
                #         class_weights[cls] *= 5
                
            #     class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            #     self.adaptive_loss = AdaptiveHeadLoss(smoothing=0.05, focal_gamma=5, class_weights=class_weights)
            #     print("[loss] class_weights:", class_weights.detach().cpu().numpy().round(3).tolist())
            # else:
            #     print("Warning: only one class in training data, using uniform weights")
            #     self.adaptive_loss = AdaptiveHeadLoss(smoothing=0.05, focal_gamma=5)
    



        # Training loop
        # ---- Training loop ----
        best_val_metric = -float("inf")
        patience_counter = 0
        early_stopping_patience = 50
        epoch_idx, train_trace, val_trace = [], [], []
        train_preds_list, train_labels_list = [], [] # For confusion matrices
        all_preds = []  # Initialize for all tasks
        all_labels = []
        epoch_idx = []
        train_trace = []
        val_trace = [] 
        import torch.nn.functional as F
        # Create task-specific directory for checkpoints
        tag = task
        task_checkpoint_dir = f"models/checkpoints/{tag}"
        os.makedirs(task_checkpoint_dir, exist_ok=True)

        print(f"Checkpoints will be saved to: {task_checkpoint_dir}")

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            all_preds, all_labels = [], []
            train_preds_epoch, train_labels_epoch = [], []  # For epoch-level storage

            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                ew = getattr(batch, "edge_weight", None)

                if detection:
                    out = self.model(batch.x, batch.edge_index, batch, task="detection", edge_weight=ew).squeeze(-1)
                    loss = self.adaptive_loss.detection_loss(out, batch.y.float())
                    preds = (torch.sigmoid(out) > 0.5).detach().cpu().numpy()
                    labels = batch.y.detach().cpu().numpy()

                elif classification:
                    out = self.model(batch.x, batch.edge_index, batch, task="classification", edge_weight=ew)
                    loss = self.adaptive_loss.classification_loss(out, batch.y,
                                                                num_classes=self.model.class_head[-1].out_features)
                    preds = out.argmax(dim=1).detach().cpu().numpy()
                    labels = batch.y.detach().cpu().numpy()

                elif early_reg:
                    out = self.model(batch.x, batch.edge_index, batch, task="forecast_time", edge_weight=ew)
                    # loss = self.adaptive_loss.regression_loss(out, batch.seq_targets.float())
                    loss = F.smooth_l1_loss(out.squeeze(), batch.seq_targets.float())
                    preds = out.detach().cpu().numpy().flatten()
                    labels = batch.seq_targets.detach().cpu().numpy().flatten()
                    all_preds.extend(preds)
                    all_labels.extend(labels)

                else:  # early_clf
                    out = self.model(batch.x, batch.edge_index, batch, task="forecast_label", edge_weight=ew)
                    loss = self.adaptive_loss.classification_loss(out, batch.seq_targets,
                                                                num_classes=self.model.label_head[-1].out_features)
                    preds = out.argmax(dim=1).detach().cpu().numpy()
                    labels = batch.seq_targets.detach().cpu().numpy()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                
                if not early_reg:
                    all_preds.extend(preds)
                    all_labels.extend(labels)
                
                train_preds_epoch.extend(preds)
                train_labels_epoch.extend(labels)

            avg_loss = epoch_loss / len(train_loader)
            
            # Store for final confusion matrix
            train_preds_list.extend(train_preds_epoch)
            train_labels_list.extend(train_labels_epoch)
            self.model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    ew = getattr(batch, "edge_weight", None)

                    if detection:
                        out = self.model(batch.x, batch.edge_index, batch, task="detection", edge_weight=ew).squeeze(-1)
                        preds = (torch.sigmoid(out) > 0.5).cpu().numpy()
                        targets = batch.y.cpu().numpy()
                    elif classification:
                        out = self.model(batch.x, batch.edge_index, batch, task="classification", edge_weight=ew)
                        preds = out.argmax(dim=1).cpu().numpy()
                        targets = batch.y.cpu().numpy()
                    elif early_reg:
                        # out_sc = self.model(batch.x, batch.edge_index, batch, task="forecast_time", edge_weight=ew).cpu().numpy().flatten()
                        # # Inverse transform to original scale
                        # preds = np.expm1(self.regression_scaler.inverse_transform(out_sc.reshape(-1, 1))).flatten()
                        # targets = batch.seq_targets.cpu().numpy().flatten()
                        # targets_raw = np.expm1(self.regression_scaler.inverse_transform(targets.reshape(-1, 1))).flatten()
                        # val_preds.extend(preds)
                        # val_labels.extend(targets_raw)
                        out = self.model(batch.x, batch.edge_index, batch, task="forecast_time", edge_weight=ew)
                        preds = out.detach().cpu().numpy().flatten()
                        targets = batch.seq_targets.detach().cpu().numpy().flatten()
                        val_preds.extend(preds)
                        val_labels.extend(targets)
                    else:  # early_clf
                        out = self.model(batch.x, batch.edge_index, batch, task="forecast_label", edge_weight=ew)
                        preds = out.argmax(dim=1).cpu().numpy()
                        targets = batch.seq_targets.cpu().numpy()

                    val_preds.extend(preds)
                    val_labels.extend(targets)

            
            # Metrics
            if early_reg:
                train_preds_raw = self.regression_scaler.inverse_transform(
                    np.array(all_preds).reshape(-1, 1)
                ).flatten()
                train_labels_raw = self.regression_scaler.inverse_transform(
                    np.array(all_labels).reshape(-1, 1)
                ).flatten()
                # R² is scale-invariant, RMSE needs original scale
                train_r2 = self.safe_r2_score(train_labels_raw, train_preds_raw)
                train_rmse = np.sqrt(mean_squared_error(train_labels_raw, train_preds_raw))
                
                val_preds_raw = self.regression_scaler.inverse_transform(
                    np.array(val_preds).reshape(-1, 1)
                ).flatten()
                val_labels_raw = self.regression_scaler.inverse_transform(
                    np.array(val_labels).reshape(-1, 1)
                ).flatten()
                val_r2 = self.safe_r2_score(val_labels_raw, val_preds_raw)
                val_rmse = np.sqrt(mean_squared_error(val_labels_raw, val_preds_raw))
                        # # ---- Metrics ----
            # if early_reg:
            #     train_preds_flat = np.array(all_preds)
            #     train_labels_flat = np.array(all_labels)
            #     train_preds_raw = train_preds_flat
            #     train_labels_raw = train_labels_flat

            #     valid_mask = np.isfinite(train_labels_raw) & np.isfinite(train_preds_raw)
            #     if valid_mask.sum() > 1:
            #         train_r2 = self.safe_r2_score(train_labels_raw[valid_mask], train_preds_raw[valid_mask])
            #         train_rmse = np.sqrt(mean_squared_error(train_labels_raw[valid_mask], train_preds_raw[valid_mask]))
            #     else:
            #         train_r2 = -float('inf')
            #         train_rmse = float('inf')

            #     val_preds_flat = np.array(val_preds)
            #     val_labels_flat = np.array(val_labels)
                
            #     valid_mask_val = np.isfinite(val_labels_flat) & np.isfinite(val_preds_flat)
            #     if valid_mask_val.sum() > 1:
            #         val_r2 = self.safe_r2_score(val_labels_flat[valid_mask_val], val_preds_flat[valid_mask_val])
            #         val_rmse = np.sqrt(mean_squared_error(val_labels_flat[valid_mask_val], val_preds_flat[valid_mask_val]))
            #     else:
            #         val_r2 = -float('inf')
            #         val_rmse = float('inf')
                
                val_metric = val_r2
                
                # Store values (replace inf/nan with 0 for plotting)
                train_trace_value = train_r2 if np.isfinite(train_r2) else 0.0
                val_trace_value = val_r2 if np.isfinite(val_r2) else 0.0
                
                epoch_idx.append(epoch)
                train_trace.append(train_r2 if not np.isnan(train_r2) else 0.0)
                val_trace.append(val_r2 if not np.isnan(val_r2) else 0.0)
                        
                print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} | "
                    f"Train R2: {train_r2:.4f}, RMSE: {train_rmse:.2f}s | "
                    f"Val R2: {val_r2:.4f}, RMSE: {val_rmse:.2f}s")
            else:
                # Detection or Classification
                train_acc = accuracy_score(all_labels, all_preds)
                val_acc = accuracy_score(val_labels, val_preds)
                train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
                val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
                val_metric = val_f1
                
                # Store values (ensure finite)
                train_trace_value = train_acc if np.isfinite(train_acc) else 0.0
                val_trace_value = val_acc if np.isfinite(val_acc) else 0.0
                
                train_acc = accuracy_score(all_labels, all_preds)
                val_acc = accuracy_score(val_labels, val_preds)
                
                epoch_idx.append(epoch)
                train_trace.append(train_acc if not np.isnan(train_acc) else 0.0)
                val_trace.append(val_acc if not np.isnan(val_acc) else 0.0)
                
                print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
                
            
            # ====================================================================
            # CHECKPOINTING - TASK SPECIFIC
            # ====================================================================
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
                
                # Save best model with task name
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": None if isinstance(self.scheduler, ReduceLROnPlateau) else self.scheduler.state_dict(),
                        "val_metric": best_val_metric,
                        "epoch": epoch,
                        "task": task,
                        "num_nodes": self.num_nodes,
                        "hidden_dim": self.num_hiddens,
                        "seq_len": self.seq_len,
                        "num_classes": self.num_classes,
                        "dropout": self.dropout,
                    },
                    f"{task_checkpoint_dir}/{tag}_best.pth"
                )
                print(f"✓ Best {task} model saved to {task_checkpoint_dir}/{tag}_best.pth (val_metric={best_val_metric:.4f})")
            else:
                patience_counter += 1

            # Save periodic checkpoint (every 10 epochs)
            if epoch % 10 == 0:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "task": task,
                        "val_metric": val_metric,
                    },
                    f"{task_checkpoint_dir}/{tag}_epoch_{epoch}.pth"
                )
                print(f"  Checkpoint saved: {task_checkpoint_dir}/{tag}_epoch_{epoch}.pth")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs).")
                break

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metric)
            else:
                self.scheduler.step()

        # After training loop, print summary of saved models
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE FOR TASK: {task.upper()}")
        print(f"{'='*60}")
        print(f"Best model saved to: {task_checkpoint_dir}/{tag}_best.pth")
        print(f"Best validation metric: {best_val_metric:.4f}")
        print(f"Checkpoints directory: {task_checkpoint_dir}")
        print(f"{'='*60}\n")

        if detection or classification or early_reg or early_clf:
            # Generate interpretability visualizations
                self.generate_interpretability_visualizations(val_loader, task)


    # ====================================================================
# POST-TRAINING VISUALIZATIONS (Complete Section)
# ====================================================================
        if val_loader is not None:
            try:
                out_prefix = f"{task}"
                
                # Create task-specific directory for visualizations
                viz_task_dir = os.path.join(self.viz.out_dir, task)
                os.makedirs(viz_task_dir, exist_ok=True)
                
                print(f"\n{'='*60}")
                print(f"GENERATING VISUALIZATIONS FOR {task.upper()}")
                print(f"{'='*60}")
                
                # ========== 1. NODE (CHANNEL) PERMUTATION IMPORTANCE ==========
                try:
                    ni = NodeInterpreter(self.model, num_nodes=self.num_nodes, device=self.device)
                    base_metric, node_drop = ni.permutation_importance(val_loader, task=task, n_repeats=8, seed=7)
                    rows = [(self.channel_names[i], float(node_drop[i])) for i in range(len(node_drop))]
                    self.viz.save_csv_columns(rows, header=["channel", "importance_drop"], 
                                            fname=os.path.join(viz_task_dir, f"{out_prefix}_node_importance.csv"))
                    self.viz.barh(node_drop, self.channel_names,
                                title=f"Node importance (perm drop) — {task} (base={base_metric:.3f})",
                                fname=os.path.join(viz_task_dir, f"{out_prefix}_node_importance.png"), top_k=20)
                    self.viz.radar(node_drop, self.channel_names,
                                title=f"Node Importance — {task} (base={base_metric:.3f})",
                                fname=os.path.join(viz_task_dir, f"{out_prefix}_radar_nodes.png"))
                    print(f"  ✓ Node permutation importance saved")
                except Exception as e:
                    print(f"  ✗ Node permutation importance skipped: {e}")

                # ========== 2. FEATURE GROUP PERMUTATION IMPORTANCE ==========
                try:
                    fi = FeatureInterpreter(self.model, num_nodes=self.num_nodes, device=self.device)
                    in_dim_actual = self.model.conv1.lin.in_channels if hasattr(self.model.conv1, 'lin') else self.num_features_cfg
                    
                    feat_groups = fi.default_groups(
                        in_dim=in_dim_actual,
                        task=task,
                        with_shapes=True,
                        with_complexity=True
                    )
                    
                    # Only use groups that fit within the actual dimension
                    max_idx = max([group[1].max() for group in feat_groups]) if feat_groups else 0
                    if max_idx < in_dim_actual:
                        group_names, group_drops, base_feat = fi.permutation_importance_by_group(
                            val_loader, feat_groups, task=task, n_repeats=5
                        )
                    rows = list(zip(group_names, map(float, group_drops)))
                    self.viz.save_csv_columns(rows, header=["feature_group", "importance_drop"],
                                            fname=os.path.join(viz_task_dir, f"{out_prefix}_feature_groups.csv"))
                    self.viz.radar(group_drops, group_names,
                                title=f"Feature-Group Importance — {task} (base={base_feat:.3f})",
                                fname=os.path.join(viz_task_dir, f"{out_prefix}_radar_features.png"))
                    print(f"  ✓ Feature group importance saved")
                except Exception as e:
                    print(f"  ✗ Feature group importance skipped: {e}")

                # ========== 3. BAND IMPORTANCE ==========
                try:
                    fi = FeatureInterpreter(self.model, num_nodes=self.num_nodes, device=self.device)
                    band_groups = fi.band_groups(task=task, in_dim=in_dim_actual, rfft_bins=100)
                    bnames, bdrops, _ = fi.permutation_importance_by_group(
                        val_loader, band_groups, task=task, n_repeats=4
                    )
                    self.viz.save_csv_columns(list(zip(bnames, map(float, bdrops))),
                                            header=["band", "importance_drop"],
                                            fname=os.path.join(viz_task_dir, f"{out_prefix}_band_importance.csv"))
                    
                    # Soft outer-product for per-node-per-band heatmap
                    node_imp = np.asarray(node_drop, float)
                    node_imp = np.maximum(node_imp, 0)
                    if node_imp.max() > 0:
                        node_imp = node_imp / (node_imp.max() + 1e-12)
                    band_imp = np.asarray(bdrops, float)
                    band_imp = np.maximum(band_imp, 0)
                    if band_imp.max() > 0:
                        band_imp = band_imp / (band_imp.max() + 1e-12)
                    band_node = {bn: (band_imp[i] * node_imp) for i, bn in enumerate(bnames)}
                    self.viz.band_node_heatmap(band_node, self.channel_names,
                                            title=f"Band × Node (soft attribution) — {task}",
                                            fname=os.path.join(viz_task_dir, f"{out_prefix}_band_node.png"))
                    print(f"  ✓ Band importance saved")
                except Exception as e:
                    print(f"  ✗ Band importance skipped: {e}")

                # ========== 4. EDGE WEIGHT VISUALIZATIONS ==========
                try:
                    if self.edge_weight is not None:
                        ew = self.edge_weight.detach().cpu().numpy()
                        n = self.num_nodes
                        A = np.zeros((n, n), dtype=float)
                        half = ew.size // 2
                        w = ew[:half]
                        r, c = np.tril_indices(n, k=-1)
                        A[r, c] = w
                        A[c, r] = w
                        self.viz.heatmap(A, self.channel_names, self.channel_names,
                                        title=f"Edge weights used by GCN — {task}",
                                        fname=os.path.join(viz_task_dir, f"{out_prefix}_edge_weights_heatmap.png"))
                        self.viz.connectogram(A, self.channel_names,
                                            title=f"Connectogram — {task}",
                                            fname=os.path.join(viz_task_dir, f"{out_prefix}_connectogram.png"),
                                            top_k=50)
                        print(f"  ✓ Edge weight visualizations saved")
                except Exception as e:
                    print(f"  ✗ Edge weight visuals skipped: {e}")

                # ========== 5. TASK-SPECIFIC PLOTS ==========
                try:
                    if detection:
                        # Detection: ROC, PR, Calibration, Confusion Matrix
                        ys, ps = [], []
                        with torch.no_grad():
                            for b in val_loader:
                                b = b.to(self.device)
                                ew = getattr(b, "edge_weight", None)
                                logit = self.model(b.x, b.edge_index, b, task="detection", edge_weight=ew)
                                ys.extend(b.y.detach().cpu().numpy().tolist())
                                ps.extend(torch.sigmoid(logit).detach().cpu().numpy().tolist())
                        ys = np.asarray(ys, int)
                        ps = np.asarray(ps, float)
                        
                        self.viz.roc_binary(ys, ps, title=f"ROC — detection", 
                                            fname=os.path.join(viz_task_dir, f"{out_prefix}_roc.png"))
                        self.viz.pr_curve(ys, ps, title=f"PR — detection", 
                                        fname=os.path.join(viz_task_dir, f"{out_prefix}_pr.png"))
                        self.viz.reliability_curve(ys, ps, title=f"Calibration — detection", 
                                                fname=os.path.join(viz_task_dir, f"{out_prefix}_calibration.png"))
                        self.viz.roc_with_ci(ys, ps, title=f"ROC (95% CI) — detection", 
                                            fname=os.path.join(viz_task_dir, f"{out_prefix}_roc_ci.png"))
                        self.viz.confusion(ys, (ps > 0.5).astype(int),
                                        class_names=["non-seizure", "seizure"],
                                        title=f"Confusion — detection", 
                                        fname=os.path.join(viz_task_dir, f"{out_prefix}_cm.png"))
                        self.viz.confusion(np.array(train_labels_list), np.array(train_preds_list),
                                        class_names=["non-seizure", "seizure"],
                                        title=f"Confusion (Train) — detection", 
                                        fname=os.path.join(viz_task_dir, f"{out_prefix}_cm_train.png"))
                        print(f"  ✓ Detection plots saved (ROC, PR, Calibration, Confusion)")
                    elif classification:
                        try:
                            # Collect validation predictions
                            ys, logits_all = [], []
                            with torch.no_grad():
                                for b in val_loader:
                                    b = b.to(self.device)
                                    ew = getattr(b, "edge_weight", None)
                                    logits = self.model(b.x, b.edge_index, b, task="classification", edge_weight=ew)
                                    ys.extend(b.y.detach().cpu().numpy().tolist())
                                    logits_all.append(logits.detach().cpu().numpy())
                            
                            ys = np.asarray(ys, int)
                            logits_all = np.concatenate(logits_all, axis=0)
                            probs = torch.softmax(torch.tensor(logits_all), dim=1).numpy()
                            yhat = probs.argmax(axis=1)
                            
                            # Get UNIQUE classes present in validation data
                            val_unique_classes = np.unique(ys)
                            val_n_classes = len(val_unique_classes)
                            print(f"  Unique classes in validation set: {val_unique_classes} (Count: {val_n_classes})")
                            print(f"  Unique classes in predictions: {np.unique(yhat)}")
                            
                            # Create mapping for validation classes (0,1,2 for classes 0,1,5)
                            val_class_to_idx = {cls: idx for idx, cls in enumerate(val_unique_classes)}
                            
                            # Create a safe mapping function that handles unknown classes
                            def safe_map(label, mapping, default=0):
                                """Map label to index, return default if label not in mapping"""
                                return mapping.get(label, default)
                            
                            # Map validation labels to consecutive indices
                            ys_mapped = np.array([safe_map(y, val_class_to_idx) for y in ys])
                            
                            # Filter predictions to only include classes in mapping
                            # For unknown predictions, we'll map to the closest class or ignore
                            valid_pred_mask = np.isin(yhat, val_unique_classes)
                            n_invalid = np.sum(~valid_pred_mask)
                            if n_invalid > 0:
                                print(f"  Warning: {n_invalid} predictions for classes not in validation set: {np.unique(yhat[~valid_pred_mask])}")
                            
                            # For confusion matrix, only use valid predictions
                            yhat_filtered = yhat[valid_pred_mask]
                            ys_filtered = ys[valid_pred_mask]
                            
                            if len(yhat_filtered) > 0:
                                yhat_mapped = np.array([safe_map(y, val_class_to_idx) for y in yhat_filtered])
                                ys_mapped_filtered = np.array([safe_map(y, val_class_to_idx) for y in ys_filtered])
                                
                                # Get class names for validation classes
                                val_class_names = []
                                for cls in val_unique_classes:
                                    if cls < len(self.real_class_names):
                                        val_class_names.append(self.real_class_names[cls])
                                    else:
                                        val_class_names.append(f"class_{cls}")
                                
                                print(f"  Validation class names: {val_class_names}")
                                
                                # Plot validation confusion matrix
                                self.viz.confusion(ys_mapped_filtered, yhat_mapped, class_names=val_class_names,
                                                title=f"Confusion — classification (Validation)",
                                                fname=os.path.join(viz_task_dir, f"{out_prefix}_cm.png"), 
                                                normalize=False)
                            else:
                                print(f"  Warning: No valid predictions for classes in validation set")
                            
                            # ========== TRAINING CONFUSION MATRIX ==========
                            if len(train_labels_list) > 0 and len(train_preds_list) > 0:
                                train_labels = np.array(train_labels_list)
                                train_preds = np.array(train_preds_list)
                                
                                # Filter to validation classes only
                                train_mask = np.isin(train_labels, val_unique_classes)
                                train_labels_filtered = train_labels[train_mask]
                                train_preds_filtered = train_preds[train_mask]
                                
                                # Also filter predictions to validation classes
                                train_pred_mask = np.isin(train_preds_filtered, val_unique_classes)
                                train_labels_filtered = train_labels_filtered[train_pred_mask]
                                train_preds_filtered = train_preds_filtered[train_pred_mask]
                                
                                if len(train_labels_filtered) > 0:
                                    train_labels_mapped = np.array([safe_map(y, val_class_to_idx) for y in train_labels_filtered])
                                    train_preds_mapped = np.array([safe_map(y, val_class_to_idx) for y in train_preds_filtered])
                                    
                                    self.viz.confusion(train_labels_mapped, train_preds_mapped, class_names=val_class_names,
                                                    title=f"Confusion — classification (Training)",
                                                    fname=os.path.join(viz_task_dir, f"{out_prefix}_cm_train.png"), 
                                                    normalize=False)
                            
                            # ========== METRICS FOR VALIDATION CLASSES ==========
                            if val_n_classes > 1 and len(yhat_filtered) > 0:
                                precision, recall, f1, _ = precision_recall_fscore_support(ys_mapped_filtered, yhat_mapped, 
                                                                                            labels=range(val_n_classes), 
                                                                                            average='weighted')
                                print(f"  Post-Training Metrics — Classification: Precision: {precision:.4f}, "
                                    f"Recall: {recall:.4f}, F1: {f1:.4f}")
                            
                            # ========== PER-CLASS CURVES (Only for validation classes) ==========
                            for idx, cls in enumerate(val_unique_classes):
                                # Create binary labels for this class (1 if this class, 0 otherwise)
                                y_bin = (ys == cls).astype(int)
                                
                                # Get probability for this class
                                if cls < probs.shape[1]:
                                    p_bin = probs[:, cls]
                                else:
                                    print(f"  Warning: class {cls} out of probs range {probs.shape[1]}, skipping")
                                    continue
                                
                                class_name = self.real_class_names[cls] if cls < len(self.real_class_names) else f"class_{cls}"
                                
                                n_pos = np.sum(y_bin)
                                n_neg = len(y_bin) - n_pos
                                print(f"    Class {cls} ({class_name}): {n_pos} positive, {n_neg} negative samples")
                                
                                # Only generate curves if we have both positive and negative samples
                                if n_pos > 0 and n_neg > 0:
                                    self.viz.pr_curve(y_bin, p_bin, title=f"PR — class {class_name}", 
                                                    fname=os.path.join(viz_task_dir, f"{out_prefix}_pr_c{cls}.png"))
                                    self.viz.reliability_curve(y_bin, p_bin, title=f"Calibration — class {class_name}", 
                                                            fname=os.path.join(viz_task_dir, f"{out_prefix}_calib_c{cls}.png"))
                                    self.viz.roc_with_ci(y_bin, p_bin, title=f"ROC (95% CI) — class {class_name}", 
                                                        fname=os.path.join(viz_task_dir, f"{out_prefix}_roc_ci_c{cls}.png"))
                                else:
                                    print(f"    Skipping curves for class {class_name} (need both positive and negative samples)")
                            
                            print(f"  ✓ Classification plots saved for {val_n_classes} classes")
                            
                        except Exception as e:
                            print(f"  ✗ Classification visuals failed: {e}")
                            import traceback
                            traceback.print_exc()
                    elif early_reg:
                        # Regression: Scatter, Residuals, Bland-Altman
                       # ========== Helper function for best‑fit line ==========
                        def add_best_fit_line(ax, x, y, color='red'):
                            """Add linear regression line and return R²."""
                            mask = np.isfinite(x) & np.isfinite(y)
                            x_clean, y_clean = x[mask], y[mask]
                            if len(x_clean) < 2:
                                return None
                            slope, intercept = np.polyfit(x_clean, y_clean, 1)
                            r2 = r2_score(y_clean, slope * x_clean + intercept)
                            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                            y_line = slope * x_line + intercept
                            ax.plot(x_line, y_line, color=color, linestyle='-', linewidth=2,
                                    label=f'Best fit: y={slope:.2f}x+{intercept:.2f} (R²={r2:.3f})')
                            return r2

                        # ========== Helper function for Bland‑Altman ==========
                        def plot_bland_altman(y_true, y_pred, title, fname):
                            mean = (y_true + y_pred) / 2
                            diff = y_true - y_pred
                            bias = np.mean(diff)
                            std_diff = np.std(diff)
                            loa_upper = bias + 1.96 * std_diff
                            loa_lower = bias - 1.96 * std_diff
                            plt.figure(figsize=(6, 4), dpi=150)
                            plt.scatter(mean, diff, s=15, alpha=0.6, edgecolors='k', linewidth=0.5)
                            plt.axhline(bias, color='red', linestyle='-', linewidth=2, label=f'Bias: {bias:.2f} s')
                            plt.axhline(loa_upper, color='gray', linestyle='--', linewidth=1.5,
                                        label=f'95% LoA: [{loa_lower:.2f}, {loa_upper:.2f}] s')
                            plt.axhline(loa_lower, color='gray', linestyle='--', linewidth=1.5)
                            plt.xlabel('Mean of true and predicted (s)', fontsize=12)
                            plt.ylabel('Difference (true - predicted) (s)', fontsize=12)
                            plt.title(title, fontsize=14)
                            plt.legend(loc='best', fontsize=9)
                            plt.grid(alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(fname, dpi=300, bbox_inches='tight')
                            plt.close()

                        # ========== VALIDATION PLOTS ==========
                        # Collect predictions
                        y_true, y_pred = [], []
                        with torch.no_grad():
                            for b in val_loader:
                                b = b.to(self.device)
                                ew = getattr(b, "edge_weight", None)
                                pred_sc = self.model(b.x, b.edge_index, b, task="forecast_time", edge_weight=ew).cpu().numpy()
                                true_sc = b.seq_targets.cpu().numpy()
                                if self.regression_scaler is not None:
                                    y_pred.extend(np.expm1(self.regression_scaler.inverse_transform(pred_sc.reshape(-1, 1))).flatten())
                                    y_true.extend(np.expm1(self.regression_scaler.inverse_transform(true_sc.reshape(-1, 1))).flatten())
                                else:
                                    y_pred.extend(pred_sc.flatten())
                                    y_true.extend(true_sc.flatten())
                        y_true = np.array(y_true, dtype=float)
                        y_pred = np.array(y_pred, dtype=float)

                        # 1. Scatter plot with best‑fit line
                        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
                        ax.scatter(y_true, y_pred, s=15, alpha=0.6, edgecolors='k', linewidth=0.5, label='Predictions')
                        add_best_fit_line(ax, y_true, y_pred)
                        # Diagonal y=x line
                        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
                        ax.plot(lims, lims, 'k--', alpha=0.5, label='Ideal (y=x)')
                        ax.set_xlabel('True TTI (s)', fontsize=12)
                        ax.set_ylabel('Predicted TTI (s)', fontsize=12)
                        ax.set_title('Early regression — validation set', fontsize=14)
                        ax.legend(loc='best', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(viz_task_dir, f"{out_prefix}_reg_scatter.png"), dpi=300, bbox_inches='tight')
                        plt.close()

                        # 2. Residuals histogram with normal fit
                        residuals = y_pred - y_true
                        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
                        n, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue',
                                                edgecolor='black', linewidth=0.8)
                        mu, std = np.mean(residuals), np.std(residuals)
                        x_fit = np.linspace(residuals.min(), residuals.max(), 200)
                        from scipy.stats import norm
                        pdf_fit = norm.pdf(x_fit, mu, std)
                        ax.plot(x_fit, pdf_fit, 'r-', linewidth=2, label=f'Normal fit (μ={mu:.2f}, σ={std:.2f})')
                        # Shapiro‑Wilk test (only if sample size reasonable)
                        if 3 <= len(residuals) <= 5000:
                            from scipy.stats import shapiro
                            _, p_val = shapiro(residuals)
                            ax.text(0.05, 0.95, f'Shapiro-Wilk p={p_val:.3f}', transform=ax.transAxes,
                                    verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        ax.set_xlabel('Residual (s)', fontsize=12)
                        ax.set_ylabel('Density', fontsize=12)
                        ax.set_title('Residuals distribution (validation)', fontsize=14)
                        ax.legend(loc='best', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(viz_task_dir, f"{out_prefix}_reg_residuals.png"), dpi=300, bbox_inches='tight')
                        plt.close()

                        # 3. Bland‑Altman plot
                        plot_bland_altman(y_true, y_pred, title="Bland–Altman — early_reg (validation)",
                                        fname=os.path.join(viz_task_dir, f"{out_prefix}_bland_altman.png"))

                        # ========== TRAINING PLOTS (identical structure) ==========
                        y_true_train, y_pred_train = [], []
                        with torch.no_grad():
                            for b in train_loader:
                                b = b.to(self.device)
                                ew = getattr(b, "edge_weight", None)
                                pred_sc = self.model(b.x, b.edge_index, b, task="forecast_time", edge_weight=ew).cpu().numpy()
                                true_sc = b.seq_targets.cpu().numpy()
                                if self.regression_scaler is not None:
                                    y_pred_train.extend(np.expm1(self.regression_scaler.inverse_transform(pred_sc.reshape(-1, 1))).flatten())
                                    y_true_train.extend(np.expm1(self.regression_scaler.inverse_transform(true_sc.reshape(-1, 1))).flatten())
                                else:
                                    y_pred_train.extend(pred_sc.flatten())
                                    y_true_train.extend(true_sc.flatten())
                        y_true_train = np.array(y_true_train, dtype=float)
                        y_pred_train = np.array(y_pred_train, dtype=float)

                        # Scatter (training)
                        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
                        ax.scatter(y_true_train, y_pred_train, s=15, alpha=0.6, edgecolors='k', linewidth=0.5, label='Predictions')
                        add_best_fit_line(ax, y_true_train, y_pred_train)
                        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
                        ax.plot(lims, lims, 'k--', alpha=0.5, label='Ideal (y=x)')
                        ax.set_xlabel('True TTI (s)', fontsize=12)
                        ax.set_ylabel('Predicted TTI (s)', fontsize=12)
                        ax.set_title('Early regression — training set', fontsize=14)
                        ax.legend(loc='best', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(viz_task_dir, f"{out_prefix}_train_reg_scatter.png"), dpi=300, bbox_inches='tight')
                        plt.close()

                        # Residuals (training)
                        residuals_train = y_pred_train - y_true_train
                        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
                        ax.hist(residuals_train, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8)
                        mu_t, std_t = np.mean(residuals_train), np.std(residuals_train)
                        x_fit_t = np.linspace(residuals_train.min(), residuals_train.max(), 200)
                        pdf_fit_t = norm.pdf(x_fit_t, mu_t, std_t)
                        ax.plot(x_fit_t, pdf_fit_t, 'r-', linewidth=2, label=f'Normal fit (μ={mu_t:.2f}, σ={std_t:.2f})')
                        if 3 <= len(residuals_train) <= 5000:
                            _, p_val_t = shapiro(residuals_train)
                            ax.text(0.05, 0.95, f'Shapiro-Wilk p={p_val_t:.3f}', transform=ax.transAxes,
                                    verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        ax.set_xlabel('Residual (s)', fontsize=12)
                        ax.set_ylabel('Density', fontsize=12)
                        ax.set_title('Residuals distribution (training)', fontsize=14)
                        ax.legend(loc='best', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(viz_task_dir, f"{out_prefix}_train_reg_residuals.png"), dpi=300, bbox_inches='tight')
                        plt.close()

                        # Bland‑Altman (training)
                        plot_bland_altman(y_true_train, y_pred_train, title="Bland–Altman — early_reg (training)",
                                        fname=os.path.join(viz_task_dir, f"{out_prefix}_train_bland_altman.png"))

                        print(f"  ✓ Regression plots saved (publication quality: best‑fit line, residuals + normal fit, Bland‑Altman)")
                    else:  # early_clf
                        try:
                            ys, logits_all = [], []
                            with torch.no_grad():
                                for b in val_loader:
                                    b = b.to(self.device)
                                    ew = getattr(b, "edge_weight", None)
                                    logits = self.model(b.x, b.edge_index, b, task="forecast_label", edge_weight=ew)
                                    ys.extend(b.seq_targets.detach().cpu().numpy().tolist())
                                    logits_all.append(logits.detach().cpu().numpy())
                            
                            ys = np.asarray(ys, int)
                            logits_all = np.concatenate(logits_all, axis=0)
                            probs = torch.softmax(torch.tensor(logits_all), dim=1).numpy()
                            yhat = probs.argmax(axis=1)
                            
                            # Get UNIQUE classes present in validation data
                            val_unique_classes = np.unique(ys)
                            val_n_classes = len(val_unique_classes)
                            print(f"  Unique classes in validation set: {val_unique_classes} (Count: {val_n_classes}")
                            print(f"  Unique classes in predictions: {np.unique(yhat)}")
                            
                            # Create mapping for validation classes
                            val_class_to_idx = {cls: idx for idx, cls in enumerate(val_unique_classes)}
                            
                            def safe_map(label, mapping, default=0):
                                return mapping.get(label, default)
                            
                            # Filter predictions to valid classes
                            valid_pred_mask = np.isin(yhat, val_unique_classes)
                            n_invalid = np.sum(~valid_pred_mask)
                            if n_invalid > 0:
                                print(f"  Warning: {n_invalid} predictions for classes not in validation set")
                            
                            yhat_filtered = yhat[valid_pred_mask]
                            ys_filtered = ys[valid_pred_mask]
                            
                            if len(yhat_filtered) > 0:
                                yhat_mapped = np.array([safe_map(y, val_class_to_idx) for y in yhat_filtered])
                                ys_mapped = np.array([safe_map(y, val_class_to_idx) for y in ys_filtered])
                                
                                # Get class names
                                val_class_names = []
                                for cls in val_unique_classes:
                                    if cls < len(self.real_class_names):
                                        val_class_names.append(self.real_class_names[cls])
                                    else:
                                        val_class_names.append(f"class_{cls}")
                                
                                # Plot confusion matrix
                                self.viz.confusion(ys_mapped, yhat_mapped, class_names=val_class_names,
                                                title=f"Confusion — early label forecast (Validation)",
                                                fname=os.path.join(viz_task_dir, f"{out_prefix}_cm.png"), normalize=False)
                            
                            # Training confusion matrix
                            if len(train_labels_list) > 0 and len(train_preds_list) > 0:
                                train_labels = np.array(train_labels_list)
                                train_preds = np.array(train_preds_list)
                                
                                train_mask = np.isin(train_labels, val_unique_classes)
                                train_labels_filtered = train_labels[train_mask]
                                train_preds_filtered = train_preds[train_mask]
                                
                                train_pred_mask = np.isin(train_preds_filtered, val_unique_classes)
                                train_labels_filtered = train_labels_filtered[train_pred_mask]
                                train_preds_filtered = train_preds_filtered[train_pred_mask]
                                
                                if len(train_labels_filtered) > 0:
                                    train_labels_mapped = np.array([safe_map(y, val_class_to_idx) for y in train_labels_filtered])
                                    train_preds_mapped = np.array([safe_map(y, val_class_to_idx) for y in train_preds_filtered])
                                    
                                    self.viz.confusion(train_labels_mapped, train_preds_mapped, class_names=val_class_names,
                                                    title=f"Confusion — early label forecast (Training)",
                                                    fname=os.path.join(viz_task_dir, f"{out_prefix}_cm_train.png"), normalize=False)
                            
                            # Per-class curves
                            for cls in val_unique_classes:
                                y_bin = (ys == cls).astype(int)
                                
                                if cls < probs.shape[1]:
                                    p_bin = probs[:, cls]
                                else:
                                    continue
                                
                                class_name = self.real_class_names[cls] if cls < len(self.real_class_names) else f"class_{cls}"
                                
                                n_pos = np.sum(y_bin)
                                n_neg = len(y_bin) - n_pos
                                
                                if n_pos > 0 and n_neg > 0:
                                    self.viz.pr_curve(y_bin, p_bin, title=f"PR — early_clf class {class_name}",
                                                    fname=os.path.join(viz_task_dir, f"{out_prefix}_pr_c{cls}.png"))
                                    self.viz.reliability_curve(y_bin, p_bin, title=f"Calibration — early_clf class {class_name}",
                                                            fname=os.path.join(viz_task_dir, f"{out_prefix}_calib_c{cls}.png"))
                                    self.viz.roc_with_ci(y_bin, p_bin, title=f"ROC (95% CI) — early_clf class {class_name}",
                                                        fname=os.path.join(viz_task_dir, f"{out_prefix}_roc_ci_c{cls}.png"))
                            
                            print(f"  ✓ Early label plots saved for {val_n_classes} classes")
                            
                        except Exception as e:
                            print(f"  ✗ Early label visuals failed: {e}")
                            import traceback
                            traceback.print_exc()
                except Exception as e:
                    print(f"  ✗ Temporal heatmap skipped: {e}")

                # ========== 7. LEARNING CURVE ==========
                if len(epoch_idx) > 0:
                    try:
                        has_valid_train = any(np.isfinite(x) for x in train_trace)
                        has_valid_val = any(np.isfinite(x) for x in val_trace)
                        
                        if has_valid_train or has_valid_val:
                            metric_name = "R²" if early_reg else "Accuracy"
                            clean_epochs = []
                            clean_train = []
                            clean_val = []
                            
                            for ep, tr, va in zip(epoch_idx, train_trace, val_trace):
                                if np.isfinite(tr) or np.isfinite(va):
                                    clean_epochs.append(ep)
                                    clean_train.append(tr if np.isfinite(tr) else 0.0)
                                    clean_val.append(va if np.isfinite(va) else 0.0)
                            
                            if len(clean_epochs) > 0:
                                self.viz.learning_curve(clean_epochs, clean_train, clean_val,
                                                        title=f"Learning Curve ({metric_name}) — {task}",
                                                        fname=os.path.join(viz_task_dir, f"{out_prefix}_learning.png"))
                                print(f"  ✓ Learning curve saved")
                            else:
                                print(f"  ✗ Learning curve skipped: No valid data points")
                        else:
                            print(f"  ✗ Learning curve skipped: No valid data (all NaN/Inf)")
                    except Exception as e:
                        print(f"  ✗ Learning curve skipped: {e}")
                else:
                    print(f"  ✗ Learning curve skipped: No epoch data")

                print(f"\n{'='*60}")
                print(f"VISUALIZATION COMPLETE FOR {task.upper()}")
                print(f"Files saved to: {viz_task_dir}")
                print(f"{'='*60}\n")

            except Exception as e:
                print(f"Visualization error: {e}")
                import traceback
                traceback.print_exc()


                # ========== STATISTICAL ANALYSIS ==========
        if val_loader is not None and self.model is not None:
            try:
                print(f"\n{'='*70}")
                print(f"COMPUTING STATISTICAL ANALYSIS FOR {task.upper()}")
                print(f"{'='*70}")
                
                if detection:
                    # For detection task
                    ys, ps = [], []
                    with torch.no_grad():
                        for b in val_loader:
                            b = b.to(self.device)
                            ew = getattr(b, "edge_weight", None)
                            logit = self.model(b.x, b.edge_index, b, task="detection", edge_weight=ew)
                            ys.extend(b.y.detach().cpu().numpy().tolist())
                            ps.extend(torch.sigmoid(logit).detach().cpu().numpy().tolist())
                    
                    ys = np.asarray(ys, int)
                    ps = np.asarray(ps, float)
                    y_pred_binary = (ps > 0.5).astype(int)
                    
                    # Compute statistics with confidence intervals
                    stats_results = self.compute_statistical_analysis(
                        ys, y_pred_binary, y_proba=ps, task='detection', n_bootstrap=1000
                    )
                    
                    # Compare with random baseline
                    random_baseline = np.random.randint(0, 2, size=len(ys))
                    self.compare_with_baseline(ys, y_pred_binary, random_baseline, task='classification')
                elif classification or early_clf:
                    # Collect predictions and probabilities
                    ys, logits_all = [], []
                    with torch.no_grad():
                        for b in val_loader:
                            b = b.to(self.device)
                            ew = getattr(b, "edge_weight", None)
                            # Use the same task string as in training
                            logits = self.model(b.x, b.edge_index, b, task="forecast_label", edge_weight=ew)
                            ys.extend(b.seq_targets.detach().cpu().numpy().tolist())
                            logits_all.append(logits.detach().cpu().numpy())
                    ys = np.asarray(ys, int)
                    logits_all = np.concatenate(logits_all, axis=0)
                    probs = torch.softmax(torch.tensor(logits_all), dim=1).numpy()
                    y_pred = probs.argmax(axis=1)

                    # ---- Bootstrap with per‑class metrics ----
                    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
                    n_classes = probs.shape[1]
                    # Get all class labels that actually appear
                    present_classes = np.unique(ys)
                    n_bootstrap = 1000

                    # Storage for global metrics and per‑class metrics
                    global_metrics = {
                        'accuracy': [], 'balanced_accuracy': [], 
                        'f1_macro': [], 'f1_weighted': []
                    }
                    per_class_metrics = {
                        'precision': {c: [] for c in present_classes},
                        'recall': {c: [] for c in present_classes},
                        'f1': {c: [] for c in present_classes}
                    }

                    np.random.seed(42)
                    n = len(ys)

                    for _ in range(n_bootstrap):
                        idx = np.random.choice(n, n, replace=True)
                        y_true_b = ys[idx]
                        y_pred_b = y_pred[idx]

                        global_metrics['accuracy'].append(accuracy_score(y_true_b, y_pred_b))
                        global_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_true_b, y_pred_b))
                        global_metrics['f1_macro'].append(f1_score(y_true_b, y_pred_b, average='macro', zero_division=0))
                        global_metrics['f1_weighted'].append(f1_score(y_true_b, y_pred_b, average='weighted', zero_division=0))

                        # Per‑class metrics (only for classes present in this bootstrap sample)
                        for c in present_classes:
                            # Convert to binary
                            y_true_c = (y_true_b == c).astype(int)
                            y_pred_c = (y_pred_b == c).astype(int)
                            # Precision (avoid division by zero)
                            p = precision_score(y_true_c, y_pred_c, zero_division=0)
                            r = recall_score(y_true_c, y_pred_c, zero_division=0)
                            f = f1_score(y_true_c, y_pred_c, zero_division=0)
                            per_class_metrics['precision'][c].append(p)
                            per_class_metrics['recall'][c].append(r)
                            per_class_metrics['f1'][c].append(f)

                    # Aggregate global metrics with confidence intervals
                    stats_results = {'metrics': {}}
                    for name, values in global_metrics.items():
                        stats_results['metrics'][name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'ci_lower': np.percentile(values, 2.5),
                            'ci_upper': np.percentile(values, 97.5)
                        }

                    # Add per‑class metrics to the same 'metrics' dict (flattened)
                    for metric_type in ['precision', 'recall', 'f1']:
                        for c in present_classes:
                            values = per_class_metrics[metric_type][c]
                            metric_name = f"{metric_type}_class_{c}"
                            stats_results['metrics'][metric_name] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'ci_lower': np.percentile(values, 2.5),
                                'ci_upper': np.percentile(values, 97.5)
                            }
                            # Also add class name if available
                            if hasattr(self, 'real_class_names') and c < len(self.real_class_names):
                                metric_name_named = f"{metric_type}_{self.real_class_names[c]}"
                                stats_results['metrics'][metric_name_named] = stats_results['metrics'][metric_name].copy()

                    # Print global metrics table
                    print("\n" + "="*70)
                    print(f"STATISTICAL ANALYSIS - {task.upper()} (validation set)")
                    print("="*70)
                    print(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'95% CI':<20}")
                    print("-"*70)
                    for name, m in stats_results['metrics'].items():
                        if not any(x in name for x in ['precision_', 'recall_', 'f1_']):  # print global first
                            print(f"{name:<25} {m['mean']:<12.4f} {m['std']:<12.4f} [{m['ci_lower']:<8.4f}, {m['ci_upper']:<8.4f}]")
                    
                    # Print per‑class metrics
                    print("\n" + "-"*70)
                    print("Per‑class metrics (validation):")
                    for c in present_classes:
                        class_name = (self.real_class_names[c] if hasattr(self, 'real_class_names') and c < len(self.real_class_names) else f"Class {c}")
                        print(f"\n{class_name}:")
                        for metric_type in ['precision', 'recall', 'f1']:
                            key = f"{metric_type}_class_{c}"
                            m = stats_results['metrics'][key]
                            print(f"  {metric_type.capitalize()}: {m['mean']:.4f} ± {m['std']:.4f} 95%CI [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")

                    # Compare with majority class baseline (as before)
                    from sklearn.dummy import DummyClassifier
                    from sklearn.metrics import accuracy_score
                    from scipy.stats import wilcoxon

                    dummy = DummyClassifier(strategy='most_frequent')
                    dummy.fit(ys, ys)
                    y_pred_baseline = dummy.predict(ys)
                    base_acc = accuracy_score(ys, y_pred_baseline)
                    model_acc = stats_results['metrics']['accuracy']['mean']
                    model_err = (y_pred != ys).astype(int)
                    baseline_err = (y_pred_baseline != ys).astype(int)
                    _, p_val = wilcoxon(model_err, baseline_err, alternative='less')
                    print("\n" + "="*70)
                    print("MODEL VS MAJORITY CLASS BASELINE")
                    print("="*70)
                    print(f"Model accuracy: {model_acc:.4f}")
                    print(f"Baseline accuracy: {base_acc:.4f}")
                    print(f"Improvement: {model_acc - base_acc:.4f}")
                    print(f"Wilcoxon p-value (model better): {p_val:.6f}")
                    print(f"Statistically Significant: {'YES' if p_val < 0.05 else 'NO'}")

                # Save CSV (same as before)
                    
                # elif classification or early_clf:
                #     # For classification task
                #     ys, logits_all = [], []
                #     with torch.no_grad():
                #         for b in val_loader:
                #             b = b.to(self.device)
                #             ew = getattr(b, "edge_weight", None)
                #             logits = self.model(b.x, b.edge_index, b, task="early_clf", edge_weight=ew)
                #             ys.extend(b.seq_targets.detach().cpu().numpy().tolist())
                #             logits_all.append(logits.detach().cpu().numpy())
                    
                #     ys = np.asarray(ys, int)
                #     logits_all = np.concatenate(logits_all, axis=0)
                #     probs = torch.softmax(torch.tensor(logits_all), dim=1).numpy()
                #     y_pred = probs.argmax(axis=1)
                    
                #     # Compute statistics with confidence intervals
                #     stats_results = self.compute_statistical_analysis(
                #         ys, y_pred, y_proba=probs, task='early_clf', n_bootstrap=1000
                #     )
                    
                #     # Compare with majority class baseline
                #     from sklearn.dummy import DummyClassifier
                #     dummy = DummyClassifier(strategy='most_frequent')
                #     dummy.fit(ys, ys)
                #     y_pred_baseline = dummy.predict(ys)
                #     self.compare_with_baseline(ys, y_pred, y_pred_baseline, task='classification')
                    
                elif early_reg:
                    # For regression task
                    y_true, y_pred = [], []
                    with torch.no_grad():
                        for b in val_loader:
                            b = b.to(self.device)
                            ew = getattr(b, "edge_weight", None)
                            pred_sc = self.model(b.x, b.edge_index, b, task="forecast_time", edge_weight=ew).cpu().numpy()
                            if self.regression_scaler is not None:
                                y_pred.extend(np.expm1(self.regression_scaler.inverse_transform(pred_sc.reshape(-1, 1))).flatten())
                            else:
                                y_pred.extend(pred_sc.flatten())
                            # y_true.extend(b.y.detach().cpu().numpy().tolist())
                            y_true.extend(b.seq_targets.detach().cpu().numpy().tolist())
                    
                    y_true = np.array(y_true, dtype=float)
                    y_pred = np.array(y_pred, dtype=float)
                    
                    if self.regression_scaler is not None:
                        y_true = np.expm1(self.regression_scaler.inverse_transform(y_true.reshape(-1, 1))).flatten()
                    
                    # Compute statistics with confidence intervals
                    stats_results = self.compute_statistical_analysis(
                        y_true, y_pred, task='regression', n_bootstrap=1000
                    )
                    
                    # Compare with mean baseline
                    mean_baseline = np.full_like(y_true, np.mean(y_true))
                    self.compare_with_baseline(y_true, y_pred, mean_baseline, task='regression')
                
                # Save statistical results to CSV
                import csv
                stats_file = os.path.join(viz_task_dir, f"{out_prefix}_statistical_analysis.csv")
                with open(stats_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Mean', 'Std', 'CI_Lower', 'CI_Upper'])
                    for metric_name, metric_data in stats_results['metrics'].items():
                        writer.writerow([
                            metric_name,
                            f"{metric_data['mean']:.4f}",
                            f"{metric_data['std']:.4f}",
                            f"{metric_data['ci_lower']:.4f}",
                            f"{metric_data['ci_upper']:.4f}"
                        ])
                
                print(f"\n  ✓ Statistical analysis saved to {stats_file}")
                
            except Exception as e:
                print(f"  ✗ Statistical analysis failed: {e}")
                import traceback
                traceback.print_exc()

