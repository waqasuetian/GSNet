__all__ = ["Trainer", "make_overview_radars", "run_stratified_kfold", "utilis"]


import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # Adjust to your CPU's physic
import warnings
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from models.model import MultiTaskGCN
from data.scripts.adjacancy_matrics import AdjacencyMatrixProcessor
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold   # <-- added StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from data.scripts.features import FeatureBuilder
from data.scripts.losses import AdaptiveHeadLoss
from data.scripts.interpreters import NodeInterpreter
from data.scripts.viz import Viz
from imblearn.over_sampling import SMOTE  # Added for oversampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import f1_score
from data.scripts.data_loader_preictal import EEGProcessorPreictal


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
                ("BandPower Δ..γ (5)", _np.arange(i, i+5)),     # 0..4
                ("BandPower Total (1)", _np.arange(i+5, i+6)),  # 5
                ("RelPower Δ..γ (5)", _np.arange(i+6, i+11)),   # 6..10
                ("Ratios α/θ, β/α (2)", _np.arange(i+11, i+13)),# 11..12
                ("Spectral Entropy (1)", _np.arange(i+13, i+14))# 13
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
        """
        Map frequency band (inclusive, 1-indexed like FeatureBuilder) to 0-indexed
        column slice for the 'bins' block starting at start_col.
        """
        lo, hi = band_span
        lo = max(1, int(np.floor(lo))); hi = min(rfft_bins, int(np.floor(hi)))
        idx = np.arange(start_col + (lo-1), start_col + hi)
        return idx

    def band_groups(self, task: str, in_dim: int, rfft_bins=100):
        """
        Build (name, idxs) for five EEG bands using the first 100 (or 200) bin columns.
        For RC, merges mean+std bins per band into a single group per band.
        """
        groups = []
        if task in ("detection", "classification"):
            base = 0
            for name, span in self.BANDS.items():
                groups.append((f"{name}", self._band_indices_for_bins(rfft_bins, span, base)))
        else:  # RC: mean(100) + std(100)
            base_mean = 0
            base_std  = 100
            for name, span in self.BANDS.items():
                idx_mean = self._band_indices_for_bins(rfft_bins, span, base_mean)
                idx_std  = self._band_indices_for_bins(rfft_bins, span, base_std)
                idx = np.concatenate([idx_mean, idx_std])
                groups.append((f"{name}", idx))
        # ensure indices exist
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
                    preds.extend(out.argmax(dim=1).cpu().numpy()); labels.extend(batch.y.cpu().numpy())
                else:
                    out = self.model(batch.x, batch.edge_index, batch, task="forecast_time", edge_weight=ew)
                    preds.extend(out.cpu().numpy()); labels.extend(batch.y.cpu().numpy())
            if task in ["early_reg", "forecast_time"]:
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
                        preds.extend(out.argmax(dim=1).cpu().numpy()); labels.extend(batch.y.cpu().numpy())
                    else:
                        out = self.model(x_perm, batch.edge_index, batch, task="forecast_time", edge_weight=ew)
                        preds.extend(out.cpu().numpy()); labels.extend(batch.y.cpu().numpy())

                    if task in ["early_reg", "forecast_time"]:
                        score = -np.sqrt(mean_squared_error(labels, preds))
                    else:
                        score = accuracy_score(labels, preds)
                    scores.append(score)
            names.append(name)
            drops.append(base - float(np.mean(scores)))
        return names, np.array(drops, dtype=float), float(base)


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
        channel_names: List[str] = [
                            "Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2",
                            "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
                            "FZ-CZ", "CZ-PZ", "P7-T5" #"P8-T6"
                        ],
         
        #   ['A1-T3','C3-CZ','C3-P3','C4-P4','C4-T4','CZ-C4',
        #            'F3-C3','F4-C4','F7-T3','F8-T4','FP1-F3','FP1-F7',
        #            'FP2-F4','FP2-F8','P3-O1','P4-O2','T3-C3','T3-T5','T4-A2'],
        out_dir: str = r"D:\PhD Research\Experiments\Gen_EEG\runs\graphs",
        data_directory: str = r"F:\tuh_data\train",
        seed: int = 42
    ):
        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs  # Reduced to 300 to mitigate overfitting
        self.batch_size = batch_size
        self.base_lr = learning_rate
        self.base_wd = 0.0001  # Increased weight decay for regularization

        # viz & labels
        self.viz = Viz(out_dir)
        self.num_nodes = 19
        assert len(channel_names) == self.num_nodes, f"channel_names length {len(channel_names)} != {self.num_nodes}"
        self.channel_names = channel_names or [f"Ch{i+1}" for i in range(self.num_nodes)]
        self.real_class_names = ['gnsz', 'fnsz', 'tcsz', 'absz', 'mysz', 'cpsz', 'tnsz']  # Real class names for confusion matrix

        # Defer model creation until we know feature dims
        self.model = None
        self.num_features_cfg = num_features
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.dropout = dropout

        self.adaptive_loss = AdaptiveHeadLoss(smoothing=0.05, focal_gamma=3.5)

        # ----- Graph topology & weights -----
        base_edge_index = torch.tril_indices(self.num_nodes, self.num_nodes, offset=-1)
        expected_pairs = self.num_nodes * (self.num_nodes - 1) // 2  # 171

        adj_proc = AdjacencyMatrixProcessor(pooled_results, data_directory=data_directory)
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

        print("[DEBUG] type(raw):", type(raw))
        if isinstance(raw, torch.Tensor):
            print("[DEBUG] raw.shape:", tuple(raw.shape), "dtype:", raw.dtype)

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

        # Scalers
        self.feature_scaler = None
        self.regression_scaler = None

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
        """
        Returns X_train, X_val, Y_train, Y_val with:
          - stratification preserved
          - guarantee: every class present in Y_val *if possible*
        If any class has <2 samples, it's impossible to put one in train and one in val;
        we then fall back to a deterministic "at least one per class in val if available".
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        # Quick exit: binary regression or continuous labels => not applicable
        if Y.dtype.kind in "fc":  # float or complex => treat as regression
            raise ValueError("Strict stratified split called for non-categorical Y.")

        counts = Counter(Y.tolist())
        # If a class has only 1 example, classic stratified split cannot guarantee presence in both.
        too_small = [c for c, n in counts.items() if n < 2]
        if len(too_small) > 0:
            # Fallback: hand-pick one example of each class (if present) into val, rest stratify best-effort
            val_idx = []
            for cls in sorted(counts):
                idx_cls = np.where(Y == cls)[0]
                if idx_cls.size > 0:
                    val_idx.append(idx_cls[0])
            val_idx = np.array(val_idx, dtype=int)

            # Fill remaining val slots (if any) using stratified shuffle on the remainder
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

        # Normal path: try a single stratified shuffle split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr, va = next(sss.split(X, Y))

        # If, by chance, some class is still missing in val (can happen in extreme imbalance + rounding),
        # retry with slight random seeds a few times.
        for seed in [random_state + k for k in range(1, 8)]:
            val_classes = set(Y[va].tolist())
            if len(val_classes) == len(counts):  # has all classes
                break
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            tr, va = next(sss.split(X, Y))

        return X[tr], X[va], Y[tr], Y[va]

    @staticmethod
    def safe_r2_score(y_true, y_pred, verbose=False):
        # to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # filter non-finite
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) == 0:
            if verbose:
                warnings.warn("No valid data after filtering NaNs/infs")
            return 0.0

        if verbose:
            print(f"Number of valid samples: {len(y_true)}")
            print(f"y_true stats: min={y_true.min():.4f}, max={y_true.max():.4f}, "
                  f"mean={y_true.mean():.4f}, std={y_true.std():.4f}")
            print(f"y_pred stats: min={y_pred.min():.4f}, max={y_pred.max():.4f}, "
                  f"mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")

        # guard degenerate case
        if np.var(y_true) == 0:
            if verbose:
                print("Warning: y_true has zero variance. R² is undefined.")
            return 0.0

        r2 = r2_score(y_true, y_pred)
        if verbose:
            print(f"R²: {r2:.6f}")
        return float(r2)  # <-- no clipping

    @staticmethod
    def _oversample(X, Y, factor=2.0):
        """
        Naive oversampling (duplicates uniformly). Consider replacing with class-balanced.
        """
        n = X.shape[0]
        if n <= 1 or factor <= 1.0:
            return X, Y
        dup = max(1, int(n * (factor - 1.0)))
        idx = np.random.choice(n, size=dup, replace=True)
        return np.concatenate([X, X[idx]], axis=0), np.concatenate([Y, Y[idx]], axis=0)
    

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
    
   
    @staticmethod
    def hybrid_oversample(X_tr, Y_tr, num_nodes, floor=6, smote_cap=0.8, seed=42):
        """
        1) Duplicate tiny classes (< floor) to reach `floor`
        2) SMOTE remaining minorities up to `smote_cap * majority_count`
        """
        # Flatten to (N, features) for imblearn
        X_flat = X_tr.reshape(X_tr.shape[0], -1)
        counts = np.bincount(Y_tr)
        counts = counts[counts > 0] if (len(counts) and counts.sum()) else counts
        if len(counts) == 0:
            return X_tr, Y_tr  # nothing to do

        # ---------- Stage 1: deal with ultra-tiny classes ----------
        class_counts = np.bincount(Y_tr)
        tiny_targets = {cls: floor for cls, c in enumerate(class_counts) if 0 < c < floor}
        if tiny_targets:
            ros = RandomOverSampler(sampling_strategy=tiny_targets, random_state=seed)
            X_flat, Y_tr = ros.fit_resample(X_flat, Y_tr)
            class_counts = np.bincount(Y_tr)  # refresh after ROS

        # ---------- Stage 2: SMOTE with cap ----------
        majority = class_counts.max()
        cap = max(floor, int(smote_cap * majority))
        # only upsample classes below the cap
        smote_targets = {cls: cap for cls, c in enumerate(class_counts) if 0 < c < cap}

        if smote_targets:
            # choose k_neighbors safely based on the new minimum class size
            min_samples = min(v for v in class_counts if v > 0)
            k_neighbors = max(1, min(5, min_samples - 1))  # k < class size
            sm = SMOTE(random_state=seed, k_neighbors=k_neighbors, sampling_strategy=smote_targets)
            X_flat, Y_tr = sm.fit_resample(X_flat, Y_tr)

        # reshape back to (N, num_nodes, features_per_node)
        X_tr = X_flat.reshape(-1, num_nodes, X_flat.shape[1] // num_nodes)
        return X_tr, Y_tr

        

        # -------------
        # Interpretability (optional graph viz)
        # -------------
    def visualize_input_graph(self, save_path="graph_input.png", threshold=0.0):
            import networkx as nx
            ei = self.edge_index.detach().cpu().numpy()
            ew = self.edge_weight.detach().cpu().numpy() if self.edge_weight is not None else np.ones(ei.shape[1])
            G = nx.Graph()
            for (u, v), w in zip(ei.T, ew):
                if w >= threshold:
                    G.add_edge(int(u), int(v), weight=float(w))
            pos = nx.spring_layout(G, seed=42)
            widths = [G[u][v]['weight'] / (ew.max() + 1e-8) * 3.0 for u, v in G.edges()]
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6))
            nx.draw(G, pos, with_labels=True, width=widths, node_size=19)
            plt.title("Input EEG Graph (edge width ∝ weight)")
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.tight_layout(); plt.savefig(save_path); plt.close()
            print(f"[viz] saved {save_path}")
    def train(
            self,
            X,
            Y,
            detection: bool = False,
            classification: bool = False,
            early_reg: bool = False,
            early_clf: bool = False,
            explain_after: bool = False,
            explain_path: str | None = None
        ):
            os.makedirs("models/checkpoints", exist_ok=True)

            X = np.array(X)  # (N,T,V,F)
            Y = np.array(Y)

            # ---- Task-aware features ----
            task = "detection" if detection else "classification" if classification else "early_reg" if early_reg else "early_clf"
            fb = FeatureBuilder(
                fs=200, rfft_bins=100,
                with_time=False,
                with_shapes=True,
                with_complexity=True,
                with_connectivity=True
            )
            X_built = fb.build(X, mode=task)
            if isinstance(X_built, tuple):
                X_feat, conn = X_built     # conn: (N,V,V)
            else:
                X_feat, conn = X_built, None

            # Reduce time to per-graph features
            X_graph = X_feat.mean(axis=1)         # (N,V,Fout)

            # --- Create/refresh model now that we know the real feature dim ---
            in_dim_actual = X_graph.shape[-1]  # Fout after feature builder (+ extras)
            self.model = MultiTaskGCN(
                hidden_dim=self.num_hiddens,
                in_dim=in_dim_actual,
                num_classes=self.num_classes,
                dropout=self.dropout
            ).to(self.device)

            if conn is not None:
                C = conn.mean(axis=0)  # (V,V)
                C = (C + C.T) / 2.0
                np.fill_diagonal(C, 0.0)
                r, c = np.tril_indices(C.shape[0], k=-1)
                w = torch.tensor(C[r, c], dtype=torch.float32, device=self.device)
                ei = torch.tril_indices(self.num_nodes, self.num_nodes, offset=-1)
                self.edge_index, self.edge_weight = self._make_undirected(ei, w)
                self.edge_index = self.edge_index.to(self.device)
                self.edge_weight = self.edge_weight.to(self.device)

            # ---- Split ----
            if early_reg:
                # regression cannot be stratified 
                qbins = np.digitize(Y, np.quantile(Y, [0.2, 0.4, 0.6, 0.8]), right=True)
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_graph, Y, test_size=0.2, random_state=42, stratify=qbins
                )
            else:
                X_train, X_val, Y_train, Y_val = self._stratified_split_strict(
                    X_graph, Y, test_size=0.2, random_state=42
                )

            # ---- Scale features ----
            self.feature_scaler = StandardScaler()
            X_train_rs = X_train.reshape(-1, X_train.shape[-1])
            X_val_rs   = X_val.reshape(-1, X_val.shape[-1])
            X_train_sc = self.feature_scaler.fit_transform(X_train_rs).reshape(X_train.shape)
            X_val_sc   = self.feature_scaler.transform(X_val_rs).reshape(X_val.shape)

            # ---- Scale regression targets (log1p + StandardScaler)
            if early_reg:
                self.regression_scaler = StandardScaler()
                Y_train_sc = self.regression_scaler.fit_transform(np.log1p(Y_train).reshape(-1, 1)).flatten()
                Y_val_sc   = self.regression_scaler.transform(np.log1p(Y_val).reshape(-1, 1)).flatten()
            else:
                Y_train_sc = Y_train
                Y_val_sc   = Y_val

            # 1---- Oversample ----
        
            # X_tr, Y_tr = X_train_sc, Y_train_sc
            # X_va, Y_va = X_val_sc, Y_val_sc
            # if detection or early_clf or early_reg:
            #     print("X_tr shape:", X_tr.shape)  # Debug
            #     print("Y_tr shape:", Y_tr.shape)  # Debug
            #     if X_tr.shape[0] != Y_tr.shape[0]:
            #         raise ValueError(f"Inconsistent samples: X_tr has {X_tr.shape[0]} samples, Y_tr has {Y_tr.shape[0]} samples")
            #     # Check number of unique classes
            #     unique_classes = np.unique(Y_tr)
            #     print("Unique classes in Y_tr:", unique_classes)
            #     if len(unique_classes) > 1:
            #         # Determine the minimum number of samples per class
            #         class_counts = np.bincount(Y_tr)
            #         min_samples = min(class_counts[class_counts > 0])
            #         k_neighbors = min(5, max(1, min_samples - 1))  # guard for tiny classes
            #         print(f"Minimum samples per class: {min_samples}, Setting k_neighbors to {k_neighbors}")
            #         # Oversample at graph level
            #         smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            #         X_tr_reshaped = X_tr.reshape(X_tr.shape[0], -1)  # Flatten to (N, 19 * F)
            #         X_tr_oversampled, Y_tr_oversampled = smote.fit_resample(X_tr_reshaped, Y_tr)
            #         X_tr = X_tr_oversampled.reshape(-1, self.num_nodes, X_tr_reshaped.shape[1] // self.num_nodes)  # Reshape back
            #         Y_tr = Y_tr_oversampled
            #     else:
            #         print("Warning: Only one class detected in Y_tr. Skipping SMOTE and relying on weighted loss.")
            #     print("X_tr shape after oversampling:", X_tr.shape)  # Debug
            #     print("Y_tr shape after oversampling:", Y_tr.shape)  # Debug
            X_tr, Y_tr = X_train_sc, Y_train_sc
            X_va, Y_va = X_val_sc,   Y_val_sc

            if detection or classification or early_clf:
                print("X_tr shape:", X_tr.shape)
                print("Y_tr shape:", Y_tr.shape)
                if X_tr.shape[0] != Y_tr.shape[0]:
                    raise ValueError(f"Inconsistent samples: X_tr has {X_tr.shape[0]}, Y_tr has {Y_tr.shape[0]}")

                uniq = np.unique(Y_tr)
                print("Unique classes in Y_tr:", uniq)
                print("Unique classes in Y_va:", np.unique(Y_va))
                if len(uniq) > 1:
                    X_tr, Y_tr = self.hybrid_oversample(X_tr, Y_tr, num_nodes=self.num_nodes, floor=10, smote_cap=1, seed=42)
                else:
                    print("Warning: single class; skip oversampling and rely on class weights.")

                print("X_tr shape after oversampling:", X_tr.shape)
                print("Y_tr shape after oversampling:", Y_tr.shape)



            # Build loaders (val loader must be deterministic) ----
            train_loader = self.create_graph_batches(X_tr, Y_tr, task=task, shuffle=True)
            val_loader   = self.create_graph_batches(X_va, Y_va, task=task, shuffle=False)

            print(f"=================== {task.capitalize()} ===================")

            # ---- Optim/sched ----
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=0.0001 if early_reg else self.base_lr,
                weight_decay=1e-5 if early_reg else self.base_wd
            )
            self.scheduler = (
                ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-7)
                if early_reg else
                StepLR(self.optimizer, step_size=10, gamma=0.5)
            )

            # Compute class weights for weighted loss
            if classification or early_clf:
                all_classes = np.arange(len(self.real_class_names))
                present = np.unique(Y_tr.astype(int))  # use TRAIN labels only
                # compute on present classes; then fill absent with max(present_weight)
                cw_present = compute_class_weight(
                    class_weight='balanced',
                    classes=present,
                    y=Y_tr.astype(int)
                ).astype(float)
                class_weights = np.ones(len(all_classes), dtype=float)
                class_weights[present] = cw_present
                if len(cw_present) > 0:
                    class_weights[np.setdiff1d(all_classes, present)] = float(cw_present.max())
                class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
                self.adaptive_loss = AdaptiveHeadLoss(smoothing=0.05, focal_gamma=5, class_weights=class_weights)
                print("[loss] class_weights:", class_weights.detach().cpu().numpy().round(3).tolist())

            best_val_metric = -float("inf")
            patience_counter = 0
            early_stopping_patience = 200

            # traces for learning curve
            epoch_idx = []
            train_trace = []
            val_trace = []

            # ---- Epoch loop ----
            for epoch in range(1, self.num_epochs + 1):
                self.model.train()
                epoch_loss = 0.0
                all_preds, all_labels = [], []

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
                        out = self.model(batch.x, batch.edge_index, batch, task="forecast_time", edge_weight=ew).squeeze(-1)
                        loss = self.adaptive_loss.regression_loss(out, batch.y.float())
                        preds = out.detach().cpu().numpy()
                        labels = batch.y.detach().cpu().numpy()

                    else:  # early_clf
                        out = self.model(batch.x, batch.edge_index, batch, task="forecast_label", edge_weight=ew)
                        loss = self.adaptive_loss.classification_loss(out, batch.y,
                                                                    num_classes=self.model.label_head[-1].out_features)
                        preds = out.argmax(dim=1).detach().cpu().numpy()
                        labels = batch.y.detach().cpu().numpy()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    all_preds.extend(preds)
                    all_labels.extend(labels)

                avg_loss = epoch_loss / max(1, len(train_loader))

                # ---- Validation ----
                self.model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        ew = getattr(batch, "edge_weight", None)

                        if detection:
                            out = self.model(batch.x, batch.edge_index, batch, task="detection", edge_weight=ew).squeeze(-1)
                            preds = (torch.sigmoid(out) > 0.5).cpu().numpy()
                        elif classification:
                            out = self.model(batch.x, batch.edge_index, batch, task="classification", edge_weight=ew)
                            preds = out.argmax(dim=1).cpu().numpy()
                        elif early_reg:
                            out_sc = self.model(batch.x, batch.edge_index, batch, task="forecast_time", edge_weight=ew).cpu().numpy()
                            preds = np.expm1(self.regression_scaler.inverse_transform(out_sc.reshape(-1, 1))).flatten()
                        else:  # early_clf
                            out = self.model(batch.x, batch.edge_index, batch, task="forecast_label", edge_weight=ew)
                            preds = out.argmax(dim=1).cpu().numpy()

                        val_preds.extend(preds)
                        val_labels.extend(batch.y.detach().cpu().numpy())
                
                # Collect predictions for train set (similar to val)
                self.model.eval()
                train_preds, train_labels = [], []
                with torch.no_grad():
                    for batch in train_loader:
                        batch = batch.to(self.device)
                        ew = getattr(batch, "edge_weight", None)
                        if detection:
                            out = self.model(batch.x, batch.edge_index, batch, task="detection", edge_weight=ew).squeeze(-1)
                            preds = torch.sigmoid(out).cpu().numpy()
                        elif classification:
                            out = self.model(batch.x, batch.edge_index, batch, task="classification", edge_weight=ew)
                            preds = out.argmax(dim=1).cpu().numpy()
                        elif early_clf:
                            out = self.model(batch.x, batch.edge_index, batch, task="forecast_label", edge_weight=ew)
                            preds = out.argmax(dim=1).cpu().numpy()
                        else:
                            continue  # Skip for early_reg
                        train_preds.extend(preds)
                        train_labels.extend(batch.y.detach().cpu().numpy())

    

                # train_preds = np.array(train_preds)
                # train_labels = np.array(train_labels)

                # ---- Metrics ----
                if early_reg:
                    inv_preds = np.expm1(self.regression_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1))).flatten()
                    inv_true  = np.expm1(self.regression_scaler.inverse_transform(np.array(all_labels).reshape(-1, 1))).flatten()
                    verbose_debug = (epoch == 1)
                    train_r2  = self.safe_r2_score(inv_true, inv_preds, verbose=verbose_debug)
                    train_rmse = np.sqrt(mean_squared_error(inv_true, inv_preds))

                    val_labels_raw = np.expm1(
                        self.regression_scaler.inverse_transform(np.array(val_labels).reshape(-1, 1))
                    ).flatten()
                    val_r2   = self.safe_r2_score(val_labels_raw, np.array(val_preds), verbose=verbose_debug)
                    val_rmse = np.sqrt(mean_squared_error(val_labels_raw, np.array(val_preds)))
                    val_metric = val_r2

                    # traces
                    epoch_idx.append(epoch); train_trace.append(train_r2); val_trace.append(val_r2)

                    print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} | "
                        f"Train R2: {train_r2:.4f}, RMSE: {train_rmse:.2f} | "
                        f"Val R2: {val_r2:.4f}, RMSE: {val_rmse:.2f}")
                else:
                    train_acc = accuracy_score(all_labels, all_preds)
                    val_acc   = accuracy_score(val_labels, val_preds)
                    # macro-F1 (treats classes equally)
                    train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
                    val_f1   = f1_score(val_labels, val_preds, average="macro", zero_division=0)
                    val_metric = val_f1  # <-- drive early stop + scheduler by macro-F1

                    # traces – keep accuracy on the plot (your choice), or switch to F1
                    epoch_idx.append(epoch); train_trace.append(train_acc); val_trace.append(val_acc)

                    print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                        f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")


                    # ---- Checkpointing ----
                tag = task
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": None if isinstance(self.scheduler, ReduceLROnPlateau) else self.scheduler.state_dict(),
                            "val_metric": best_val_metric,
                            "epoch": epoch,
                        },
                        f"models/checkpoints/{tag}_best.pth"
                    )
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs).")
                    break

                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metric)
                else:
                    self.scheduler.step()

                ckpt_dir = f"models/checkpoints/{tag}"
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(ckpt_dir, f"{tag}_epoch_{epoch}.pth"))

            # =======================
            # Post-training visuals
            # =======================
            try:
                out_prefix = f"{task}"

                # 1) Node (channel) permutation importance on validation set
                try:
                    ni = NodeInterpreter(self.model, num_nodes=self.num_nodes, device=self.device)
                    base_metric, node_drop = ni.permutation_importance(val_loader, task=task, n_repeats=8, seed=7)
                    rows = [(self.channel_names[i], float(node_drop[i])) for i in range(len(node_drop))]
                    self.viz.save_csv_columns(rows, header=["channel", "importance_drop"], fname=f"{out_prefix}_node_importance.csv")
                    self.viz.barh(node_drop, self.channel_names,
                                title=f"Node importance (perm drop) — {task} (base={base_metric:.3f})",
                                fname=f"{out_prefix}_node_importance.png", top_k=20)
                    # radar for nodes
                    self.viz.radar(node_drop, self.channel_names,
                                title=f"Node Importance — {task} (base={base_metric:.3f})",
                                fname=f"{out_prefix}_radar_nodes.png")
                except Exception as e:
                    print(f"[viz] node permutation importance skipped: {e}")

                # 1b) FEATURE GROUP permutation importance + radar
                try:
                    fi = FeatureInterpreter(self.model, num_nodes=self.num_nodes, device=self.device)
                    feat_groups = fi.default_groups(
                        in_dim=in_dim_actual,
                        task=task,
                        with_shapes=True,
                        with_complexity=True
                    )
                    group_names, group_drops, base_feat = fi.permutation_importance_by_group(
                        val_loader, feat_groups, task=task, n_repeats=5
                    )
                    rows = list(zip(group_names, map(float, group_drops)))
                    self.viz.save_csv_columns(rows, header=["feature_group", "importance_drop"],
                                            fname=f"{out_prefix}_feature_groups.csv")
                    self.viz.radar(group_drops, group_names,
                                title=f"Feature-Group Importance — {task} (base={base_feat:.3f})",
                                fname=f"{out_prefix}_radar_features.png")
                except Exception as e:
                    print(f"[viz] feature-group importance skipped: {e}")

                # 1c) Band importance from bin groups + Band×Node heatmap (soft attribution)
                try:
                    fi = FeatureInterpreter(self.model, num_nodes=self.num_nodes, device=self.device)
                    band_groups = fi.band_groups(task=task, in_dim=in_dim_actual, rfft_bins=100)
                    bnames, bdrops, _ = fi.permutation_importance_by_group(
                        val_loader, band_groups, task=task, n_repeats=4
                    )
                    # save CSV for band importances
                    self.viz.save_csv_columns(list(zip(bnames, map(float, bdrops))),
                                            header=["band", "importance_drop"],
                                            fname=f"{out_prefix}_band_importance.csv")
                    # soft outer-product for per-node-per-band heatmap (for visualization only)
                    node_imp = np.asarray(node_drop, float)
                    node_imp = np.maximum(node_imp, 0)
                    if node_imp.max() > 0: node_imp = node_imp / (node_imp.max() + 1e-12)
                    band_imp = np.asarray(bdrops, float)
                    band_imp = np.maximum(band_imp, 0)
                    if band_imp.max() > 0: band_imp = band_imp / (band_imp.max() + 1e-12)
                    band_node = {bn: (band_imp[i] * node_imp) for i, bn in enumerate(bnames)}
                    self.viz.band_node_heatmap(band_node, self.channel_names,
                                            title=f"Band × Node (soft attribution) — {task}",
                                            fname=f"{out_prefix}_band_node.png")
                except Exception as e:
                    print(f"[viz] band×node heatmap skipped: {e}")

                # 2) Edge weight heatmap (graph used by GCN) + connectogram
                try:
                    if self.edge_weight is not None:
                        ew = self.edge_weight.detach().cpu().numpy()
                        n = self.num_nodes
                        A = np.zeros((n, n), dtype=float)
                        half = ew.size // 2
                        w = ew[:half]
                        r, c = np.tril_indices(n, k=-1)
                        A[r, c] = w; A[c, r] = w
                        self.viz.heatmap(A, self.channel_names, self.channel_names,
                                        title=f"Edge weights used by GCN — {task}",
                                        fname=f"{out_prefix}_edge_weights_heatmap.png")
                        self.viz.connectogram(A, self.channel_names,
                                            title=f"Connectogram — {task}",
                                            fname=f"{out_prefix}_connectogram.png",
                                            top_k=50)
                except Exception as e:
                    print(f"[viz] edge weight visuals skipped: {e}")

                # 3) Task-specific plots + PR/Calibration/ROC-CI/Bland–Altman
                try:
                    if detection:
                        ys, ps = [], []
                        with torch.no_grad():
                            for b in val_loader:
                                b = b.to(self.device)
                                ew = getattr(b, "edge_weight", None)
                                logit = self.model(b.x, b.edge_index, b, task="detection", edge_weight=ew)
                                ys.extend(b.y.detach().cpu().numpy().tolist())
                                ps.extend(torch.sigmoid(logit).detach().cpu().numpy().tolist())
                        ys = np.asarray(ys, int); ps = np.asarray(ps, float)
                        self.viz.roc_binary(ys, ps, title=f"ROC — detection", fname=f"{out_prefix}_roc.png")
                        self.viz.pr_curve(ys, ps, title=f"PR — detection", fname=f"{out_prefix}_pr.png")
                        self.viz.reliability_curve(ys, ps, title=f"Calibration — detection", fname=f"{out_prefix}_calibration.png")
                        self.viz.roc_with_ci(ys, ps, title=f"ROC (95% CI) — detection", fname=f"{out_prefix}_roc_ci.png")
                        self.viz.confusion(ys, (ps > 0.5).astype(int),
                                        class_names=["non-seizure", "seizure"],
                                        title=f"Confusion — detection", fname=f"{out_prefix}_cm.png")
                        self.viz.confusion(train_labels, (train_preds > 0.5).astype(int),
                                            class_names=["non-seizure", "seizure"],
                                            title=f"Confusion (Train) — detection", fname=f"{out_prefix}_cm_train.png")

                    elif classification:
                        #  ys, logits_all = [], []
                        #  with torch.no_grad():
                        #      for b in val_loader:
                        #          b = b.to(self.device)
                        #          ew = getattr(b, "edge_weight", None)
                        #          logits = self.model(b.x, b.edge_index, b, task="forecast_label", edge_weight=ew)
                        #          ys.extend(b.y.detach().cpu().numpy().tolist())
                        #          logits_all.append(logits.detach().cpu().numpy())
                        #  ys = np.asarray(ys, int)
                        #  # === APPLY LOGIT DE-BIAS FOR PLOTTING ===
                        #  logits_all = np.concatenate(logits_all, axis=0)
                        #  logits_t = torch.tensor(logits_all, device=self.device)
                        #  if getattr(self, "_log_prior", None) is not None:
                        #      logits_t = logits_t - self._tau * self._log_prior
                        #  probs = torch.softmax(logits_t, dim=1).detach().cpu().numpy()
                        #  yhat = probs.argmax(axis=1)
                        #  class_names = [f"c{i}" for i in range(self.model.label_head[-1].out_features)]
                        #  self.viz.confusion(ys, yhat,
                        #                     class_names=class_names,
                        #                     title=f"Confusion — early label forecast",
                        #                     fname=f"{out_prefix}_cm.png")
                        #  # Optional per-class diagnostics
                        #  C = probs.shape[1]
                        #  for c in range(C):
                        #      y_bin = (ys == c).astype(int)
                        #      p_bin = probs[:, c]
                        #      self.viz.pr_curve(y_bin, p_bin, title=f"PR — early_clf class {c}", fname=f"{out_prefix}_pr_c{c}.png")
                        #      self.viz.reliability_curve(y_bin, p_bin, title=f"Calibration — early_clf class {c}", fname=f"{out_prefix}_calib_c{c}.png")
                        #      self.viz.roc_with_ci(y_bin, p_bin, title=f"ROC (95% CI) — early_clf class {c}", fname=f"{out_prefix}_roc_ci_c{c}.png")
                

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

                        # Ensure all classes are represented, even those with zero samples
                        all_classes = np.array(self.real_class_names)
                        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
                        ys_mapped = np.array([class_to_idx[all_classes[y]] for y in ys])
                        yhat_mapped = np.array([class_to_idx[all_classes[np.argmax(p)]] for p in probs])

                        # Plot confusion matrix with raw counts
                        self.viz.confusion(
                            ys_mapped,  # y_true
                            yhat_mapped,  # y_pred
                            class_names=all_classes,
                            title=f"Confusion — classification (Raw Counts)",
                            fname=f"{out_prefix}_cm.png",
                            normalize=False
                        )

                        # Map for train (reuse all_classes, class_to_idx from val)
                        print("tainingCMMMMMMMMMMMMMM")
                        self.model.eval()
                        train_labels, logits_all = [], []
                        with torch.no_grad():
                            for batch in train_loader:
                                batch = batch.to(self.device)
                                ew = getattr(batch, "edge_weight", None)
                                out = self.model(batch.x, batch.edge_index, batch, task="classification", edge_weight=ew)
                                logits_all.append(out.detach().cpu().numpy())
                                train_labels.extend(batch.y.detach().cpu().numpy())

                        logits_all = np.concatenate(logits_all, axis=0)
                        train_labels_mapped = np.array([class_to_idx[all_classes[int(y)]] for y in train_labels])
                        train_probs = torch.softmax(torch.tensor(logits_all), dim=1).numpy()
                        train_yhat_mapped = np.array([class_to_idx[all_classes[np.argmax(p)]] for p in train_probs])
                        self.viz.confusion(
                            train_labels_mapped,
                            train_yhat_mapped,
                            class_names=all_classes,
                            title=f"Confusion — classification (Testing, Raw Counts)",
                            fname=f"{out_prefix}_cm_train.png",
                            normalize=False
                        )
                        # train_labels_mapped = np.array([class_to_idx[all_classes[int(y)]] for y in train_labels])
                        # train_yhat_mapped = np.array([class_to_idx[all_classes[np.argmax(p)]] for p in torch.softmax(torch.tensor(np.array(train_preds)), dim=1).numpy()])  # Adjust if no probs
                        # self.viz.confusion(train_labels_mapped, train_yhat_mapped,
                        #                 class_names=all_classes,
                        #                 title=f"Confusion — classification (Raw Counts)",
                        #                 fname=f"{out_prefix}_cm_train.png", normalize=False)

                        # Add precision, recall, F1 metrics
                        precision, recall, f1, _ = precision_recall_fscore_support(ys_mapped, yhat_mapped, labels=range(len(all_classes)), average='weighted')
                        print(f"Post-Training Metrics — Classification: Precision: {precision:.4f}, "
                            f"Recall: {recall:.4f}, F1: {f1:.4f}")

                        # One-vs-rest PR/Cal/ROC-CI for each class
                        for c in range(len(all_classes)):
                            y_bin = (ys_mapped == c).astype(int)
                            p_bin = probs[:, c]
                            self.viz.pr_curve(y_bin, p_bin, title=f"PR — class {all_classes[c]}", fname=f"{out_prefix}_pr_c{c}.png")
                            self.viz.reliability_curve(y_bin, p_bin, title=f"Calibration — class {all_classes[c]}", fname=f"{out_prefix}_calib_c{c}.png")
                            self.viz.roc_with_ci(y_bin, p_bin, title=f"ROC (95% CI) — class {all_classes[c]}", fname=f"{out_prefix}_roc_ci_c{c}.png")

                    elif early_reg:
                        y_true, y_pred = [], []
                        with torch.no_grad():
                            for b in val_loader:
                                b = b.to(self.device)
                                ew = getattr(b, "edge_weight", None)
                                pred_sc = self.model(b.x, b.edge_index, b, task="forecast_time", edge_weight=ew).cpu().numpy()
                                y_pred.extend(np.expm1(self.regression_scaler.inverse_transform(pred_sc.reshape(-1, 1))).flatten())
                                y_true.extend(b.y.detach().cpu().numpy().tolist())
                        y_true = np.array(y_true, dtype=float)
                        y_true = np.expm1(self.regression_scaler.inverse_transform(y_true.reshape(-1,1))).flatten()

                        plt.figure(figsize=(5,5))
                        plt.scatter(y_true, y_pred, s=10, alpha=0.6)
                        plt.xlabel("True TTI (s)"); plt.ylabel("Pred TTI (s)")
                        plt.title("Early regression — predictions vs truth")
                        self.viz._savefig(f"{out_prefix}_reg_scatter.png")

                        plt.figure(figsize=(6,4))
                        plt.hist(np.array(y_pred) - np.array(y_true), bins=40)
                        plt.title("Early regression — residuals (pred - true) [s]")
                        plt.xlabel("Residual [s]"); plt.ylabel("Count")
                        self.viz._savefig(f"{out_prefix}_reg_residuals.png")

                        # Bland–Altman (agreement)
                        self.viz.bland_altman(y_true, y_pred,
                                            title="Bland–Altman — early_reg",
                                            fname=f"{out_prefix}_bland_altman.png")
                        
                        #training
                        y_true, y_pred = [], []
                        with torch.no_grad():
                            for b in train_loader:
                                b = b.to(self.device)
                                ew = getattr(b, "edge_weight", None)
                                pred_sc = self.model(b.x, b.edge_index, b, task="forecast_time", edge_weight=ew).cpu().numpy()
                                y_pred.extend(np.expm1(self.regression_scaler.inverse_transform(pred_sc.reshape(-1, 1))).flatten())
                                y_true.extend(b.y.detach().cpu().numpy().tolist())
                        y_true = np.array(y_true, dtype=float)
                        y_true = np.expm1(self.regression_scaler.inverse_transform(y_true.reshape(-1,1))).flatten()

                        plt.figure(figsize=(5,5))
                        plt.scatter(y_true, y_pred, s=10, alpha=0.6)
                        plt.xlabel("True TTI (s)"); plt.ylabel("Pred TTI (s)")
                        plt.title("Time To Ictal — predictions vs truth")
                        self.viz._savefig(f"{out_prefix}_train_reg_scatter.png")

                        plt.figure(figsize=(6,4))
                        plt.hist(np.array(y_pred) - np.array(y_true), bins=40)
                        plt.title("Train regression — residuals (pred - true) [s]")
                        plt.xlabel("Residual [s]"); plt.ylabel("Count")
                        self.viz._savefig(f"{out_prefix}_train_reg_residuals.png")

                        # Bland–Altman (agreement)
                        self.viz.bland_altman(y_true, y_pred,
                                            title="Bland–Altman — train_reg",
                                            fname=f"{out_prefix}_train_bland_altman.png")

                    else:  # early_clf
                        # ys, logits_all = [], []
                        # with torch.no_grad():
                        #     for b in val_loader:
                        #         b = b.to(self.device)
                        #         ew = getattr(b, "edge_weight", None)
                        #         logits = self.model(b.x, b.edge_index, b, task="classification", edge_weight=ew)
                        #         ys.extend(b.y.detach().cpu().numpy().tolist())
                        #         logits_all.append(logits.detach().cpu().numpy())
                        # ys = np.asarray(ys, int)
                        # logits_all = np.concatenate(logits_all, axis=0)
                        # probs = torch.softmax(torch.tensor(logits_all), dim=1).numpy()
                        # yhat = probs.argmax(axis=1)

                        # # Ensure all classes are represented, even those with zero samples
                        # all_classes = np.array(self.real_class_names)
                        # class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
                        # ys_mapped = np.array([class_to_idx[all_classes[y]] for y in ys])
                        # yhat_mapped = np.array([class_to_idx[all_classes[np.argmax(p)]] for p in probs])

                        # # Plot confusion matrix with raw counts
                        # self.viz.confusion(
                        #     ys_mapped,  # y_true
                        #     yhat_mapped,  # y_pred
                        #     class_names=all_classes,
                        #     title=f"Confusion — classification (Raw Counts)",
                        #     fname=f"{out_prefix}_cm.png",
                        #     normalize=False
                        # )

                        # # Add precision, recall, F1 metrics
                        # precision, recall, f1, _ = precision_recall_fscore_support(ys_mapped, yhat_mapped, labels=range(len(all_classes)), average='weighted')
                        # print(f"Post-Training Metrics — Classification: Precision: {precision:.4f}, "
                        #     f"Recall: {recall:.4f}, F1: {f1:.4f}")

                        # # One-vs-rest PR/Cal/ROC-CI for each class
                        # for c in range(len(all_classes)):
                        #     y_bin = (ys_mapped == c).astype(int)
                        #     p_bin = probs[:, c]
                        #     self.viz.pr_curve(y_bin, p_bin, title=f"PR — class {all_classes[c]}", fname=f"{out_prefix}_pr_c{c}.png")
                        #     self.viz.reliability_curve(y_bin, p_bin, title=f"Calibration — class {all_classes[c]}", fname=f"{out_prefix}_calib_c{c}.png")
                        #     self.viz.roc_with_ci(y_bin, p_bin, title=f"ROC (95% CI) — class {all_classes[c]}", fname=f"{out_prefix}_roc_ci_c{c}.png")

                        #  ys, logits_all = [], []
                        #  with torch.no_grad():
                        #      for b in val_loader:
                        #          b = b.to(self.device)
                        #          ew = getattr(b, "edge_weight", None)
                        #          logits = self.model(b.x, b.edge_index, b, task="forecast_label", edge_weight=ew)
                        #          ys.extend(b.y.detach().cpu().numpy().tolist())
                        #          logits_all.append(logits.detach().cpu().numpy())
                        #  ys = np.asarray(ys, int)
                        #  # === APPLY LOGIT DE-BIAS FOR PLOTTING ===
                        #  logits_all = np.concatenate(logits_all, axis=0)
                        #  logits_t = torch.tensor(logits_all, device=self.device)
                        #  if getattr(self, "_log_prior", None) is not None:
                        #      logits_t = logits_t - self._tau * self._log_prior
                        #  probs = torch.softmax(logits_t, dim=1).detach().cpu().numpy()
                        #  yhat = probs.argmax(axis=1)
                        #  class_names = [f"c{i}" for i in range(self.model.label_head[-1].out_features)]
                        #  self.viz.confusion(ys, yhat,
                        #                     class_names=class_names,
                        #                     title=f"Confusion — early label forecast",
                        #                     fname=f"{out_prefix}_cm.png")
                        
                        #  self.viz.confusion(train_labels, train_preds.argmax(axis=1) if train_preds.ndim > 1 else train_preds,
                        #                     class_names=class_names,
                        #                     title=f"Confusion  — early label forecast",
                        #                     fname=f"{out_prefix}_cm_train.png")
                        #  # Optional per-class diagnostics
                        #  C = probs.shape[1]
                        #  for c in range(C):
                        #      y_bin = (ys == c).astype(int)
                        #      p_bin = probs[:, c]
                        #      self.viz.pr_curve(y_bin, p_bin, title=f"PR — early_clf class {c}", fname=f"{out_prefix}_pr_c{c}.png")
                        #      self.viz.reliability_curve(y_bin, p_bin, title=f"Calibration — early_clf class {c}", fname=f"{out_prefix}_calib_c{c}.png")
                        #      self.viz.roc_with_ci(y_bin, p_bin, title=f"ROC (95% CI) — early_clf class {c}", fname=f"{out_prefix}_roc_ci_c{c}.png")
                
                        ys, logits_all = [], []
                    with torch.no_grad():
                        for b in val_loader:
                            b = b.to(self.device)
                            ew = getattr(b, "edge_weight", None)
                            logits = self.model(b.x, b.edge_index, b, task="forecast_label", edge_weight=ew)
                            ys.extend(b.y.detach().cpu().numpy().tolist())
                            logits_all.append(logits.detach().cpu().numpy())
                    ys = np.asarray(ys, int)
                    # === APPLY LOGIT DE-BIAS FOR PLOTTING ===
                    logits_all = np.concatenate(logits_all, axis=0)
                    logits_t = torch.tensor(logits_all, device=self.device)
                    if getattr(self, "_log_prior", None) is not None:
                        logits_t = logits_t - self._tau * self._log_prior
                    probs = torch.softmax(logits_t, dim=1).detach().cpu().numpy()
                    yhat = probs.argmax(axis=1)

                    # Ensure all classes are represented, even those with zero samples
                    all_classes = np.array(self.real_class_names)
                    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
                    ys_mapped = np.array([class_to_idx[all_classes[y]] for y in ys])
                    yhat_mapped = np.array([class_to_idx[all_classes[np.argmax(p)]] for p in probs])

                    # Plot confusion matrix with raw counts for validation
                    self.viz.confusion(
                        ys_mapped,  # y_true
                        yhat_mapped,  # y_pred
                        class_names=all_classes,
                        title=f"Confusion — early label forecast (Raw Counts)",
                        fname=f"{out_prefix}_cm.png",
                        normalize=False
                    )

                    # Compute training confusion matrix
                    print("Training Confusion Matrix for Early Classification")
                    self.model.eval()
                    train_labels, logits_all = [], []
                    with torch.no_grad():
                        for batch in train_loader:
                            batch = batch.to(self.device)
                            ew = getattr(batch, "edge_weight", None)
                            out = self.model(batch.x, batch.edge_index, batch, task="forecast_label", edge_weight=ew)
                            logits_all.append(out.detach().cpu().numpy())
                            train_labels.extend(batch.y.detach().cpu().numpy())

                    logits_all = np.concatenate(logits_all, axis=0)
                    train_logits_t = torch.tensor(logits_all, device=self.device)
                    if getattr(self, "_log_prior", None) is not None:
                        train_logits_t = train_logits_t - self._tau * self._log_prior
                    train_probs = torch.softmax(train_logits_t, dim=1).detach().cpu().numpy()
                    train_labels_mapped = np.array([class_to_idx[all_classes[int(y)]] for y in train_labels])
                    train_yhat_mapped = np.array([class_to_idx[all_classes[np.argmax(p)]] for p in train_probs])

                    # Plot confusion matrix with raw counts for training
                    self.viz.confusion(
                        train_labels_mapped,
                        train_yhat_mapped,
                        class_names=all_classes,
                        title=f"Confusion — early label forecast (Training, Raw Counts)",
                        fname=f"{out_prefix}_cm_train.png",
                        normalize=False
                    )
                    C = len(all_classes)
                    for c in range(C):
                        y_bin = (ys_mapped == c).astype(int)
                        p_bin = probs[:, c]
                        self.viz.pr_curve(y_bin, p_bin, title=f"PR — early_clf class {all_classes[c]}", fname=f"{out_prefix}_pr_c{c}.png")
                        self.viz.reliability_curve(y_bin, p_bin, title=f"Calibration — early_clf class {all_classes[c]}", fname=f"{out_prefix}_calib_c{c}.png")
                        self.viz.roc_with_ci(y_bin, p_bin, title=f"ROC (95% CI) — early_clf class {all_classes[c]}", fname=f"{out_prefix}_roc_ci_c{c}.png")
                except Exception as e:
                    print(f"[viz] task visuals skipped: {e}")

                # 4) Learning curve (epoch traces)
                try:
                    metric_name = "R²" if early_reg else "Accuracy"
                    self.viz.learning_curve(epoch_idx, train_trace, val_trace,
                                            title=f"Learning Curve ({metric_name}) — {task}",
                                            fname=f"{out_prefix}_learning.png")
                except Exception as e:
                    print(f"[viz] learning-curve skipped: {e}")

            except Exception as e:
                print(f"[viz] visuals skipped (outer): {e}")

            print("Training complete!")
            return all_preds


# =======================
# Multi-task radar stitchers (robust CSV parsing)
# =======================
def _load_csv_rows(csv_path):
    """Returns (header:list[str] | None, rows:list[list[str]]). Skips empty lines."""
    import csv
    rows = []
    header = None
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if not row or all(c.strip() == "" for c in row):
                continue
            if i == 0:
                header = row
            else:
                rows.append(row)
    return header, rows

def _collect_panels_for_kind(out_dir: str, tasks: list, kind: str):
    """
    kind: 'features' -> reads '{task}_feature_groups.csv' with columns [feature_group, ..., importance_drop]
          'nodes'    -> reads '{task}_node_importance.csv'  with columns [channel, ..., importance_drop]
    Returns: (labels, panels) where panels = [(title, values_np), ...]
    """
    import os, numpy as np

    file_for = {
        "features": lambda t: os.path.join(out_dir, f"{t}_feature_groups.csv"),
        "nodes":    lambda t: os.path.join(out_dir, f"{t}_node_importance.csv"),
    }[kind]

    title_for = {
        "detection": "Detection",
        "classification": "Classification",
        "early_reg": "Time Forecasting",
        "early_clf": "Type Forecasting",
    }

    # 1) Seed label order from the first existing file
    global_labels, label_index, first_found = [], {}, None
    for t in tasks:
        fp = file_for(t)
        if os.path.exists(fp):
            _, rows = _load_csv_rows(fp)
            if rows:
                names = [row[0] for row in rows]  # first col is the name
                global_labels = names[:]
                label_index = {n: i for i, n in enumerate(global_labels)}
                first_found = t
                break
    if first_found is None:
        return [], []

    # 2) Extend labels with unseen names from other tasks
    for t in tasks:
        fp = file_for(t)
        if not os.path.exists(fp):
            continue
        _, rows = _load_csv_rows(fp)
        for row in rows:
            name = row[0]
            if name not in label_index:
                label_index[name] = len(global_labels)
                global_labels.append(name)

    # 3) Build aligned panels
    panels = []
    for t in tasks:
        fp = file_for(t)
        if not os.path.exists(fp):
            continue
        _, rows = _load_csv_rows(fp)
        vec = np.zeros(len(global_labels), dtype=float)
        for row in rows:
            try:
                name = row[0]
                val = row[-1]  # last column = importance value
                vec[label_index[name]] = float(val)
            except Exception:
                pass
        panels.append((title_for.get(t, t), vec))
    return global_labels, panels

def make_overview_radars(out_dir: str,
                         tasks: list = ("detection", "classification", "early_reg", "early_clf")):
    """
    Creates 2×2 overview radar charts from per-task CSVs saved by Trainer.train().
    Outputs in out_dir:
      - radar_features_2x2.png
      - radar_nodes_2x2.png
    """
    viz = Viz(out_dir)

    # Feature-group overview
    feat_labels, feat_panels = _collect_panels_for_kind(out_dir, list(tasks), kind="features")
    if feat_panels:
        viz.radar_grid(
            panels=feat_panels,
            labels=feat_labels,
            fname="radar_features_2x2.png",
            suptitle="Feature Importance Across Tasks"
        )
    else:
        print("[stitch] No feature-group CSVs found; skipping radar_features_2x2.png")

    # Node/channel overview
    node_labels, node_panels = _collect_panels_for_kind(out_dir, list(tasks), kind="nodes")
    if node_panels:
        viz.radar_grid(
            panels=node_panels,
            labels=node_labels,
            fname="radar_nodes_2x2.png",
            suptitle="Node Importance Across Tasks (EEG Channels)"
        )
    else:
        print("[stitch] No node-importance CSVs found; skipping radar_nodes_2x2.png")


# =========================================================
# K-FOLD CROSS-VALIDATION WRAPPER (non-invasive enhancement)
# =========================================================

def _build_graph_features_for_task(X: np.ndarray, task: str) -> np.ndarray:
    """
    Rebuild features in the same way Trainer.train() does, so we can
    evaluate folds with the scaler learned in that fold.
    """
    fb = FeatureBuilder(
        fs=200, rfft_bins=100,
        with_time=False,
        with_shapes=True,
        with_complexity=True,
        with_connectivity=True
    )
    X_built = fb.build(X, mode=task)
    X_feat = X_built[0] if isinstance(X_built, tuple) else X_built
    if X_feat.ndim == 4:
        return X_feat.mean(axis=1)   # (N, V, Fout)
    elif X_feat.ndim == 3:
        return X_feat
    raise ValueError(f"Unexpected feature shape from FeatureBuilder: {X_feat.shape}")


def run_stratified_kfold(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
    seed: int = 42,
    task_flags: dict | None = None,
    trainer_kwargs: dict | None = None,
):
    """
    Run StratifiedKFold CV using Trainer without altering its internals.
    We temporarily patch Trainer._stratified_split_strict to feed the
    train/val indices of each fold.
    """
    if task_flags is None:
        task_flags = dict(classification=True)
    if trainer_kwargs is None:
        raise ValueError("trainer_kwargs must be provided (same args you use to build Trainer).")

    # Determine task string
    if task_flags.get("detection"):
        task = "detection"
    elif task_flags.get("classification"):
        task = "classification"
    elif task_flags.get("early_reg"):
        task = "early_reg"
    else:
        task = "early_clf"

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(skf.split(X, Y))

    # Keep original splitter to restore after each fold
    original_splitter = Trainer._stratified_split_strict

    metrics = []
    print(f"\n===== Stratified {k}-Fold CV — task: {task} =====")

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold}/{k} (train={len(tr_idx)}, val={len(va_idx)}) ---")

        # Patch split so Trainer uses our fold indices
        def _patched_split(X_in, Y_in, test_size=0.2, random_state=42):
            return X_in[tr_idx], X_in[va_idx], Y_in[tr_idx], Y_in[va_idx]
        Trainer._stratified_split_strict = staticmethod(_patched_split)

        # Fresh trainer per fold
        tr = Trainer(**trainer_kwargs)
        _ = tr.train(
            X=X, Y=Y,
            detection=task_flags.get("detection", False),
            classification=task_flags.get("classification", False),
            early_reg=task_flags.get("early_reg", False),
            early_clf=task_flags.get("early_clf", False),
            explain_after=False
        )

        # Evaluate on this fold's validation split
        with torch.no_grad():
            tr.model.eval()
            X_graph = _build_graph_features_for_task(X, task)
            X_val = X_graph[va_idx]
            X_val_rs = X_val.reshape(-1, X_val.shape[-1])
            X_val_sc = tr.feature_scaler.transform(X_val_rs).reshape(X_val.shape)

            Y_val = Y[va_idx]
            val_loader = tr.create_graph_batches(X_val_sc, Y_val, task=task, shuffle=False)

            y_pred, y_true = [], []
            for batch in val_loader:
                batch = batch.to(tr.device)
                ew = getattr(batch, "edge_weight", None)
                if task == "detection":
                    out = tr.model(batch.x, batch.edge_index, batch, task="detection", edge_weight=ew).squeeze(-1)
                    p = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
                elif task == "classification":
                    out = tr.model(batch.x, batch.edge_index, batch, task="classification", edge_weight=ew)
                    p = out.argmax(dim=1).cpu().numpy()
                elif task == "early_reg":
                    out_sc = tr.model(batch.x, batch.edge_index, batch, task="forecast_time", edge_weight=ew).cpu().numpy()
                    p = out_sc.squeeze()
                else:  # early_clf
                    out = tr.model(batch.x, batch.edge_index, batch, task="forecast_label", edge_weight=ew)
                    p = out.argmax(dim=1).cpu().numpy()
                y_pred.extend(p.tolist())
                y_true.extend(batch.y.detach().cpu().numpy().tolist())

        if task == "early_reg":
            print("Fold finished (regression). Add R²/RMSE here if desired.")
            metrics.append({})
        else:
            acc = accuracy_score(y_true, y_pred)
            f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
            print(f"Fold {fold} — Acc: {acc:.4f} | Macro-F1: {f1m:.4f}")
            metrics.append(dict(acc=acc, f1=f1m))

        # Restore original splitter
        Trainer._stratified_split_strict = original_splitter

    # Summary
    if task != "early_reg":
        accs = np.array([m["acc"] for m in metrics], float)
        f1s  = np.array([m["f1"]  for m in metrics], float)
        print("\n===== CV Summary =====")
        print(f"Accuracy:  mean {accs.mean():.4f} ± {accs.std(ddof=1):.4f}")
        print(f"Macro-F1:  mean {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}")
    else:
        print("\nCV finished for regression task.")
