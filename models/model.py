"""
GSNet MultiTaskGCN — Path B (true weight-shared unified model).

This is a drop-in replacement for models/model.py. It is 100% backward
compatible with the existing Trainer (same __init__ args, same forward
signature and behaviour), and ADDS the machinery needed for Path B:

  * a single, explicitly-defined GCN "backbone" (conv1..conv3 + graph_norm)
    that can be saved, loaded, and frozen independently of the heads;
  * an explicitly-defined temporal encoder (GRU) that can likewise be
    saved / loaded / frozen and SHARED across the two forecasting heads;
  * `encode_windows()` so inference can run the backbone ONCE and reuse the
    pooled embedding for every head (this is what makes the "unified"
    efficiency claim true);
  * `run_detection / run_classification / run_forecast` head calls that
    operate on a precomputed window embedding.

Backbone  = {conv1, bn1, conv2, bn2, conv3, bn3, graph_norm}
Temporal  = {temporal_encoder}   (GRU, shared by both forecasting heads)
Heads     = {detect_head, class_head, time_head, label_head}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MultiTaskGCN(nn.Module):
    # Submodule names that constitute the shared spatial backbone.
    BACKBONE_MODULES = ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "graph_norm")
    # Submodule name of the shared temporal encoder (forecasting).
    TEMPORAL_MODULES = ("temporal_encoder",)

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        num_classes: int,
        forecast_classes: int = None,
        dropout: float = 0.3,
        seq_len: int = 10,
        use_uncertainty: bool = False,
    ):
        super().__init__()
        if forecast_classes is None:
            forecast_classes = num_classes
        self.dropout = float(dropout)
        self.seq_len = seq_len
        self.use_uncertainty = use_uncertainty

        # flags controlling frozen behaviour (see train() override below)
        self._backbone_frozen = False
        self._temporal_frozen = False

        # --- GCN backbone (shared spatial encoder) ---
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.graph_norm = nn.LayerNorm(hidden_dim)

        # --- Temporal encoder for forecasting tasks (shared by time & type) ---
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # --- Task heads ---
        def mlp_head(in_size, widths, out_size, use_norm=True):
            layers = []
            prev = in_size
            for w in widths:
                layers += [nn.Linear(prev, w)]
                if use_norm:
                    layers += [nn.LayerNorm(w)]
                layers += [nn.ReLU(inplace=True), nn.Dropout(self.dropout)]
                prev = w
            layers += [nn.Linear(prev, out_size)]
            return nn.Sequential(*layers)

        self.detect_head = mlp_head(
            hidden_dim, [hidden_dim, hidden_dim // 2, hidden_dim // 4], out_size=1
        )
        self.class_head = mlp_head(
            hidden_dim, [hidden_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4], out_size=num_classes
        )
        self.time_head = mlp_head(
            hidden_dim, [hidden_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4], out_size=1
        )
        self.label_head = mlp_head(
            hidden_dim, [hidden_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4], out_size=forecast_classes
        )

        if self.use_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(4))

    # ------------------------------------------------------------------ #
    #  Shared backbone: encode                                           #
    # ------------------------------------------------------------------ #
    def _backbone(self, x, edge_index, edge_weight=None):
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = self.bn1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.conv2(x1, edge_index, edge_weight=edge_weight)
        x2 = self.bn2(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index, edge_weight=edge_weight)
        x3 = self.bn3(x3)
        x3 = F.relu(x3 + 0.1 * x2, inplace=True)          # scaled residual (beta=0.1)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        return x3

    def _get_batch_index(self, batch):
        if hasattr(batch, "batch"):
            return batch.batch
        return batch

    def encode_windows(self, x, edge_index, batch, edge_weight=None):
        """Run the shared backbone ONCE and return the pooled, normalised
        per-window graph embedding of shape (num_windows, hidden_dim)."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        node_feat = self._backbone(x, edge_index, edge_weight=edge_weight)
        window_feat = global_mean_pool(node_feat, self._get_batch_index(batch))
        window_feat = self.graph_norm(window_feat)
        return window_feat

    # ------------------------------------------------------------------ #
    #  Heads operating on a precomputed window embedding                 #
    # ------------------------------------------------------------------ #
    def _sequence_from_windows(self, window_feat):
        """Reshape (num_windows, H) -> (B, L, H), truncating to a multiple of L."""
        num_windows = window_feat.size(0)
        if num_windows % self.seq_len != 0:
            n_keep = (num_windows // self.seq_len) * self.seq_len
            if n_keep == 0:
                raise ValueError(
                    f"Too few windows ({num_windows}) for a sequence of length {self.seq_len}")
            window_feat = window_feat[:n_keep]
            num_windows = n_keep
        batch_size = num_windows // self.seq_len
        return window_feat.view(batch_size, self.seq_len, -1)

    def run_detection(self, window_feat):
        return self.detect_head(window_feat).squeeze(-1)

    def run_classification(self, window_feat):
        return self.class_head(window_feat)

    def _forecast_feat(self, window_feat):
        seq_feat = self._sequence_from_windows(window_feat)     # (B, L, H)
        _, hidden = self.temporal_encoder(seq_feat)
        return hidden[-1]                                       # (B, H)

    def run_forecast_time(self, window_feat):
        return self.time_head(self._forecast_feat(window_feat)).squeeze(-1)

    def run_forecast_label(self, window_feat):
        return self.label_head(self._forecast_feat(window_feat))

    # ------------------------------------------------------------------ #
    #  Original forward (unchanged behaviour) — keeps Trainer working    #
    # ------------------------------------------------------------------ #
    def forward(self, x, edge_index, batch, edge_weight=None, task: str = None):
        window_feat = self.encode_windows(x, edge_index, batch, edge_weight)

        if task in ("detection", "classification"):
            last_feat = window_feat
            forecast_feat = None
        else:
            seq_feat = self._sequence_from_windows(window_feat)
            last_feat = seq_feat[:, -1, :]
            _, hidden = self.temporal_encoder(seq_feat)
            forecast_feat = hidden[-1]

        if task is not None:
            if task == "detection":
                return self.detect_head(last_feat).squeeze(-1)
            elif task == "classification":
                return self.class_head(last_feat)
            elif task == "forecast_time":
                return self.time_head(forecast_feat).squeeze(-1)
            elif task == "forecast_label":
                return self.label_head(forecast_feat)
            else:
                raise ValueError(f"Unknown task: {task}")

        outputs = {
            "detection": self.detect_head(last_feat).squeeze(-1) if last_feat is not None else None,
            "classification": self.class_head(last_feat) if last_feat is not None else None,
            "forecast_time": self.time_head(forecast_feat).squeeze(-1) if forecast_feat is not None else None,
            "forecast_label": self.label_head(forecast_feat) if forecast_feat is not None else None,
        }
        if self.use_uncertainty:
            outputs["log_vars"] = self.log_vars
        return outputs

    # ------------------------------------------------------------------ #
    #  Parameter groups                                                  #
    # ------------------------------------------------------------------ #
    def _modules_by_names(self, names):
        return [getattr(self, n) for n in names if hasattr(self, n)]

    def backbone_parameters(self):
        for m in self._modules_by_names(self.BACKBONE_MODULES):
            yield from m.parameters()

    def temporal_parameters(self):
        for m in self._modules_by_names(self.TEMPORAL_MODULES):
            yield from m.parameters()

    def trainable_parameters(self):
        """Only params with requires_grad=True (i.e. non-frozen)."""
        return (p for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ #
    #  Freeze / unfreeze                                                 #
    # ------------------------------------------------------------------ #
    def freeze_backbone(self):
        for p in self.backbone_parameters():
            p.requires_grad = False
        self._backbone_frozen = True
        for m in self._modules_by_names(self.BACKBONE_MODULES):
            m.eval()                       # keep BatchNorm running stats fixed
        return self

    def unfreeze_backbone(self):
        for p in self.backbone_parameters():
            p.requires_grad = True
        self._backbone_frozen = False
        return self

    def freeze_temporal(self):
        for p in self.temporal_parameters():
            p.requires_grad = False
        self._temporal_frozen = True
        return self

    def unfreeze_temporal(self):
        for p in self.temporal_parameters():
            p.requires_grad = True
        self._temporal_frozen = False
        return self

    def train(self, mode: bool = True):
        """Ensure frozen submodules stay in eval() mode even when the
        Trainer calls model.train() at the start of each epoch."""
        super().train(mode)
        if self._backbone_frozen:
            for m in self._modules_by_names(self.BACKBONE_MODULES):
                m.eval()
        if self._temporal_frozen:
            for m in self._modules_by_names(self.TEMPORAL_MODULES):
                m.eval()
        return self

    # ------------------------------------------------------------------ #
    #  Save / load specific sub-networks                                 #
    # ------------------------------------------------------------------ #
    def _state_dict_for(self, names):
        prefixes = tuple(n + "." for n in names)
        return {k: v for k, v in self.state_dict().items() if k.startswith(prefixes)}

    def _load_state_dict_subset(self, sub, names, strict=True):
        prefixes = tuple(n + "." for n in names)
        own = self.state_dict()
        matched, missing = {}, []
        for k in own:
            if k.startswith(prefixes):
                if k in sub and sub[k].shape == own[k].shape:
                    matched[k] = sub[k]
                else:
                    missing.append(k)
        own.update(matched)
        self.load_state_dict(own)
        if strict and missing:
            raise RuntimeError(f"Missing/shape-mismatched params while loading {names}: {missing[:6]} ...")
        return len(matched), missing

    def save_backbone(self, path):
        torch.save({"backbone_state": self._state_dict_for(self.BACKBONE_MODULES)}, path)

    def load_backbone(self, path, strict=True):
        ckpt = torch.load(path, map_location="cpu")
        sub = ckpt.get("backbone_state", ckpt)
        n, missing = self._load_state_dict_subset(sub, self.BACKBONE_MODULES, strict=strict)
        return n

    def save_temporal(self, path):
        torch.save({"temporal_state": self._state_dict_for(self.TEMPORAL_MODULES)}, path)

    def load_temporal(self, path, strict=True):
        ckpt = torch.load(path, map_location="cpu")
        sub = ckpt.get("temporal_state", ckpt)
        n, missing = self._load_state_dict_subset(sub, self.TEMPORAL_MODULES, strict=strict)
        return n

    def load_head_from_full_ckpt(self, path, head_module_name, strict=False):
        """Pull a single head's params out of a full model checkpoint saved
        by the Trainer (which stores {'model_state_dict': ...})."""
        ckpt = torch.load(path, map_location="cpu")
        sub = ckpt.get("model_state_dict", ckpt)
        n, missing = self._load_state_dict_subset(sub, (head_module_name,), strict=strict)
        return n
