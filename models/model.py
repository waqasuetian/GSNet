import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class MultiTaskGCN(nn.Module):
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

        # --- GCN backbone (shared spatial encoder) ---
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.graph_norm = nn.LayerNorm(hidden_dim)

        # --- Temporal encoder for forecasting tasks ---
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,                      # increased to 2 for dropout to work
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
        x3 = F.relu(x3 + 0.1 * x2, inplace=True)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        return x3

    def _get_batch_index(self, batch):
        if hasattr(batch, "batch"):
            return batch.batch
        return batch
   
    def forward(self, x, edge_index, batch, edge_weight=None, task: str = None):
        # Ensure x is at least 2D (nodes, features)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Shared GCN backbone
        node_feat = self._backbone(x, edge_index, edge_weight=edge_weight)
        window_feat = global_mean_pool(node_feat, self._get_batch_index(batch))
        window_feat = self.graph_norm(window_feat)

        num_windows = window_feat.size(0)
        
        # --------------------------------------------------------------
        # Single‑window tasks (detection / classification)
        # --------------------------------------------------------------
        if task in ('detection', 'classification'):
            last_feat = window_feat
            forecast_feat = None
        else:
            # ----------------------------------------------------------
            # Forecasting tasks (time / label) – require sequences
            # ----------------------------------------------------------
            if num_windows % self.seq_len != 0:
                n_keep = (num_windows // self.seq_len) * self.seq_len
                if n_keep == 0:
                    raise ValueError(f"Too few windows ({num_windows}) to form a sequence of length {self.seq_len}")
                window_feat = window_feat[:n_keep]
                num_windows = n_keep
                print(f"Warning: Truncated {num_windows - n_keep} windows to align with seq_len={self.seq_len}")

            batch_size = num_windows // self.seq_len
            seq_feat = window_feat.view(batch_size, self.seq_len, -1)   # (B, L, H)
            last_feat = seq_feat[:, -1, :]                             # last window
            
            # Use temporal_encoder (not gru_encoder)
            if hasattr(self, 'temporal_encoder'):
                _, hidden = self.temporal_encoder(seq_feat)
                forecast_feat = hidden[-1]      # Take last layer's hidden state -> (B, H)
            else:
                # Fallback if no temporal encoder (should not happen)
                forecast_feat = seq_feat.mean(dim=1)

        # --------------------------------------------------------------
        # Task dispatch
        # --------------------------------------------------------------
        if task is not None:
            if task == 'detection':
                return self.detect_head(last_feat).squeeze(-1)
            elif task == 'classification':
                return self.class_head(last_feat)
            elif task == 'forecast_time':
                return self.time_head(forecast_feat).squeeze(-1)
            elif task == 'forecast_label':
                return self.label_head(forecast_feat)
            else:
                raise ValueError(f"Unknown task: {task}")

        # Multi‑task mode – return all outputs
        outputs = {
            'detection': self.detect_head(last_feat).squeeze(-1) if last_feat is not None else None,
            'classification': self.class_head(last_feat) if last_feat is not None else None,
            'forecast_time': self.time_head(forecast_feat).squeeze(-1) if forecast_feat is not None else None,
            'forecast_label': self.label_head(forecast_feat) if forecast_feat is not None else None,
        }
        if self.use_uncertainty:
            outputs['log_vars'] = self.log_vars
        return outputs
