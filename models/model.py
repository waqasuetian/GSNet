
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
        dropout: float = 0.3,  # Increased to reduce overfitting
    ):
        super().__init__()
        if forecast_classes is None:
            forecast_classes = num_classes
        self.dropout = float(dropout)

        # --- GCN backbone ---
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Optional projector/norm after graph pooling
        self.graph_norm = nn.LayerNorm(hidden_dim)

        # --- Heads ---
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

        # Detection: binary logit
        self.detect_head = mlp_head(
            hidden_dim, [hidden_dim, hidden_dim // 2, hidden_dim // 4], out_size=1
        )

        # Classification: logits over classes
        self.class_head = mlp_head(
            hidden_dim, [hidden_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4], out_size=num_classes
        )

        # Time-to-event regression: scalar
        self.time_head = mlp_head(
            hidden_dim, [hidden_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4], out_size=1
        )

        # Forecast next label: logits over forecast_classes
        self.label_head = mlp_head(
            hidden_dim, [hidden_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4], out_size=forecast_classes
        )

    def _backbone(self, x, edge_index, edge_weight=None):
        # Block 1
        #print(f"[DEBUG] In _backbone: x shape {x.shape}, edge_index max {edge_index.max().item()}")
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = self.bn1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Block 2
        x2 = self.conv2(x1, edge_index, edge_weight=edge_weight)
        x2 = self.bn2(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Block 3 with scaled residual from x2
        x3 = self.conv3(x2, edge_index, edge_weight=edge_weight)
        x3 = self.bn3(x3)
        x3 = F.relu(x3 + 0.1 * x2, inplace=True)  # Scaled residual
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        return x3

    def _get_batch_index(self, batch):
        """
        Accept either:
          - torch_geometric.data.Batch object (has .batch), or
          - a LongTensor of size [num_nodes_total] with graph IDs.
        """
        if hasattr(batch, "batch"):
            return batch.batch
        return batch  # assume it's already a tensor

    def forward(self, x, edge_index, batch, task: str = None, edge_weight=None):
        """
        x:           [num_nodes_total, in_dim]
        edge_index:  [2, num_edges]
        batch:       Batch object or LongTensor[num_nodes_total]
        edge_weight: Optional FloatTensor[num_edges]
        """
        x = self._backbone(x, edge_index, edge_weight=edge_weight)
        batch_index = self._get_batch_index(batch)
        graph_feat = global_mean_pool(x, batch_index)  # [num_graphs, hidden_dim]
        graph_feat = self.graph_norm(graph_feat)

        if task == 'detection':
            return self.detect_head(graph_feat).squeeze(-1)   # [B]
        if task == 'classification':
            return self.class_head(graph_feat)                # [B, C]
        if task == 'forecast_time':
            return self.time_head(graph_feat).squeeze(-1)     # [B]
        if task == 'forecast_label':
            return self.label_head(graph_feat)                # [B, C_forecast]

        # If no task specified, return all heads
        return {
            'detection': self.detect_head(graph_feat).squeeze(-1),
            'classification': self.class_head(graph_feat),
            'forecast_time': self.time_head(graph_feat).squeeze(-1),
            'forecast_label': self.label_head(graph_feat),
        }
