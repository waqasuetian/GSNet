

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
        dropout: float = 0.5,
    ):
        super().__init__()
        if forecast_classes is None:
            forecast_classes = num_classes
        self.dropout = dropout

        # GCN backbone
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Detection head
        self.detect_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        # Forecast time (regression)
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        # Forecast label (classification)
        self.label_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_classes)
        )

    def forward(self, x, edge_index, batch, task: str = None):
        # Backbone
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        # Pool graph using the batch vector
        graph_feat = global_mean_pool(x, batch.batch)

        # Task-specific output
        if task == 'detection':
            return self.detect_head(graph_feat).view(-1)
        if task == 'classification':
            return self.class_head(graph_feat)
        if task == 'forecast_time':
            return self.time_head(graph_feat).view(-1)
        if task == 'forecast_label':
            return self.label_head(graph_feat)

        # If no task specified, return all
        return {
            'detection': self.detect_head(graph_feat).view(-1),
            'classification': self.class_head(graph_feat),
            'forecast_time': self.time_head(graph_feat).view(-1),
            'forecast_label': self.label_head(graph_feat)
        }
