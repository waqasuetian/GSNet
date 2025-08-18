# #old one

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool

# class MultiTaskGCN(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int,
#         in_dim: int,
#         num_classes: int,
#         forecast_classes: int = None,
#         dropout: float = 0.5,
#     ):
#         super().__init__()
#         if forecast_classes is None:
#             forecast_classes = num_classes
#         self.dropout = dropout

#         # GCN backbone
#         self.conv1 = GCNConv(in_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)

#         # Detection head
#         self.detect_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#         )
#         # Classification head
#         self.class_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, num_classes)
#         )
#         # Forecast time (regression)
#         self.time_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )
#         # Forecast label (classification)
#         self.label_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, forecast_classes)
#         )

#     def forward(self, x, edge_index, batch, task: str = None):
#         # Backbone
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)

#         # Pool graph using the batch vector
#         graph_feat = global_mean_pool(x, batch.batch)

#         # Task-specific output
#         if task == 'detection':
#             return self.detect_head(graph_feat).view(-1)
#         if task == 'classification':
#             return self.class_head(graph_feat)
#         if task == 'forecast_time':
#             return self.time_head(graph_feat).view(-1)
#         if task == 'forecast_label':
#             return self.label_head(graph_feat)

#         # If no task specified, return all
#         return {



##new one


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, AttentionalAggregation

# class MultiTaskGCN(nn.Module):
#     """
#     A multi-task Graph Convolutional Network (GCN) model optimized for EEG data with temporal modeling.
    
#     Args:
#         in_dim (int): Input feature dimension (e.g., 100 for pooled EEG features).
#         hidden_dim (int): Dimension of hidden layers in GCN and MLP heads.
#         num_classes (int): Number of classes for classification task.
#         forecast_classes (int, optional): Number of classes for forecast_label task. Defaults to num_classes.
#         dropout (float, optional): Dropout rate for regularization. Defaults to 0.2.
#     """
#     def __init__(
#         self,
#         in_dim: int,
#         hidden_dim: int,
#         num_classes: int,
#         forecast_classes: int = None,
#         dropout: float = 0.2,
#     ):
#         super().__init__()
#         if forecast_classes is None:
#             forecast_classes = num_classes
#         self.dropout = dropout

#         # GCN backbone
#         self.conv1 = GCNConv(in_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.bn3 = nn.BatchNorm1d(hidden_dim)
#         # Attention-based pooling
#         self.attention_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1), nn=None)

#         # Detection head (binary classification)
#         self.detect_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.LayerNorm(hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.LayerNorm(hidden_dim // 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 4, 1)
#         )

#         # Classification head (multi-class classification)
#         self.class_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 4, num_classes)
#         )

#         # Time head (regression with LSTM)
#         self.time_head = nn.Sequential(
#             nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 4, 1)
#         )

#         # Label head (multi-class classification)
#         self.label_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 4, forecast_classes)
#         )

#         # Learnable log-variances for task weighting
#         self.s_detect = nn.Parameter(torch.zeros(1))
#         self.s_class = nn.Parameter(torch.zeros(1))
#         self.s_time = nn.Parameter(torch.zeros(1))
#         self.s_label = nn.Parameter(torch.zeros(1))

#     def forward(self, x, edge_index, batch=None, data=None, task: str = None):
#         """
#         Forward pass of the model.

#         Args:
#             x (torch.Tensor): Node feature matrix, shape [num_nodes, in_dim].
#             edge_index (torch.Tensor): Graph edge indices, shape [2, num_edges].
#             batch (torch.Tensor, optional): Batch vector, shape [num_nodes].
#             data (torch_geometric.data.Data, optional): Data object containing x, edge_index, batch.
#             task (str, optional): Task to compute ('detection', 'classification', 'forecast_time', 'forecast_label').

#         Returns:
#             tuple: (task_output, attn_weights) for specific task, or (dict of outputs, attn_weights) if task=None.
#         """
#         if data is not None and batch is None:
#             batch = getattr(data, 'batch', None)
#         if batch is None:
#             raise ValueError("Batch tensor must be provided via batch or data.batch")

#         # Backbone
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout, training=self.training)

#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout, training=self.training)

#         # Capture attention weights
#         gate = self.attention_pool.gate_nn(x)
#         attn_weights = torch.softmax(gate / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32, device=x.device)), dim=0)
#         graph_feat = self.attention_pool(x, batch)

#         if task == 'detection':
#             return self.detect_head(graph_feat).view(-1), attn_weights
#         elif task == 'classification':
#             return self.class_head(graph_feat), attn_weights
#         elif task == 'forecast_time':
#             graph_feat = graph_feat.unsqueeze(1)
#             lstm_out, _ = self.time_head[0](graph_feat)
#             lstm_out = lstm_out[:, -1, :]
#             out = self.time_head[1:](lstm_out)
#             return out.view(-1), attn_weights
#         elif task == 'forecast_label':
#             return self.label_head(graph_feat), attn_weights
#         else:
#             lstm_out, _ = self.time_head[0](graph_feat.unsqueeze(1))
#             lstm_out = lstm_out[:, -1, :]
#             time_out = self.time_head[1:](lstm_out)
#             return {
#                 'detection': self.detect_head(graph_feat).view(-1),
#                 'classification': self.class_head(graph_feat),
#                 'forecast_time': time_out.view(-1),
#                 'forecast_label': self.label_head(graph_feat)
#             }, attn_weights
# #             'detection': self.detect_head(graph_feat).view(-1),
# #             'classification': self.class_head(graph_feat),
# #             'forecast_time': self.time_head(graph_feat).view(-1),
# #             'forecast_label': self.label_head(graph_feat)
# #         }



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, AttentionalAggregation
from torch_scatter import scatter_softmax  # needed for better attention

class MultiTaskGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, forecast_classes=None, dropout=0.2):
        super().__init__()
        if forecast_classes is None:
            forecast_classes = num_classes
        self.dropout = dropout

        # GCN backbone
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Attention pooling
        self.attention_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))

        # Detection head
        self.detect_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )

        # Time head (LSTM outside Sequential)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True)
        self.time_regressor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Forecast label head
        self.label_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, forecast_classes)
        )

        # Learnable loss weights
        self.s_detect = nn.Parameter(torch.zeros(1))
        self.s_class = nn.Parameter(torch.zeros(1))
        self.s_time = nn.Parameter(torch.zeros(1))
        self.s_label = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, batch=None, data=None, task=None):
        if data is not None and batch is None:
            batch = getattr(data, 'batch', None)
        if batch is None:
            raise ValueError("Batch tensor must be provided via batch or data.batch")

        # GCN backbone
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Attention weights per graph
        gate = self.attention_pool.gate_nn(x)
        attn_weights = scatter_softmax(gate, batch, dim=0)
        graph_feat = self.attention_pool(x, batch)

        if task == 'detection':
            return self.detect_head(graph_feat).view(-1), attn_weights
        elif task == 'classification':
            return self.class_head(graph_feat), attn_weights
        elif task == 'forecast_time':
            lstm_input = graph_feat.unsqueeze(1)  # (batch_size, seq_len=1, features)
            lstm_out, _ = self.lstm(lstm_input)
            last_out = lstm_out[:, -1, :]
            return self.time_regressor(last_out).view(-1), attn_weights
        elif task == 'forecast_label':
            return self.label_head(graph_feat), attn_weights
        else:
            lstm_input = graph_feat.unsqueeze(1)
            lstm_out, _ = self.lstm(lstm_input)
            last_out = lstm_out[:, -1, :]
            time_out = self.time_regressor(last_out)
            return {
                'detection': self.detect_head(graph_feat).view(-1),
                'classification': self.class_head(graph_feat),
                'forecast_time': time_out.view(-1),
                'forecast_label': self.label_head(graph_feat)
            }, attn_weights




#version 2


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
        self.dropout = float(dropout)

        # --- GCN backbone ---
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3   = nn.BatchNorm1d(hidden_dim)

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
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1 = self.bn1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Block 2
        x2 = self.conv2(x1, edge_index, edge_weight=edge_weight)
        x2 = self.bn2(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Block 3 with residual from x2
        x3 = self.conv3(x2, edge_index, edge_weight=edge_weight)
        x3 = self.bn3(x3)
        x3 = F.relu(x3 + x2, inplace=True)  # simple residual
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




