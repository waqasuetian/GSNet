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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, AttentionalAggregation

class MultiTaskGCN(nn.Module):
    """
    A multi-task Graph Convolutional Network (GCN) model optimized for EEG data with temporal modeling.
    
    Args:
        in_dim (int): Input feature dimension (e.g., 100 for pooled EEG features).
        hidden_dim (int): Dimension of hidden layers in GCN and MLP heads.
        num_classes (int): Number of classes for classification task.
        forecast_classes (int, optional): Number of classes for forecast_label task. Defaults to num_classes.
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.2.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        forecast_classes: int = None,
        dropout: float = 0.2,
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
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        # Attention-based pooling
        self.attention_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1), nn=None)

        # Detection head (binary classification)
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

        # Classification head (multi-class classification)
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

        # Time head (regression with LSTM)
        self.time_head = nn.Sequential(
            nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Label head (multi-class classification)
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

        # Learnable log-variances for task weighting
        self.s_detect = nn.Parameter(torch.zeros(1))
        self.s_class = nn.Parameter(torch.zeros(1))
        self.s_time = nn.Parameter(torch.zeros(1))
        self.s_label = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, batch=None, data=None, task: str = None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Node feature matrix, shape [num_nodes, in_dim].
            edge_index (torch.Tensor): Graph edge indices, shape [2, num_edges].
            batch (torch.Tensor, optional): Batch vector, shape [num_nodes].
            data (torch_geometric.data.Data, optional): Data object containing x, edge_index, batch.
            task (str, optional): Task to compute ('detection', 'classification', 'forecast_time', 'forecast_label').

        Returns:
            tuple: (task_output, attn_weights) for specific task, or (dict of outputs, attn_weights) if task=None.
        """
        if data is not None and batch is None:
            batch = getattr(data, 'batch', None)
        if batch is None:
            raise ValueError("Batch tensor must be provided via batch or data.batch")

        # Backbone
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

        # Capture attention weights
        gate = self.attention_pool.gate_nn(x)
        attn_weights = torch.softmax(gate / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32, device=x.device)), dim=0)
        graph_feat = self.attention_pool(x, batch)

        if task == 'detection':
            return self.detect_head(graph_feat).view(-1), attn_weights
        elif task == 'classification':
            return self.class_head(graph_feat), attn_weights
        elif task == 'forecast_time':
            graph_feat = graph_feat.unsqueeze(1)
            lstm_out, _ = self.time_head[0](graph_feat)
            lstm_out = lstm_out[:, -1, :]
            out = self.time_head[1:](lstm_out)
            return out.view(-1), attn_weights
        elif task == 'forecast_label':
            return self.label_head(graph_feat), attn_weights
        else:
            lstm_out, _ = self.time_head[0](graph_feat.unsqueeze(1))
            lstm_out = lstm_out[:, -1, :]
            time_out = self.time_head[1:](lstm_out)
            return {
                'detection': self.detect_head(graph_feat).view(-1),
                'classification': self.class_head(graph_feat),
                'forecast_time': time_out.view(-1),
                'forecast_label': self.label_head(graph_feat)
            }, attn_weights
#             'detection': self.detect_head(graph_feat).view(-1),
#             'classification': self.class_head(graph_feat),
#             'forecast_time': self.time_head(graph_feat).view(-1),
#             'forecast_label': self.label_head(graph_feat)
#         }
