# """
# Clip crazy values: first cap your raw targets to a sensible range (e.g. 0.001–300) so a few huge outliers don’t wreck everything.

# Take a log: apply y_log = log1p(y) so your heavily skewed data looks more “bell-shaped.”

# Fit scaler on train only: use StandardScaler on those log-values from your training set—never peek at the test data when fitting.

# Train on “scaled log”: teach your model to predict the standardized log values, minimizing a smooth-L1 (Huber) loss.

# Undo transforms in order: after prediction, first do scaler.inverse_transform(...) (back to log space), then expm1(...) (back to original units).

# Compare apples to apples: compute R² on these final, back-transformed predictions vs. your original raw targets.

# Schedule by R²: drive your learning-rate scheduler off the R² itself (via ReduceLROnPlateau) so you directly optimize the metric you care about.

# Reset each epoch: clear out your lists of preds/true values at the end of each epoch so metrics don’t “leak” between epochs.
# """

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch_geometric.data import Data, DataLoader
# from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
# from models.model import MultiTaskGCN
# from data.scripts.adjacancy_matrics import AdjacencyMatrixProcessor

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction="mean"):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         ce_loss = F.cross_entropy(logits, targets, reduction="none")
#         pt = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
#         return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


# class Trainer:
#     def __init__(self, num_features, num_hiddens, num_classes, dropout, num_heads,
#                  learning_rate, batch_size, num_epochs, pooled_results, DC, RC):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.hidden_channels = 125
#         self.num_features = num_features
#         self.num_classes = num_classes
#         self.model = MultiTaskGCN(self.hidden_channels, num_features, num_classes).to(self.device)
#         self.criterion = FocalLoss(alpha=0.75, gamma=2)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
#         self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
#         self.pooled_results = pooled_results

#         # Build adjacency
#         num_nodes = 19
#         edge_index = torch.tril_indices(num_nodes, num_nodes, offset=-1)
#         adj_proc = AdjacencyMatrixProcessor(pooled_results, data_directory=r'E:\tuh_data\train')
#         if DC:
#             ew = adj_proc.compute_all_edge_weights(DC=True, RC=False).mean(dim=0)
#         else:
#             ew = adj_proc.compute_all_edge_weights(DC=False, RC=True).mean(dim=0)
#         exp_size = num_nodes * (num_nodes - 1) // 2
#         if ew.numel() > exp_size:
#             ew = ew[:exp_size]
#         elif ew.numel() < exp_size:
#             ew = torch.cat([ew, torch.zeros(exp_size - ew.numel())])
#         self.edge_index = edge_index
#         self.edge_attr = ew

#     def create_graph_batches(self, X, Y, task):
#         data_list = []
#         for i in range(len(X)):
#             x = torch.tensor(X[i].mean(axis=0), dtype=torch.float)
#             y = torch.tensor(Y[i], dtype=torch.float if task == "early_reg" else torch.long)
#             data_list.append(Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y))
#         return DataLoader(data_list, batch_size=self.batch_size, shuffle=False)

#     def train(self, X, Y,
#               detection=False, classification=False,
#               early_clf=False, early_reg=False):

#         if detection:
#             loader = self.create_graph_batches(X, Y, task="detection")
#             self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
#             print(">>> Detection task")

#         elif classification:
#             loader = self.create_graph_batches(X, Y, task="classification")
#             self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
#             print(">>> Classification task")

#         elif early_clf:
#             loader = self.create_graph_batches(X, Y, task="early_clf")
#             self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
#             print(">>> Early Classification task")

#         elif early_reg:
#             print(">>> Early Regression task")
#             Y = np.clip(Y, 0.001, 300.0)  # avoid outliers and negatives
#             self.regression_scaler = RobustScaler()
#             Y_scaled = self.regression_scaler.fit_transform(Y.reshape(-1, 1)).flatten()

#             loader = self.create_graph_batches(X, Y_scaled, task="early_reg")
#             self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
#             self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5,
#                                                patience=3, verbose=True, min_lr=1e-6)
#             print("Sample Y (raw):", Y[:5])
#             print("Sample Y (scaled):", Y_scaled[:5])
#         else:
#             raise ValueError("Specify exactly one task flag.")

#         for epoch in range(1, self.num_epochs + 1):
#             self.model.train()
#             total_loss = 0.0
#             scaled_preds = []
#             scaled_trues = []

#             for batch in loader:
#                 batch = batch.to(self.device)
#                 self.optimizer.zero_grad()

#                 if detection:
#                     out = self.model(batch.x, batch.edge_index, batch, task="detection").view(-1)
#                     loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
#                     preds = (torch.sigmoid(out) > 0.5).cpu().numpy()
#                     trues = batch.y.cpu().numpy()

#                 elif classification:
#                     out = self.model(batch.x, batch.edge_index, batch, task="classification")
#                     loss = F.cross_entropy(out, batch.y)
#                     preds = out.argmax(dim=1).cpu().numpy()
#                     trues = batch.y.cpu().numpy()

#                 elif early_clf:
#                     out = self.model(batch.x, batch.edge_index, batch, task="forecast_label")
#                     loss = F.cross_entropy(out, batch.y)
#                     preds = out.argmax(dim=1).cpu().numpy()
#                     trues = batch.y.cpu().numpy()

#                 elif early_reg:
#                     out = self.model(batch.x, batch.edge_index, batch, task="forecast_time").view(-1)
#                     loss = F.smooth_l1_loss(out, batch.y.float())
#                     scaled_preds.extend(out.detach().cpu().numpy())
#                     scaled_trues.extend(batch.y.cpu().numpy())
#                     print("pred" * 10, np.round(scaled_preds, 3))
#                     print("true" * 10, np.round(scaled_trues, 3))

#                 loss.backward()
#                 self.optimizer.step()
#                 total_loss += loss.item()

#             avg_loss = total_loss / len(loader)

#             if early_reg:
#                 preds = self.regression_scaler.inverse_transform(
#                     np.array(scaled_preds).reshape(-1, 1)).flatten()
#                 trues = self.regression_scaler.inverse_transform(
#                     np.array(scaled_trues).reshape(-1, 1)).flatten()

#                 r2 = r2_score(trues, preds)
#                 rmse = np.sqrt(mean_squared_error(trues, preds))
#                 print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} "
#                       f"- R²: {r2:.4f} - RMSE: {rmse:.2f}")
#                 self.scheduler.step(r2)

#             else:
#                 metric = accuracy_score(trues, preds)
#                 print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} "
#                       f"- Acc: {metric:.4f}")
#                 self.scheduler.step()

#         print("Training complete!\n")
#         return preds  # Final predictions

        

    

# # ─── Example usage ─────────────────────────────────────────────
# # trainer = Trainer(num_features=64,
# #                   num_classes=2,
# #                   learning_rate=1e-3,
# #                   batch_size=32,
# #                   num_epochs=30,
# #                   pooled_results=my_pooled_results,
# #                   DC=True)
# # preds = trainer.train(X_data, Y_data, early_reg=True)



## --------------------- old version --------------------






# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from torch_geometric.data import Data, DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, accuracy_score
# from sklearn.model_selection import train_test_split

# from models.model import MultiTaskGCN
# from data.scripts.adjacancy_matrics import AdjacencyMatrixProcessor

# class FocalLoss(torch.nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction="mean"):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         ce_loss = F.cross_entropy(logits, targets, reduction="none")
#         pt = torch.exp(-ce_loss)
#         loss = self.alpha * (1 - pt)**self.gamma * ce_loss
#         return loss.mean() if self.reduction=="mean" else loss.sum()

# class Trainer:
#     def __init__(
#         self,
#         num_features,
#         num_hiddens,
#         num_classes,
#         dropout,
#         num_heads,
#         learning_rate,
#         batch_size,
#         num_epochs,
#         pooled_results,
#         DC=False,
#         RC=False
#     ):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.base_lr = learning_rate
#         self.base_wd = 5e-4

#         # Model & loss
#         self.model = MultiTaskGCN(num_hiddens, num_features, num_classes).to(self.device)
#         self.criterion = FocalLoss(alpha=0.75, gamma=2)

#         # Build edge_index & edge_attr
#         num_nodes = 19
#         edge_index = torch.tril_indices(num_nodes, num_nodes, offset=-1)
#         adj_proc = AdjacencyMatrixProcessor(pooled_results, data_directory=r"E:\tuh_data\train")
#         weights = (
#             adj_proc.compute_all_edge_weights(DC=DC, RC=False).mean(dim=0)
#             if DC else
#             adj_proc.compute_all_edge_weights(DC=False, RC=RC).mean(dim=0)
#         )
#         expected = num_nodes*(num_nodes-1)//2
#         if weights.numel()>expected:
#             weights = weights[:expected]
#         elif weights.numel()<expected:
#             weights = torch.cat([weights, torch.zeros(expected-weights.numel())])
#         self.edge_index = edge_index.to(self.device)
#         self.edge_attr  = weights.to(self.device)

#         # Scalers
#         self.feature_scaler = None
#         self.regression_scaler = None

#     @staticmethod
#     def safe_r2_score(y_true, y_pred):
#         y_true = np.array(y_true).flatten()
#         y_pred = np.array(y_pred).flatten()
#         mask = np.isfinite(y_true)&np.isfinite(y_pred)
#         y_true, y_pred = y_true[mask], y_pred[mask]
#         if len(y_true)==0:
#             return 0.0
#         ss_res = np.sum((y_true-y_pred)**2)
#         ss_tot = np.sum((y_true-np.mean(y_true))**2)
#         if ss_tot<=0 or np.isnan(ss_res):
#             return 0.0
#         return max(1 - ss_res/ss_tot, 0.0)

#     def create_graph_batches(self, X, Y, task=None):
#         data_list = []
#         for i in range(len(X)):
#             x_vec = X[i]  # already collapsed and scaled if early_reg
#             x = torch.tensor(x_vec, dtype=torch.float, device=self.device)
#             y_val = Y[i]
#             if task=="early_reg":
#                 y = torch.tensor(y_val, dtype=torch.float, device=self.device)
#             else:
#                 y = torch.tensor(y_val, dtype=torch.long, device=self.device)
#             if i<3:
#                 print(f"[{i}] Label: {y_val} → Tensor: {y.item()}")
#             data_list.append(Data(
#                 x=x,
#                 edge_index=self.edge_index,
#                 edge_attr=self.edge_attr,
#                 y=y
#             ))
#         return DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

#     def train(
#         self,
#         X,              # np.ndarray shape (N, T, F)
#         Y,              # list or array length N
#         detection=False,
#         classification=False,
#         early_reg=False,
#         early_clf=False
#     ):
#         os.makedirs("models/checkpoints", exist_ok=True)

#         # === Prepare data/loaders ===
#         if early_reg:
#             X = np.array(X)
#             Y = np.array(Y)
#             Y = np.maximum(Y, 1e-3)
#             X_mean = X.mean(axis=1)
#             X_train, X_val, Y_train, Y_val = train_test_split(
#                 X_mean, Y, test_size=0.2, random_state=42
#             )
#             # scale features
#             self.feature_scaler = StandardScaler()
#             X_train = self.feature_scaler.fit_transform(X_train)
#             X_val   = self.feature_scaler.transform(X_val)
#             # log+scale targets
#             Y_log = np.log1p(Y_train).reshape(-1,1)
#             self.regression_scaler = StandardScaler()
#             Y_train_scaled = self.regression_scaler.fit_transform(Y_log).flatten()
#             Y_val_scaled   = self.regression_scaler.transform(
#                 np.log1p(Y_val).reshape(-1,1)
#             ).flatten()
#             train_loader = self.create_graph_batches(X_train, Y_train_scaled, task="early_reg")
#             val_loader   = self.create_graph_batches(X_val, Y_val_scaled,   task="early_reg")
#             print("=================== (Early Regression) ===================")
#             self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
#             self.scheduler = ReduceLROnPlateau(
#                 self.optimizer, mode="max", factor=0.5, patience=3, verbose=True, min_lr=1e-6
#             )
#         else:
#             task = ("detection" if detection else
#                     "classification" if classification else
#                     "early_clf")
#             train_loader = self.create_graph_batches(X, Y, task=task)
#             print(f"=================== ({task}) ===================")
#             lr = self.base_lr
#             self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.base_wd)
#             self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

#         # === Training loop ===
#         for epoch in range(1, self.num_epochs+1):
#             self.model.train()
#             epoch_loss = 0.0
#             all_preds, all_labels = [], []

#             for batch in train_loader:
#                 batch = batch.to(self.device)
#                 self.optimizer.zero_grad()

#                 if detection:
#                     out = self.model(batch.x, batch.edge_index, batch, task="detection").view(-1)
#                     loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
#                     preds = (torch.sigmoid(out)>0.5).cpu().numpy()
#                     labels = batch.y.cpu().numpy()
#                 elif classification:
#                     out = self.model(batch.x, batch.edge_index, batch, task="classification")
#                     loss = F.cross_entropy(out, batch.y)
#                     preds = out.argmax(dim=1).cpu().numpy()
#                     labels = batch.y.cpu().numpy()
#                 elif early_reg:
#                     out = self.model(batch.x, batch.edge_index, batch, task="forecast_time").view(-1)
#                     loss = F.smooth_l1_loss(out, batch.y.float())
#                     preds = out.detach().cpu().numpy()
#                     labels = batch.y.cpu().numpy()
#                 else:  # early_clf
#                     out = self.model(batch.x, batch.edge_index, batch, task="forecast_label")
#                     loss = F.cross_entropy(out, batch.y)
#                     preds = out.argmax(dim=1).cpu().numpy()
#                     labels = batch.y.cpu().numpy()

#                 loss.backward()
#                 self.optimizer.step()
#                 epoch_loss += loss.item()
#                 all_preds.extend(preds)
#                 all_labels.extend(labels)

#             avg_loss = epoch_loss / len(train_loader)

#             # === Metrics & scheduling ===
#             if early_reg:
#                 # inverse-transform for metrics
#                 train_preds = np.expm1(self.regression_scaler.inverse_transform(
#                     np.array(all_preds).reshape(-1,1)
#                 ).flatten())
#                 train_true  = np.expm1(self.regression_scaler.inverse_transform(
#                     np.array(all_labels).reshape(-1,1)
#                 ).flatten())
#                 train_r2  = self.safe_r2_score(train_true, train_preds)
#                 train_rmse = np.sqrt(mean_squared_error(train_true, train_preds))

#                 # validation
#                 self.model.eval()
#                 val_preds, val_true = [], []
#                 with torch.no_grad():
#                     for batch in val_loader:
#                         batch = batch.to(self.device)
#                         out = self.model(batch.x, batch.edge_index, batch, task="forecast_time").view(-1).cpu().numpy()
#                         inv_pred = np.expm1(self.regression_scaler.inverse_transform(out.reshape(-1,1)).flatten())
#                         inv_true = np.expm1(self.regression_scaler.inverse_transform(
#                             batch.y.cpu().numpy().reshape(-1,1)
#                         ).flatten())
#                         val_preds.extend(inv_pred)
#                         val_true.extend(inv_true)

#                 val_r2   = self.safe_r2_score(val_true, val_preds)
#                 val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))

#                 print(
#                     f"Epoch {epoch}/{self.num_epochs} – "
#                     f"Loss: {avg_loss:.4f} | "
#                     f"Train R²: {train_r2:.4f} RMSE: {train_rmse:.2f} | "
#                     f"Val R²: {val_r2:.4f} RMSE: {val_rmse:.2f}"
#                 )
#                 self.scheduler.step(val_r2)
#             else:
#                 acc = accuracy_score(all_labels, all_preds)
#                 print(f"Epoch {epoch}/{self.num_epochs} – Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
#                 if isinstance(self.scheduler, ReduceLROnPlateau):
#                     self.scheduler.step(acc)
#                 else:
#                     self.scheduler.step()

#             # === Checkpointing ===
#             if detection:
#                 ckpt_dir = "models/checkpoints/detection_checkpoints"
#                 ckpt_path = f"{ckpt_dir}/detection_epoch_{epoch}.pth"
#             elif classification:
#                 ckpt_dir = "models/checkpoints/classification_checkpoints"
#                 ckpt_path = f"{ckpt_dir}/classification_epoch_{epoch}.pth"
#             elif early_reg:
#                 ckpt_dir = "models/checkpoints/early_regression_checkpoints"
#                 ckpt_path = f"{ckpt_dir}/early_reg_epoch_{epoch}.pth"
#             elif early_clf:
#                 ckpt_dir = "models/checkpoints/early_classification_checkpoints"
#                 ckpt_path = f"{ckpt_dir}/early_clf_epoch_{epoch}.pth"
#             else:
#                 ckpt_dir = "models/checkpoints/mixed_task"
#                 ckpt_path = f"{ckpt_dir}/epoch_{epoch}.pth"

#             os.makedirs(ckpt_dir, exist_ok=True)
#             torch.save(self.model.state_dict(), ckpt_path)

#         print("Training complete!")
#         return all_preds










import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

from models.model import MultiTaskGCN
from data.scripts.adjacancy_matrics import AdjacencyMatrixProcessor


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


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
        RC: bool = False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_lr = learning_rate
        self.base_wd = 0.0

        self.model = MultiTaskGCN(num_hiddens, num_features, num_classes, dropout=0.2).to(self.device)
        self.criterion = FocalLoss(alpha=0.75, gamma=2)

        num_nodes = 19
        edge_index = torch.tril_indices(num_nodes, num_nodes, offset=-1)
        adj_proc = AdjacencyMatrixProcessor(pooled_results, data_directory=r"E:\tuh_data\train")
        weights = (
            adj_proc.compute_all_edge_weights(DC=False, RC=True).mean(dim=0)
            if RC else
            adj_proc.compute_all_edge_weights(DC=True, RC=False).mean(dim=0)
        )

        valid_weights = []
        if isinstance(weights, list):
            for i, w in enumerate(weights):
                if w is not None and isinstance(w, torch.Tensor):
                    valid_weights.append(w)
                else:
                    print(f"⚠️ Skipping invalid edge weights at index {i}")
        else:
            valid_weights.append(weights)

        expected = num_nodes * (num_nodes - 1) // 2
        if valid_weights:
            weights = torch.stack(valid_weights).mean(dim=0)
            if weights.numel() > expected:
                weights = weights[:expected]
            elif weights.numel() < expected:
                weights = torch.cat([weights, torch.zeros(expected - weights.numel())])
        else:
            print("⚠️ No valid edge weights found. Using default uniform weights.")
            weights = torch.ones(expected)

        self.edge_index = edge_index.to(self.device)
        self.edge_attr = weights.to(self.device)
        self.feature_scaler = None
        self.regression_scaler = None

    @staticmethod
    def safe_r2_score(y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) == 0:
            return 0.0
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot <= 0 or np.isnan(ss_res):
            return 0.0
        return max(1 - ss_res / ss_tot, 0.0)

    def create_graph_batches(self, X, Y, task: str = None):
        data_list = []
        for i in range(len(X)):
            x_np = X[i]
            if task != "early_reg":
                x_np = x_np.mean(axis=0)
            x = torch.tensor(x_np, dtype=torch.float, device=self.device)

            y_val = Y[i]
            if task == "early_reg":
                y = torch.tensor(y_val, dtype=torch.float, device=self.device)
            else:
                y = torch.tensor(y_val, dtype=torch.long, device=self.device)

            if i < 3:
                print(f"[{i}] Label: {y_val} -> Tensor: {y.item()}")

            data_list.append(Data(
                x=x,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=y
            ))

        return DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

    def train(
        self,
        X,
        Y,
        detection: bool = False,
        classification: bool = False,
        early_reg: bool = False,
        early_clf: bool = False,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        os.makedirs("models/checkpoints", exist_ok=True)

        if early_reg:
            X = np.array(X)
            Y = np.array(Y)
            Y = np.maximum(Y, 1e-3)
            X_mean = X.mean(axis=1)
            N, nodes, feats = X_mean.shape
            X_flat = X_mean.reshape(N, -1)

            X_train, X_val, Y_train, Y_val = train_test_split(
                X_flat, Y, test_size=test_size, random_state=random_state
            )
            self.feature_scaler = StandardScaler()
            X_train = self.feature_scaler.fit_transform(X_train)
            X_val = self.feature_scaler.transform(X_val)

            Y_log = np.log1p(Y_train).reshape(-1, 1)
            self.regression_scaler = StandardScaler()
            Y_train_scaled = self.regression_scaler.fit_transform(Y_log).flatten()
            Y_val_scaled = self.regression_scaler.transform(
                np.log1p(Y_val).reshape(-1, 1)
            ).flatten()

            train_loader = self.create_graph_batches(
                X_train.reshape(-1, nodes, feats), Y_train_scaled, task="early_reg"
            )
            val_loader = self.create_graph_batches(
                X_val.reshape(-1, nodes, feats), Y_val_scaled, task="early_reg"
            )

            print("=================== Early Regression ===================")
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=1e-4, weight_decay=0.0
            )
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5,
                patience=3, verbose=True, min_lr=1e-6
            )
        else:
            task = (
                "detection" if detection else
                "classification" if classification else
                "early_clf"
            )

            stratify = Y if (classification or early_clf) else None
            idx = np.arange(len(X))
            train_idx, val_idx = train_test_split(
                idx, test_size=test_size, random_state=random_state, stratify=stratify
            )
            X_train = [X[i] for i in train_idx]
            Y_train = [Y[i] for i in train_idx]
            X_val   = [X[i] for i in val_idx]
            Y_val   = [Y[i] for i in val_idx]

            train_loader = self.create_graph_batches(X_train, Y_train, task=task)
            val_loader = self.create_graph_batches(X_val, Y_val, task=task)

            print(f"=================== {task.capitalize()} ===================")
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.base_lr, weight_decay=self.base_wd
            )
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            all_preds, all_labels = [], []

            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                if detection:
                    out = self.model(batch.x, batch.edge_index, batch, task="detection").view(-1)
                    loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
                    preds = (torch.sigmoid(out) > 0.5).cpu().numpy()
                    labels = batch.y.cpu().numpy()
                elif classification:
                    out = self.model(batch.x, batch.edge_index, batch, task="classification")
                    loss = F.cross_entropy(out, batch.y)
                    preds = out.argmax(dim=1).cpu().numpy()
                    labels = batch.y.cpu().numpy()
                elif early_reg:
                    out = self.model(batch.x, batch.edge_index, batch, task="forecast_time").view(-1)
                    loss = F.smooth_l1_loss(out, batch.y.float())
                    preds = out.detach().cpu().numpy()
                    labels = batch.y.cpu().numpy()
                else:
                    out = self.model(batch.x, batch.edge_index, batch, task="forecast_label")
                    loss = F.cross_entropy(out, batch.y)
                    preds = out.argmax(dim=1).cpu().numpy()
                    labels = batch.y.cpu().numpy()

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                all_preds.extend(preds)
                all_labels.extend(labels)

            avg_loss = epoch_loss / len(train_loader)

            if early_reg:
                inv_preds = np.expm1(self.regression_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1))).flatten()
                inv_true = np.expm1(self.regression_scaler.inverse_transform(np.array(all_labels).reshape(-1, 1))).flatten()
                train_r2 = self.safe_r2_score(inv_true, inv_preds)
                train_rmse = np.sqrt(mean_squared_error(inv_true, inv_preds))

                self.model.eval()
                val_preds, val_true = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        out = self.model(batch.x, batch.edge_index, batch, task="forecast_time").view(-1).cpu().numpy()
                        p = np.expm1(self.regression_scaler.inverse_transform(out.reshape(-1, 1))).flatten()
                        t = np.expm1(self.regression_scaler.inverse_transform(batch.y.cpu().numpy().reshape(-1, 1))).flatten()
                        val_preds.extend(p)
                        val_true.extend(t)

                val_r2 = self.safe_r2_score(val_true, val_preds)
                val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))

                print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} | Train R2: {train_r2:.4f}, RMSE: {train_rmse:.2f} | Val R2: {val_r2:.4f}, RMSE: {val_rmse:.2f}")
                self.scheduler.step(val_r2)
            else:
                acc = accuracy_score(all_labels, all_preds)

                self.model.eval()
                v_preds, v_labels = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        if detection:
                            out = self.model(batch.x, batch.edge_index, batch, task="detection").view(-1)
                            v_preds.extend((torch.sigmoid(out) > 0.5).cpu().numpy())
                            v_labels.extend(batch.y.cpu().numpy())
                        elif classification:
                            out = self.model(batch.x, batch.edge_index, batch, task="classification")
                            v_preds.extend(out.argmax(dim=1).cpu().numpy())
                            v_labels.extend(batch.y.cpu().numpy())
                        else:
                            out = self.model(batch.x, batch.edge_index, batch, task="forecast_label")
                            v_preds.extend(out.argmax(dim=1).cpu().numpy())
                            v_labels.extend(batch.y.cpu().numpy())
                val_acc = accuracy_score(v_labels, v_preds)

                print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Val Acc: {val_acc:.4f}")
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            tag = "detection" if detection else "classification" if classification else "early_reg" if early_reg else "early_clf"
            ckpt_dir = f"models/checkpoints/{tag}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, f"{tag}_epoch_{epoch}.pth"))

        print("Training complete!")
        return all_preds
