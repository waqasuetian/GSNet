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










# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
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
#         loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
#         return loss.mean() if self.reduction == "mean" else loss.sum()


# class Trainer:
#     def __init__(
#         self,
#         num_features: int,
#         num_hiddens: int,
#         num_classes: int,
#         dropout: float,
#         num_heads: int,
#         learning_rate: float,
#         batch_size: int,
#         num_epochs: int,
#         pooled_results,
#         DC: bool = False,
#         RC: bool = False
#     ):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.base_lr = learning_rate
#         self.base_wd = 0.0

#         self.model = MultiTaskGCN(num_hiddens, num_features, num_classes, dropout=0.2).to(self.device)
#         self.criterion = FocalLoss(alpha=0.75, gamma=2)

#         num_nodes = 19
#         edge_index = torch.tril_indices(num_nodes, num_nodes, offset=-1)
#         adj_proc = AdjacencyMatrixProcessor(pooled_results, data_directory=r"E:\tuh_data\train")
#         weights = (
#             adj_proc.compute_all_edge_weights(DC=False, RC=True).mean(dim=0)
#             if RC else
#             adj_proc.compute_all_edge_weights(DC=True, RC=False).mean(dim=0)
#         )

#         valid_weights = []
#         if isinstance(weights, list):
#             for i, w in enumerate(weights):
#                 if w is not None and isinstance(w, torch.Tensor):
#                     valid_weights.append(w)
#                 else:
#                     print(f"⚠️ Skipping invalid edge weights at index {i}")
#         else:
#             valid_weights.append(weights)

#         expected = num_nodes * (num_nodes - 1) // 2
#         if valid_weights:
#             weights = torch.stack(valid_weights).mean(dim=0)
#             if weights.numel() > expected:
#                 weights = weights[:expected]
#             elif weights.numel() < expected:
#                 weights = torch.cat([weights, torch.zeros(expected - weights.numel())])
#         else:
#             print("⚠️ No valid edge weights found. Using default uniform weights.")
#             weights = torch.ones(expected)

#         self.edge_index = edge_index.to(self.device)
#         self.edge_attr = weights.to(self.device)
#         self.feature_scaler = None
#         self.regression_scaler = None

#     @staticmethod
#     def safe_r2_score(y_true, y_pred):
#         y_true = np.array(y_true).flatten()
#         y_pred = np.array(y_pred).flatten()
#         mask = np.isfinite(y_true) & np.isfinite(y_pred)
#         y_true, y_pred = y_true[mask], y_pred[mask]
#         if len(y_true) == 0:
#             return 0.0
#         ss_res = np.sum((y_true - y_pred) ** 2)
#         ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#         if ss_tot <= 0 or np.isnan(ss_res):
#             return 0.0
#         return max(1 - ss_res / ss_tot, 0.0)

#     def create_graph_batches(self, X, Y, task: str = None):
#         data_list = []
#         for i in range(len(X)):
#             x_np = X[i]
#             if task != "early_reg":
#                 x_np = x_np.mean(axis=0)
#             x = torch.tensor(x_np, dtype=torch.float, device=self.device)

#             y_val = Y[i]
#             if task == "early_reg":
#                 y = torch.tensor(y_val, dtype=torch.float, device=self.device)
#             else:
#                 y = torch.tensor(y_val, dtype=torch.long, device=self.device)

#             if i < 3:
#                 print(f"[{i}] Label: {y_val} -> Tensor: {y.item()}")

#             data_list.append(Data(
#                 x=x,
#                 edge_index=self.edge_index,
#                 edge_attr=self.edge_attr,
#                 y=y
#             ))

#         return DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

#     def train(
#         self,
#         X,
#         Y,
#         detection: bool = False,
#         classification: bool = False,
#         early_reg: bool = False,
#         early_clf: bool = False,
#         test_size: float = 0.2,
#         random_state: int = 42
#     ):
#         os.makedirs("models/checkpoints", exist_ok=True)

#         if early_reg:
#             X = np.array(X)
#             Y = np.array(Y)
#             Y = np.maximum(Y, 1e-3)
#             X_mean = X.mean(axis=1)
#             N, nodes, feats = X_mean.shape
#             X_flat = X_mean.reshape(N, -1)

#             X_train, X_val, Y_train, Y_val = train_test_split(
#                 X_flat, Y, test_size=test_size, random_state=random_state
#             )
#             self.feature_scaler = StandardScaler()
#             X_train = self.feature_scaler.fit_transform(X_train)
#             X_val = self.feature_scaler.transform(X_val)

#             Y_log = np.log1p(Y_train).reshape(-1, 1)
#             self.regression_scaler = StandardScaler()
#             Y_train_scaled = self.regression_scaler.fit_transform(Y_log).flatten()
#             Y_val_scaled = self.regression_scaler.transform(
#                 np.log1p(Y_val).reshape(-1, 1)
#             ).flatten()

#             train_loader = self.create_graph_batches(
#                 X_train.reshape(-1, nodes, feats), Y_train_scaled, task="early_reg"
#             )
#             val_loader = self.create_graph_batches(
#                 X_val.reshape(-1, nodes, feats), Y_val_scaled, task="early_reg"
#             )

#             print("=================== Early Regression ===================")
#             self.optimizer = optim.Adam(
#                 self.model.parameters(), lr=1e-4, weight_decay=0.0
#             )
#             self.scheduler = ReduceLROnPlateau(
#                 self.optimizer, mode="max", factor=0.5,
#                 patience=3, verbose=True, min_lr=1e-6
#             )
#         else:
#             task = (
#                 "detection" if detection else
#                 "classification" if classification else
#                 "early_clf"
#             )

#             stratify = Y if (classification or early_clf) else None
#             idx = np.arange(len(X))
#             train_idx, val_idx = train_test_split(
#                 idx, test_size=test_size, random_state=random_state, stratify=stratify
#             )
#             X_train = [X[i] for i in train_idx]
#             Y_train = [Y[i] for i in train_idx]
#             X_val   = [X[i] for i in val_idx]
#             Y_val   = [Y[i] for i in val_idx]

#             train_loader = self.create_graph_batches(X_train, Y_train, task=task)
#             val_loader = self.create_graph_batches(X_val, Y_val, task=task)

#             print(f"=================== {task.capitalize()} ===================")
#             self.optimizer = optim.Adam(
#                 self.model.parameters(), lr=self.base_lr, weight_decay=self.base_wd
#             )
#             self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

#         for epoch in range(1, self.num_epochs + 1):
#             self.model.train()
#             epoch_loss = 0.0
#             all_preds, all_labels = [], []

#             for batch in train_loader:
#                 batch = batch.to(self.device)
#                 self.optimizer.zero_grad()

#                 if detection:
#                     out = self.model(batch.x, batch.edge_index, batch, task="detection").view(-1)
#                     loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
#                     preds = (torch.sigmoid(out) > 0.5).cpu().numpy()
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
#                 else:
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

#             if early_reg:
#                 inv_preds = np.expm1(self.regression_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1))).flatten()
#                 inv_true = np.expm1(self.regression_scaler.inverse_transform(np.array(all_labels).reshape(-1, 1))).flatten()
#                 train_r2 = self.safe_r2_score(inv_true, inv_preds)
#                 train_rmse = np.sqrt(mean_squared_error(inv_true, inv_preds))

#                 self.model.eval()
#                 val_preds, val_true = [], []
#                 with torch.no_grad():
#                     for batch in val_loader:
#                         batch = batch.to(self.device)
#                         out = self.model(batch.x, batch.edge_index, batch, task="forecast_time").view(-1).cpu().numpy()
#                         p = np.expm1(self.regression_scaler.inverse_transform(out.reshape(-1, 1))).flatten()
#                         t = np.expm1(self.regression_scaler.inverse_transform(batch.y.cpu().numpy().reshape(-1, 1))).flatten()
#                         val_preds.extend(p)
#                         val_true.extend(t)

#                 val_r2 = self.safe_r2_score(val_true, val_preds)
#                 val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))

#                 print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} | Train R2: {train_r2:.4f}, RMSE: {train_rmse:.2f} | Val R2: {val_r2:.4f}, RMSE: {val_rmse:.2f}")
#                 self.scheduler.step(val_r2)
#             else:
#                 acc = accuracy_score(all_labels, all_preds)

#                 self.model.eval()
#                 v_preds, v_labels = [], []
#                 with torch.no_grad():
#                     for batch in val_loader:
#                         batch = batch.to(self.device)
#                         if detection:
#                             out = self.model(batch.x, batch.edge_index, batch, task="detection").view(-1)
#                             v_preds.extend((torch.sigmoid(out) > 0.5).cpu().numpy())
#                             v_labels.extend(batch.y.cpu().numpy())
#                         elif classification:
#                             out = self.model(batch.x, batch.edge_index, batch, task="classification")
#                             v_preds.extend(out.argmax(dim=1).cpu().numpy())
#                             v_labels.extend(batch.y.cpu().numpy())
#                         else:
#                             out = self.model(batch.x, batch.edge_index, batch, task="forecast_label")
#                             v_preds.extend(out.argmax(dim=1).cpu().numpy())
#                             v_labels.extend(batch.y.cpu().numpy())
#                 val_acc = accuracy_score(v_labels, v_preds)

#                 print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Val Acc: {val_acc:.4f}")
#                 if isinstance(self.scheduler, ReduceLROnPlateau):
#                     self.scheduler.step(val_acc)
#                 else:
#                     self.scheduler.step()

#             tag = "detection" if detection else "classification" if classification else "early_reg" if early_reg else "early_clf"
#             ckpt_dir = f"models/checkpoints/{tag}"
#             os.makedirs(ckpt_dir, exist_ok=True)
#             torch.save(self.model.state_dict(), os.path.join(ckpt_dir, f"{tag}_epoch_{epoch}.pth"))

#         print("Training complete!")
#         return all_preds


##new one

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
        # Device and hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_lr = learning_rate
        self.base_wd = 0.0

        # Model and loss
        self.model = MultiTaskGCN(
            in_dim=100,  # Match input feature dimension (100 features)
            hidden_dim=num_hiddens,
            num_classes=num_classes,
            dropout=0.5  # Match model.py default
        ).to(self.device)
        self.criterion = FocalLoss(alpha=0.75, gamma=2)

        # Build edge_index and edge_attr
        num_nodes = 19
        edge_index = torch.tril_indices(num_nodes, num_nodes, offset=-1)
        adj_proc = AdjacencyMatrixProcessor(pooled_results, data_directory=r"E:\tuh_data\train")
        try:
            weights = adj_proc.compute_all_edge_weights(DC=DC, RC=RC).mean(dim=0)
            expected = num_nodes * (num_nodes - 1) // 2
            if weights.numel() > expected:
                weights = weights[:expected]
            elif weights.numel() < expected:
                weights = torch.cat([weights, torch.zeros(expected - weights.numel(), device=weights.device)])
        except Exception as e:
            print(f"[WARNING] Edge weight computation failed: {e}. Using uniform weights.")
            weights = torch.ones(num_nodes * (num_nodes - 1) // 2)

        self.edge_index = edge_index.to(self.device)
        self.edge_attr = weights.to(self.device)

        # Scalers for regression
        self.feature_scaler = None
        self.regression_scaler = None
    
    @staticmethod
    def safe_r2_score(y_true, y_pred, verbose=False):
        """
        Compute a robust R² score for regression tasks, optimized for EEG data.

        Args:
            y_true (torch.Tensor): True target values.
            y_pred (torch.Tensor): Predicted values.
            verbose (bool): If True, print diagnostic information.

        Returns:
            float: R² score, capped at 0.0 to avoid negative values.
        """
        y_true = torch.as_tensor(y_true).flatten()
        y_pred = torch.as_tensor(y_pred).flatten()
        mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) == 0:
            if verbose:
                print("No valid data after filtering NaNs/infs")
            return 0.0
        y_true_mean = torch.mean(y_true)
        y_true_std = torch.std(y_true, unbiased=False)
        y_pred_mean = torch.mean(y_pred)
        y_pred_std = torch.std(y_pred, unbiased=False)
        epsilon = 1e-8
        if y_true_std < epsilon or y_pred_std < epsilon:
            if verbose:
                print(f"Variance too low: y_true_std={y_true_std.item()}, y_pred_std={y_pred_std.item()}")
            return 0.0
        y_true = (y_true - y_true_mean) / (y_true_std + epsilon)
        y_pred = (y_pred - y_pred_mean) / (y_pred_std + epsilon)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        if ss_tot <= epsilon or torch.isnan(ss_res):
            if verbose:
                print(f"Invalid SS_tot ({ss_tot.item()}) or SS_res is NaN")
            return 0.0
        r2 = 1 - ss_res / (ss_tot + epsilon)
        return max(r2.item(), 0.0)

    def create_graph_batches(self, X, Y, task: str = None):
        """
        Build a PyTorch Geometric DataLoader from input features X and labels Y,
        applying 30% data duplication (augmentation) for all tasks:
            - detection, classification: average over time dimension.
            - early_reg (forecast time): regression on (nodes, features).
            - early_class (forecast label): classification on (nodes, features).
        """
        original_len = X.shape[0]
        augment_count = max(1, int(original_len * 0.3)) if original_len > 1 else 0
        print(f"[Before Augmentation] X: {original_len}, Y: {Y.shape[0]}")
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y)
        if augment_count > 0:
            dup_indices = np.random.choice(original_len, size=augment_count, replace=True)
            X_aug = X[dup_indices]
            Y_aug = Y[dup_indices]
            X = np.concatenate([X, X_aug], axis=0)
            Y = torch.cat([Y, Y_aug], dim=0)
        print(f"[After Augmentation] X: {X.shape[0]}, Y: {Y.shape[0]}")
        data_list = []
        for i in range(len(X)):
            x_np = X[i]
            if task not in ["early_reg", "early_class"]:
                x_np = x_np.mean(axis=0)  # (nodes, features)
            x = torch.tensor(x_np, dtype=torch.float, device=self.device)
            y_val = Y[i]
            if task == "early_reg":
                y = y_val.clone().detach().to(dtype=torch.float, device=self.device)
            else:
                y = y_val.clone().detach().to(dtype=torch.long, device=self.device)
            if i < 3:
                print(f"[{i}] Label: {y_val} -> Tensor: {y.item()}")
            data = Data(
                x=x,
                edge_index=self.edge_index,
                y=y
            )
            if hasattr(self, 'edge_attr') and self.edge_attr is not None:
                data.edge_attr = self.edge_attr
            data_list.append(data)
        return DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

    def train(
        self,
        X,
        Y,
        detection: bool = False,
        classification: bool = False,
        early_reg: bool = False,
        early_clf: bool = False
    ):
        os.makedirs("models/checkpoints", exist_ok=True)

        # Debug input shapes and model parameters
        print(f"[DEBUG] Input X shape: {np.array(X).shape}")
        print(f"[DEBUG] Input Y shape: {np.array(Y).shape}")
        print(f"[DEBUG] Model conv1 weight shape: {self.model.conv1.lin.weight.shape}")

        # Prepare data loaders
        if early_reg:
            X = np.array(X)
            Y = np.array(Y)
            Y = np.maximum(Y, 1e-3)
            X_mean = X.mean(axis=1)  # (N, nodes, features)
            N, nodes, feats = X_mean.shape
            print(f"[DEBUG] X_mean shape: {X_mean.shape}, nodes: {nodes}, feats: {feats}")
            X_flat = X_mean.reshape(N, -1)
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_flat, Y, test_size=0.2, random_state=42
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
            train_loader = self.create_graph_batches(X, Y, task=task)
            print(f"=================== {task.capitalize()} ===================")
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.base_lr, weight_decay=self.base_wd
            )
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            all_preds, all_labels = [], []

            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward + loss + preds
                if detection:
                    out, _ = self.model(
                        batch.x, batch.edge_index, batch.batch, task="detection"
                    )
                    out = out.view(-1)
                    loss = F.binary_cross_entropy_with_logits(
                        out, batch.y.float()
                    )
                    preds = (torch.sigmoid(out) > 0.5).cpu().numpy()
                    labels = batch.y.cpu().numpy()
                elif classification:
                    out, _ = self.model(
                        batch.x, batch.edge_index, batch.batch, task="classification"
                    )
                    loss = self.criterion(out, batch.y)  # Use FocalLoss
                    preds = out.argmax(dim=1).cpu().numpy()
                    labels = batch.y.cpu().numpy()
                elif early_reg:
                    out, _ = self.model(
                        batch.x, batch.edge_index, batch.batch, task="forecast_time"
                    )
                    out = out.view(-1)
                    loss = F.smooth_l1_loss(out, batch.y.float())
                    preds = out.detach().cpu().numpy()
                    labels = batch.y.cpu().numpy()
                else:  # early_clf
                    out, _ = self.model(
                        batch.x, batch.edge_index, batch.batch, task="forecast_label"
                    )
                    loss = self.criterion(out, batch.y)  # Use FocalLoss
                    preds = out.argmax(dim=1).cpu().numpy()
                    labels = batch.y.cpu().numpy()

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                all_preds.extend(preds)
                all_labels.extend(labels)

            avg_loss = epoch_loss / len(train_loader)

            # Metrics & scheduler
            if early_reg:
                inv_preds = np.expm1(
                    self.regression_scaler.inverse_transform(
                        np.array(all_preds).reshape(-1, 1)
                    )
                ).flatten()
                inv_true = np.expm1(
                    self.regression_scaler.inverse_transform(
                        np.array(all_labels).reshape(-1, 1)
                    )
                ).flatten()
                train_r2 = self.safe_r2_score(inv_true, inv_preds)
                train_rmse = np.sqrt(mean_squared_error(inv_true, inv_preds))
                self.model.eval()
                val_preds, val_true = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        out, _ = self.model(
                            batch.x, batch.edge_index, batch.batch, task="forecast_time"
                        )
                        out = out.view(-1).cpu().numpy()
                        p = np.expm1(
                            self.regression_scaler.inverse_transform(
                                out.reshape(-1, 1)
                            )
                        ).flatten()
                        t = np.expm1(
                            self.regression_scaler.inverse_transform(
                                batch.y.cpu().numpy().reshape(-1, 1)
                            )
                        ).flatten()
                        val_preds.extend(p)
                        val_true.extend(t)
                val_r2 = self.safe_r2_score(val_true, val_preds)
                val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))
                print(
                    f"Epoch {epoch}/{self.num_epochs} - "
                    f"Loss: {avg_loss:.4f} | "
                    f"Train R2: {train_r2:.4f}, RMSE: {train_rmse:.2f} | "
                    f"Val   R2: {val_r2:.4f}, RMSE: {val_rmse:.2f}"
                )
                self.scheduler.step(val_r2)
            else:
                acc = accuracy_score(all_labels, all_labels)
                print(
                    f"Epoch {epoch}/{self.num_epochs} - "
                    f"Loss: {avg_loss:.4f} | Acc: {acc:.4f}"
                )
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(acc)
                else:
                    self.scheduler.step()

            # Checkpointing
            tag = (
                "detection" if detection else
                "classification" if classification else
                "early_reg" if early_reg else
                "early_clf"
            )
            ckpt_dir = f"models/checkpoints/{tag}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                self.model.state_dict(),
                os.path.join(ckpt_dir, f"{tag}_epoch_{epoch}.pth")
            )

        print("Training complete!")
        return all_preds


#version 2


import os
import warnings
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.model import MultiTaskGCN
from data.scripts.adjacancy_matrics import AdjacencyMatrixProcessor


# =========================
# Task-aware feature builder
# =========================
class FeatureBuilder:
    """
    Builds task-specific per-node features from RFFT (log-amplitude) bins.
    Assumes 200 Hz, 1 s => 100 bins (DC dropped). Adds compact spectral summaries.
    DC tasks:     [100 bins] + [bandpowers, relative bandpowers, ratios, entropy] = 100 + 14 = 114
    RC tasks:     mean⊕std over time => 200, plus the same compact summary = 214
    """
    BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 70)}

    def __init__(self, fs=200, rfft_bins=100):
        self.fs = fs
        self.rfft_bins = rfft_bins  # 1..100 Hz after DC drop

    def _band_indices(self, lo, hi):
        lo = max(1, int(np.floor(lo)))
        hi = min(self.rfft_bins, int(np.floor(hi)))
        return slice(lo - 1, hi)  # zero-based for numpy

    def _bandpowers(self, x_bins):
        # x_bins: (..., 100) log amplitude → convert back to magnitude before band sums
        p_lin = np.exp(x_bins)
        out = {}
        for k, (lo, hi) in self.BANDS.items():
            s = self._band_indices(lo, hi)
            out[f"bp_{k}"] = p_lin[..., s].sum(axis=-1, keepdims=True)
        out["bp_total"] = p_lin.sum(axis=-1, keepdims=True)
        # relative
        for k in list(self.BANDS.keys()):
            out[f"rp_{k}"] = out[f"bp_{k}"] / np.maximum(out["bp_total"], 1e-12)
        # simple ratios
        out["ratio_alpha_theta"] = out["bp_alpha"] / np.maximum(out["bp_theta"], 1e-12)
        out["ratio_beta_alpha"] = out["bp_beta"] / np.maximum(out["bp_alpha"], 1e-12)
        return out  # each (..., 1)

    def _spectral_entropy(self, x_bins):
        p = np.exp(x_bins)
        p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
        ent = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=-1, keepdims=True)
        return ent

    @staticmethod
    def expected_out_dim_dc(in_bins=100):
        # 100 bins + 5 bp + 1 total + 5 rel + 2 ratios + 1 entropy = 114
        return in_bins + 14

    @staticmethod
    def expected_out_dim_rc(in_bins=100):
        # mean (100) + std (100) + 14 summary = 214
        return 2 * in_bins + 14

    def build(self, X, mode: str):
        """
        X: (N, T, V, F). For DC, F should be 100 log-RFFT bins (DC dropped).
           For RC, F should be 100 per timestep; we compute mean⊕std over T here.
        Returns:
            DC modes: (N, T, V, 114)
            RC modes: (N, 1, V, 214)   (T collapsed)
        """
        assert X.ndim == 4, f"Expected 4D (N,T,V,F), got {X.shape}"
        N, T, V, F = X.shape

        if mode in ("detection", "classification"):
            if F != 100:
                warnings.warn(f"[FeatureBuilder] Expected F=100 RFFT bins for DC; got F={F}. Proceeding.")
            bp = self._bandpowers(X)                      # dict of (N,T,V,1)
            ent = self._spectral_entropy(X)               # (N,T,V,1)
            summary = np.concatenate([*bp.values(), ent], axis=-1)  # (N,T,V,14)
            return np.concatenate([X, summary], axis=-1)            # (N,T,V,114)

        else:  # early_reg / early_clf
            if F != 100:
                warnings.warn(f"[FeatureBuilder] Expected F=100 RFFT bins for RC; got F={F}. Proceeding.")
            X_mean = X.mean(axis=1)                       # (N,V,100)
            X_std = X.std(axis=1)                         # (N,V,100)
            bp = self._bandpowers(X_mean)                 # dict of (N,V,1)
            ent = self._spectral_entropy(X_mean)          # (N,V,1)
            summary = np.concatenate([*bp.values(), ent], axis=-1)  # (N,V,14)
            feats = np.concatenate([X_mean, X_std, summary], axis=-1)  # (N,V,214)
            return feats[:, None, ...]                    # (N,1,V,214)


# ---------------------------
# Feature name helper for CSV
# ---------------------------
def make_feature_names(is_rc: bool, rfft_bins: int = 100):
    """
    Returns a list of per-node feature names in the same order used by FeatureBuilder.
    DC  -> 100 RFFT bins + 14 summary = 114
    RC  -> 100 mean + 100 std + 14 summary = 214
    """
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    bp = [f"bp_{b}" for b in bands] + ["bp_total"]
    rp = [f"rp_{b}" for b in bands]
    ratios = ["ratio_alpha_theta", "ratio_beta_alpha"]
    summary = bp + rp + ratios + ["spec_entropy"]

    if not is_rc:
        core = [f"rfft_bin_{i}" for i in range(1, rfft_bins + 1)]
        return core + summary
    else:
        mean_feats = [f"mean_bin_{i}" for i in range(1, rfft_bins + 1)]
        std_feats = [f"std_bin_{i}" for i in range(1, rfft_bins + 1)]
        return mean_feats + std_feats + summary


# ===========================
# Adaptive, head-aware losses
# ===========================
class AdaptiveHeadLoss(torch.nn.Module):
    def __init__(self, smoothing=0.05, focal_gamma=2.0):
        super().__init__()
        self.smoothing = smoothing
        self.focal_gamma = focal_gamma
        self.reg_running_mad = None  # scale-invariant normalization

    def _label_smooth_ce(self, logits, targets, num_classes, smoothing):
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smoothing / max(1, (num_classes - 1)))
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
        logp = torch.log_softmax(logits, dim=1)
        return -(true_dist * logp).sum(dim=1).mean()

    def _focal_from_ce(self, ce_per_sample, gamma):
        pt = torch.exp(-ce_per_sample)
        return ((1 - pt) ** gamma * ce_per_sample).mean()

    def detection_loss(self, logits, y):
        # BCE with adaptive pos_weight from batch balance
        p = y.mean().clamp_min(1e-6)
        pos_weight = (1 - p) / p
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

    def classification_loss(self, logits, y, num_classes):
        with torch.no_grad():
            ce_per = F.cross_entropy(logits, y, reduction='none')
        ls_ce = self._label_smooth_ce(logits, y, num_classes, self.smoothing)
        focal = self._focal_from_ce(ce_per, self.focal_gamma)
        return 0.5 * ls_ce + 0.5 * focal

    def regression_loss(self, pred, y):
        # SmoothL1 with adaptive delta (IQR/2) and scale-invariant division by running MAD
        with torch.no_grad():
            if y.numel() >= 4:
                q1, q3 = torch.quantile(y, 0.25), torch.quantile(y, 0.75)
                delta = torch.clamp((q3 - q1) / 2.0, min=0.1)
            else:
                delta = torch.tensor(1.0, device=y.device)
            mad = torch.median(torch.abs(y - torch.median(y)))
            if self.reg_running_mad is None:
                self.reg_running_mad = mad
            else:
                self.reg_running_mad = 0.9 * self.reg_running_mad + 0.1 * mad
            scale = torch.clamp(self.reg_running_mad, min=1e-3)
        loss = F.smooth_l1_loss(pred, y, beta=float(delta))
        return loss / scale


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

        # Determine expected input dim per task *before* model init (based on FeatureBuilder)
        if RC:
            in_dim = FeatureBuilder.expected_out_dim_rc(num_features)   # 214 if num_features=100
        else:
            in_dim = FeatureBuilder.expected_out_dim_dc(num_features)   # 114 if num_features=100

        self.model = MultiTaskGCN(
            hidden_dim=num_hiddens,
            in_dim=in_dim,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)

        self.adaptive_loss = AdaptiveHeadLoss(smoothing=0.05, focal_gamma=2.0)

        # ----- Graph topology & weights -----
        self.num_nodes = 19
        base_edge_index = torch.tril_indices(self.num_nodes, self.num_nodes, offset=-1)
        expected_pairs = self.num_nodes * (self.num_nodes - 1) // 2  # 171 for n=19

        adj_proc = AdjacencyMatrixProcessor(pooled_results, data_directory=r"G:\tuh_data\train")

        # NOTE: get raw weights as returned (do NOT average before we know the shape)
        raw = adj_proc.compute_all_edge_weights(DC=not RC, RC=RC)  # shape: [B,E] or [] or [B,190] etc.

        def _coerce_to_offdiag_lower_tri_1d(w: torch.Tensor, n: int) -> torch.Tensor | None:
            """
            Accepts:
            - [E] where E ∈ {n(n-1)/2, n(n+1)/2, n*n}
            - [B,E] or [B,n(n+1)/2] or [B,n*n]
            Returns:
            - [n(n-1)/2] off-diagonal lower-tri vector, averaged across batch if needed
            """
            if w.ndim == 1:
                num = w.numel()
                if num == n * (n - 1) // 2:
                    return w  # already offdiag lower-tri
                if num == n * (n + 1) // 2:
                    # drop diagonal
                    r0, c0 = torch.tril_indices(n, n, offset=0)
                    mask = (r0 != c0)
                    return w[mask]
                if num == n * n:
                    W = w.view(n, n)
                    r, c = torch.tril_indices(n, n, offset=-1)
                    return W[r, c]
                return None

            # Average any leading batch dims → [E]
            lead_dims = tuple(range(0, w.ndim - 1))
            w1 = w.mean(dim=lead_dims)

            # Recurse on 1D
            return _coerce_to_offdiag_lower_tri_1d(w1, n)

        # ---- Debug: show raw type/shape
        print("[DEBUG] type(raw):", type(raw))
        if isinstance(raw, torch.Tensor):
            print("[DEBUG] raw.shape:", tuple(raw.shape), "dtype:", raw.dtype)

        # ---- Build final weights
        if isinstance(raw, torch.Tensor) and raw.numel() > 0:
            weights = _coerce_to_offdiag_lower_tri_1d(raw, self.num_nodes)
            if weights is None:
                print("⚠️ Could not coerce edge weights; using uniform.")
                weights = torch.ones(expected_pairs, dtype=torch.float32)
            else:
                # Safety: exact length
                if weights.numel() != expected_pairs:
                    print(f"⚠️ Edge weight length {weights.numel()} != expected {expected_pairs}. "
                        "Truncating/padding for safety.")
                    if weights.numel() > expected_pairs:
                        weights = weights[:expected_pairs]
                    else:
                        weights = torch.cat([weights, torch.zeros(expected_pairs - weights.numel(), dtype=weights.dtype)])
        else:
            print("⚠️ No valid edge weights found. Using default uniform weights.")
            weights = torch.ones(expected_pairs, dtype=torch.float32)

        # ---- Undirected graph & device move
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
    def safe_r2_score(y_true, y_pred, verbose=False):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) == 0:
            if verbose:
                warnings.warn("No valid data after filtering NaNs/infs")
            return 0.0

        if verbose:
            print(f"Number of valid samples: {len(y_true)}")
            print(f"y_true stats: min={y_true.min():.4f}, max={y_true.max():.4f}, mean={y_true.mean():.4f}, std={y_true.std():.4f}")
            print(f"y_pred stats: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")

        if np.var(y_true) == 0:
            if verbose:
                print("Warning: y_true has zero variance. R² is undefined.")
            return 0.0

        r2 = r2_score(y_true, y_pred)
        return max(r2, 0.0)

    @staticmethod
    def _oversample(X, Y, factor=3.0):
        n = X.shape[0]
        if n <= 1 or factor <= 1.0:
            return X, Y
        dup = max(1, int(n * (factor - 1.0)))
        idx = np.random.choice(n, size=dup, replace=True)
        return np.concatenate([X, X[idx]], axis=0), np.concatenate([Y, Y[idx]], axis=0)

    def create_graph_batches(self, X, Y, task: str = None):
        """
        X: (N, V, F)  — per-graph node features (time already pooled)
        Y: (N,)
        """
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
                y = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
            else:
                y = torch.as_tensor(y_val, dtype=torch.long, device=self.device)

            data = Data(x=x, edge_index=self.edge_index, y=y)
            if self.edge_weight is not None:
                data.edge_weight = self.edge_weight
            data_list.append(data)

        return DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

    # -------------
    # Interpretability
    # -------------
    def visualize_input_graph(self, save_path="graph_input.png", threshold=0.0):
        import networkx as nx
        import matplotlib.pyplot as plt
        ei = self.edge_index.detach().cpu().numpy()
        ew = self.edge_weight.detach().cpu().numpy() if self.edge_weight is not None else np.ones(ei.shape[1])
        G = nx.Graph()
        for (u, v), w in zip(ei.T, ew):
            if w >= threshold:
                G.add_edge(int(u), int(v), weight=float(w))
        pos = nx.spring_layout(G, seed=42)
        widths = [G[u][v]['weight'] / (ew.max() + 1e-8) * 3.0 for u, v in G.edges()]
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, width=widths, node_size=420)
        plt.title("Input EEG Graph (edge width ∝ weight)")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout(); plt.savefig(save_path); plt.close()
        print(f"[viz] saved {save_path}")

    def explain_one_batch(self, loader, task="classification", save_path="explain.png", csv_prefix=None):
        """
        Saves: a PNG plot, plus CSVs:
          - {csv_prefix}_feature_importance.csv
          - {csv_prefix}_edge_importance.csv
          - {csv_prefix}_node_importance.csv
        Uses new PyG API: no epochs in __init__; pass epochs to explain_graph().
        """
        from torch_geometric.explain import GNNExplainer
        import matplotlib.pyplot as plt
        import networkx as nx
        import csv
        import os
        
        self.model.eval()
        batch = next(iter(loader))
        batch = batch.to(self.device)

        # Decide return_type per task/head
        # - 'regression' for scalar heads (detection logit, forecast_time)
        # - 'classification' for multi-class heads (classification, forecast_label)
        if task in ["detection", "early_reg", "forecast_time"]:
            return_type = "regression"
        else:
            return_type = "classification"

        def forward_for_explain(x, edge_index):
            return self.model(x, edge_index, batch, task=task, edge_weight=getattr(batch, "edge_weight", None))

        # New API: no epochs in __init__
        explainer = GNNExplainer(self.model, return_type=return_type)

        # Pass epochs here
        node_feat_mask, edge_mask = explainer.explain_graph(
            x=batch.x, edge_index=batch.edge_index, forward=forward_for_explain, epochs=200
        )

        # ---- Plot edge importance graph
        ei = batch.edge_index.detach().cpu().numpy()
        em = edge_mask.detach().cpu().numpy()
        G = nx.Graph()
        for (u, v), w in zip(ei.T, em):
            G.add_edge(int(u), int(v), weight=float(w))
        pos = nx.spring_layout(G, seed=7)
        widths = [G[u][v]['weight'] / (em.max() + 1e-8) * 3.0 for u, v in G.edges()]
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, width=widths, node_size=420)
        plt.title(f"GNNExplainer — edge importance ({task})")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout(); plt.savefig(save_path); plt.close()
        print(f"[explain] plot saved {save_path}")

        # ---- CSV exports
        prefix = csv_prefix or os.path.splitext(save_path)[0]

        # 1) Feature importance (global mask over node features)
        feat_mask = node_feat_mask.detach().cpu().numpy().reshape(-1)
        is_rc = (task in ["early_reg", "forecast_time", "early_clf", "forecast_label"])
        feat_names = make_feature_names(is_rc=is_rc, rfft_bins=100)
        L = min(len(feat_names), len(feat_mask))
        rows = list(zip(range(L), feat_names[:L], feat_mask[:L]))
        with open(f"{prefix}_feature_importance.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["feature_index", "feature_name", "importance"])
            w.writerows(rows)
        print(f"[explain] feature importances -> {prefix}_feature_importance.csv")

        # 2) Edge importance
        with open(f"{prefix}_edge_importance.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["u", "v", "importance"])
            for (u, v), wgt in zip(ei.T, em):
                w.writerow([int(u), int(v), float(wgt)])
        print(f"[explain] edge importances -> {prefix}_edge_importance.csv")

        # 3) Node importance (sum of incident edge importances as a proxy)
        node_scores = np.zeros(self.num_nodes, dtype=float)
        for (u, v), wgt in zip(ei.T, em):
            node_scores[int(u)] += float(wgt)
            node_scores[int(v)] += float(wgt)
        with open(f"{prefix}_node_importance.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["node_index", "importance_sum_incident_edges"])
            for i, s in enumerate(node_scores):
                w.writerow([i, s])
        print(f"[explain] node importances -> {prefix}_node_importance.csv")

    # -------------
    # Training
    # -------------
    def train(
        self,
        X,
        Y,
        detection: bool = False,
        classification: bool = False,
        early_reg: bool = False,
        early_clf: bool = False,
        explain_after: bool = False,          # NEW
        explain_path: str | None = None       # NEW
    ):
        os.makedirs("models/checkpoints", exist_ok=True)

        X = np.array(X)  # (N,T,V,F)
        Y = np.array(Y)

        # ---- Task-aware feature categorization ----
        fb = FeatureBuilder(fs=200, rfft_bins=100)
        task = "detection" if detection else "classification" if classification else "early_reg" if early_reg else "early_clf"
        X_feat = fb.build(X, mode=task)        # DC: (N,T,V,114) ; RC: (N,1,V,214)

        # Reduce time to per-graph features
        X_graph = X_feat.mean(axis=1)          # (N,V,Fout)

        # ---- Split ----
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_graph, Y, test_size=0.2, random_state=42, stratify=Y if not early_reg else None
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

        # ---- Oversample (train only) for classification-like tasks ----
        X_tr, Y_tr = X_train_sc, Y_train_sc
        X_va, Y_va = X_val_sc,   Y_val_sc
        if detection or classification or early_clf:
            X_tr, Y_tr = self._oversample(X_tr, Y_tr, factor=3.0)

        # ---- Build loaders (no augmentation inside) ----
        train_loader = self.create_graph_batches(X_tr, Y_tr, task=task)
        val_loader   = self.create_graph_batches(X_va, Y_va, task=task)

        print(f"=================== {task.capitalize()} ===================")

        # ---- Optim/sched ----
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001 if early_reg else self.base_lr,
            weight_decay=1e-5 if early_reg else self.base_wd
        )
        self.scheduler = (
            ReduceLROnPlateau(self.optimizer, mode="max", factor=0.3, patience=15, min_lr=1e-7)
            if early_reg else
            StepLR(self.optimizer, step_size=10, gamma=0.5)
        )

        best_val_metric = -float("inf")
        patience_counter = 0
        early_stopping_patience = 100

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
                        out = self.model(batch.x, batch.edge_index, batch, task="forecast_time", edge_weight=ew).squeeze(-1).cpu().numpy()
                        preds = np.expm1(self.regression_scaler.inverse_transform(out.reshape(-1, 1))).flatten()
                    else:  # early_clf
                        out = self.model(batch.x, batch.edge_index, batch, task="forecast_label", edge_weight=ew)
                        preds = out.argmax(dim=1).cpu().numpy()

                    val_preds.extend(preds)
                    val_labels.extend(batch.y.detach().cpu().numpy())

            # ---- Metrics ----
            if early_reg:
                inv_preds = np.expm1(self.regression_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1))).flatten()
                inv_true  = np.expm1(self.regression_scaler.inverse_transform(np.array(all_labels).reshape(-1, 1))).flatten()
                verbose_debug = (epoch == 1)
                train_r2  = self.safe_r2_score(inv_true, inv_preds, verbose=verbose_debug)
                train_rmse = np.sqrt(mean_squared_error(inv_true, inv_preds))
                val_r2   = self.safe_r2_score(val_labels, val_preds, verbose=verbose_debug)
                val_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
                val_metric = val_r2
                print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} | "
                      f"Train R2: {train_r2:.4f}, RMSE: {train_rmse:.2f} | "
                      f"Val R2: {val_r2:.4f}, RMSE: {val_rmse:.2f}")
            else:
                train_acc = accuracy_score(all_labels, all_preds)
                val_acc   = accuracy_score(val_labels, val_preds)
                val_metric = val_acc
                print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

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

            # epoch checkpoint (weights only)
            ckpt_dir = f"models/checkpoints/{tag}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, f"{tag}_epoch_{epoch}.pth"))

        # Post-train explanation on the validation batch (optional)
        if explain_after:
            try:
                csv_prefix = os.path.splitext(explain_path or f"explain_{task}.png")[0]
                self.explain_one_batch(val_loader, task=task, save_path=explain_path or f"explain_{task}.png", csv_prefix=csv_prefix)
            except Exception as e:
                print(f"[explain] skipped due to error: {e}")

        print("Training complete!")
        return all_preds


