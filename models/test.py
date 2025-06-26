

# def run_pipeline(num_classes, detection=True, classification=False):
#     directory = r'D:\PhD Research\Experiments\test'
#     processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
#     processor.convert_edf_to_h5()
#     results = processor.process_directory()

#     # Step 2: Apply Adaptive Pooling
#     pooler = EEGPooler(results, target_time_points=100)
#     pooled_results = pooler.apply_pooling()

#     # Processed EEG Data
#     data_processor = EEGDataProcessor(pooled_results, detection=detection, classification=classification)
#     X, Y = data_processor.process_fixed_length_clips()

#     # Compute adjacency matrix
#     data_dir = r'C:\Users\MyPC\AppData\Roaming\MobaXterm\slash\home\Documents\edf\train'
#     adjacency_processor = AdjacencyMatrixProcessor(pooled_results, data_directory=data_dir)
#     edge_weights_tensor = adjacency_processor.compute_all_edge_weights()

#     # Edge Index for Graph Representation
#     num_nodes = 19
#     edge_index = torch.tril_indices(num_nodes, num_nodes, offset=-1)

#     # Convert Data to PyG Format
#     def create_graph_batches(X, Y, batch_size):
#         data_list = []
#         for i in range(X.shape[0]):
#             x = torch.tensor(X[i].mean(dim=0), dtype=torch.float)
#             y = torch.tensor(Y[i], dtype=torch.long)
#             data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights_tensor, y=y)
#             data_list.append(data)
#         return DataLoader(data_list, batch_size=batch_size, shuffle=True)

#     train_loader = create_graph_batches(X, Y, batch_size)

#     # Train the Model
#     trainer = Trainer(num_features, num_hiddens, num_classes, dropout, num_heads, learning_rate, batch_size, num_epochs)
#     if detection == True and classification == False:
#         return trainer.train(X, Y, True, False)

#     else: 
#         return trainer.train(X, Y, False, True)
# # Step 1: Run Detection
# print("Running detection...")
# detection_result = run_pipeline(num_classes=2, detection=True, classification=False)

# # Step 2: If detected class is 1, proceed to classification
# if detection_result != 0:
#     print("Detected class 1, proceeding to classification...")
#     run_pipeline(num_classes=7, detection=False, classification=True)
# else:
#     print("Passed: No further classification required.")




# import torch
# import os
# import mne 
# from torch_geometric.data import Data, DataLoader
# from models.trainer import Trainer 
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import StepLR
# from models.model import GATModel
# from models.results import ResultsHandler
# from sklearn.metrics import accuracy_score
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, confusion_matrix
# from models.results import ResultsHandler
# from data.scripts.data_loader import EEGProcessor
# from data.scripts.pooling import EEGPooler
# from data.scripts.data_generation import EEGDataProcessor
#  # For handling EEG (EDF) files

# class EEGModelHandler:
#     def __init__(self, model, device='cpu'):
#         """
#         Initializes the EEGModelHandler with a model.
#         :param model: The PyTorch model to use for inference.
#         :param device: The device to run inference on ('cpu' or 'cuda').
#         """
#         num_nodes = 19
#         self.batch_size = 16
#         edge_index = torch.tril_indices(num_nodes, num_nodes, offset=-1)
#         edge_weights_tensor = torch.randn(166, 190).mean(dim=0)  # Replace with actual weights
#         expected_edge_size = (num_nodes * (num_nodes - 1)) // 2

#         if edge_weights_tensor.numel() > expected_edge_size:
#             edge_weights_tensor = edge_weights_tensor[:expected_edge_size]
#         elif edge_weights_tensor.numel() < expected_edge_size:
#             pad_size = expected_edge_size - edge_weights_tensor.numel()
#             edge_weights_tensor = torch.cat([edge_weights_tensor, torch.zeros(pad_size)])

#         edge_weight_matrix = torch.zeros((num_nodes, num_nodes))
#         xs, ys = torch.tril_indices(num_nodes, num_nodes, offset=-1)
#         edge_weight_matrix[xs, ys] = edge_weights_tensor
#         edge_weight_matrix[ys, xs] = edge_weights_tensor  # Symmetric adjacency matrix
#         self.edge_index = edge_index
#         self.edge_attr = edge_weights_tensor

#         self.model = model.to(device)
#         self.device = device
        
        


#     def create_graph_batches(self, X, Y):
#         data_list = []
#         for i in range(X.shape[0]):
#             x = torch.tensor(X[i].mean(dim=0), dtype=torch.float)
#             y = torch.tensor(Y[i], dtype=torch.long)
#             data = Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y)
#             data_list.append(data)
#         return DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

    
    
#     def infer(self,detection, classification):
#         # Load EEG data
#         #raw = mne.io.read_raw_edf(edf_path, preload=True)
#         #data = raw.get_data()
        
#         directory = r'C:\Users\MyPC\AppData\Roaming\MobaXterm\slash\home\Documents\edf\train'
#         processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
#         processor.convert_edf_to_h5()
#         results = processor.process_directory()

#         # Step 2: Apply Adaptive Pooling
#         pooler = EEGPooler(results, target_time_points=100)
#         pooled_results = pooler.apply_pooling()

#         # Processed EEG Data
#         data_processor = EEGDataProcessor(pooled_results, detection=detection, classification=classification)
#         X, Y = data_processor.process_fixed_length_clips()
#         # Process EEG data into graph batches (Implement this method as per training pipeline)
#         test_loader = self.create_graph_batches(X,Y)

        

#         # Load the appropriate model checkpoint
#         if detection and not classification:
#             checkpoint_dir = "models/checkpoints/detection_checkpoints"
#         elif classification and not detection:
#             checkpoint_dir = "models/checkpoints/classification_checkpoints"
#         else:
#             raise ValueError("Specify either detection or classification, not both.")
        
#         checkpoint_files = sorted(os.listdir(checkpoint_dir))
#         latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
#         self.model.load_state_dict(torch.load(latest_checkpoint, map_location=self.device))
#         self.model.eval()
        
#         all_preds = []
        
#         with torch.no_grad():
#             for batch in  test_loader:
#                 batch = batch.to(self.device)
#                 outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
#                 preds = outputs.argmax(dim=1).cpu().numpy()
#                 all_preds.extend(preds)
        
#         print("Predicted Labels:", all_preds)
#         return all_preds






# ======================================================================


import torch
from torch_geometric.data import Data
from data.scripts.preprocessors import EEGProcessor, EEGPooler, AdjacencyMatrixProcessor
from model import MultiTaskGCN  # make sure this points to your GCN model definition
import numpy as np
import os

# -------- Configuration -------- #
NUM_NODES = 19
HIDDEN_CHANNELS = 64
NUM_CLASSES = 4
FORECAST_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load Models -------- #
def load_model(task, path):
    model = MultiTaskGCN(HIDDEN_CHANNELS, num_features=NUM_NODES, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

detection_model = load_model("detection", "models/checkpoints/detection_checkpoints/detection_epoch_10.pth")
classification_model = load_model("classification", "models/checkpoints/classification_checkpoints/classification_epoch_10.pth")
early_reg_model = load_model("forecast_time", "models/checkpoints/early_regression_checkpoints/early_reg_epoch_10.pth")
early_clf_model = load_model("forecast_label", "models/checkpoints/early_classification_checkpoints/early_clf_epoch_10.pth")

# -------- Preprocessing Function -------- #
def preprocess_clip(eeg_clip_path):
    # processor = EEGProcessor(resampled_freq=200, time_step_size=1, apply_fft=True)
    # clip = processor.load_and_preprocess_clip(eeg_clip_path)  # Custom method to handle single clip
    # pooler = EEGPooler({0: clip}, target_time_points=100)
    # pooled_clip = pooler.apply_pooling()[0]  # Get pooled array

    # Mean across time dimension for graph node features
    # x = torch.tensor(pooled_clip.mean(axis=0), dtype=torch.float).to(DEVICE)

    # Graph edges and edge attributes
    # edge_index = torch.tril_indices(NUM_NODES, NUM_NODES, offset=-1).to(DEVICE)
    # adj_processor = AdjacencyMatrixProcessor({0: clip})
    # edge_attr = adj_processor.compute_all_edge_weights().mean(dim=0).to(DEVICE)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# -------- Inference Function -------- #
def run_inference(eeg_clip_path):
    # graph_data = preprocess_clip(eeg_clip_path)
    # graph_data = graph_data.to(DEVICE)

    # Step 1: Run Detection
    detection_output = detection_model(graph_data.x, graph_data.edge_index, task="detection")
    seizure_detected = torch.sigmoid(detection_output.view(-1)).item() > 0.5
    print(f"Seizure Detected: {seizure_detected}")

    if seizure_detected:
        # Step 2A: Run Classification
        class_output = classification_model(graph_data.x, graph_data.edge_index, task="classification")
        predicted_class = class_output.argmax(dim=1).item()
        print(f"Seizure Class: {predicted_class}")
    else:
        # Step 2B: Run Early Forecasting
        reg_output = early_reg_model(graph_data.x, graph_data.edge_index, task="forecast_time")
        forecast_time = reg_output.view(-1).item()

        label_output = early_clf_model(graph_data.x, graph_data.edge_index, task="forecast_label")
        forecast_label = label_output.argmax(dim=1).item()

        print(f"Forecast Time Until Seizure: {forecast_time:.2f} seconds")
        print(f"Forecasted Seizure Type: {forecast_label}")

# -------- Run -------- #
if __name__ == "__main__":
    eeg_clip_file = r"E:\tuh_data\test_clip.edf"  # Change this path to your test clip
    run_inference(eeg_clip_file)
