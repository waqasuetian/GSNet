# import torch
# from models.trainer import Trainer
# from torch_geometric.data import Data, DataLoader
# from data.scripts.data_generation import EEGDataProcessor
# from data.scripts.data_loader import EEGProcessor
# from data.scripts.data_loader_preictal import EEGProcessorPreictal 
# from data.scripts.pooling import EEGPooler
# from data.scripts.adjacancy_matrics import AdjacencyMatrixProcessor
# from data.scripts.seizure_dataset import SeizureDataset
# from data.scripts.preictel_datageneration import preictal_dataLoader


# # 📌 Other Hyperparameters
# num_hiddens = 128
# dropout = 0.4
# num_heads = 8
# learning_rate = 0.001
# batch_size = 16
# num_epochs = 30
# num_features = 100

# def run_pipeline(num_classes, detection, classification, early_reg, early_label):
#     directory = r'G:\tuh_data\train'

#     if detection:
#         processor = EEGProcessor(directory, resampled_freq = 200, time_step_size=1, apply_fft=True)
#         processor.convert_edf_to_h5()
#         results = processor.process_directory()
#         pooler = EEGPooler(results, target_time_points=100)
#         pooled_results = pooler.apply_pooling_DC()
#         data_processor = EEGDataProcessor(pooled_results, detection=True, classification=False)
#         X, Y = data_processor.process_fixed_length_clips()
#     elif classification:
#         processor = EEGProcessor(directory, resampled_freq = 200, time_step_size=1, apply_fft=True)
#         processor.convert_edf_to_h5()
#         results = processor.process_directory()
#         pooler = EEGPooler(results, target_time_points=100)
#         pooled_results = pooler.apply_pooling_DC()
#         data_processor = EEGDataProcessor(pooled_results, detection=False, classification=True)
#         X, Y = data_processor.process_fixed_length_clips()
#     elif early_reg: 
#         processor = EEGProcessorPreictal(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
#         processor.convert_edf_to_h5()
#         results = processor.process_directory()
#         pooler = EEGPooler(results, target_time_points=100)
#         pooled_results = pooler.apply_pooling_RC()
#         data_processor = preictal_dataLoader(pooled_results, early_reg=True, early_label=False)
#         X, Y = data_processor.get_data()
#         print('Waqas')
#         print(Y)
#     elif early_label:
#         processor = EEGProcessorPreictal(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
#         processor.convert_edf_to_h5()
#         results = processor.process_directory()
#         pooler = EEGPooler(results, target_time_points=100)
#         pooled_results = pooler.apply_pooling_RC()
#         data_processor = preictal_dataLoader(pooled_results, early_reg=False, early_label=True)
#         X, Y = data_processor.get_data()
#     else:
#         print("Neither detection, classification, early_reg nor early_label selected...")
    
#     # Compute adjacency matrix
#     data_dir = r'G:\tuh_data\train'
#     adjacency_processor = AdjacencyMatrixProcessor(pooled_results, data_directory=data_dir)
#     if detection or classification:
#         edge_weights_tensor = adjacency_processor.compute_all_edge_weights(DC=True, RC=False)
#     elif early_reg or early_label:
#         edge_weights_tensor = adjacency_processor.compute_all_edge_weights(DC=False, RC=True)

#     # Edge Index for Graph Representation
#     num_nodes = 19
#     edge_index = torch.tril_indices(num_nodes, num_nodes, offset=-1)


#     if detection == True:
#         trainer = Trainer(num_features, num_hiddens, num_classes, dropout, num_heads, learning_rate, batch_size, num_epochs, pooled_results, DC=True, RC=False)
#         return trainer.train(X, Y, detection = True, classification = False, early_reg=False, early_clf=False)
#     elif classification == True: 
#         trainer = Trainer(num_features, num_hiddens, num_classes, dropout, num_heads, learning_rate, batch_size, num_epochs, pooled_results, DC=True, RC=False)
#         return trainer.train(X, Y, detection = False, classification = True, early_reg=False, early_clf=False)
#     elif early_reg == True: 
#         trainer = Trainer(num_features, num_hiddens, num_classes, dropout, num_heads, learning_rate, batch_size, num_epochs, pooled_results, DC=False, RC=True)
#         return trainer.train(X, Y, detection = False, classification = False, early_reg=True, early_clf=False)
#     elif early_label == True: 
#         trainer = Trainer(num_features, num_hiddens, num_classes, dropout, num_heads, learning_rate, batch_size, num_epochs, pooled_results, DC=False, RC=True)
#         return trainer.train(X, Y, detection = False, classification = False, early_reg=False, early_clf=True)
#     else:
#         pass

# num_classes = 7
# # Train Detection Head
# run_pipeline(num_classes=num_classes, detection=True, classification=False, early_reg=False, early_label=False)

# # Train Classification Head
# run_pipeline(num_classes=num_classes, detection=False, classification=True, early_reg=False, early_label=False)

# # Train Early Regression Head
# run_pipeline(num_classes=num_classes, detection=False, classification=False, early_reg=True, early_label=False)

# # Train Early Classification Head
# run_pipeline(num_classes=num_classes, detection=False, classification=False, early_reg=False, early_label=True)






# """
# I WANT to use this pipeline for two purposes, first of all this pipeline will be called for detection. In this case num of 
# class will be two, and the argument in EEGdataprocessor class will be this detection= True, classification=False
# By seeing the results of detection task. If the model returns bckg/0 then Passed else if the model retturs 1 then  
# num of classes will be 7, and the argument in EEGdataprocessor class will be this detection= False, classification=True

# """


import torch
import numpy as np
from models.trainer import Trainer
from data.scripts.data_generation import EEGDataProcessor
from data.scripts.data_loader import EEGProcessor
from data.scripts.data_loader_preictal import EEGProcessorPreictal
from data.scripts.pooling import EEGPooler
from data.scripts.preictel_datageneration import preictal_dataLoader


def run_pipeline(
    num_classes: int,
    detection: bool = False,
    classification: bool = False,
    early_reg: bool = False,
    early_label: bool = False,
    directory: str = r'G:\tuh_data\train',
    num_features: int = 100,
    num_hiddens: int = 128,
    dropout: float = 0.4,
    num_heads: int = 8,
    learning_rate: float = 0.001,
    batch_size: int = 16,
    num_epochs: int = 30
):
    """
    Runs the end-to-end pipeline for EEG-based tasks.
    """
    # 1️⃣ Data Loading & Preprocessing
    if detection:
        processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
        processor.convert_edf_to_h5()
        results = processor.process_directory()
        pooler = EEGPooler(results, target_time_points=100)
        pooled_results = pooler.apply_pooling_DC()
        data_processor = EEGDataProcessor(pooled_results, detection=True, classification=False)
        X, Y = data_processor.process_fixed_length_clips()

    elif classification:
        processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
        processor.convert_edf_to_h5()
        results = processor.process_directory()
        pooler = EEGPooler(results, target_time_points=100)
        pooled_results = pooler.apply_pooling_DC()
        data_processor = EEGDataProcessor(pooled_results, detection=False, classification=True)
        X, Y = data_processor.process_fixed_length_clips()

    elif early_reg:
        processor = EEGProcessorPreictal(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
        processor.convert_edf_to_h5()
        results = processor.process_directory()
        pooler = EEGPooler(results, target_time_points=100)
        pooled_results = pooler.apply_pooling_RC()
        data_processor = preictal_dataLoader(pooled_results, early_reg=True, early_label=False)
        X, Y = data_processor.get_data()

    elif early_label:
        processor = EEGProcessorPreictal(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
        processor.convert_edf_to_h5()
        results = processor.process_directory()
        pooler = EEGPooler(results, target_time_points=100)
        pooled_results = pooler.apply_pooling_RC()
        data_processor = preictal_dataLoader(pooled_results, early_reg=False, early_label=True)
        X, Y = data_processor.get_data()

    else:
        raise ValueError("One of detection, classification, early_reg, or early_label must be True.")

    # 2️⃣ Sanity-check and convert feature dimensions
    # Expecting X of shape (N, T, nodes, features) where features == num_features
    if isinstance(X, torch.Tensor):
        X_arr = X.cpu().numpy()
    elif isinstance(X, (list, np.ndarray)):
        X_arr = np.array(X)
    else:
        raise TypeError(f"X should be a Tensor, list, or ndarray; got {type(X)}.")

    if X_arr.ndim != 4:
        raise AssertionError(
            f"Expected X to have 4 dims (N, T, nodes, features), got {X_arr.shape}."
        )
    N, T, num_nodes, feat_dim = X_arr.shape
    if feat_dim != num_features:
        raise AssertionError(
            f"Expected last dim to be {num_features} features, but got {feat_dim} (shape {X_arr.shape})."
        )

    # 3️⃣ Initialize and run Trainer
    trainer = Trainer(
        num_features=num_features,
        num_hiddens=num_hiddens,
        num_classes=num_classes,
        dropout=dropout,
        num_heads=num_heads,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        pooled_results=pooled_results,
        DC=detection,
        RC=(early_reg or early_label)
    )

    # 4️⃣ Execute training
    return trainer.train(
        X_arr, Y,
        detection=detection,
        classification=classification,
        early_reg=early_reg,
        early_clf=early_label
    )


if __name__ == "__main__":
    num_classes = 7

    # Train Detection Head
    run_pipeline(
        num_classes=num_classes,
        detection=True
    )

    # Train Classification Head
    run_pipeline(
        num_classes=num_classes,
        classification=True
    )

    # Train Early Regression Head
    run_pipeline(
        num_classes=num_classes,
        early_reg=True
    )

    # Train Early Classification Head
    run_pipeline(
        num_classes=num_classes,
        early_label=True
    )
