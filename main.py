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




#version 2


## with validation scores

import os
import sys
import logging
from typing import Tuple, Union

import torch
import numpy as np

from models.trainer import Trainer
from data.scripts.data_generation import EEGDataProcessor
from data.scripts.data_loader import EEGProcessor
from data.scripts.data_loader_preictal import EEGProcessorPreictal
from data.scripts.pooling import EEGPooler
from data.scripts.preictel_datageneration import preictal_dataLoader  # fixed spelling

ArrayLike = Union[np.ndarray, torch.Tensor, list]

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)

def _check_not_empty(pooled_results: dict, stage: str):
    if not pooled_results:
        raise RuntimeError(f"{stage}: pooled_results is empty. "
                           f"Check directory path, EDF/H5 availability, and preprocessing filters.")

def _inspect_labels_rc(pooled_results: dict) -> Tuple[int, list]:
    try:
        uniq = sorted({lab for _, (_, labs, *_rest) in pooled_results.items() for lab in labs})
    except Exception:
        uniq = []
    total = sum(len(v[0]) for v in pooled_results.values()) if pooled_results else 0
    return total, uniq

def run_pipeline(
    num_classes: int,
    detection: bool = False,
    classification: bool = False,
    early_reg: bool = False,
    early_label: bool = False,
    directory: str = r'G:\tuh_data\train',
    num_features: int = 100,   # matches RFFT(200Hz,1s) after DC drop
    num_hiddens: int = 100,
    dropout: float = 0.5,
    num_heads: int = 8,
    learning_rate: float = 0.005,
    batch_size: int = 32,
    num_epochs: int = 200
):
    """
    Runs the end-to-end pipeline for EEG-based tasks.
    Expects X to be shaped (N, T, nodes, features) with features == num_features.
    """

    if not (detection or classification or early_reg or early_label):
        raise ValueError("One of detection, classification, early_reg, or early_label must be True.")

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Data directory does not exist: {directory}")

    logging.info("Starting pipeline | modes: "
                 f"detection={detection}, classification={classification}, "
                 f"early_reg={early_reg}, early_label={early_label}")

    # -------------------------------
    # 1) Data Loading & Preprocessing
    # -------------------------------
    if detection or classification:
        # DC branch
        logging.info("Loading DC data (detection/classification)…")
        processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
        processor.convert_edf_to_h5()  # consider caching/skip-existing in your implementation
        results = processor.process_directory()
        pooler = EEGPooler(results, target_time_points=100)
        pooled_results = pooler.apply_pooling_DC()
        _check_not_empty(pooled_results, "DC pooling")

        data_processor = EEGDataProcessor(
            pooled_results,
            detection=detection,
            classification=classification
        )
        X, Y = data_processor.process_fixed_length_clips()

    else:
        # RC branch (early_reg or early_label)
        logging.info("Loading RC data (early tasks)…")
        processor = EEGProcessorPreictal(
            directory,
            resampled_freq=200,
            time_step_size=1,     # -> window_sec=1.0
            apply_fft=True,       # -> feature_mode="rfft"
            overlap=0.5,
            min_channels_per_event=1,
            topk_events_by_duration=10,
            pad_short_segments=True,
            min_short_frac=0.30
        )
        processor.convert_edf_to_h5()
        results = processor.process_directory()
        pooler = EEGPooler(results, target_time_points=100)
        pooled_results = pooler.apply_pooling_RC()
        _check_not_empty(pooled_results, "RC pooling")

        total_clips, uniq_labels = _inspect_labels_rc(pooled_results)
        logging.info(f"[RC] pooled clips={total_clips}, unique labels={uniq_labels}")

        data_processor = preictal_dataLoader(
            pooled_results,
            early_reg=early_reg, early_label=early_label,
            allow_intermediate_labels={"artf"},
            max_gap_between_bckg_and_ictal_sec=2.0,
            min_preictal_clip_sec=0.0
        )
        X, Y = data_processor.get_data()

    # -------------------------------
    # 2) Sanity checks / type guards
    # -------------------------------
    X_arr = _to_numpy(X)
    Y_arr = _to_numpy(Y)

    if X_arr.ndim != 4:
        raise AssertionError(f"Expected X to have 4 dims (N, T, nodes, features), got {X_arr.shape}.")

    N, T, num_nodes, feat_dim = X_arr.shape
    if feat_dim != num_features:
        raise AssertionError(
            f"Expected last dim to be {num_features} features, but got {feat_dim} (shape {X_arr.shape})."
        )

    # Basic Y checks (best effort; specific shapes depend on your loaders)
    if early_reg:
        # regression: Y can be (N,) or (N, T) or (N, 1); warn if it's obviously class-like
        if Y_arr.dtype.kind in ("i", "u") and Y_arr.ndim == 1 and np.unique(Y_arr).size <= num_classes:
            logging.warning("Y looks like discrete labels, but early_reg=True suggests regression targets.")
    elif classification or early_label or detection:
        # classification-like: expect ints or one-hots
        if Y_arr.dtype.kind not in ("i", "u", "b", "f"):
            raise AssertionError(f"Classification-style heads expect numeric labels; got dtype={Y_arr.dtype}")

    logging.info(f"Data OK | X: {X_arr.shape} (N={N}, T={T}, nodes={num_nodes}, feat={feat_dim}) | "
                 f"Y: {Y_arr.shape} dtype={Y_arr.dtype}")

    # -------------------------------
    # 3) Initialize Trainer
    # -------------------------------
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
        DC=(detection or classification),
        RC=(early_reg or early_label)
    )

    # -------------------------------
    # 4) Train
    # -------------------------------
    return trainer.train(
        X_arr, Y_arr,
        detection=detection,
        classification=classification,
        early_reg=early_reg,
        early_clf=early_label
    )

if __name__ == "__main__":
    num_classes = 7

    # Tip: if EDF→H5 is heavy, consider running one head at a time or
    # pre-converting once offline to avoid repeated work.

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
