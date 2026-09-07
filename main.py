from typing import List, Tuple, Union, Dict, Any  # Add Dict and Any
import sys; print(sys.path)
import os
import sys
import logging
from typing import Tuple, Union, List
import torch
import numpy as np
from models.trainer import Trainer
from data.scripts.data_generation import EEGDataProcessor
from data.scripts.data_loader import EEGProcessor
from data.scripts.pooling import EEGPooler
from data.scripts.preictel_datageneration import preictal_dataLoader
from data.scripts.data_loader_preictal import EEGProcessorPreictal

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

def _check_not_empty(pooled_results, stage: str):
    if not pooled_results:
        raise RuntimeError(f"{stage}: pooled_results is empty. "
                           f"Check directory path, EDF/H5 availability, and preprocessing filters.")

def _inspect_labels_rc(pooled_results) -> Tuple[int, list]:
    if isinstance(pooled_results, list):
        return len(pooled_results), []
    try:
        uniq = sorted({lab for _, (_, labs, *_rest) in pooled_results.items() for lab in labs})
    except Exception:
        uniq = []
    total = sum(len(v[0]) for v in pooled_results.values()) if pooled_results else 0
    return total, uniq
X_val = None
Y_val = None
def run_pipeline(
    num_classes: int,
    detection: bool = False,
    classification: bool = False,
    early_reg: bool = False,
    early_label: bool = False,
    directory: str = r'E:\tuh_data\train',
    num_features: int = 100,
    num_hiddens: int = 64,
    dropout: float = 0.2,
    num_heads: int = 8,
    learning_rate: float = 0.005,
    batch_size: int = 64,
    num_epochs: int = 250,
    max_files: int = 50,
    lazy_loading: bool = True,
    file_batch_size: int = 5,
    min_channels_per_event: int = 0,
    topk_events_by_duration: int = 20,
    max_gap_between_bckg_and_ictal_sec: float = 5.0,
    preictal_window_sec: float = 600.0,
    auto_expand_windows: tuple = (1200.0, 1800.0),
    seq_len: int = 10,
    channel_names: List[str] = None,
    graph_method: str = 'hybrid',
    use_augmentation: bool = True,
    augmentation_strength: float = 15,   # 0.5 = weaker, 1.0 = normal, 1.5 = stronger
    graph_params: Dict[str, Any] = None,
    # diffusion_steps = 2 ,      # Diffusion steps (K)
    # use_diffusion = True,     # Enable diffusion
    # bidirectional_diffusion = True, 
    patient_wise_split: bool = True,      # NEW
    test_patient_ratio: float = 0.2,      # NEW
    random_seed: int = 42,                # NEW
        # ---- Path B pass-throughs ----
    freeze_backbone: bool = False,
    backbone_ckpt: str = None,
    save_backbone_to: str = None,
    freeze_gru: bool = False,
    gru_ckpt: str = None,
    save_gru_to: str = None,
):
    
    """
    
    Runs the end-to-end pipeline for EEG-based tasks.
    Expects X to be shaped (N, T, nodes, features) with features == num_features.
    """
    if not (detection or classification or early_reg or early_label):
        raise ValueError("One of detection, classification, early_reg, or early_label must be True.")
    if graph_params is None:
        graph_params = {}
    
    X_val = None
    Y_val = None
    print(f"Directory being checked: {directory}")
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Data directory does not exist: {directory}\n"
            f"Please update the 'directory' parameter in run_pipeline() to match your system's path.\n"
            f"Current working directory: {os.getcwd()}"
        )

    logging.info("Starting pipeline | modes: "
                 f"detection={detection}, classification={classification}, "
                 f"early_reg={early_reg}, early_label={early_label}")

    # -------------------------------
    # Data Loading & Preprocessing
    # -------------------------------
    if detection or classification:
        # DC branch (unchanged)
        logging.info("Loading DC data (detection/classification)…")
        processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)
        all_have_h5 = all(os.path.exists(os.path.join(root, f.replace(".edf", ".h5")))
                          for root, _, files in os.walk(directory)
                          for f in files if f.endswith(".edf"))
        if not all_have_h5:
            logging.info("Converting EDF to H5 files...")
            processor.convert_edf_to_h5()
        else:
            logging.info("All EDF files already have corresponding H5 files. Skipping conversion.")
        results = processor.process_directory(max_files=max_files)
        pooler = EEGPooler(results, target_time_points=100)
        pooled_results = pooler.apply_pooling_DC()
        _check_not_empty(pooled_results, "DC pooling") if not isinstance(pooled_results, list) else None
        data_processor = EEGDataProcessor(
            pooled_results,
            detection=detection,
            classification=classification,
            lazy_loading=lazy_loading,
            processor=processor,
            pooler=pooler,
            patient_wise_split=patient_wise_split,      # NEW
            test_patient_ratio=test_patient_ratio,      # NEW
            random_seed=random_seed  
        )
        # X, Y = data_processor.process_fixed_length_clips(batch_size=file_batch_size)
        # file_ids = None
        
        # Get validation data if patient-wise split was used
        if patient_wise_split and lazy_loading:
            X_val, Y_val = data_processor.get_validation_data()
        else:
            X_val, Y_val = None, None
        X, Y = data_processor.process_fixed_length_clips(batch_size=file_batch_size)
        file_ids = None  # DC tasks do not need file IDs

    else:
        # RC branch (early_reg or early_label)
        logging.info("Loading RC data (early tasks)…")

        # If no channel list is provided, use the 33‑channel H5 list
        if channel_names is None:
            channel_names = [
                      'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
            ]
            print(f"Using 19‑channel H5 list ({len(channel_names)} channels)")

        # Create the real processor with channel filtering enabled
        processor = EEGProcessorPreictal(
            root_directory=directory,
            resampled_freq=200,
            time_step_size=12,               # 12‑second windows
            apply_fft=True, 
            #feature_mode='raw',  
            overlap=0.5,
            min_channels_per_event=min_channels_per_event,
            topk_events_by_duration=topk_events_by_duration,
            pad_short_segments=True,
            min_short_frac=0.1,
            enable_channel_filter=True,      # <-- important: filter channels
            included_channels=channel_names  # <-- use the provided list
        )

        results = processor.process_directory(max_files=max_files)
        pooler = EEGPooler(results, target_time_points=100)
        pooled_results = pooler.apply_pooling_RC()
        _check_not_empty(pooled_results, "RC pooling")

        if isinstance(pooled_results, list):
            logging.info(f"[RC] Found {len(pooled_results)} files for lazy processing")
        else:
            total_clips, uniq_labels = _inspect_labels_rc(pooled_results)
            logging.info(f"[RC] pooled clips={total_clips}, unique labels={uniq_labels}")

        data_processor = preictal_dataLoader(
            pooled_results,
            early_reg=early_reg,
            early_label=early_label,
            allow_intermediate_labels={"artf"},
            max_gap_between_bckg_and_ictal_sec=max_gap_between_bckg_and_ictal_sec,
            min_preictal_clip_sec=0.0,
            preictal_window_sec=preictal_window_sec,
            auto_expand_windows=auto_expand_windows,
            lazy_loading=lazy_loading,
            processor=processor,
            pooler=pooler
        )
        # X, Y, file_ids = data_processor.get_data()
        X, Y, file_ids, patient_ids = data_processor.get_data()  # Updated
        print(Y)


    # -------------------------------
    # Sanity checks / type guards
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
    if early_reg:
        if Y_arr.dtype.kind in ("i", "u") and Y_arr.ndim == 1 and np.unique(Y_arr).size <= num_classes:
            logging.warning("Y looks like discrete labels, but early_reg=True suggests regression targets.")
    elif classification or early_label or detection:
        if Y_arr.dtype.kind not in ("i", "u", "b", "f"):
            raise AssertionError(f"Classification-style heads expect numeric labels; got dtype={Y_arr.dtype}")

    logging.info(f"Data OK | X: {X_arr.shape} (N={N}, T={T}, nodes={num_nodes}, feat={feat_dim}) | "
                 f"Y: {Y_arr.shape} dtype={Y_arr.dtype}")

    # -------------------------------
    # Initialize Trainer
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
        RC=(early_reg or early_label),
        channel_names=channel_names,      # <-- pass channel list to trainer
        seq_len=seq_len,
        graph_method=graph_method  
    )

       # Determine which file_ids to use (patient_ids for RC tasks, None for DC)
    if early_reg or early_label:
        file_ids_to_use = patient_ids if 'patient_ids' in locals() else None
    else:
        file_ids_to_use = None
    # -------------------------------
    # Train
    # -------------------------------
    return trainer.train(
        X_arr, Y_arr,
        detection=detection,
        classification=classification,
        early_reg=early_reg,
        early_clf=early_label,
        #file_ids=patient_ids,
        X_val=X_val,
        Y_val=Y_val,
        # ---- Path B ----
        freeze_backbone=freeze_backbone,
        backbone_ckpt=backbone_ckpt,
        save_backbone_to=save_backbone_to,
        freeze_gru=freeze_gru,
        gru_ckpt=gru_ckpt,
        save_gru_to=save_gru_to,
    )
if __name__ == "__main__":
    # ===== CONFIGURE YOUR DATA PATH HERE =====
    DATA_DIRECTORY = r"F:\tuh_data\train"  # Change this as needed
    # =========================================

    num_classes = 7
    real_class_names = ['gnsz', 'fnsz', 'tcsz', 'absz', 'mysz', 'cpsz', 'tnsz']

   # Train Detection Head (uncomment to run)
    run_pipeline(
        directory=r"E:\tuh_data\train",
        num_classes=num_classes,
        detection=True,
        max_files=200,
        graph_method='hybrid',
        graph_params={'alpha': 0.5}

    )

#     # # Train Classification Head (uncomment to run)
    # run_pipeline(
    #     directory=DATA_DIRECTORY,
    #     num_classes=num_classes,
    #     classification=True,
    #     max_files=7000,
    #      graph_method='hybrid',
    #     graph_params={'alpha': 0.5}
    # )

#    # Train Early Regression Head (uncomment to run)
#     run_pipeline(
#         directory=r"F:\tuh_data\train",
#         num_classes=num_classes,
#         early_reg=True,
#         #graph_params={'alpha': 0.5, 'distance_threshold': 0.9 },
#         max_files=1000,
#         graph_method='hybrid',
#         graph_params={'alpha': 0.1}
        
#     )

# # #    # Train Early Classification Head
#     run_pipeline(
#         directory=r"E:\tuh_data\train",
#         num_classes=num_classes,
#         early_label=True,
#         max_files=1000,
#         graph_method='hybrid',
#         graph_params={'alpha': 0.5},
        

#  )
