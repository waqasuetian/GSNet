import sys; print(sys.path)
import os
import sys
import logging
from typing import Tuple, Union
import torch
import numpy as np
from models.trainer import Trainer
from data.scripts.data_generation import EEGDataProcessor
from data.scripts.data_loader import EEGProcessor
from data.scripts.pooling import EEGPooler
from data.scripts.preictel_datageneration import preictal_dataLoader  # fixed spelling
from data.scripts.data_loader_preictal import EEGProcessorPreictal
#from models.trainer import make_overview_radars
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
    # Handle lazy loading mode (list of file paths)
    if isinstance(pooled_results, list):
        return len(pooled_results), []

    # Traditional mode (dictionary)
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
    directory: str = r'F:\tuh_data\train',
    num_features: int = 100,
    num_hiddens: int = 100,
    dropout: float = 0.2,
    num_heads: int = 8,
    learning_rate: float = 0.005,
    batch_size: int = 32,
    num_epochs: int = 200,
    max_files: int = 50,  # Increased from 13
    lazy_loading: bool = True,
    file_batch_size: int = 5,
    min_channels_per_event: int = 0,
    topk_events_by_duration: int = 20,
    max_gap_between_bckg_and_ictal_sec: float = 5.0,
    preictal_window_sec: float = 600.0,
    auto_expand_windows: tuple = (1200.0, 1800.0)
):
    """
    Runs the end-to-end pipeline for EEG-based tasks.
    Expects X to be shaped (N, T, nodes, features) with features == num_features.
    """
    if not (detection or classification or early_reg or early_label):
        raise ValueError("One of detection, classification, early_reg, or early_label must be True.")

    print(f"Directory being checked: {directory}")  # Debug statement to confirm the directory value
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Data directory does not exist: {directory}\n"
            f"Please update the 'directory' parameter in run_pipeline() to match your system's path.\n"
            f"Current working directory: {os.getcwd()}"
        )

    logging.info("Starting pipeline | modes: "
                 f"detection={detection}, classification={classification}, "
                 f"early_reg={early_reg}, early_label={early_label}")

    #-------------------------------
    #1) Data Loading & Preprocessing
    #-------------------------------
    if detection or classification:
        # DC branch
        logging.info("Loading DC data (detection/classification)…")
        processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)  # Explicitly pass directory
        # Skip conversion if all H5 files exist
        all_have_h5 = all(os.path.exists(os.path.join(root, f.replace(".edf", ".h5")))
                          for root, _, files in os.walk(directory)
                          for f in files if f.endswith(".edf"))
        if not all_have_h5:
            logging.info("Converting EDF to H5 files...")
            processor.convert_edf_to_h5()
        else:
            logging.info("All EDF files already have corresponding H5 files. Skipping conversion.")
        results = processor.process_directory(max_files=max_files)  # Pass max_files here
        pooler = EEGPooler(results, target_time_points=100)
        pooled_results = pooler.apply_pooling_DC()
        _check_not_empty(pooled_results, "DC pooling") if not isinstance(pooled_results, list) else None
        data_processor = EEGDataProcessor(
            pooled_results,
            detection=detection,
            classification=classification,
            lazy_loading=lazy_loading,
            processor=processor,
            pooler=pooler
        )
        X, Y = data_processor.process_fixed_length_clips(batch_size=file_batch_size)
    else:
        # RC branch (early_reg or early_label)
        logging.info("Loading RC data (early tasks)…")
        processor = EEGProcessorPreictal(
            root_directory=directory,
            resampled_freq=200,
            time_step_size=1,
            apply_fft=True,
            overlap=0.5,
            min_channels_per_event=min_channels_per_event,
            topk_events_by_duration=topk_events_by_duration,
            pad_short_segments=True,
            min_short_frac=0.1
        )
       
        results = processor.process_directory(max_files=max_files)  # Pass max_files here
        pooler = EEGPooler(results, target_time_points=100)
        pooled_results = pooler.apply_pooling_RC()
        _check_not_empty(pooled_results, "RC pooling")

        # Only inspect labels if not in lazy mode
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
        X, Y = data_processor.get_data()

    # if detection or classification:
    #     # DC branch
    #     logging.info("Loading DC data (detection/classification)…")
    #     processor = EEGProcessor(directory, resampled_freq=200, time_step_size=1, apply_fft=True)  # Explicitly pass directory
    #    # processor.convert_edf_to_h5()  # consider caching/skip-existing in your implementation
    #     results = processor.process_directory()
    #     pooler = EEGPooler(results, target_time_points=100)
    #     pooled_results = pooler.apply_pooling_DC()
    #     _check_not_empty(pooled_results, "DC pooling")
    #     data_processor = EEGDataProcessor(
    #         pooled_results,
    #         detection=detection,
    #         classification=classification
    #     )
    #     X, Y = data_processor.process_fixed_length_clips()
    # else:
    #     # RC branch (early_reg or early_label)
    #     logging.info("Loading RC data (early tasks)…")
    #     processor = EEGProcessorPreictal(
    #         directory,  # Explicitly pass directory
    #         resampled_freq=200,
    #         time_step_size=1,     # -> window_sec=1.0
    #         apply_fft=True,       # -> feature_mode="rfft"
    #         overlap=0.5,
    #         min_channels_per_event=1,
    #         topk_events_by_duration=10,
    #         pad_short_segments=True,
    #         min_short_frac=0.30
    #     )
    #     processor.convert_edf_to_h5()
    #     results = processor.process_directory()
    #     pooler = EEGPooler(results, target_time_points=100)
    #     pooled_results = pooler.apply_pooling_RC()
    #     _check_not_empty(pooled_results, "RC pooling")
    #     total_clips, uniq_labels = _inspect_labels_rc(pooled_results)
    #     logging.info(f"[RC] pooled clips={total_clips}, unique labels={uniq_labels}")
    #     data_processor = preictal_dataLoader(
    #         pooled_results,
    #         early_reg=early_reg, early_label=early_label,
    #         allow_intermediate_labels={"artf"},
    #         max_gap_between_bckg_and_ictal_sec=2.0,
    #         min_preictal_clip_sec=0.0
    #     )
    #     X, Y = data_processor.get_data()
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
#make_overview_radars(out_dir=r"D:\PhD Research\Experiments\Gen_EEG\runs\graphs")
if __name__ == "__main__":
    # ===== CONFIGURE YOUR DATA PATH HERE =====
    # Update this path to match where your EEG data is stored on THIS machine
    DATA_DIRECTORY = r"F:\tuh_data\train"  # Change this as needed
    # =========================================

    num_classes = 7
    real_class_names = ['gnsz', 'fnsz', 'tcsz', 'absz', 'mysz', 'cpsz', 'tnsz']  # Added for confusion matrix

    # Tip: if EDF→H5 is heavy, consider running one head at a time or
    # pre-converting once offline to avoid repeated work.

#    #Train Detection Head
    # run_pipeline(
    #     directory=r"F:\tuh_data\train",
    #     num_classes=num_classes,
    #     detection=True,
    #     max_files=6000
    # )
    #Train Classification Head
    # run_pipeline(
    #     directory=DATA_DIRECTORY,  # Uses the configured path above
    #     num_classes=num_classes,
    #     classification=True,
    #     max_files=7500   # achieved F1 79%
    # )

# #     #Train Early Regression Headc
#     run_pipeline(
#         directory=r"F:\tuh_data\train",
#         num_classes=num_classes,
#         early_reg=True,
#         max_files=100
    
#   )
# #     # Train Early Classification Head
    run_pipeline(
        directory=r"F:\tuh_data\train",
        num_classes=num_classes,
        early_label=True,
        max_files=8000
    
    )
