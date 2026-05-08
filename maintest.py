

# import os
# import torch
# import numpy as np
# import h5py
# import mne
# from itertools import combinations
# from torch.nn.functional import adaptive_avg_pool1d
# from torch_geometric.data import Data

# from models.model import MultiTaskGCN
# from utils import comp_xcorr, keep_topk  # your cross‐corr & top‐k utilities

# # ─── Configuration ─────────────────────────────────────────────────────────────
# EDF_PATH         = r"G:\tuh_data\eval\aaaaaaaq\s007_2014\01_tcp_ar\aaaaaaaq_s007_t003.edf"
# CSV_PATH         = r"G:\tuh_data\eval\aaaaaaaq\s007_2014\01_tcp_ar\aaaaaaaq_s007_t003.csv"
# CHECKPOINT_EPOCH = 30

# DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RESAMPLED_FREQ = 200     # Hz, must match training
# WINDOW_SEC     = 1.0     # seconds per clip
# POOL_FEATS     = 100     # features per node (matches training num_features)
# NUM_NODES      = 19
# THRESHOLD      = 0.5     # detection threshold

# # ─── Helpers ────────────────────────────────────────────────────────────────────

# def preprocess_edf_to_h5(edf_path):
#     raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
#     if raw.info["sfreq"] != RESAMPLED_FREQ:
#         raw.resample(RESAMPLED_FREQ)
#     data = raw.get_data()  # (channels, samples)
#     h5_path = edf_path.replace(".edf", ".h5")
#     with h5py.File(h5_path, "w") as f:
#         f.create_dataset("data", data=data)
#     print(f"[INFO] Wrote H5: {h5_path}")
#     return h5_path

# def extract_raw_clips(h5_path, csv_path):
#     import pandas as pd
#     df = pd.read_csv(csv_path, header=5, on_bad_lines="skip")
#     with h5py.File(h5_path, "r") as f:
#         sig = f["data"][()]  # (channels, total_samples)
#     clips = []
#     step = int(RESAMPLED_FREQ * WINDOW_SEC)
#     for _, row in df.iterrows():
#         start = int(float(row["start_time"]) * RESAMPLED_FREQ)
#         stop  = min(int(float(row["stop_time"]) * RESAMPLED_FREQ), sig.shape[1])
#         segment = sig[:, start:stop]
#         for i in range(0, segment.shape[1] - step + 1, step):
#             clips.append(segment[:, i : i + step])
#     print(f"[INFO] Extracted {len(clips)} raw clips")
#     return clips

# def pool_clip(clip):
#     t = torch.tensor(clip, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, ch, samples)
#     p = adaptive_avg_pool1d(t, POOL_FEATS)                                 # (1, ch, POOL_FEATS)
#     return p.squeeze(0)  # (ch, POOL_FEATS)

# def build_adjacency(clip, top_k=8):
#     """
#     Build a 19×19 adjacency using the max absolute cross-correlation
#     between each channel pair over the raw clip samples.
#     """
#     ch, samples = clip.shape
#     adj = np.eye(ch, dtype=np.float32)
#     for i, j in combinations(range(ch), 2):
#         seq = comp_xcorr(clip[i], clip[j], mode="valid", normalize=True)
#         val = float(np.abs(seq).max())   # scalar
#         adj[i, j] = adj[j, i] = val
#     # sparsify top_k edges per node
#     return keep_topk(adj, top_k=top_k, directed=True)

# def load_head(head_name):
#     sub = {
#         "detection":      f"detection_checkpoints/detection_epoch_{CHECKPOINT_EPOCH}.pth",
#         "classification": f"classification_checkpoints/classification_epoch_{CHECKPOINT_EPOCH}.pth",
#         "forecast_time":  f"early_regression_checkpoints/early_reg_epoch_{CHECKPOINT_EPOCH}.pth",
#     }[head_name]
#     path = os.path.join("models/checkpoints", sub)
#     model = MultiTaskGCN(125, POOL_FEATS, 7).to(DEVICE)
#     checkpoint = torch.load(path, map_location=DEVICE)

#     model_dict = model.state_dict()
#     # only keep parameters that match in shape
#     loaded = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
#     model_dict.update(loaded)
#     model.load_state_dict(model_dict)
#     print(f"[INFO] Loaded '{head_name}' head ({len(loaded)} params)")
#     model.eval()
#     return model

# # ─── Main Inference ─────────────────────────────────────────────────────────────

# # 1️⃣ Preprocess and extract raw clips
# h5_path   = preprocess_edf_to_h5(EDF_PATH)
# raw_clips = extract_raw_clips(h5_path, CSV_PATH)

# # 2️⃣ Load model heads
# det_model = load_head("detection")
# cls_model = load_head("classification")
# reg_model = load_head("forecast_time")

# # 3️⃣ Prepare graph indices
# xs, ys     = torch.tril_indices(NUM_NODES, NUM_NODES, offset=-1)
# edge_index = torch.stack([xs, ys], dim=0).to(DEVICE)

# # 4️⃣ Run per-clip inference
# print("\n--- Inference Results ---\n")
# for idx, clip in enumerate(raw_clips):
#     x         = pool_clip(clip)               # (19, 100)
#     adj_mat   = build_adjacency(clip)         # (19, 19)
#     edge_attr = torch.tensor(adj_mat[xs, ys], dtype=torch.float32, device=DEVICE)

#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

#     with torch.no_grad():
#         out_det = det_model(data.x, data.edge_index, data, task="detection").view(-1)
#         score   = torch.sigmoid(out_det).item()
#         seiz    = score > THRESHOLD

#     if seiz:
#         logits = cls_model(data.x, data.edge_index, data, task="classification")
#         label  = int(logits.argmax(dim=1).item())
#         print(f"Clip {idx}: DETECTION=1 (score={score:.3f}) → CLASSIFICATION={label}")
#     else:
#         ft = reg_model(data.x, data.edge_index, data, task="forecast_time").item()
#         print(f"Clip {idx}: DETECTION=0 (score={score:.3f}) → FORECAST_TIME={ft:.2f}")



#new one 

import os
import torch
import numpy as np
import h5py
import mne
from torch.nn.functional import adaptive_avg_pool1d
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

from models.model import MultiTaskGCN
from data.scripts.adjacancy_matrics import AdjacencyMatrixProcessor
from utils import comp_xcorr, keep_topk  # your cross-corr & top-k utilities

# ─── Configuration ─────────────────────────────────────────────────────────────
EDF_PATH         = r"G:\tuh_data\eval\aaaaaaaq\s007_2014\01_tcp_ar\aaaaaaaq_s007_t003.edf"
CSV_PATH         = r"G:\tuh_data\eval\aaaaaaaq\s007_2014\01_tcp_ar\aaaaaaaq_s007_t003.csv"
CHECKPOINT_EPOCH = 30

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESAMPLED_FREQ = 200     # Hz, must match training
WINDOW_SEC     = 1.0     # seconds per clip
POOL_FEATS     = 100     # features per node (matches training num_features)
NUM_NODES      = 19      # Align with trainer.py (19 nodes)
THRESHOLD      = 0.5     # detection threshold

# Seizure type mapping for interpretability
SEIZURE_TYPES = {
    0: "Focal Non-Specific Seizure (fnsz)",
    1: "Tonic-Clonic Seizure (tcsz)",
    2: "Absence Seizure (absz)",
    3: "Complex Partial Seizure (cpsz)",
    4: "General Seizure (seiz)",
    5: "Tonic Seizure (tnsz)",
    6: "Generalized Non-Specific Seizure (gnsz)"
}

# EEG channels to match dataloader.py
INCLUDED_CHANNELS = [
    'A1-T3', 'C3-CZ', 'C3-P3', 'C4-P4', 'C4-T4', 'CZ-C4',
    'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'FP1-F7',
    'FP2-F4', 'FP2-F8', 'P3-O1', 'P4-O2', 'T3-C3', 'T3-T5',
    'T4-A2', 'T4-T6', 'T5-O1', 'T6-O2'
]

# ─── Helpers ────────────────────────────────────────────────────────────────────

def preprocess_edf_to_h5(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    if raw.info["sfreq"] != RESAMPLED_FREQ:
        raw.resample(RESAMPLED_FREQ)
    
    # Debug: Print available channels
    available_channels = raw.ch_names
    print(f"[DEBUG] Available channels in EDF: {available_channels}")
    
    # Select matching channels
    selected_channels = [ch for ch in INCLUDED_CHANNELS if ch in available_channels]
    if len(selected_channels) < NUM_NODES:
        print(f"[WARNING] Only {len(selected_channels)}/{NUM_NODES} expected channels found.")
        if len(available_channels) >= NUM_NODES:
            selected_channels = available_channels[:NUM_NODES]
            print(f"[INFO] Using first {NUM_NODES} channels: {selected_channels}")
        else:
            raise ValueError(f"EDF file has only {len(available_channels)} channels, need {NUM_NODES}")
    
    raw.pick_channels(selected_channels)
    data = raw.get_data()  # (19, samples)
    h5_path = edf_path.replace(".edf", ".h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=data)
    print(f"[INFO] Wrote H5: {h5_path}")
    return h5_path

def extract_raw_clips(h5_path, csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, header=5, on_bad_lines="skip")
    with h5py.File(h5_path, "r") as f:
        sig = f["data"][()]  # (19, total_samples)
    clips = []
    labels = []  # Store (detection, classification, forecast_time) tuples
    step = int(RESAMPLED_FREQ * WINDOW_SEC)
    
    # Map seizure type abbreviations to numerical indices
    seizure_to_index = {name: idx for idx, name in enumerate(["fnsz", "tcsz", "absz", "cpsz", "seiz", "tnsz", "gnsz"])}
    print(f"[INFO] Seizure type mapping: {seizure_to_index}")
    
    for _, row in df.iterrows():
        start = int(float(row["start_time"]) * RESAMPLED_FREQ)
        stop = min(int(float(row["stop_time"]) * RESAMPLED_FREQ), sig.shape[1])
        segment = sig[:, start:stop]
        label = row.get("label", "bckg")
        det_label = 1 if label in seizure_to_index else 0
        cls_label = seizure_to_index.get(label, 0)
        time_label = float(row.get("forecast_time", 0.0))
        
        # Create 3D clips to match datageneration.py
        time_steps = []
        for i in range(0, segment.shape[1] - step + 1, step):
            chunk = segment[:, i:i + step]
            time_steps.append(chunk)
        if time_steps:
            clips.append(np.stack(time_steps, axis=0))  # (time_steps, 19, samples)
            labels.append((det_label, cls_label, time_label))
    
    print(f"[INFO] Extracted {len(clips)} clips with shape {clips[0].shape if clips else 'N/A'}")
    return clips, labels

def pool_clip(clip):
    t = torch.tensor(clip, dtype=torch.float32, device=DEVICE)  # (time_steps, 19, samples)
    t = t.mean(dim=0, keepdim=True)  # (1, 19, samples)
    p = adaptive_avg_pool1d(t, POOL_FEATS)  # (1, 19, POOL_FEATS)
    return p.squeeze(0)  # (19, POOL_FEATS)

def load_head(head_name):
    sub = {
        "detection": f"detection_checkpoints/detection_epoch_{CHECKPOINT_EPOCH}.pth",
        "classification": f"classification_checkpoints/classification_epoch_{CHECKPOINT_EPOCH}.pth",
        "forecast_time": f"early_regression_checkpoints/early_reg_epoch_{CHECKPOINT_EPOCH}.pth",
    }[head_name]
    path = os.path.join("models/checkpoints", sub)
    model = MultiTaskGCN(100, POOL_FEATS, 7, dropout=0.5).to(DEVICE)
    print(f"[DEBUG] Model class: {model.__class__.__name__}, Forward args: {model.forward.__code__.co_varnames}")
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        if isinstance(checkpoint, torch.nn.Module):
            checkpoint = checkpoint.state_dict()
        model_dict = model.state_dict()
        loaded = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        if len(loaded) < len(model_dict):
            print(f"[WARNING] Checkpoint missing {len(model_dict) - len(loaded)} parameters. Initializing with new model.")
        model_dict.update(loaded)
        model.load_state_dict(model_dict)
        print(f"[INFO] Loaded '{head_name}' head ({len(loaded)} params)")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint for {head_name}: {e}")
        raise
    model.eval()
    return model

# ─── Main Inference ─────────────────────────────────────────────────────────────

# 1️⃣ Preprocess and extract raw clips with labels
h5_path = preprocess_edf_to_h5(EDF_PATH)
raw_clips, labels = extract_raw_clips(h5_path, CSV_PATH)

# 2️⃣ Load model heads
det_model = load_head("detection")
cls_model = load_head("classification")
reg_model = load_head("forecast_time")

# 3️⃣ Prepare graph indices and edge attributes
xs, ys = torch.tril_indices(NUM_NODES, NUM_NODES, offset=-1)
edge_index = torch.stack([xs, ys], dim=0).to(DEVICE)

# Compute edge attributes
adj_proc = AdjacencyMatrixProcessor({EDF_PATH: (raw_clips, [l[1] for l in labels], [0], [1])}, data_directory=r"G:\tuh_data\eval")
edge_attr = adj_proc.compute_all_edge_weights(DC=False, RC=True).mean(dim=0)
expected_edges = NUM_NODES * (NUM_NODES - 1) // 2
if edge_attr.numel() > expected_edges:
    edge_attr = edge_attr[:expected_edges]
elif edge_attr.numel() < expected_edges:
    edge_attr = torch.cat([edge_attr, torch.zeros(expected_edges - edge_attr.numel(), device=DEVICE)])
edge_attr = edge_attr.to(DEVICE)

# 4️⃣ Run per-clip inference and collect metrics
print("\n--- Inference Results ---\n")
det_preds, det_trues = [], []
cls_preds, cls_trues = [], []
time_preds, time_trues = [], []

for idx, (clip, (det_label, cls_label, time_label)) in enumerate(zip(raw_clips, labels)):
    x = pool_clip(clip)  # (19, 100)
    # Use precomputed edge_attr instead of rebuilding
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(NUM_NODES, dtype=torch.long, device=DEVICE)
    )

    # Debug: Verify Data object contents
    print(f"[DEBUG] Clip {idx}: x.shape={data.x.shape}, edge_index.shape={data.edge_index.shape}, "
          f"edge_attr.shape={data.edge_attr.shape}, batch.shape={data.batch.shape}")

    with torch.no_grad():
        out_det, attn_weights = det_model(data.x, data.edge_index, data.batch, task="detection")
        out_det = out_det.view(-1)
        score = torch.sigmoid(out_det).item()
        seiz = int(score > THRESHOLD)
        det_preds.append(seiz)
        det_trues.append(det_label)

        if seiz:
            logits, attn_weights = cls_model(data.x, data.edge_index, data.batch, task="classification")
            probs = torch.softmax(logits, dim=1)
            label = int(probs.argmax(dim=1).item())
            confidence = probs[0, label].item()
            cls_preds.append(label)
            cls_trues.append(cls_label)
            pred_seizure_type = SEIZURE_TYPES.get(label, f"Unknown ({label})")
            true_seizure_type = SEIZURE_TYPES.get(cls_label, f"Unknown ({cls_label})")
            top_channels = torch.topk(attn_weights.flatten(), k=5).indices.cpu().numpy()
            print(f"Clip {idx}: DETECTION=1 (score={score:.3f}, true={det_label}) → "
                  f"CLASSIFICATION={pred_seizure_type} (confidence={confidence:.3f}, true={true_seizure_type})")
            print(f"Top contributing channels: {top_channels}")
        else:
            ft, attn_weights = reg_model(data.x, data.edge_index, data.batch, task="forecast_time")
            ft = ft.item()
            time_preds.append(ft)
            time_trues.append(time_label)
            top_channels = torch.topk(attn_weights.flatten(), k=5).indices.cpu().numpy()
            print(f"Clip {idx}: DETECTION=0 (score={score:.3f}, true={det_label}) → "
                  f"FORECAST_TIME={ft:.2f} (true={time_label:.2f})")
            print(f"Top contributing channels: {top_channels}")

# 5️⃣ Compute and print metrics
if det_preds:
    det_accuracy = accuracy_score(det_trues, det_preds)
    det_f1 = f1_score(det_trues, det_preds, average='binary')
    print(f"\n--- Detection Metrics ---")
    print(f"Accuracy: {det_accuracy:.4f}")
    print(f"F1 Score: {det_f1:.4f}")
else:
    print("\n--- Detection Metrics ---")
    print("No detection predictions to evaluate.")

if cls_preds:
    cls_accuracy = accuracy_score(cls_trues, cls_preds)
    cls_f1 = f1_score(cls_trues, cls_preds, average='weighted')
    print(f"\n--- Classification Metrics ---")
    print(f"Accuracy: {cls_accuracy:.4f}")
    print(f"F1 Score: {cls_f1:.4f}")
else:
    print("\n--- Classification Metrics ---")
    print("No classification predictions (no seizures detected).")

if time_preds:
    time_r2 = r2_score(time_trues, time_preds)
    time_mse = mean_squared_error(time_trues, time_preds)
    print(f"\n--- Forecast Time Metrics ---")
    print(f"R² Score: {time_r2:.4f}")
    print(f"Mean Squared Error: {time_mse:.4f}")
else:
    print("\n--- Forecast Time Metrics ---")
    print("No forecast time predictions (all clips classified as seizures).")
