# test_pipeline.py

import os
import torch
import numpy as np
import h5py
import mne
from itertools import combinations
from torch.nn.functional import adaptive_avg_pool1d
from torch_geometric.data import Data

from models.model import MultiTaskGCN
from utils import comp_xcorr, keep_topk  # your cross‐corr & top‐k utilities

# ─── Configuration ─────────────────────────────────────────────────────────────
EDF_PATH         = r"G:\tuh_data\eval\aaaaaaaq\s007_2014\01_tcp_ar\aaaaaaaq_s007_t003.edf"
CSV_PATH         = r"G:\tuh_data\eval\aaaaaaaq\s007_2014\01_tcp_ar\aaaaaaaq_s007_t003.csv"
CHECKPOINT_EPOCH = 30

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESAMPLED_FREQ = 200     # Hz, must match training
WINDOW_SEC     = 1.0     # seconds per clip
POOL_FEATS     = 100     # features per node (matches training num_features)
NUM_NODES      = 19
THRESHOLD      = 0.5     # detection threshold

# ─── Helpers ────────────────────────────────────────────────────────────────────

def preprocess_edf_to_h5(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    if raw.info["sfreq"] != RESAMPLED_FREQ:
        raw.resample(RESAMPLED_FREQ)
    data = raw.get_data()  # (channels, samples)
    h5_path = edf_path.replace(".edf", ".h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=data)
    print(f"[INFO] Wrote H5: {h5_path}")
    return h5_path

def extract_raw_clips(h5_path, csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, header=5, on_bad_lines="skip")
    with h5py.File(h5_path, "r") as f:
        sig = f["data"][()]  # (channels, total_samples)
    clips = []
    step = int(RESAMPLED_FREQ * WINDOW_SEC)
    for _, row in df.iterrows():
        start = int(float(row["start_time"]) * RESAMPLED_FREQ)
        stop  = min(int(float(row["stop_time"]) * RESAMPLED_FREQ), sig.shape[1])
        segment = sig[:, start:stop]
        for i in range(0, segment.shape[1] - step + 1, step):
            clips.append(segment[:, i : i + step])
    print(f"[INFO] Extracted {len(clips)} raw clips")
    return clips

def pool_clip(clip):
    t = torch.tensor(clip, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, ch, samples)
    p = adaptive_avg_pool1d(t, POOL_FEATS)                                 # (1, ch, POOL_FEATS)
    return p.squeeze(0)  # (ch, POOL_FEATS)

def build_adjacency(clip, top_k=8):
    """
    Build a 19×19 adjacency using the max absolute cross-correlation
    between each channel pair over the raw clip samples.
    """
    ch, samples = clip.shape
    adj = np.eye(ch, dtype=np.float32)
    for i, j in combinations(range(ch), 2):
        seq = comp_xcorr(clip[i], clip[j], mode="valid", normalize=True)
        val = float(np.abs(seq).max())   # scalar
        adj[i, j] = adj[j, i] = val
    # sparsify top_k edges per node
    return keep_topk(adj, top_k=top_k, directed=True)

def load_head(head_name):
    sub = {
        "detection":      f"detection_checkpoints/detection_epoch_{CHECKPOINT_EPOCH}.pth",
        "classification": f"classification_checkpoints/classification_epoch_{CHECKPOINT_EPOCH}.pth",
        "forecast_time":  f"early_regression_checkpoints/early_reg_epoch_{CHECKPOINT_EPOCH}.pth",
    }[head_name]
    path = os.path.join("models/checkpoints", sub)
    model = MultiTaskGCN(125, POOL_FEATS, 7).to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)

    model_dict = model.state_dict()
    # only keep parameters that match in shape
    loaded = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(loaded)
    model.load_state_dict(model_dict)
    print(f"[INFO] Loaded '{head_name}' head ({len(loaded)} params)")
    model.eval()
    return model

# ─── Main Inference ─────────────────────────────────────────────────────────────

# 1️⃣ Preprocess and extract raw clips
h5_path   = preprocess_edf_to_h5(EDF_PATH)
raw_clips = extract_raw_clips(h5_path, CSV_PATH)

# 2️⃣ Load model heads
det_model = load_head("detection")
cls_model = load_head("classification")
reg_model = load_head("forecast_time")

# 3️⃣ Prepare graph indices
xs, ys     = torch.tril_indices(NUM_NODES, NUM_NODES, offset=-1)
edge_index = torch.stack([xs, ys], dim=0).to(DEVICE)

# 4️⃣ Run per-clip inference
print("\n--- Inference Results ---\n")
for idx, clip in enumerate(raw_clips):
    x         = pool_clip(clip)               # (19, 100)
    adj_mat   = build_adjacency(clip)         # (19, 19)
    edge_attr = torch.tensor(adj_mat[xs, ys], dtype=torch.float32, device=DEVICE)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    with torch.no_grad():
        out_det = det_model(data.x, data.edge_index, data, task="detection").view(-1)
        score   = torch.sigmoid(out_det).item()
        seiz    = score > THRESHOLD

    if seiz:
        logits = cls_model(data.x, data.edge_index, data, task="classification")
        label  = int(logits.argmax(dim=1).item())
        print(f"Clip {idx}: DETECTION=1 (score={score:.3f}) → CLASSIFICATION={label}")
    else:
        ft = reg_model(data.x, data.edge_index, data, task="forecast_time").item()
        print(f"Clip {idx}: DETECTION=0 (score={score:.3f}) → FORECAST_TIME={ft:.2f}")
