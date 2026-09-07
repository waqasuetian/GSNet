"""
Path B unified inference — assembles ONE model from the shared backbone,
the shared GRU, and the four trained heads, then runs a SINGLE backbone
encode per clip and routes conditionally (Algorithm 3, now literally true).

Assumes the Path B checkpoints produced by train_shared.py:
  models/checkpoints/shared_backbone.pth
  models/checkpoints/shared_gru.pth
  models/checkpoints/detection/detection_best.pth
  models/checkpoints/classification/classification_best.pth
  models/checkpoints/early_reg/early_reg_best.pth
  models/checkpoints/early_clf/early_clf_best.pth
"""

import os
import torch
from torch_geometric.data import Data, Batch

from models.model import MultiTaskGCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT = "models/checkpoints"
BACKBONE_CKPT = os.path.join(CKPT, "shared_backbone.pth")
GRU_CKPT      = os.path.join(CKPT, "shared_gru.pth")
HEAD_CKPTS = {
    "detect_head": os.path.join(CKPT, "detection", "detection_best.pth"),
    "class_head":  os.path.join(CKPT, "classification", "classification_best.pth"),
    "time_head":   os.path.join(CKPT, "early_reg", "early_reg_best.pth"),
    "label_head":  os.path.join(CKPT, "early_clf", "early_clf_best.pth"),
}

NUM_NODES   = 19
IN_DIM      = 1200          # per-node feature dim used at train time — MUST match training
HIDDEN_DIM  = 64            # MUST match training (num_hiddens)
NUM_CLASSES = 7
SEQ_LEN     = 10
DET_THRESHOLD = 0.5
SUPPRESS_SECONDS = 60.0     # forecasts beyond this horizon are treated as "no imminent seizure"


def build_unified_model():
    """Instantiate one model and load the shared backbone + shared GRU + all heads."""
    model = MultiTaskGCN(
        hidden_dim=HIDDEN_DIM,
        in_dim=IN_DIM,
        num_classes=NUM_CLASSES,
        forecast_classes=NUM_CLASSES,
        dropout=0.0,               # eval
        seq_len=SEQ_LEN,
    ).to(DEVICE)

    n_b = model.load_backbone(BACKBONE_CKPT, strict=True)
    print(f"[unified] loaded shared backbone ({n_b} tensors)")
    n_g = model.load_temporal(GRU_CKPT, strict=True)
    print(f"[unified] loaded shared GRU ({n_g} tensors)")
    for head_name, path in HEAD_CKPTS.items():
        n_h = model.load_head_from_full_ckpt(path, head_name, strict=False)
        print(f"[unified] loaded {head_name} ({n_h} tensors) from {os.path.basename(path)}")

    model.eval()
    return model


@torch.no_grad()
def infer_clip(model, seq_windows, edge_index, edge_weight):
    """
    seq_windows : list/tensor of L window node-feature matrices, each (NUM_NODES, IN_DIM)
    Returns a dict with detection + (classification | time+type) via conditional routing.
    A SINGLE backbone encode is shared by every head.
    """
    # Build one PyG batch of the L windows so the backbone runs once.
    graphs = []
    for w in seq_windows:
        x = torch.as_tensor(w, dtype=torch.float32, device=DEVICE)
        g = Data(x=x, edge_index=edge_index)
        g.edge_weight = edge_weight
        graphs.append(g)
    batch = Batch.from_data_list(graphs).to(DEVICE)

    # ---- shared encode ONCE ----
    window_feat = model.encode_windows(batch.x, batch.edge_index, batch,
                                       getattr(batch, "edge_weight", None))   # (L, H)
    last_feat = window_feat[-1:].clone()                                      # (1, H)

    # ---- detection ----
    p_det = torch.sigmoid(model.run_detection(last_feat)).item()
    result = {"p_detection": p_det, "seizure": p_det > DET_THRESHOLD}

    if result["seizure"]:
        # classify the ongoing seizure (reuses the SAME embedding, no re-encode)
        logits = model.run_classification(last_feat)
        result["class"] = int(logits.argmax(dim=-1).item())
        result["time_forecast"] = None
        result["type_forecast"] = None
    else:
        # forecast onset time + type from the SAME window sequence via shared GRU
        t = model.run_forecast_time(window_feat).item()          # transformed target space if you used log1p
        type_logits = model.run_forecast_label(window_feat)
        result["class"] = None
        result["time_forecast"] = t
        result["type_forecast"] = int(type_logits.argmax(dim=-1).item())
        if t is not None and t > SUPPRESS_SECONDS:
            result["type_forecast"] = None
            result["time_forecast"] = ">60s (suppressed)"
    return result


def build_edges(adj_edge_weight=None):
    """Lower-triangular edge_index (matches training) + optional hybrid weights."""
    xs, ys = torch.tril_indices(NUM_NODES, NUM_NODES, offset=-1)
    edge_index = torch.stack([xs, ys], dim=0).to(DEVICE)
    expected = NUM_NODES * (NUM_NODES - 1) // 2
    if adj_edge_weight is None:
        edge_weight = torch.ones(expected, device=DEVICE)
    else:
        edge_weight = adj_edge_weight.to(DEVICE)
        if edge_weight.numel() > expected:
            edge_weight = edge_weight[:expected]
        elif edge_weight.numel() < expected:
            edge_weight = torch.cat(
                [edge_weight, torch.zeros(expected - edge_weight.numel(), device=DEVICE)])
    return edge_index, edge_weight


if __name__ == "__main__":
    # Minimal smoke test with random data to verify the graph assembles and routes.
    model = build_unified_model()
    edge_index, edge_weight = build_edges(None)
    dummy_seq = [torch.randn(NUM_NODES, IN_DIM) for _ in range(SEQ_LEN)]
    out = infer_clip(model, dummy_seq, edge_index, edge_weight)
    print("\n[smoke test] unified routed output:")
    for k, v in out.items():
        print(f"  {k}: {v}")
    print("\nParameter accounting (unified vs. 4 separate models):")
    n_backbone = sum(p.numel() for p in model.backbone_parameters())
    n_gru = sum(p.numel() for p in model.temporal_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  shared backbone params : {n_backbone:,}")
    print(f"  shared GRU params      : {n_gru:,}")
    print(f"  unified model params   : {n_total:,}")
    print(f"  Path A (4x backbone)   : ~{n_total + 3*n_backbone:,}  (backbone duplicated per task)")
    print(f"  Path B saving          : ~{3*n_backbone:,} params not duplicated")
