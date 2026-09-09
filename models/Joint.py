
"""
Joint multi-head baseline for GSNet (Reviewer 2, comments 11 & 13).

This is the *naive unification* that reviewers expect as a comparison point:
ONE shared GCN backbone (NOT frozen) with all four heads trained
SIMULTANEOUSLY. On every optimizer step we draw one batch per task, compute
each task's loss, sum them, and backpropagate once -- so the backbone receives
gradients from all four tasks jointly. Contrast this with GSNet's modular
design, where the backbone is trained once and then frozen while heads are
trained on the frozen representation.

Why round-robin (and not one big multi-task batch): the four tasks have
heterogeneous input structure (detection/classification operate on single
windows; forecasting on length-L window sequences) and disjoint label
availability (a background clip has no seizure type). A single jointly-labeled
batch therefore does not exist; round-robin joint optimization is the standard
way to train one shared trunk from several task-specific datasets.

>>> This produces the "Joint multi-head" row of Table tab:jointbaseline.
>>> I cannot run it for you; wire in your per-task loaders (see PREP HOOKS)
    and run it. Do not paste numbers you have not produced.

Optional: set TEMPORAL="transformer" to make the temporal encoder a small
Transformer instead of the GRU, i.e. a literal multi-task GNN-Transformer,
which is the exact phrasing of Reviewer 2 comment 11.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle

from models.model import MultiTaskGCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
HIDDEN_DIM   = 64
IN_DIM       = 121          # per-node feature dim (match your training)
NUM_CLASSES  = 7
SEQ_LEN      = 10
EPOCHS       = 100
LR           = 1e-3
TASK_WEIGHTS = {"detection": 1.0, "classification": 1.0,
                "forecast_time": 1.0, "forecast_label": 1.0}
TEMPORAL     = "gru"        # "gru" (matches GSNet) or "transformer" (GNN-Transformer variant)


# ----------------------------------------------------------------------
# Optional Transformer temporal encoder (for the GNN-Transformer variant)
# ----------------------------------------------------------------------
class TransformerTemporal(nn.Module):
    def __init__(self, dim, heads=4, layers=2, dropout=0.2):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                         dim_feedforward=2 * dim,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, seq):                 # seq: (B, L, dim)
        h = self.encoder(seq)
        return h[:, -1, :]                  # last-step representation


def build_model():
    model = MultiTaskGCN(hidden_dim=HIDDEN_DIM, in_dim=IN_DIM,
                         num_classes=NUM_CLASSES, forecast_classes=NUM_CLASSES,
                         dropout=0.3, seq_len=SEQ_LEN).to(DEVICE)
    if TEMPORAL == "transformer":
        # swap the GRU for a Transformer; the forecasting head calls
        # model.temporal_encoder, so we monkey-patch a compatible module.
        model.temporal_encoder = TransformerTemporal(HIDDEN_DIM).to(DEVICE)
        model._temporal_is_transformer = True
    return model


# ----------------------------------------------------------------------
# Per-task losses (mirror GSNet's settings)
# ----------------------------------------------------------------------
def detection_loss(logit, y, pos_weight=None):
    return F.binary_cross_entropy_with_logits(
        logit, y.float(), pos_weight=pos_weight)

def focal_ce(logits, y, gamma=3.5, eps=0.05, weight=None):
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    C = logits.size(-1)
    yoh = F.one_hot(y, C).float()
    yoh = (1 - eps) * yoh + eps / C
    focal = (1 - p) ** gamma
    loss = -(weight if weight is not None else 1.0) * (yoh * focal * logp)
    return loss.sum(-1).mean()

def time_loss(pred, y):
    return F.smooth_l1_loss(pred, y.float())


# ----------------------------------------------------------------------
# One joint optimizer step over all tasks (round-robin)
# ----------------------------------------------------------------------
def joint_step(model, batches, class_weight=None):
    total = 0.0
    logs = {}
    for task, batch in batches.items():
        if batch is None:
            continue
        batch = batch.to(DEVICE)
        out = model(batch.x, batch.edge_index, batch, getattr(batch, "edge_weight", None), task=task)
        if task == "detection":
            loss = detection_loss(out, batch.y)
        elif task == "classification":
            loss = focal_ce(out, batch.y.long(), weight=class_weight)
        elif task == "forecast_time":
            loss = time_loss(out, batch.y)
        elif task == "forecast_label":
            loss = focal_ce(out, batch.y.long(), weight=class_weight)
        else:
            continue
        total = total + TASK_WEIGHTS[task] * loss
        logs[task] = float(loss.detach())
    return total, logs


def train_joint(model, loaders, val_loaders, class_weight=None):
    """loaders: dict task -> torch_geometric DataLoader (train)
       val_loaders: dict task -> DataLoader (val)"""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    iters = {t: cycle(dl) for t, dl in loaders.items()}
    steps = max(len(dl) for dl in loaders.values())

    for ep in range(1, EPOCHS + 1):
        model.train()
        for _ in range(steps):
            batches = {t: next(iters[t]) for t in loaders}
            opt.zero_grad()
            loss, logs = joint_step(model, batches, class_weight)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:3d}  " + "  ".join(f"{k}:{v:.3f}" for k, v in logs.items()))
    return evaluate_joint(model, val_loaders)


@torch.no_grad()
def evaluate_joint(model, val_loaders):
    """Returns {task: metric}; plug in your project's exact metric functions."""
    model.eval()
    results = {}
    # ---- PREP HOOK: replace the stubs below with your metric computations ----
    # For each task, iterate val_loaders[task], collect preds/targets, and
    # compute: detection->AUROC, classification->weighted F1,
    # forecast_time->R2, forecast_label->weighted F1.
    for task in val_loaders:
        results[task] = None   # <-- FILL via your metric code
    print("[joint baseline] validation metrics:", results)
    return results


# ----------------------------------------------------------------------
# PREP HOOKS — wire your existing pipeline in here
# ----------------------------------------------------------------------
def build_loaders():
    """
    Return (train_loaders, val_loaders, class_weight) where each is a dict:
        {"detection": DataLoader, "classification": DataLoader,
         "forecast_time": DataLoader, "forecast_label": DataLoader}

    Reuse your existing data preparation. In your codebase, run_pipeline builds
    per-task graph datasets; expose those DataLoaders here rather than
    re-implementing feature/graph construction. Each batch must carry
    .x, .edge_index, .edge_weight, .batch, and .y for its task.
    """
    raise NotImplementedError(
        "Wire in your per-task DataLoaders from the existing data pipeline. "
        "detection/classification batches = individual window graphs; "
        "forecast_* batches = length-L window sequences.")


if __name__ == "__main__":
    train_loaders, val_loaders, class_weight = build_loaders()
    model = build_model()
    metrics = train_joint(model, train_loaders, val_loaders, class_weight)
    print("\n% Paste into Table tab:jointbaseline (Joint multi-head row):")
    d = metrics.get("detection"); c = metrics.get("classification")
    t = metrics.get("forecast_time"); y = metrics.get("forecast_label")
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else r"\FILL{}"
    print(f"Joint multi-head & {fmt(d)} & {fmt(c)} & {fmt(t)} & {fmt(y)}\\\\")
