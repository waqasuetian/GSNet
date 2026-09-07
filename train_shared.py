"""
Path B driver — trains ONE shared GCN backbone + ONE shared GRU + four heads.

Run order (each call reuses our existing run_pipeline / Trainer):

  Phase 1  detection      -> trains backbone + detection head, SAVES backbone
  Phase 2a classification -> freezes backbone, trains classification head
  Phase 2b early_reg      -> freezes backbone, trains GRU + time head, SAVES GRU
  Phase 2c early_label    -> freezes backbone + GRU, trains type head

Prerequisite edits :
  * models/model.py            -> replaced with the Path B version
  * models/trainer.py          -> 4 small insertions (new kwargs threaded to build/opt/save)
  * main.py run_pipeline(...)   -> 3 new kwargs threaded to trainer.train(...)

Adjust DATA_* paths, max_files, epochs, and alpha to your setup.
"""

import os
from main import run_pipeline

CKPT_DIR       = "models/checkpoints"
BACKBONE_CKPT  = os.path.join(CKPT_DIR, "shared_backbone.pth")
GRU_CKPT       = os.path.join(CKPT_DIR, "shared_gru.pth")

DATA_DC = r"E:\tuh_data\train"     # detection / classification data root
DATA_RC = r"E:\tuh_data\train"     # forecasting (preictal) data root
NUM_CLASSES = 7
ALPHA = 0.3                        # your chosen hybrid-graph fusion weight

os.makedirs(CKPT_DIR, exist_ok=True)


def phase1_pretrain_backbone():
    """Train backbone + detection head jointly; persist the backbone."""
    print("\n==================== PHASE 1: pretrain shared backbone (detection) ====================")
    run_pipeline(
        directory=DATA_DC,
        num_classes=NUM_CLASSES,
        detection=True,
        max_files=100,
        graph_method="hybrid",
        graph_params={"alpha": ALPHA},
        # ---- Path B ----
        freeze_backbone=False,
        backbone_ckpt=None,
        save_backbone_to=BACKBONE_CKPT,     # <-- persists the shared backbone
    )


def phase2a_classification():
    print("\n==================== PHASE 2a: classification head (frozen backbone) ====================")
    run_pipeline(
        directory=DATA_DC,
        num_classes=NUM_CLASSES,
        classification=True,
        max_files=100,
        graph_method="hybrid",
        graph_params={"alpha": ALPHA},
        # ---- Path B ----
        freeze_backbone=True,
        backbone_ckpt=BACKBONE_CKPT,        # <-- reuse the SAME backbone
    )


def phase2b_time():
    """Train the shared GRU + time head on the frozen backbone; persist the GRU."""
    print("\n==================== PHASE 2b: forecast-time (frozen backbone, train+save GRU) ====================")
    run_pipeline(
        directory=DATA_RC,
        num_classes=NUM_CLASSES,
        early_reg=True,
        max_files=1000,
        graph_method="hybrid",
        graph_params={"alpha": ALPHA},
        # ---- Path B ----
        freeze_backbone=True,
        backbone_ckpt=BACKBONE_CKPT,
        freeze_gru=False,
        gru_ckpt=None,
        save_gru_to=GRU_CKPT,               # <-- persists the shared temporal encoder
    )


def phase2c_type():
    """Train the type head on the frozen backbone + frozen shared GRU."""
    print("\n==================== PHASE 2c: forecast-type (frozen backbone + frozen GRU) ====================")
    run_pipeline(
        directory=DATA_RC,
        num_classes=NUM_CLASSES,
        early_label=True,
        max_files=100,
        graph_method="hybrid",
        graph_params={"alpha": ALPHA},
        # ---- Path B ----
        freeze_backbone=True,
        backbone_ckpt=BACKBONE_CKPT,
        freeze_gru=True,
        gru_ckpt=GRU_CKPT,                  # <-- reuse the SAME GRU as time head
    )


if __name__ == "__main__":
    phase1_pretrain_backbone()
    phase2a_classification()
    phase2b_time()
    phase2c_type()
    print("\nPath B training complete.")
    print(f"  shared backbone : {BACKBONE_CKPT}")
    print(f"  shared GRU      : {GRU_CKPT}")
    print(f"  heads           : {CKPT_DIR}/<task>/<task>_best.pth")
    print("Run infer_shared.py to evaluate the unified (single-encode, routed) model.")
