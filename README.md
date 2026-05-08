# GSNet: A Unified Graph-Based Deep Learning Framework for Multi-Stage Seizure Monitoring

## Overview
GSNet is a novel, hybrid graph-based deep learning (DL) framework designed for real-time epileptic seizure monitoring using Electroencephalography (EEG) signals. Developed as part of the research paper *"A Unified Graph-Based Deep Learning Framework for Multi-Stage Seizure Monitoring"* by Waqas Ali and Muhammad Shahbaz (Department of Computer Engineering, University of Engineering and Technology, Lahore, Pakistan), GSNet addresses key challenges in epilepsy care by integrating seizure detection, classification, and early forecasting of onset and type within a single, efficient architecture.


- **Core Innovations:**
  - Transforms non-Euclidean EEG signals into hybrid graph structures to capture spatiotemporal dependencies.
  - Employs a shared Graph Convolutional Network (GCN) backbone with task-specific heads for multi-stage processing.
  - Provides interpretable band×node soft attribution maps to highlight critical EEG channels and frequency contributions.
  
- **Performance Highlights (on Temple University Seizure Corpus - TUSZ):**
  - Seizure Detection: 94% Area Under the Receiver Operating Characteristic Curve (AUROC).
  - Seizure Classification: 0.81 weighted F1-score.
  - Onset Forecasting: 0.785 R²-score.
  - Seizure Type Forecasting: 0.83 weighted F1-score.

This framework advances computational neuroscience by enabling holistic EEG analysis, supporting precise healthcare interventions for epilepsy patients affecting over 70 million worldwide.
# 🏗️ GSNet Architecture

## Overall Framework

<p align="center">
  <img src="figures/gsnet_architecture.png" width="900">
</p>

<p align="center">
<b>Figure 1.</b> Overall GSNet framework for multi-stage seizure monitoring.
</p>

## Installation

### Prerequisites
- Python 3.8 or higher.
- PyTorch 1.10+ (with CUDA support for GPU acceleration).
- Additional libraries: NumPy, SciPy, Scikit-learn, Pandas, Matplotlib, NetworkX, PyTorch Geometric (for GCN layers).

### Quick Setup
1. Clone the repository (once created):
   ```
   git clone https://github.com/waqasuetian/GSNet.git
   cd GSNet
   ```
2. Install dependencies:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA version
   pip install -r requirements.txt
   ```
3. Download the TUSZ dataset (v2.0.3) from the [official site](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) and place EDF files and annotations in `data/tusz/`.

## Usage

### Data Preparation
- Preprocess EEG data into 12-second clips (512 samples at 256 Hz sampling rate) with band-pass filtering (0.5-50 Hz).
- Run the synthesis script to generate hybrid graph structures and task-specific labels (binary for detection, multiclass for classification, temporal for forecasting):
  ```
  python src/data/preprocess.py --input_dir data/tusz --output_dir data/processed --clip_length 2 --sample_rate 256
  ```

### Training the Model
- Configure hyperparameters (e.g., learning rate: 0.001, batch size: 32, GCN layers: 3) in `config.yaml`.
- Train GSNet end-to-end:
  ```
  python src/train.py --config config.yaml --device cuda --epochs 100
  ```
- The model supports multi-task learning with weighted losses (BCE for detection, CE for classification, MSE for TTI forecasting).

### Evaluation and Inference
- Load a trained checkpoint and evaluate on test data:
  ```
  python src/evaluate.py --model_path models/gsnet_best.pth --data_dir data/processed --task all
  ```
- Generate predictions and attribution maps:
  ```
  python src/infer.py --model_path models/gsnet_best.pth --input_file data/test_clip.edf --output_dir results/
  ```

### Key Scripts
- `src/models/gsnet.py`: Core GSNet implementation (GCN backbone + heads).
- `src/utils/graph_construction.py`: Functions for node feature generation, edge computation (cosine similarity or cross-correlation), and adjacency matrix sparsification (top-k=3).
- `src/visualize.py`: Plots band×node attribution maps and graph visualizations.

## Code Structure
```
GSNet/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── config.yaml            # Hyperparameters
├── data/                  # Raw and processed TUSZ data
│   └── tusz/              # EDF files and annotations
├── src/                   # Source code
│   ├── data/              # Loaders and preprocessors
│   ├── models/            # GSNet architecture (GCN, heads)
│   ├── utils/             # Graph utils, attribution, metrics
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation and metrics (AUROC, F1, R²)
│   └── infer.py           # Inference and visualization
├── models/                # Saved checkpoints (.pth)
├── results/               # Logs, metrics, and plots
└── docs/                  # Paper draft (PDF) and figures
    └── GSNet_Paper.pdf    # Full manuscript
```

## Reproducibility
- **Environment:** Tested on Ubuntu 20.04 with NVIDIA RTX 3080 (11 GB VRAM).
- **Seeds:** Set random seeds (e.g., 42) in `config.yaml` for deterministic results.
- **Metrics Logging:** Uses TensorBoard; run `tensorboard --logdir results/logs` to visualize training curves.
- **Dataset Compliance:** TUSZ usage adheres to its non-commercial research license.

## Contributing
Contributions are welcome! For bug fixes, new features (e.g., support for CHB-MIT dataset), or improvements:
1. Fork the repo and create a branch (`git checkout -b feature-branch`).
2. Commit changes (`git commit -m "Add feature"`).
3. Push and open a Pull Request.

Please reference the paper's methodology for any modifications.

## License


## Citation
If you use GSNet in your work, please cite:
```
@article{ali2025gsnet,
  title={A Unified Graph-Based Deep Learning Framework for Multi-Stage Seizure Monitoring},
  author={Ali, Waqas and Shahbaz, Muhammad},
  journal={Medical & Biological & Engineering & Computing (Under Review)},
  year={2025},
  doi={...}  % Update upon acceptance
}
```

## Authors and Contact
- **Waqas Ali** (Corresponding): waqas.ali2@uet.edu.pk
- **Muhammad Shahbaz**: m.shahbaz@uet.edu.pk

## Acknowledgements
- Supported by the Department of Computer Engineering, UET Lahore.
- Grateful to the TUH EEG team for the dataset.

## Future Plans
- Integration with real-time EEG wearables.
- Extension to cross-dataset generalization (e.g., CHB-MIT).
- Enhanced interpretability via advanced XAI techniques.
