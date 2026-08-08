# Deep Learning-Based Hierarchical Insect Classification Using Camera Trap Imagery

Source code for dataset preprocessing, model training, and evaluation for hierarchical insect classification from camera trap imagery.

## Associated Resources

- **Paper:** Mahfoud et al., *Deep learning-based hierarchical insect classification using camera trap imagery* (submitted for publication, 2026). Pre-print: https://arxiv.org/abs/2607.28005
- **Zenodo Archive** (dataset, metadata database, trained weights, splits, reproducibility artefacts): https://doi.org/10.5281/zenodo.21765014

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/smAIL-WS/HierarchicalInsectClassification.git
   cd HierarchicalInsectClassification
   ```

2. Create and activate a Python virtual environment (developed and tested on Python 3.13.2):

   ```bash
   python3.13 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   For exact reproduction of the published environment, install the package versions listed in `environment.txt` instead:

   ```bash
   pip install -r environment.txt
   ```

   `environment.txt` pins CUDA 12.8 builds of PyTorch (e.g. `torch==2.9.1+cu128`), which are only published for 64-bit Linux and require a Linux/x86_64 machine with an NVIDIA GPU (driver supporting CUDA 12.8 or newer). The download is several GB, so make sure you have at least 10 GB of free disk space before installing.

## Reproducing the Published Results

This is the recommended route to reproduce the paper's results directly, without re-splitting data or re-running hyperparameter optimisation.

1. Download and extract everything from Zenodo into a single `data/` folder at the repository root:

   ```bash
   ./download_data.sh
   ```

   This fetches the dataset, metadata database, splits, and trained checkpoint, and extracts the archives, so `data/` looks like this:

   ```
   data/
   ├── IHC_dataset/                                                    # the insect image dataset
   ├── insect_images_public.duckdb                                     # the DuckDB metadata database
   ├── 2026-01-19_portable/                                            # the published split/hierarchy files
   └── model_Insect_100_96_bz1024_resnet18_OneCycle_2026-01-19_01-09.pth  # the trained model checkpoint
   ```

   `insect_hier_class/config.py` points `DATASET_ROOT`, `DUCKDB_PATH`, and `RUN_DATE`/`RUN_FOLDER` at this `data/` folder by default. If you extract the downloaded files to a different location, update these variables in `config.py` accordingly.

2. Run evaluation:

   ```bash
   python insect_hier_class/analysis_metrics.py
   ```

   This defaults `--weights_path` to `data/model_Insect_100_96_bz1024_resnet18_OneCycle_2026-01-19_01-09.pth`; pass `--weights_path` explicitly to point at a different checkpoint.

This performs the complete evaluation procedure reported in the paper (hierarchical MAP inference, confidence-thresholded backoff, coverage analysis, per-level and per-class metrics, confusion matrices) and writes the results described below.

## Training a Model from Scratch

If you want to train rather than use the provided checkpoint:

```bash
python insect_hier_class/main.py --backbone resnet18 --use_pretrained
```

The paper's final model was trained with this backbone for 100 epochs; see `insect_hier_class/config.py` for the full training configuration. Training saves checkpoints into `data/` as `model_<dataset>_<epochs>_<img_size>_bz<batch>_<backbone>_<lr_adjt>_<timestamp>.pth`. Once training completes, pass the resulting `.pth` checkpoint to `analysis_metrics.py` via `--weights_path`:

```bash
python insect_hier_class/analysis_metrics.py --weights_path data/model_<dataset>_<epochs>_<img_size>_bz<batch>_<backbone>_<lr_adjt>_<timestamp>.pth
```

The results will be saved to `/analysis_paper`:

```
level_summary.csv          # per-level metrics
class_metrics.csv          # per-class metrics
confusion_coverage.xlsx
confusion_extended.xlsx
confusion_matrices.pdf
confusion_matrices_extended.pdf
```
