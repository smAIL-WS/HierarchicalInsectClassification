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
   python3 -m venv venv
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

## Reproducing the Published Results

This is the recommended route to reproduce the paper's results directly, without re-splitting data or re-running hyperparameter optimisation.

1. Download from Zenodo:
   - The insect image dataset
   - The DuckDB metadata database (`insect_images_public.duckdb`)
   - The published split/hierarchy files (`2026-01-19_portable`)
   - The trained model checkpoint

2. In `insect_hier_class/config.py`, set:

   ```python
   DATASET_ROOT = Path("/path/to/IHC_dataset")
   DUCKDB_PATH = Path("/path/to/insect_images_public.duckdb")
   RUN_DATE = "2026-01-19_portable"
   ```

3. Place the downloaded split/hierarchy files in:

   ```
   insect_hier_class/runs/2026-01-19_portable/
   ```

4. Place the downloaded model checkpoint in:

   ```
   insect_hier_class/models_Insect/
   ```

5. Run evaluation, pointing `--weights_path` at the downloaded checkpoint:

   ```bash
   python insect_hier_class/analysis_metrics.py --weights_path insect_hier_class/models_Insect/model_Insect_100_96_bz1024_resnet18_OneCycle_2026-01-19_01-09.pth
   ```

This performs the complete evaluation procedure reported in the paper (hierarchical MAP inference, confidence-thresholded backoff, coverage analysis, per-level and per-class metrics, confusion matrices) and writes the results described below.

## Training a Model from Scratch

If you want to train rather than use the provided checkpoint:

```bash
python main.py --backbone resnet18 --use_pretrained
```

The paper's final model was trained with this backbone for 100 epochs; see `insect_hier_class/config.py` for the full training configuration. Once training completes, pass the resulting `.pth` checkpoint to `analysis_metrics.py` via `--weights_path` as shown in the evaluation step above.

## Evaluation Outputs

Running `analysis_metrics.py` produces:

```
level_summary.csv          # per-level metrics
class_metrics.csv          # per-class metrics
confusion_coverage.xlsx
confusion_extended.xlsx
confusion_matrices.pdf
confusion_matrices_extended.pdf
```
