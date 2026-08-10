# effusionCytoAI

`effusionCytoAI` is a research pipeline for cytology images of pleural and
peritoneal effusions. It covers slide tiling, cell detection, binary
benign/malignant classification, false-positive removal by cluster matching,
and multi-cancer classification.

> **Research use only.** This repository is not a medical device and must not
> be used as the sole basis for clinical diagnosis or treatment decisions.

## Repository layout

```text
.
├── train/                         # Training scripts, from SVS tiling to cancer classification
├── Soft/
│   ├── shilab_pipeline/           # YOLO-based SVS-to-cell-image inference step
│   ├── shilab-binary-classifier/  # Benign/malignant classifier package
│   ├── shilab-cluster-algorithm/  # Cluster matching / false-positive removal package
│   └── shilab-cancer-classifier/  # Multi-cancer classifier package
└── models/
    ├── yolo/                      # Cell detection model
    ├── binary-classifier/         # Binary-classification checkpoints
    └── cluster/                   # Checkpoints used by the matching workflow
```

## Workflow

```text
SVS whole-slide image
  → tile extraction and YOLO detection
  → benign/malignant classification
  → cluster matching and false-positive removal
  → matched_malignant_cells/*.png
  → multi-cancer classification
```

The training scripts are ordered by stage:

| Stage | Script | Purpose |
| --- | --- | --- |
| 1 | `train/step1.svsSplit.py` | Split SVS images into patches |
| 2.1 | `train/step2.1.prepare_dataset.py` | Prepare a YOLO dataset |
| 2.2 | `train/step2.2.train_yolo.py` | Train the cell detector |
| 3 | `train/step3.train_classifier.py` | Train the binary classifier |
| 4 | `train/step4.top4_models_clustering.py` | Train/evaluate matching models |
| 5 | `train/step5.train_cancer_classifier.py` | Train the 12-class cancer classifier |

The cancer classes used in `step5` are: `Bile_duct`, `Breast`, `Cervix`,
`Colorectum`, `Endometrium`, `Esophagus`, `Gastric`, `Mesothelioma`, `NSCLC`,
`Ovary`, `Pancreas`, and `SCLC`.

## Installation

Use a supported Python version (the code was developed with Python 3.10 or a
compatible version). Install PyTorch and TorchVision builds that match your
CUDA/driver environment first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install -e Soft/shilab-binary-classifier
pip install -e Soft/shilab-cluster-algorithm
pip install -e Soft/shilab-cancer-classifier
```

`openslide-python` also requires the native OpenSlide library. On Windows,
install the OpenSlide binaries and make their DLL directory available on
`PATH` before processing SVS files.

## Training workflow

Run the following commands from the repository root. The training scripts use
public placeholder paths such as `/path/to/your/raw_data`; no developer-local
drive paths are stored in `train/`. Before running a script without command
line arguments, replace only the path constants at the top of that script with
your own local paths. Use a new output directory for each experiment: the
dataset split functions recreate their output folders.

Install the three local packages once before Steps 3--5:

```bash
pip install -e Soft/shilab-binary-classifier
pip install -e Soft/shilab-cluster-algorithm
pip install -e Soft/shilab-cancer-classifier
```

### Step 1 — extract representative SVS patches

`train/step1.svsSplit.py` accepts either one `.svs` file or a directory of
`.svs` files. It reads level-0 pixels and retains the centre patch from each
`grid_size × grid_size` group; it is therefore a sampling/extraction utility,
not an exhaustive tiler.

```bash
python train/step1.svsSplit.py \
  --input_file /path/to/svs_files \
  --output_dir /path/to/patches \
  --size 1024 --overlap 0.05 --grid_size 3 --threads 8
```

Output PNG patches are written as `/path/to/patches/<slide_name>/tile_<row>_<col>.png`.
OpenSlide must be installed for this step.

### Step 2.1 — convert LabelMe annotations to a YOLO dataset

Prepare one flat input directory containing image files and LabelMe JSON files
with matching basenames. Set `input_dir` and `output_dir` at the top of
`train/step2.1.prepare_dataset.py`, then run:

```bash
python train/step2.1.prepare_dataset.py
```

The script uses a 70%/20%/10% train/validation/test split with seed 42, maps
`single_cell`, `cluster`, `impurity`, `part`, and `vague` to class IDs 0--4,
and writes `images/`, YOLO `labels/`, `classes.txt`, `custom.yaml`, and a split
chart below the output directory. By default `keep_json=True`. For an image
without a JSON file the script creates an empty LabelMe JSON **in the input
directory**; use a writable copy of the annotations if that is undesirable.

### Step 2.2 — train the YOLO detector

In `train/step2.2.train_yolo.py`, set `YAML_PATH` to the `custom.yaml` produced
by Step 2.1 and set `TRAIN_ARGS['project']` to an experiment output directory.
Optionally replace `MODEL_CONFIG = "yolov12s.yaml"` with a pretrained checkpoint.

```bash
python train/step2.2.train_yolo.py
```

The current defaults train for 400 epochs at image size 1024, batch size 8, on
CUDA device 0. Ultralytics writes the run (including `weights/best.pt`) under
`<project>/train_exp01/`.

### Step 3 — train and evaluate the benign/malignant classifier

Set `RAW_DATA_DIR`, `BASE_OUTPUT_DIR`, and, if needed,
`ORIGINAL_BENIGN_TEST_DIR` in `train/step3.train_classifier.py`. The raw data
root must contain the following folders:

```text
/path/to/your/raw_data/
├── benign/
└── malignant/
```

Then run:

```bash
python train/step3.train_classifier.py
```

The script performs a 90% train/validation split, creates five cross-validation
folds, trains every model in `SUPPORTED_MODELS` for 50 epochs by default, and
evaluates them on the held-out test set. Outputs include `split_data/`,
`cross_validation_data/`, `train_val_models/`, and `evaluation_results/` below
`BASE_OUTPUT_DIR`. To train fewer architectures, replace `MODELS_TO_TRAIN` with
a list such as `["DenseNet161"]`.

### Step 4 — analyse the top four binary models by reference clustering

This is a post-training model-selection and reference-clustering analysis; it
does not train a new classifier. Set `MODELS_FOLDER`, `EVALUATION_FOLDER`,
`OUTPUT_BASE_DIR`, `MALIGNANT_CELLS_DIR`, and `BENIGN_CELLS_DIR` in
`train/step4.top4_models_clustering.py`, then run:

```bash
python train/step4.top4_models_clustering.py
```

`EVALUATION_FOLDER` must contain `top4_models_summary.csv` and each selected
model's `<ModelName>/<ModelName>_detailed_results.csv`; `MODELS_FOLDER` must
contain the corresponding `<ModelName>/<modelname>_fold_<n>.pth` files. The
malignant reference directory may contain `LUAD/`, `LUSC/`, and `SCLC/`
subdirectories; the benign reference directory contains benign cell images.
For every selected model, the script reads same-name `.log` normalization
statistics when available, extracts features, applies PCA plus KMeans, and
saves reference-cluster visualizations.

### Step 5 — train and evaluate the multi-cancer classifier

Set `RAW_DATA_DIR` and `BASE_OUTPUT_DIR` in
`train/step5.train_cancer_classifier.py`. The input root must have one image
folder for each of these exact class names:

```text
Bile_duct/  Breast/  Cervix/  Colorectum/  Endometrium/  Esophagus/
Gastric/    Mesothelioma/  NSCLC/  Ovary/  Pancreas/  SCLC/
```

Run the complete pipeline with:

```bash
python train/step5.train_cancer_classifier.py
```

It validates that every class has at least five images, creates a 90%/10%
train-validation/test split, generates five folds, trains the architectures in
`MODELS_TO_TRAIN`, and writes evaluation results. The boolean switches
`RUN_SPLIT`, `RUN_CV`, `RUN_TRAIN`, and `RUN_EVAL` let you run only the needed
stage when resuming an experiment.

## Data, configuration, and checkpoints

No source images, patient information, labels, or reference-cell libraries are
included. Before running a workflow, configure local paths for:

- input SVS files and output directory;
- the YOLO and Python environments;
- Step 4 reference-cell folders/cache; and
- model checkpoints and normalization statistics.

The packages expose command-line inference entry points. For example, inspect
their options with:

```bash
python -m shilab_classifier.infer.binary_infer --help
python -m shilab_cancer_classifier.infer.cancer_infer --help
python -m cross_cluster_matching.application.pipeline --help
```

The multi-cancer inference stage consumes PNG images below the Step 4 output
folder `matched_malignant_cells/`. The appropriate cancer-model checkpoint is
not currently present in `models/`; add the released checkpoint together with
its class order, image-normalization mean, and standard deviation before
publishing a fully reproducible multi-cancer workflow.

## Large model files

The repository contains model checkpoints larger than GitHub's ordinary Git
file-size limit. Install and use Git LFS before committing them:

```bash
git lfs install
git lfs track "*.pt" "*.pth"
git add .gitattributes models/
```

Alternatively, publish checkpoints as versioned GitHub Release assets and
document their download locations here. Do not commit raw clinical data or
private reference images.

For reference, the 102.21 MiB DenseNet161 checkpoint was tested with ZIP
compression and became 97.48 MiB, so an archive can fit below GitHub's 100 MiB
ordinary-Git file limit. This is only a fallback: users must extract the archive
before inference, and the original `.pth` must not be committed alongside it.
Git LFS is the recommended way to publish runnable checkpoints.

## Before public release

- Replace every `/path/to/...` placeholder in local copies or a private config;
  do not upload machine-specific paths, account names, or patient identifiers.
- Add the missing end-to-end runner/configuration template and the released
  multi-cancer checkpoint metadata if you want one-command reproducibility.
- Verify that all model, dataset, and third-party-code licenses permit public
  redistribution.
- Choose and add a repository license only after all rights holders agree on
  it. Until then, no permission to reuse the code is granted by this repository.
