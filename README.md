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
├── demo/                          # Example SVS inputs for detection-stage validation
├── train/                         # Training scripts, from SVS tiling to cancer classification
├── Soft/
│   ├── shilab_pipeline/           # YOLO-based SVS-to-cell-image inference step
│   ├── shilab-binary-classifier/  # Benign/malignant classifier package
│   ├── shilab-cluster-algorithm/  # Cluster matching / false-positive removal package
│   └── shilab-cancer-classifier/  # Four-class cancer classifier package
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
| 5 | `train/step5.train_cancer_classifier.py` | Train the four-class cancer classifier |

The cancer classes used in `step5` are: `Gastrointestinal_Breast`,
`Gynecologic`, `Lung`, and `Mesothelioma`.

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

Steps 3--5 also require the shared `shilab-binary-classifier`,
`shilab-cluster-algorithm`, and `shilab-cancer-classifier` packages installed
from the local `Soft/` directory above.

`openslide-python` also requires the native OpenSlide library. On Windows,
install the OpenSlide binaries and make their DLL directory available on
`PATH` before processing SVS files.

## Run the included demo

The repository includes two SVS files in `demo/` (`malignant.svs` and
`benign.svs`) for validating the streaming SVS-to-cell-detection stage.
They are example inputs only: no labels, clinical metadata, reference-cell
library, or expected diagnostic result is supplied.

## End-to-end inference workflow

Run every command from the repository root. The commands below use
`malignant.svs` as the example and are equally applicable to
`benign.svs` after replacing `malignant` in all paths. Output names are
illustrative; use a fresh output root for each run.

### 1. SVS to candidate cell crops

```bash
python Soft/shilab_pipeline/application/step1_2_yolo_model_effusion.py \
  --input_file demo/malignant.svs \
  --output_dir demo/run_malignant \
  --model_path models/yolo/best.pt \
  --grid_size 3 \
  --read_workers 4 \
  --yolo_batch_size 8 \
  --imgsz 1024 \
  --infer_conf 0.1 \
  --infer_iou 0.3 \
  --conf 0.5 \
  --no_save_json
```

This is the demo command shown above. Its required hand-off directories are:

```text
demo/run_malignant/malignant/result/single_cell/
demo/run_malignant/malignant/result/cluster/
```

### 2. Binary benign/malignant classification

Use the deployment parameters that correspond to each released checkpoint.

```bash
python -m shilab_classifier.infer.binary_infer \
  --model DenseNet161 \
  --weights models/binary-classifier/densenet161_fold_4.pth \
  --input demo/run_malignant/malignant/result/single_cell \
  --output demo/run_malignant/binary_infer/single_cell/malignant \
  --mean 0.5665 0.7001 0.7650 \
  --std 0.3161 0.2253 0.1554 \
  --threshold 0.5 --batch-size 16 --workers 0

python -m shilab_classifier.infer.binary_infer \
  --model MobileNetV2 \
  --weights models/binary-classifier/mobilenetv2_fold_2.pth \
  --input demo/run_malignant/malignant/result/cluster \
  --output demo/run_malignant/binary_infer/cluster/malignant \
  --mean 0.6252 0.7208 0.7815 \
  --std 0.3338 0.2548 0.1815 \
  --threshold 0.5 --batch-size 16 --workers 0
```

### 3. Cluster matching and false-positive removal

The package expects a root directory whose immediate children are sample
directories, each containing `malignant_images/`. `positive_folder_path`
and `negative_folder_path` only label rows in the output statistics; they do
not alter matching. For an unlabelled sample, put it below either root and
provide an existing empty directory (shown as `/path/to/empty_cases`) for the
other.

```bash
python -m cross_cluster_matching.application.pipeline \
  --positive_folder_path demo/run_malignant/binary_infer/single_cell \
  --negative_folder_path /path/to/empty_cases \
  --reference_cache_dir /path/to/released_reference_cache/single_cell_densenet161_pca32_top4_article_v1 \
  --model_path models/cluster/densenet161_fold_4.pth \
  --base_save_dir demo/run_malignant/cluster_fp_removal/single_cell \
  --model_type DenseNet161 \
  --mean 0.5665 0.7001 0.7650 --std 0.3161 0.2253 0.1554 \
  --malignant_ref_clusters 7,0 \
  --rules_str "7:0:0.8:3:none@0:7:0.6:3:none" \
  --use_cell_level_rule_filter

python -m cross_cluster_matching.application.pipeline \
  --positive_folder_path demo/run_malignant/binary_infer/cluster \
  --negative_folder_path /path/to/empty_cases \
  --reference_cache_dir /path/to/released_reference_cache/cluster_mobilenetv2_pca16_top4_article_v1 \
  --model_path models/cluster/mobilenetv2_fold_2.pth \
  --base_save_dir demo/run_malignant/cluster_fp_removal/cluster \
  --model_type MobileNetV2 --pca_dim 16 \
  --mean 0.6252 0.7208 0.7815 --std 0.3338 0.2548 0.1815 \
  --malignant_ref_clusters 2,0 --max_candidate_k 10 \
  --rules_str "2:0:0.65:20:none@0:2:0.5:20:0" \
  --use_cell_level_rule_filter
```

The reference cache must match the checkpoint, preprocessing, PCA dimension,
and reference-cluster IDs in the command. Without a cache, replace
`--reference_cache_dir` with private `--malignant_cells_dir` and
`--benign_cells_dir` paths.

For each processed cell type, the retained images are written to:

```text
demo/run_malignant/cluster_fp_removal/<cell_type>/malignant/matched_malignant_cells/
```

### 4. Four-class cancer classification

Run this step only for a cell type whose `matched_malignant_cells/` image
format matches the cancer model's training data. The model has four outputs in
this exact order: `Gastrointestinal_Breast`, `Gynecologic`, `Lung`, and
`Mesothelioma`.

```bash
python -m shilab_cancer_classifier.infer.cancer_infer \
  --model ViT_L16 \
  --weights models/cancer-classifier/vit_l16.pth \
  --input demo/run_malignant/cluster_fp_removal/single_cell/malignant/matched_malignant_cells \
  --output demo/run_malignant/cancer_infer/single_cell \
  --classes Gastrointestinal_Breast,Gynecologic,Lung,Mesothelioma \
  --mean <training_mean_r> <training_mean_g> <training_mean_b> \
  --std <training_std_r> <training_std_g> <training_std_b> \
  --threshold 0.5 --batch-size 16 --workers 0
```

The cancer classifier requires its own fold-specific training `mean` and
`std`; these values are not interchangeable with the two binary-classifier
parameter sets above. Do **not** substitute ImageNet defaults. Obtain the
values from the log of the exact `ViT_L16` checkpoint, replace the six
placeholders, and retain that log with the released model. Outputs include
`prediction_results.csv`, patient-level summaries, probability plots, and
high-confidence image copies.

## Quick start: training

Run the commands from the repository root. Before each script, replace its
`/path/to/...` settings with your local input and output paths. Steps 3--5
require the three packages installed from `Soft/`.

```bash
pip install -e Soft/shilab-binary-classifier
pip install -e Soft/shilab-cluster-algorithm
pip install -e Soft/shilab-cancer-classifier

# 1. Extract SVS patches
python train/step1.svsSplit.py --input_file /path/to/svs --output_dir /path/to/patches

# 2. Prepare annotations and train the YOLO detector
python train/step2.1.prepare_dataset.py
python train/step2.2.train_yolo.py

# 3. Train the binary benign/malignant classifier
python train/step3.train_classifier.py

# 4. Build/evaluate the reference clustering used for false-positive removal
python train/step4.top4_models_clustering.py

# 5. Train the four-class cancer classifier
python train/step5.train_cancer_classifier.py
```

| Script | Configure before running |
| --- | --- |
| `step2.1.prepare_dataset.py` | `input_dir`, `output_dir` |
| `step2.2.train_yolo.py` | `YAML_PATH`, `TRAIN_ARGS['project']` |
| `step3.train_classifier.py` | binary data and output paths |
| `step4.top4_models_clustering.py` | trained binary models, evaluation results, and reference-cell paths |
| `step5.train_cancer_classifier.py` | four-class data and output paths |

The Step 5 training data must use these exact folders:
`Gastrointestinal_Breast/`, `Gynecologic/`, `Lung/`, and
`Mesothelioma/`.

## Data, configuration, and checkpoints

The repository includes the two SVS demo inputs described above. No image
labels, clinical metadata, training source images, or reference-cell libraries
are included. Before running the full inference workflow, provide and
configure:

- an input SVS file and a writable output directory;
- a Python environment containing the packages in `requirements.txt`, plus the
  native OpenSlide library;
- binary-classifier checkpoint architecture, fold-specific `mean`/`std`, and
  its corresponding training log;
- Step 4 malignant/benign reference-cell folders or a versioned
  `reference_cache`, matching rules, and the matching-model checkpoint; and
- the four-class cancer checkpoint, its fold-specific `mean`/`std`, and
  training log.

The available repository files are sufficient for the detection demo above,
but the omitted reference assets and normalization records are required for a
scientifically reproducible end-to-end prediction.

The packages expose command-line inference entry points. For example, inspect
their options with:

```bash
python -m shilab_classifier.infer.binary_infer --help
python -m shilab_cancer_classifier.infer.cancer_infer --help
python -m cross_cluster_matching.application.pipeline --help
```

The multi-cancer inference stage consumes PNG images below the Step 4 output
folder `matched_malignant_cells/`. The released model must have four outputs in
this exact ImageFolder order: `Gastrointestinal_Breast`, `Gynecologic`,
`Lung`, and `Mesothelioma`.

```bash
shilab-cancer-infer \
  --model ViT_L16 \
  --weights /path/to/vit_l16.pth \
  --input /path/to/step4_output \
  --output /path/to/cancer_inference \
  --classes Gastrointestinal_Breast Gynecologic Lung Mesothelioma \
  --mean <training_mean_r> <training_mean_g> <training_mean_b> \
  --std <training_std_r> <training_std_g> <training_std_b>
```

Replace the normalization placeholders with the values in the log for the
specific training fold. Publish the checkpoint and its corresponding log via
Git LFS or a versioned release before claiming a fully reproducible workflow.

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
- Add an end-to-end runner/configuration template, released reference cache,
  fold-specific normalization logs, and multi-cancer checkpoint metadata to
  support one-command reproducibility beyond the detection demo.
- Verify that all model, dataset, and third-party-code licenses permit public
  redistribution.
- Choose and add a repository license only after all rights holders agree on
  it. Until then, no permission to reuse the code is granted by this repository.
