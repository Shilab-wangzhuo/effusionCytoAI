# effusionCytoAI

`effusionCytoAI` is a research pipeline for pleural and peritoneal effusion
cytology whole-slide images. It covers slide tiling, cell detection, binary
benign/malignant classification, clustering-based false-positive removal, and
four-class cancer classification.

> **Research use only.** This software is not a medical device and must not be
> used as the sole basis for clinical diagnosis or treatment decisions.

## Repository layout

```text
.
├── demo/                          # Example SVS inputs
├── train/                         # Training scripts for all workflow stages
├── Soft/
│   ├── shilab_pipeline/           # YOLO-based SVS-to-cell-image inference
│   ├── shilab-binary-classifier/  # Benign/malignant classifier package
│   ├── shilab-cluster-algorithm/  # Cluster matching / false-positive removal
│   └── shilab-cancer-classifier/  # Four-class cancer classifier package
├── models/
│   ├── yolo/                      # Cell-detection checkpoint
│   ├── binary-classifier/         # Binary-classification checkpoints
│   ├── cluster/                   # Matching checkpoints and reference caches
│   └── cancer-classifier/         # Four-class cancer checkpoint
├── environment.yml                # Reproducible server inference environment
└── requirements.txt               # Pinned direct pip dependencies
```

## Workflow

```text
SVS whole-slide image
  → patch extraction and YOLO detection
  → benign/malignant classification
  → cluster matching and false-positive removal
  → matched_malignant_cells/*.png
  → four-class cancer classification
```

| Stage | Script | Purpose |
| --- | --- | --- |
| 1 | `train/step1.svsSplit.py` | Extract representative patches from SVS slides |
| 2.1 | `train/step2.1.prepare_dataset.py` | Convert annotations into a YOLO dataset |
| 2.2 | `train/step2.2.train_yolo.py` | Train the cell detector |
| 3 | `train/step3.train_classifier.py` | Train the binary classifier |
| 4 | `train/step4.top4_models_clustering.py` | Build and evaluate reference-cell clustering models |
| 5 | `train/step5.train_cancer_classifier.py` | Train the four-class cancer classifier |

The cancer classes, in model-output order, are `Gastrointestinal_Breast`,
`Gynecologic`, `Lung`, and `Mesothelioma`.

## Installation

The released end-to-end inference workflow was run on the server in a single
`patho_pipeline` environment. Create the same environment with:

```bash
conda env create -f environment.yml
conda activate patho_pipeline
```

Install the repository versions of the three local packages:

```bash
pip install -e Soft/shilab-binary-classifier
pip install -e Soft/shilab-cluster-algorithm
pip install -e Soft/shilab-cancer-classifier
```

`environment.yml` is the authoritative specification for the released
inference workflow. `requirements.txt` contains the corresponding pinned
direct pip dependencies for users who manage native libraries separately.
`openslide-python` requires the native OpenSlide library; the Conda environment
installs OpenSlide 4.0.1.

The workflow was validated on CentOS Linux 7 with an NVIDIA GeForce RTX 3080
(10 GB), NVIDIA driver 550.54.14, Python 3.10.20, PyTorch 2.2.1 with CUDA 11.8,
TorchVision 0.17.1, Ultralytics 8.3.63, NumPy 1.26.4, pandas 2.0.3,
scikit-learn 1.3.0, SciPy 1.10.1, OpenCV 4.9.0, OpenSlide 4.0.1,
openslide-python 1.4.1, UMAP-learn 0.5.7, and Supervision 0.25.1.

## Run the included demo

The repository includes `demo/malignant.svs` and `demo/benign.svs`. The
commands below use `malignant.svs`; replace `malignant` with `benign` in the
input and output paths to process the other slide. Run all commands from the
repository root. The slides are example inputs and do not include diagnostic
labels or clinical metadata.

## End-to-end inference

### 1. SVS to candidate cell crops

```bash
python Soft/shilab_pipeline/application/step1_2_yolo_model_effusion.py \
  --input_file demo/malignant.svs \
  --output_dir demo/run_malignant \
  --model_path models/yolo/best.pt \
  --grid_size 1 \
  --read_workers 4 \
  --yolo_batch_size 8 \
  --imgsz 1024 \
  --infer_conf 0.1 \
  --infer_iou 0.3 \
  --conf 0.5 \
  --no_save_json
```

The directories passed to the next stage are:

```text
demo/run_malignant/malignant/result/single_cell/
demo/run_malignant/malignant/result/cluster/
```

### 2. Binary benign/malignant classification

```bash
python -m shilab_classifier.infer.binary_infer \
  --model DenseNet161 \
  --weights models/binary-classifier/densenet161_fold_4.pth \
  --input demo/run_malignant/malignant/result/single_cell \
  --output demo/run_malignant/binary_infer/single_cell/malignant \
  --mean 0.5665 0.7001 0.7650 \
  --std 0.3161 0.2253 0.1554

python -m shilab_classifier.infer.binary_infer \
  --model MobileNetV2 \
  --weights models/binary-classifier/mobilenetv2_fold_2.pth \
  --input demo/run_malignant/malignant/result/cluster \
  --output demo/run_malignant/binary_infer/cluster/malignant \
  --mean 0.6252 0.7208 0.7815 \
  --std 0.3338 0.2548 0.1815
```

Each output directory contains `prediction_results.csv`, a probability plot,
and `malignant_images/` containing images retained as malignant.

### 3. Cluster matching and false-positive removal

The matching program expects each immediate child of
`positive_folder_path` or `negative_folder_path` to be a sample directory
containing `malignant_images/`. For this unlabelled demo, the sample is placed
below the positive root and an empty negative root is created. These roots
only label summary rows; they do not change the matching algorithm.

```bash
mkdir -p demo/empty_cases

python -m cross_cluster_matching.application.pipeline \
  --positive_folder_path demo/run_malignant/binary_infer/single_cell \
  --negative_folder_path demo/empty_cases \
  --reference_cache_dir models/cluster/reference_cache/single_cell_densenet161_pca32_top4_article_v1 \
  --model_path models/cluster/densenet161_fold_4.pth \
  --base_save_dir demo/run_malignant/cluster_fp_removal/single_cell \
  --model_type DenseNet161 \
  --mean 0.5665 0.7001 0.7650 \
  --std 0.3161 0.2253 0.1554 \
  --malignant_ref_clusters 7,0 \
  --rules_str "7:0:0.8:3:none@0:7:0.6:3:none" \
  --use_cell_level_rule_filter

python -m cross_cluster_matching.application.pipeline \
  --positive_folder_path demo/run_malignant/binary_infer/cluster \
  --negative_folder_path demo/empty_cases \
  --reference_cache_dir models/cluster/reference_cache/cluster_mobilenetv2_pca16_top4_article_v1 \
  --model_path models/cluster/mobilenetv2_fold_2.pth \
  --base_save_dir demo/run_malignant/cluster_fp_removal/cluster \
  --model_type MobileNetV2 \
  --pca_dim 16 \
  --mean 0.6252 0.7208 0.7815 \
  --std 0.3338 0.2548 0.1815 \
  --malignant_ref_clusters 2,0 \
  --rules_str "2:0:0.65:20:none@0:2:0.5:20:0" \
  --use_cell_level_rule_filter
```

The final retained images are written to:

```text
demo/run_malignant/cluster_fp_removal/single_cell/malignant/matched_malignant_cells/
demo/run_malignant/cluster_fp_removal/cluster/malignant/matched_malignant_cells/
```

### 4. Four-class cancer classification

The released Fold 3 `ViT_L16` checkpoint uses the following exact class order
and fold-specific normalization values:

```bash
python -m shilab_cancer_classifier.infer.cancer_infer \
  --model ViT_L16 \
  --weights models/cancer-classifier/vit_l16.pth \
  --input demo/run_malignant/cluster_fp_removal/single_cell/malignant/matched_malignant_cells \
  --output demo/run_malignant/cancer_infer/single_cell \
  --classes Gastrointestinal_Breast Gynecologic Lung Mesothelioma \
  --mean 0.4908 0.5972 0.6659 \
  --std 0.3262 0.2490 0.1852 \
  --threshold 0.5 --batch-size 4 --workers 0
```

Outputs include `prediction_results.csv`, patient-level summaries,
probability plots, and high-confidence image copies. Run this stage only on
the single-cell `matched_malignant_cells/` output, which matches the cancer
model's training image type.

## Quick start: training

The released models were trained locally. YOLO development used Python 3.11.11,
PyTorch 2.5.1 with CUDA 12.1, and Ultralytics 8.3.63; binary-classifier,
reference-clustering, and cancer-classifier development used PyTorch 2.2.1
with CUDA 11.8 and TorchVision 0.17.1.

```bash
# 1. Extract representative SVS patches
python train/step1.svsSplit.py --input_file /path/to/svs --output_dir /path/to/patches

# 2. Prepare annotations and train the YOLO detector
python train/step2.1.prepare_dataset.py
python train/step2.2.train_yolo.py

# 3. Train the binary benign/malignant classifier
python train/step3.train_classifier.py

# 4. Build and evaluate reference-cell clustering
python train/step4.top4_models_clustering.py

# 5. Train the four-class cancer classifier
python train/step5.train_cancer_classifier.py
```

| Script | Configure before running |
| --- | --- |
| `step2.1.prepare_dataset.py` | `input_dir`, `output_dir` |
| `step2.2.train_yolo.py` | `YAML_PATH`, `TRAIN_ARGS['project']` |
| `step3.train_classifier.py` | raw binary dataset and output paths |
| `step4.top4_models_clustering.py` | trained binary models, evaluation results, and effusion reference-cell paths |
| `step5.train_cancer_classifier.py` | four-class dataset and output paths |

Step 3 expects `benign/` and `malignant/` directories. Step 5 expects the
four class directories listed above. These scripts intentionally require the
user to set project-specific data and output paths before running.

## Data, configuration, and checkpoints

The repository supplies example SVS inputs, all inference checkpoints, and
frozen reference caches for both single cells and cell clusters. The caches
contain the fixed numerical representations required by the matching
algorithm; the original reference images are not required for inference.

Training source images, clinical metadata, and patient-level annotations are
not distributed. Users training new models must recalculate normalization
statistics and keep the model architecture, class order, reference cache,
cluster IDs, and matching rules synchronized with each new checkpoint.

Model checkpoints and SVS demo files are stored with Git LFS. Install Git LFS
before cloning or downloading these assets:

```bash
git lfs install
git lfs pull
```

Verify that all model, dataset, and third-party-code licenses permit the
intended use. A repository license should be added only after all rights
holders agree on the redistribution terms.
