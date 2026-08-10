# Cross-domain Cluster Matching Pipeline

This project implements a cross-domain cluster matching pipeline for analyzing cell images.

## Installation

Install the package in development mode:

```bash
cd  path/to/shilab-cluster-algorithm
pip install -e .
```

## Usage

After installation, you can run the analysis pipeline:

```python
from cross_cluster_matching.application.pipeline import run_pipeline

# Run the pipeline
run_pipeline(
    positive_folder_path="path/to/positive/folders",
    negative_folder_path="path/to/negative/folders", 
    malignant_cells_dir="path/to/malignant/cells",
    benign_cells_dir="path/to/benign/cells",
    model_path="path/to/model",
    base_save_dir="path/to/save/results",
    feature_layer='penultimate',
    reference_k=10,
    max_candidate_k=20,
    batch_size=16,
    num_workers=0,
    cell_size_weight=1.0,
    model_type='ResNeXt',
    pca_dim=32
)
```

## Features

### Module Structure

The package is organized into modular folders for easy maintenance and development:

- **`cross_cluster_matching/`**: Main package directory containing all modules

- **`clustering/`**: Clustering-related functions
  - `calculate_consensus_score.py`: Functions for calculating matching cluster consensus scores
  - `K_optimizer.py`: K-value optimization related functions
  - `matching_rule_application.py`: Matching rule setting and application functions

- **`data/`**: Image data processing functions
  - `data_utils.py`: Image data processing related utilities

- **`models/`**: Model construction and feature extraction
  - `models.py`: Model class construction functions
  - `feature_extractor.py`: Feature extraction related functions

- **`visualization/`**: Visualization functions
  - `visualization.py`: Visualization related functions

- **`application/`**: Code implementation functions (under active development)
  - `qrr_pipeline.py`: Main pipeline function for QRR graduate thesis running the complete analysis
  - *Additional functions to be added as development progresses*

## Project Structure

```
cross_cluster_matching/
├── clustering/           # Clustering algorithms and optimization
├── data/                # Image data processing utilities  
├── models/              # Feature extraction models
├── visualization/       # Visualization functions
├── application/         # Main implementation functions (active development)
└── __init__.py         # Package initialization
```
