"""Model evaluation utilities for the 5-class cancer classifier."""

import os
import re
import time
import traceback

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from shilab_cancer_classifier.config import CANCER_CLASSES, NUM_CLASSES
from shilab_cancer_classifier.evaluation.metrics import (
    calculate_metrics,
    flatten_metrics,
    print_metrics,
)
from shilab_cancer_classifier.models.model_definitions import create_model
from shilab_cancer_classifier.training.preprocessing import (
    calculate_mean_std,
    pad_to_square_299_transform,
    pad_to_square_transform,
)

# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def _extract_logits(outputs):
    if hasattr(outputs, 'logits'):
        return outputs.logits
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _safe_class_name(class_name):
    safe_name = re.sub(r'\W+', '_', class_name.strip().lower())
    return safe_name.strip('_') or 'class'


def _safe_class_column_names(class_names):
    safe_names = [_safe_class_name(class_name) for class_name in class_names]
    duplicates = sorted({name for name in safe_names if safe_names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Class names produce duplicate output column names after normalization: "
            f"{duplicates}. Original class names: {class_names}"
        )
    return safe_names


def _build_transform(model_name, train_dir):
    mean, std = calculate_mean_std(train_dir, model_name)
    pad_transform = (
        pad_to_square_299_transform if 'Inception' in model_name
        else pad_to_square_transform
    )
    transform = transforms.Compose([
        transforms.Lambda(pad_transform),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform, mean, std


def _find_model_path(model_dir, model_name, fold):
    possible_names = [
        f'{model_name.lower()}_fold_{fold}.pth',
        f'{model_name}_fold_{fold}.pth',
        f'{model_name.lower()}_fold{fold}.pth',
        f'{model_name}_fold{fold}.pth',
        f'fold_{fold}.pth',
        f'fold{fold}.pth',
        f'model_fold_{fold}.pth',
    ]
    for name in possible_names:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return None


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

_CLASS_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
    '#66c2a5', '#fc8d62',
]


def _compute_ovr_roc(prob_csv_path, class_names):
    """Compute one-versus-rest ROC curves."""
    if not os.path.exists(prob_csv_path):
        return None

    try:
        df = pd.read_csv(prob_csv_path)
    except Exception as exc:
        print(f"  [ROC] Failed to read {prob_csv_path}: {exc}")
        return None

    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if len(prob_cols) != len(class_names):
        print(
            f"  [ROC] prob column count ({len(prob_cols)}) != "
            f"class_names count ({len(class_names)}), skipping."
        )
        return None

    true_labels = df['true_label'].values.astype(int)
    # label_binarize: shape (n_samples, n_classes)
    bin_labels = label_binarize(true_labels, classes=list(range(len(class_names))))

    mean_fpr = np.linspace(0, 1, 300)
    all_tpr_interp = []

    class_curves = {}
    for idx, (cls_name, prob_col) in enumerate(zip(class_names, prob_cols)):
        y_true_bin = bin_labels[:, idx]
        y_score = df[prob_col].values

        if len(np.unique(y_true_bin)) < 2:
            print(f"  [ROC] Class '{cls_name}' has only one label value, skipped.")
            continue

        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)
        class_curves[cls_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        all_tpr_interp.append(tpr_interp)

    if not all_tpr_interp:
        return None

    mean_tpr = np.mean(all_tpr_interp, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = auc(mean_fpr, mean_tpr)

    return {
        'class_curves': class_curves,
        'macro': {'fpr': mean_fpr, 'tpr': mean_tpr, 'auc': macro_auc},
        'n_samples': len(true_labels),
    }


def _draw_roc_axes(ax, roc_data, class_names, title, colors):
    """Draw ROC axes and baseline annotations."""
    # ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random (AUC = 0.50)', zorder=1)

    if roc_data is None:
        ax.text(0.5, 0.5, 'Data Not Available',
                ha='center', va='center', fontsize=11,
                color='gray', transform=ax.transAxes)
    else:
        class_curves = roc_data['class_curves']
        for idx, cls_name in enumerate(class_names):
            if cls_name not in class_curves:
                continue
            curve = class_curves[cls_name]
            color = colors[idx % len(colors)]
            display_name = cls_name.replace('_', ' ').title()
            ax.plot(
                curve['fpr'], curve['tpr'],
                color=color, lw=1.8, alpha=0.85,
                label=f'{display_name} (AUC = {curve["auc"]:.4f})',
            )

        # macro = roc_data['macro']
        # ax.plot(
        #     macro['fpr'], macro['tpr'],
        #     color='black', lw=2.5, linestyle='-.',
        #     label=f'Macro Average (AUC = {macro["auc"]:.4f})',
        # )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.3)


def plot_roc_curve_multiclass(
    model_results_dir,
    model_name,
    class_names,
    num_folds=5,
    save_dir=None,
    figsize_per_fold=(18, 6),
    dpi=200,
):
    """Plot multiclass ROC curves."""
    if save_dir is None:
        save_dir = os.path.join(model_results_dir, 'roc_curves')
    os.makedirs(save_dir, exist_ok=True)

    colors = _CLASS_COLORS[:len(class_names)]
    datasets_info = ['train', 'val', 'test']
    auc_records = []

    for fold in range(1, num_folds + 1):
        fold_dir = os.path.join(model_results_dir, f'fold_{fold}')

        fig, axes = plt.subplots(1, 3, figsize=figsize_per_fold)
        fig.suptitle(
            f'{model_name}  —  Fold {fold}  OvR ROC Curves',
            fontsize=13, fontweight='bold', y=1.01,
        )

        for ax, ds_name in zip(axes, datasets_info):
            csv_path = os.path.join(fold_dir, f'{ds_name}_probabilities.csv')
            roc_data = _compute_ovr_roc(csv_path, class_names)
            _draw_roc_axes(
                ax, roc_data, class_names,
                title=f'{ds_name.capitalize()} Set',
                colors=colors,
            )

            if roc_data is not None:
                row = {
                    'model': model_name,
                    'fold': fold,
                    'dataset': ds_name,
                    'macro_auc': roc_data['macro']['auc'],
                    'n_samples': roc_data['n_samples'],
                }
                for cls_name, curve in roc_data['class_curves'].items():
                    row[f'auc_{cls_name}'] = curve['auc']
                auc_records.append(row)

        plt.tight_layout()
        fold_save_path = os.path.join(save_dir, f'{model_name}_fold{fold}_roc.tif')
        plt.savefig(fold_save_path, format='tiff', dpi=300, pil_kwargs={'compression': 'none'})        
        plt.close()
        print(f"[ROC] Fold {fold} figure saved → {fold_save_path}")

    if auc_records:
        summary_df = pd.DataFrame(auc_records)

        fig, ax = plt.subplots(figsize=(9, 5))
        ds_styles = {
            'train': ('o-',  '#2196F3', 'Train'),
            'val':   ('s--', '#4CAF50', 'Validation'),
            'test':  ('D-.', '#F44336', 'Test'),
        }
        for ds_name, (style, color, label) in ds_styles.items():
            sub = summary_df[summary_df['dataset'] == ds_name].sort_values('fold')
            if sub.empty:
                continue
            ax.plot(
                sub['fold'], sub['macro_auc'],
                style, color=color, lw=2, ms=7,
                label=f'{label} Macro AUC',
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    f'{row["macro_auc"]:.4f}',
                    xy=(row['fold'], row['macro_auc']),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=7.5, color=color,
                )

        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel('Macro AUC (OvR)', fontsize=11)
        ax.set_title(f'{model_name} — Macro AUC per Fold', fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, num_folds + 1))
        ax.set_ylim([max(0, summary_df['macro_auc'].min() - 0.05), 1.02])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        summary_fig_path = os.path.join(save_dir, f'{model_name}_macro_auc_summary.png')
        plt.savefig(summary_fig_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"[ROC] Macro AUC summary figure saved → {summary_fig_path}")

        auc_csv_path = os.path.join(save_dir, f'{model_name}_roc_auc_summary.csv')
        summary_df.to_csv(auc_csv_path, index=False, encoding='utf-8-sig')
        print(f"[ROC] AUC summary CSV saved → {auc_csv_path}")

        return summary_df

    print(f"[ROC] No valid probability files found for {model_name}.")
    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def predict_and_save_probabilities(
    model,
    dataloader,
    dataset_name,
    save_dir,
    device='cuda',
    class_names=None,
):
    """Predict a dataset, save logits/probabilities, and return metrics."""
    class_names = list(class_names or CANCER_CLASSES)
    model.eval()

    all_logits, all_probs, all_preds, all_labels, all_paths = [], [], [], [], []

    print(f"Predicting {dataset_name} and saving probabilities...")

    with torch.no_grad():
        sample_offset = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  {dataset_name} batch {batch_idx}/{len(dataloader)}")

            inputs = inputs.to(device)
            logits = _extract_logits(model(inputs))
            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(logits, dim=1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.numpy())

            if hasattr(dataloader.dataset, 'samples'):
                batch_count = inputs.size(0)
                end_idx = min(sample_offset + batch_count, len(dataloader.dataset))
                all_paths.extend([
                    dataloader.dataset.samples[i][0]
                    for i in range(sample_offset, end_idx)
                ])
                sample_offset = end_idx

    all_logits = np.concatenate(all_logits, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if len(class_names) != all_logits.shape[1]:
        raise ValueError(
            f"class_names length {len(class_names)} != logits dim {all_logits.shape[1]}"
        )
    safe_class_names = _safe_class_column_names(class_names)

    results_data = {
        'image_path': all_paths if all_paths else [''] * len(all_labels),
        'filename': (
            [os.path.basename(p) if p else '' for p in all_paths]
            if all_paths else [''] * len(all_labels)
        ),
        'predicted_idx':   all_preds,
        'predicted_class': [class_names[int(i)] for i in all_preds],
        'true_label':       all_labels,
        'true_class':       [class_names[int(i)] for i in all_labels],
    }
    for idx, safe_name in enumerate(safe_class_names):
        results_data[f'logit_{safe_name}'] = all_logits[:, idx]
        results_data[f'prob_{safe_name}']  = all_probs[:,  idx]

    results_df = pd.DataFrame(results_data)
    prob_save_path = os.path.join(save_dir, f'{dataset_name}_probabilities.csv')
    results_df.to_csv(prob_save_path, index=False, encoding='utf-8-sig')
    print(f"Probabilities saved to {prob_save_path}")

    metrics = calculate_metrics(all_labels, all_preds, all_probs, class_names=class_names)
    metrics_row = {'dataset': dataset_name}
    metrics_row.update(flatten_metrics(metrics))
    metrics_save_path = os.path.join(save_dir, f'{dataset_name}_metrics.csv')
    pd.DataFrame([metrics_row]).to_csv(metrics_save_path, index=False, encoding='utf-8-sig')
    print(f"Metrics saved to {metrics_save_path}")

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        metrics['confusion_matrix'],
        annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_save_path = os.path.join(save_dir, f'{dataset_name}_confusion_matrix.png')
    plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_save_path}")

    return metrics


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def evaluate_single_model(
    model_name,
    cv_data_dir,
    test_dir,
    models_dir,
    output_dir,
    num_folds=5,
    batch_size=16,
    num_workers=0,
    num_classes=NUM_CLASSES,
    device='cuda',
):
    """Evaluate one model across all folds, then plot OvR ROC curves."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating model: {model_name}")
    print(f"{'=' * 60}")

    model_results_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    all_results = []
    class_names_used = None

    for fold in range(1, num_folds + 1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold}")
        print(f"{'=' * 60}")

        train_dir = os.path.join(cv_data_dir, f'fold_{fold}', 'train')
        val_dir   = os.path.join(cv_data_dir, f'fold_{fold}', 'val')
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print(f"Skipping fold {fold}: missing train or val directory.")
            continue

        fold_results_dir = os.path.join(model_results_dir, f'fold_{fold}')
        os.makedirs(fold_results_dir, exist_ok=True)

        transform, mean, std = _build_transform(model_name, train_dir)
        print(f"Normalization mean={mean}, std={std}")

        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset   = datasets.ImageFolder(val_dir,   transform=transform)
        test_dataset  = datasets.ImageFolder(test_dir,  transform=transform)
        class_names   = train_dataset.classes

        if len(class_names) != num_classes:
            print(f"num_classes={num_classes}, but data has {len(class_names)}: {class_names}")
            continue
        if val_dataset.classes != class_names or test_dataset.classes != class_names:
            print(
                "Class folder order differs between train/val/test. "
                f"train={class_names}, val={val_dataset.classes}, test={test_dataset.classes}"
            )
            continue

        class_names_used = class_names
        print(f"class_to_idx: {train_dataset.class_to_idx}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)

        model_dir  = os.path.join(models_dir, model_name)
        model_path = _find_model_path(model_dir, model_name, fold)
        if model_path is None:
            print(f"Could not find weights for fold {fold}; skipping.")
            continue

        print(f"Loading weights: {model_path}")
        model = create_model(model_name, num_classes=num_classes)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
        except Exception as exc:
            print(f"Failed to load weights: {exc}")
            traceback.print_exc()
            continue

        try:
            train_metrics = predict_and_save_probabilities(
                model, train_loader, 'train', fold_results_dir, device, class_names)
            val_metrics = predict_and_save_probabilities(
                model, val_loader, 'val', fold_results_dir, device, class_names)
            test_metrics = predict_and_save_probabilities(
                model, test_loader, 'test', fold_results_dir, device, class_names)
        except Exception as exc:
            print(f"Failed to evaluate {model_name} fold {fold}: {exc}")
            traceback.print_exc()
            continue

        print_metrics(train_metrics, "Train")
        print_metrics(val_metrics,   "Validation")
        print_metrics(test_metrics,  "Test")

        fold_result = {'model': model_name, 'fold': fold}
        fold_result.update(flatten_metrics(train_metrics, prefix='train_'))
        fold_result.update(flatten_metrics(val_metrics,   prefix='val_'))
        fold_result.update(flatten_metrics(test_metrics,  prefix='test_'))
        all_results.append(fold_result)

    if not all_results:
        print(f"Warning: no folds were evaluated for {model_name}")
        return None

    results_df = pd.DataFrame(all_results)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    avg_result = results_df[numeric_cols].mean().to_dict()
    avg_result['model'] = model_name
    avg_result['fold']  = 'Average'
    results_df = pd.concat([results_df, pd.DataFrame([avg_result])], ignore_index=True)

    detailed_results_path = os.path.join(
        model_results_dir, f'{model_name}_detailed_results.csv')
    results_df.to_csv(detailed_results_path, index=False, encoding='utf-8-sig')
    print(f"Detailed results saved to {detailed_results_path}")

    cv_results_path = os.path.join(
        model_results_dir, f'{model_name}_cross_validation_results.csv')
    pd.DataFrame([avg_result]).to_csv(cv_results_path, index=False, encoding='utf-8-sig')
    print(f"Cross-validation summary saved to {cv_results_path}")

    if class_names_used is not None:
        print(f"\n[ROC] Plotting OvR ROC curves for {model_name}...")
        try:
            plot_roc_curve_multiclass(
                model_results_dir=model_results_dir,
                model_name=model_name,
                class_names=class_names_used,
                num_folds=num_folds,
            )
        except Exception as exc:
            print(f"[ROC] Failed to plot ROC curves: {exc}")
            traceback.print_exc()

    return results_df


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def evaluate_all_models(
    model_list,
    cv_data_dir,
    test_dir,
    models_dir,
    output_dir,
    num_folds=5,
    batch_size=16,
    num_workers=0,
    num_classes=NUM_CLASSES,
    device='cuda',
    save_summary=True,
):
    """Evaluate all requested models."""
    print(f"\n{'=' * 60}")
    print("Starting model evaluation")
    print(f"{'=' * 60}")
    os.makedirs(output_dir, exist_ok=True)

    if model_list is None:
        if not os.path.exists(models_dir):
            raise ValueError(f"Model directory does not exist: {models_dir}")
        model_list = [
            d for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d))
        ]
        if not model_list:
            raise ValueError(f"No model folders found in {models_dir}")

    print(f"Found {len(model_list)} models: {model_list}")
    all_model_results = []
    start_time = time.time()

    for model_name in model_list:
        try:
            results_df = evaluate_single_model(
                model_name=model_name,
                cv_data_dir=cv_data_dir,
                test_dir=test_dir,
                models_dir=models_dir,
                output_dir=output_dir,
                num_folds=num_folds,
                batch_size=batch_size,
                num_workers=num_workers,
                num_classes=num_classes,
                device=device,
            )
            if results_df is not None:
                avg_result = results_df[results_df['fold'] == 'Average'].iloc[0].to_dict()
                all_model_results.append(avg_result)
        except Exception as exc:
            print(f"Evaluation failed for {model_name}: {exc}")
            traceback.print_exc()

    total_time = (time.time() - start_time) / 60
    if not all_model_results:
        print("Error: no models were evaluated successfully.")
        return None

    if save_summary:
        _save_evaluation_summary(all_model_results, output_dir)

    print(f"\nTotal evaluation time: {total_time:.2f} minutes")
    return {
        'total_time':  total_time,
        'num_models':  len(all_model_results),
        'results':     all_model_results,
        'output_dir':  output_dir,
    }


# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────

def _save_evaluation_summary(all_model_results, output_dir):
    summary_df = pd.DataFrame(all_model_results)
    test_cols = ['model'] + [c for c in summary_df.columns if c.startswith('test_')]
    summary_df_test = summary_df[test_cols]
    sort_col = 'test_mcc' if 'test_mcc' in summary_df_test.columns else 'test_f1_macro'
    summary_df_test = summary_df_test.sort_values(sort_col, ascending=False)

    summary_path = os.path.join(output_dir, 'all_models_summary.csv')
    summary_df_test.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"All-model summary saved to {summary_path}")

    key_cols = [
        c for c in [
            'model', 'test_accuracy', 'test_balanced_accuracy',
            'test_f1_macro', 'test_f1_weighted', 'test_mcc', 'test_auc_macro_ovr'
        ]
        if c in summary_df_test.columns
    ]
    heatmap_df = summary_df_test[key_cols].set_index('model')
    if len(heatmap_df) < 2:
        print("Only one model evaluated, skipping heatmap.")
    else:
        plt.figure(figsize=(10, max(6, len(heatmap_df) * 0.6)))
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5)
        plt.title('Model Performance Heatmap (Test Set)')
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, 'all_models_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {heatmap_path}")
