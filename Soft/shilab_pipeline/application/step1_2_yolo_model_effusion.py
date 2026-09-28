#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Step 1 and Step 2 streaming inference for pleural/peritoneal effusion cytology:
SVS -> in-memory OpenSlide patches -> batched YOLO inference -> Step 2-compatible output.

"""
import os
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json
import random
import threading
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
import openslide
import torch
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

# Reproducibility configuration
def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming SVS YOLO inference (effusion): combines Steps 1 and 2 without writing intermediate patches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Path arguments
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to a single input SVS file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for inference outputs, e.g. data_svsModle")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to YOLO model weights")

    # Step 1 patch parameters
    parser.add_argument("--size", type=int, default=1024,
                        help="Patch size")
    parser.add_argument("--overlap", type=float, default=0.05,
                        help="Patch overlap ratio")
    parser.add_argument("--grid_size", type=int, default=1,
                        help="Center-sampling grid size; matches the original Step 1")
    parser.add_argument("--read_workers", type=int, default=4,
                        help="Number of OpenSlide reader threads; each thread opens its own slide object")

    # YOLO inference parameters
    parser.add_argument("--infer_conf", type=float, default=0.1,
                        help="YOLO confidence threshold (lower values retain more candidates)")
    parser.add_argument("--infer_iou", type=float, default=0.3,
                        help="YOLO NMS IoU threshold")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold for saving crops, confidence JSON, and confidence visualizations")
    parser.add_argument("--yolo_batch_size", type=int, default=8,
                        help="YOLO batch size")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="YOLO inference image size; defaults to the patch size")
    parser.add_argument("--half", action="store_true",
                        help="Enable FP16 inference; disabled by default for output consistency")

    # Effusion-specific cell-completeness filtering parameters
    parser.add_argument("--border_threshold", type=int, default=5,
                        help="Border threshold in pixels for truncated single_cell detections; border-touching cells are filtered")
    parser.add_argument("--small_cell_size", type=int, default=40,
                        help="Small single_cell threshold in pixels; cells with both dimensions below this value are counted but not cropped or written to JSON")
    parser.add_argument("--crop_padding", type=int, default=5,
                        help="Padding in pixels added around saved crops")

    # Output controls
    parser.add_argument("--save_json", action="store_true", default=True,
                        help="Save patch-level JSON files (default: enabled)")
    parser.add_argument("--no_save_json", dest="save_json", action="store_false",
                        help="Do not save patch-level JSON files; save crops and statistics only")
    # Disable annotation images by default to match the effusion workflow
    parser.add_argument("--save_annotation", action="store_true", default=False,
                        help="Save annotation visualizations (default: disabled to match the effusion workflow)")

    # Optional background filtering: disabled by default to avoid missed detections
    parser.add_argument("--skip_background", action="store_true",
                        help="Skip obvious background patches; disabled by default to avoid missed detections")
    parser.add_argument("--bg_sat_thr", type=int, default=15,
                        help="HSV saturation threshold for background filtering")
    parser.add_argument("--bg_val_thr", type=int, default=230,
                        help="HSV brightness threshold for background filtering")
    parser.add_argument("--bg_ratio_thr", type=float, default=0.95,
                        help="Skip a patch when its background-pixel ratio exceeds this threshold")

    return parser.parse_args()


# Generate center-patch coordinates consistent with the original Step 1
def generate_center_patch_coords(width, height, patch_size=1024, overlap_rate=0.05, grid_size=3):
    overlap_pixels = int(patch_size * overlap_rate)
    step_size = patch_size - overlap_pixels

    num_cols = (width - overlap_pixels) // step_size
    num_rows = (height - overlap_pixels) // step_size

    grid_rows = (num_rows + grid_size - 1) // grid_size
    grid_cols = (num_cols + grid_size - 1) // grid_size

    coords = []
    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            center_row = grid_row * grid_size + grid_size // 2
            center_col = grid_col * grid_size + grid_size // 2

            if center_row >= num_rows or center_col >= num_cols:
                continue

            x = center_col * step_size
            y = center_row * step_size
            coords.append((x, y, center_row, center_col))

    return coords, num_rows, num_cols, grid_rows, grid_cols


# Optional background filtering: disabled by default
def is_background_patch_bgr(img_bgr, sat_thr=15, val_thr=230, ratio_thr=0.95):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    bg_mask = (s < sat_thr) & (v > val_thr)
    return float(np.mean(bg_mask)) >= ratio_thr

_thread_local = threading.local()

def get_slide(slide_path):
    """Each worker thread opens OpenSlide once and reuses it."""
    if not hasattr(_thread_local, "slide") or _thread_local.slide_path != slide_path:
        if hasattr(_thread_local, "slide"):
            try:
                _thread_local.slide.close()
            except Exception:
                pass   # Ignore close failures and continue reopening the slide
        _thread_local.slide = openslide.OpenSlide(slide_path)
        _thread_local.slide_path = slide_path
    return _thread_local.slide

def read_patch_worker(slide_path, coord_item, patch_size):
    x, y, row, col = coord_item
    slide = get_slide(slide_path)   # Reuse one OpenSlide object within each worker thread
    region = slide.read_region((x, y), level=0, size=(patch_size, patch_size))
    region_rgb = region.convert("RGB")
    region_np = np.ascontiguousarray(np.asarray(region_rgb, dtype=np.uint8))
    image_bgr = cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR)

    image_name = f"tile_{row}_{col}"
    return {
        "image": image_bgr,
        "image_name": image_name,
        "row": row,
        "col": col,
        "x": x,
        "y": y,
    }

# Effusion single-cell completeness filtering (legacy Step 2 logic)
# Preserve the original Step 2 filtering behavior
def is_complete_single_cell(xyxy, image_shape, border_threshold=5):
    x1, y1, x2, y2 = map(int, xyxy)
    img_h, img_w = image_shape[:2]

    if x1 < border_threshold:         return False
    if y1 < border_threshold:         return False
    if x2 > img_w - border_threshold: return False
    if y2 > img_h - border_threshold: return False

    return True


# Filter incomplete single cells using the effusion workflow rules
def filter_incomplete_detections(detections, class_names, image_shape, border_threshold=5):
    if len(detections) == 0:
        return detections, 0

    valid_mask    = []
    skipped_count = 0

    for xyxy, class_id in zip(detections.xyxy, detections.class_id):
        class_name = class_names[class_id]

        if class_name == "single_cell":
            if not is_complete_single_cell(xyxy, image_shape, border_threshold):
                valid_mask.append(False)
                skipped_count += 1
                continue

        valid_mask.append(True)

    valid_mask          = np.array(valid_mask, dtype=bool)
    filtered_detections = detections[valid_mask]

    return filtered_detections, skipped_count

# Small-cell rule: both dimensions below the threshold are treated as small cells
# Keep this rule centralized to ensure consistent behavior
def is_small_single_cell(class_name: str, w: float, h: float, small_cell_size: int) -> bool:
    """Return whether a detection is a small single cell.

    Both dimensions must be below ``small_cell_size``.
    Floating-point comparison avoids integer-truncation boundary errors.
    """
    if "single_cell" not in class_name.lower():
        return False
    return (w < small_cell_size) and (h < small_cell_size)

# JSON template compatible with the original effusion Step 2 format
def make_json_template(image_name, image_ext, image_shape, model_path, description=""):
    return {
        "version":     "3.0.3",
        "flags":       {},
        "shapes":      [],
        "imagePath":   f"{image_name}{image_ext}",
        "imageData":   None,
        "imageHeight": image_shape[0],
        "imageWidth":  image_shape[1],
        "description": description,
    }


# Process and save results for one patch using the original Step 2 layout
def process_and_save_one_patch(
    image,
    image_name,
    sample_output_dir,
    model_path,
    result,
    class_names,
    box_annotator,
    label_annotator,
    conf,
    border_threshold,
    small_cell_size,
    crop_padding,
    save_json=True,
    save_annotation=False,
):
    image_ext   = ".png"
    image_shape = image.shape

    sample_annotation_dir = os.path.join(sample_output_dir, "annotation")
    sample_json_dir       = os.path.join(sample_output_dir, "json")
    sample_result_dir     = os.path.join(sample_output_dir, "result")

    if save_annotation:
        os.makedirs(sample_annotation_dir, exist_ok=True)
    if save_json:
        os.makedirs(sample_json_dir, exist_ok=True)
    for class_name_val in class_names.values():
        os.makedirs(os.path.join(sample_result_dir, class_name_val), exist_ok=True)

    detections = sv.Detections.from_ultralytics(result)

    # Initialize the statistics record with Step 2-compatible fields
    stats_record = {
        "image_name":                     image_name,
        "total_detections":               0,
        "detections_conf":                0,
        "skipped_incomplete_single_cell": 0,
        "small_single_cell":              0,
        "small_single_cell_conf":         0,
    }
    for cn in class_names.values():
        stats_record[f"total_{cn}"] = 0
        stats_record[f"{cn}_conf"]  = 0

    # Handle patches with no detections
    if len(detections) == 0:
        if save_json:
            desc_all  = f"Detected by YOLOv12 model (all detections, incomplete single_cell removed): {model_path}"
            desc_conf = f"Detected by YOLOv12 model (conf>={conf}, incomplete single_cell removed): {model_path}"
            with open(os.path.join(sample_json_dir, f"{image_name}_all.json"), "w", encoding="utf-8") as f:
                json.dump(make_json_template(image_name, image_ext, image_shape, model_path, desc_all), f, indent=2, ensure_ascii=False)
            with open(os.path.join(sample_json_dir, f"{image_name}_conf{conf}.json"), "w", encoding="utf-8") as f:
                json.dump(make_json_template(image_name, image_ext, image_shape, model_path, desc_conf), f, indent=2, ensure_ascii=False)

        if save_annotation:
            cv2.imwrite(os.path.join(sample_annotation_dir, f"{image_name}_all_annotated.png"), image)
            cv2.imwrite(os.path.join(sample_annotation_dir, f"{image_name}_conf{conf}_annotated.png"), image)

        return stats_record

    # Filter detections: incomplete single cells are removed
    detections, skipped_count = filter_incomplete_detections(
        detections, class_names, image_shape, border_threshold
    )

    # Apply the confidence threshold and optionally create annotations
    detections_conf = detections[detections.confidence >= conf]

    # Initialize statistics counters
    class_counts      = defaultdict(int)
    class_counts_conf = defaultdict(int)
    small_sc_count    = 0
    small_sc_conf     = 0

    # Initialize JSON templates
    desc_all  = f"Detected by YOLOv12 model (all detections, incomplete single_cell removed): {model_path}"
    desc_conf = f"Detected by YOLOv12 model (conf>={conf}, incomplete single_cell removed): {model_path}"
    json_data_all  = make_json_template(image_name, image_ext, image_shape, model_path, desc_all)
    json_data_conf = make_json_template(image_name, image_ext, image_shape, model_path, desc_conf)

    # Continuous detection counters
    # json_all_i: shape index in _all.json, including small cells at every confidence level
    # crop_conf_i: shape index in _conf.json and saved crop filenames
    # Small cells with confidence above the threshold are filtered consistently
    json_all_i  = 0
    crop_conf_i = 0

    # Single pass: update statistics, JSON, and saved crops
    for xyxy, class_id, confidence in zip(
        detections.xyxy, detections.class_id, detections.confidence
    ):
        x1, y1, x2, y2 = map(float, xyxy)
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        w          = x2 - x1
        h          = y2 - y1
        class_name = class_names[class_id]
        is_small   = is_small_single_cell(class_name, w, h, small_cell_size)

        # Track small cells in the statistics
        class_counts[class_name] += 1
        if is_small:
            small_sc_count += 1
        if confidence >= conf:
            class_counts_conf[class_name] += 1
            if is_small:
                small_sc_conf += 1

        # Small cells are recorded in statistics only and are not saved as JSON or crops
        if is_small:
            continue

        # Compute padded coordinates once for both JSON and crop export
        x1_pad = max(0,              x1i - crop_padding)
        y1_pad = max(0,              y1i - crop_padding)
        x2_pad = min(image_shape[1], x2i + crop_padding)
        y2_pad = min(image_shape[0], y2i + crop_padding)

        shape_base = {
            "kie_linking": [],
            "label":       class_name,
            "score":       float(confidence),
            "points": [
                [x1_pad, y1_pad],
                [x2_pad, y1_pad],
                [x2_pad, y2_pad],
                [x1_pad, y2_pad],
            ],
            "group_id":    None,
            "description": f"Confidence: {confidence:.4f}",
            "difficult":   False,
            "shape_type":  "rectangle",
            "flags":       {},
        }

        # Write _all.json
        shape_all = {
            **shape_base,
            "attributes": {
                "confidence":   float(confidence),
                "class_id":     int(class_id),
                "detection_id": json_all_i,
            }
        }
        json_data_all["shapes"].append(shape_all)
        json_all_i += 1

        # Confidence filtering: write only _all.json below the threshold
        if confidence < conf:
            continue

        # Write _conf.json
        shape_conf = {
            **shape_base,
            "attributes": {
                "confidence":   float(confidence),
                "class_id":     int(class_id),
                "detection_id": crop_conf_i,
            }
        }
        json_data_conf["shapes"].append(shape_conf)

        # Save crops for single_cell and cluster detections only
        SAVE_CROP_CLASSES = {"single_cell", "cluster"}
        if class_name in SAVE_CROP_CLASSES:
            cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
            cv2.imwrite(
                os.path.join(
                    sample_result_dir, class_name,
                    f"{image_name}_{crop_conf_i}_conf{confidence:.2f}.png"
                ),
                cropped
            )
        crop_conf_i += 1

    # Save JSON files
    if save_json:
        with open(os.path.join(sample_json_dir, f"{image_name}_all.json"), "w", encoding="utf-8") as f:
            json.dump(json_data_all, f, indent=2, ensure_ascii=False)
        with open(os.path.join(sample_json_dir, f"{image_name}_conf{conf}.json"), "w", encoding="utf-8") as f:
            json.dump(json_data_conf, f, indent=2, ensure_ascii=False)

    # Save annotation images when enabled
    if save_annotation:
        # Visualize all detections
        ann_all    = image.copy()
        labels_all = [f"{class_names[c]} {cf:.2f}" for c, cf in zip(detections.class_id, detections.confidence)]
        ann_all    = box_annotator.annotate(scene=ann_all, detections=detections)
        ann_all    = label_annotator.annotate(scene=ann_all, detections=detections, labels=labels_all)
        cv2.imwrite(os.path.join(sample_annotation_dir, f"{image_name}_all_annotated.png"), ann_all)

        # Visualize detections above the confidence threshold
        if len(detections_conf) > 0:
            ann_conf    = image.copy()
            labels_conf = [f"{class_names[c]} {cf:.2f}" for c, cf in zip(detections_conf.class_id, detections_conf.confidence)]
            ann_conf    = box_annotator.annotate(scene=ann_conf, detections=detections_conf)
            ann_conf    = label_annotator.annotate(scene=ann_conf, detections=detections_conf, labels=labels_conf)
        else:
            ann_conf = image.copy()
        cv2.imwrite(os.path.join(sample_annotation_dir, f"{image_name}_conf{conf}_annotated.png"), ann_conf)

    # Update the statistics record
    stats_record.update({
        "total_detections":               len(detections),
        "detections_conf":                len(detections_conf),
        "skipped_incomplete_single_cell": skipped_count,
        "small_single_cell":              small_sc_count,
        "small_single_cell_conf":         small_sc_conf,
    })
    for cn in class_names.values():
        stats_record[f"total_{cn}"] = class_counts[cn]
        stats_record[f"{cn}_conf"]  = class_counts_conf[cn]

    return stats_record



# Main workflow
def main():
    args = parse_args()

    svs_path          = args.input_file
    sample_name       = os.path.splitext(os.path.basename(svs_path))[0]
    sample_output_dir = os.path.join(args.output_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)

    print("============================================================")
    print("  Combined Step 1 + Step 2 streaming inference started (effusion)")
    print(f"  SVS file         : {svs_path}")
    print(f"  Sample name      : {sample_name}")
    print(f"  Output directory : {sample_output_dir}")
    print(f"  patch size     : {args.size}")
    print(f"  overlap        : {args.overlap}")
    print(f"  grid_size      : {args.grid_size}")
    print(f"  read_workers   : {args.read_workers}")
    print(f"  YOLO batch     : {args.yolo_batch_size}")
    print(f"  imgsz          : {args.imgsz}")
    print(f"  half           : {args.half}")
    print(f"  infer_conf     : {args.infer_conf}")
    print(f"  conf(save)     : {args.conf}")
    print(f"  small_cell_size: {args.small_cell_size}")
    print(f"  save_json      : {args.save_json}")
    print(f"  save_annotation: {args.save_annotation}")
    print(f"  skip_background: {args.skip_background}")
    print("============================================================")

    total_start = time.time()

    # Read slide dimensions once for metadata
    slide = openslide.OpenSlide(svs_path)
    width, height = slide.dimensions
    slide.close()

    coords, num_rows, num_cols, grid_rows, grid_cols = generate_center_patch_coords(
        width=width, height=height,
        patch_size=args.size, overlap_rate=args.overlap, grid_size=args.grid_size
    )

    print(f"Slide dimensions : {width} x {height}")
    print(f"Original grid    : {num_rows} x {num_cols} = {num_rows * num_cols} patches")
    print(f"Center patches   : {grid_rows} x {grid_cols} = {len(coords)} patches")

    # Load the YOLO model
    model           = YOLO(args.model_path)
    class_names     = model.names
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    stats_records      = []
    skipped_bg         = 0
    batch_items        = []

    # Run YOLO inference for the current batch
    def flush_batch(items):

        if not items:
            return

        images  = [it["image"] for it in items]
        results = model(
            images,
            verbose=False,
            conf=args.infer_conf,
            iou=args.infer_iou,
            imgsz=args.imgsz,
            half=args.half,
        )

        for item, result in zip(items, results):
            rec = process_and_save_one_patch(
                image             = item["image"],
                image_name        = item["image_name"],
                sample_output_dir = sample_output_dir,
                model_path        = args.model_path,
                result            = result,
                class_names       = class_names,
                box_annotator     = box_annotator,
                label_annotator   = label_annotator,
                conf              = args.conf,
                border_threshold  = args.border_threshold,
                small_cell_size   = args.small_cell_size,
                crop_padding      = args.crop_padding,
                save_json         = args.save_json,
                save_annotation   = args.save_annotation,
            )
            stats_records.append(rec)

    # Read patches in worker threads and run batched YOLO inference on the main thread
    with ThreadPoolExecutor(max_workers=args.read_workers) as executor:
        # Submit futures in coordinate order
        futures = [
            executor.submit(read_patch_worker, svs_path, coord, args.size)
            for coord in coords
        ]

        # Consume futures in submission order to preserve deterministic output ordering
        for future in tqdm(futures, total=len(futures), desc="OpenSlide reading + YOLO inference"):
            try:
                item = future.result()   # Read and process each patch
            except Exception as e:
                print(f"  Warning: failed to read patch: {e}")
                continue

            if args.skip_background and is_background_patch_bgr(
                item["image"],
                sat_thr=args.bg_sat_thr, val_thr=args.bg_val_thr, ratio_thr=args.bg_ratio_thr
            ):
                skipped_bg += 1
                continue

            batch_items.append(item)

            if len(batch_items) >= args.yolo_batch_size:
                flush_batch(batch_items)
                batch_items = []

    # Save patch-level detection statistics
    flush_batch(batch_items)

    # Save sample-level confidence statistics
    if stats_records:
        stats_df   = pd.DataFrame(stats_records)
        stats_df   = stats_df.sort_values("image_name").reset_index(drop=True)
        stats_path = os.path.join(sample_output_dir, "detection_statistics.csv")
        stats_df.to_csv(stats_path, index=False)

        total_images   = len(stats_df)
        total_det      = stats_df["total_detections"].sum()
        total_det_conf = stats_df["detections_conf"].sum()
        total_skipped  = stats_df["skipped_incomplete_single_cell"].sum()
        total_small    = stats_df["small_single_cell"].sum()
        total_small_c  = stats_df["small_single_cell_conf"].sum()

        print("\nDetection statistics:")
        print(f"- Processed patches                         : {total_images}")
        print(f"- Skipped background patches                : {skipped_bg}")
        print(f"- Total detections                          : {total_det}")
        print(f"- Detections with conf >= {args.conf}             : {total_det_conf}",
              f" ({total_det_conf/total_det*100:.1f}%)" if total_det > 0 else "")
        print(f"- Removed incomplete single_cell detections : {total_skipped}")
        print(f"- Small single_cell detections              : {total_small}")
        print(f"- Small single_cell with conf >= {args.conf}      : {total_small_c}")
        print("\nStatistics by class:")
        for cn in sorted(class_names.values()):
            t = stats_df[f"total_{cn}"].sum()
            c = stats_df[f"{cn}_conf"].sum()
            print(f"  - {cn}: total {t}; conf >= {args.conf}: {c}")
        print(f"\n- Patch-level statistics saved to: {stats_path}")

    elapsed = time.time() - total_start
    print("\nProcessing complete!")
    print(f"Results saved to: {sample_output_dir}")
    print(f"Total elapsed time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
