#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1 + Step2 合并优化版（胸腹水）：SVS → OpenSlide内存patch → YOLO batch推理 → 原Step2输出结构

目标：
  1. 不再把中间 patch 保存为 png，减少 Step1/Step2 重复 IO
  2. YOLO 支持 batch 推理，提高 GPU 利用率
  3. 保留原 step2.yolo_modele_infer_effusion.py 的医学处理逻辑：
     - single_cell 完整性过滤（贴边过滤）
     - single_cell 小细胞过滤（不裁剪保存、不写入JSON，但仍统计）
     - conf 筛选
     - crop_padding 裁剪
     - JSON 保存（_all.json / _conf{conf}.json）
     - result/{class_name}/ 输出结构
  4. 输出保持与原 Step2 一致，确保 Step3 可直接衔接

输出结构：
  output_dir/
    {sample_name}/
      annotation/          ← 仅 --save_annotation 时生成
      json/
      result/
        {class_name}/      ← Step3 输入目录
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

# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="SVS流式YOLO推理（胸腹水）：合并Step1和Step2，避免中间patch落盘",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── 路径参数 ──────────────────────────────────────────────
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入单个SVS文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出结果根目录，例如 data_svsModle")
    parser.add_argument("--model_path", type=str, required=True,
                        help="YOLO模型权重路径")

    # ── Step1 patch 参数 ──────────────────────────────────────
    parser.add_argument("--size", type=int, default=1024,
                        help="patch大小")
    parser.add_argument("--overlap", type=float, default=0.05,
                        help="patch overlap比例")
    parser.add_argument("--grid_size", type=int, default=1,
                        help="中心采样grid大小，与原Step1一致")
    parser.add_argument("--read_workers", type=int, default=4,
                        help="OpenSlide读取线程数，每个线程独立打开OpenSlide对象")

    # ── YOLO 推理参数 ─────────────────────────────────────────
    parser.add_argument("--infer_conf", type=float, default=0.1,
                        help="YOLO推理置信度阈值（低阈值保留更多候选）")
    parser.add_argument("--infer_iou", type=float, default=0.3,
                        help="YOLO NMS IoU阈值")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="保存/裁剪/JSON(conf版)/可视化(conf版) 的置信度阈值")
    parser.add_argument("--yolo_batch_size", type=int, default=8,
                        help="YOLO batch size")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="YOLO推理输入尺寸，默认与patch size一致")
    parser.add_argument("--half", action="store_true",
                        help="启用FP16推理，默认关闭以保持结果一致性")

    # ── 胸腹水完整性过滤参数 ──────────────────────────────────
    parser.add_argument("--border_threshold", type=int, default=5,
                        help="single_cell 贴边截断判断阈值（像素），贴边即过滤")
    parser.add_argument("--small_cell_size", type=int, default=40,
                        help="小尺寸single_cell判断阈值（像素），长和宽均小于此值则视为小细胞，不裁剪保存、不写入JSON，但仍统计")
    parser.add_argument("--crop_padding", type=int, default=5,
                        help="裁剪保存时四周扩展padding（像素）")

    # ── 输出控制 ──────────────────────────────────────────────
    parser.add_argument("--save_json", action="store_true", default=True,
                        help="保存patch级JSON，默认保存")
    parser.add_argument("--no_save_json", dest="save_json", action="store_false",
                        help="不保存patch级JSON，仅保存crop和统计")
    # 胸腹水原版 annotation 默认关闭（原代码中注释掉了）
    parser.add_argument("--save_annotation", action="store_true", default=False,
                        help="保存annotation可视化图，默认关闭（与原胸腹水step2一致）")

    # ── 背景过滤（可选，默认关闭）────────────────────────────
    parser.add_argument("--skip_background", action="store_true",
                        help="跳过明显背景patch，默认关闭避免漏检")
    parser.add_argument("--bg_sat_thr", type=int, default=15,
                        help="背景过滤HSV饱和度阈值")
    parser.add_argument("--bg_val_thr", type=int, default=230,
                        help="背景过滤HSV亮度阈值")
    parser.add_argument("--bg_ratio_thr", type=float, default=0.95,
                        help="背景像素比例超过该值则跳过patch")

    return parser.parse_args()


# ============================================================
# 与原 Step1 一致的中心 patch 坐标生成
# ============================================================
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


# ============================================================
# 可选背景过滤：默认不启用
# ============================================================
def is_background_patch_bgr(img_bgr, sat_thr=15, val_thr=230, ratio_thr=0.95):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    bg_mask = (s < sat_thr) & (v > val_thr)
    return float(np.mean(bg_mask)) >= ratio_thr

_thread_local = threading.local()

def get_slide(slide_path):
    """每个线程只打开一次 OpenSlide，线程内复用"""
    if not hasattr(_thread_local, "slide") or _thread_local.slide_path != slide_path:
        if hasattr(_thread_local, "slide"):
            try:
                _thread_local.slide.close()
            except Exception:
                pass                        # close 失败不阻止重新 open
        _thread_local.slide = openslide.OpenSlide(slide_path)
        _thread_local.slide_path = slide_path
    return _thread_local.slide

def read_patch_worker(slide_path, coord_item, patch_size):
    x, y, row, col = coord_item
    slide = get_slide(slide_path)           # ← 线程内复用，不重复 open/close
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

# ============================================================
# 胸腹水：single_cell 完整性检查（贴边即过滤）
# 其他类别不做贴边过滤，与原 step2 保持一致
# ============================================================
def is_complete_single_cell(xyxy, image_shape, border_threshold=5):
    x1, y1, x2, y2 = map(int, xyxy)
    img_h, img_w = image_shape[:2]

    if x1 < border_threshold:         return False
    if y1 < border_threshold:         return False
    if x2 > img_w - border_threshold: return False
    if y2 > img_h - border_threshold: return False

    return True


# ============================================================
# 胸腹水：过滤不完整 single_cell（其他类别全部保留）
# ============================================================
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

# ============================================================
# 小细胞判断：长和宽均小于阈值才视为小细胞（and 条件）
# 统一在此函数中定义，避免多处散落导致不一致
# ============================================================
def is_small_single_cell(class_name: str, w: float, h: float, small_cell_size: int) -> bool:
    """
    判断是否为小尺寸 single_cell。
    条件：类别包含 single_cell，且宽和高均小于 small_cell_size（and）。
    统一使用 float 类型比较，避免 int 截断导致边界值不一致。
    """
    if "single_cell" not in class_name.lower():
        return False
    return (w < small_cell_size) and (h < small_cell_size)

# ============================================================
# JSON 模板：保持原胸腹水 step2 格式
# ============================================================
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


# ============================================================
# 单 patch 结果处理与保存：保留原胸腹水 step2 输出结构
# ============================================================
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

    # ── 初始化统计记录（与原胸腹水 step2 字段一致）──────────
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

    # ── 无检测结果 ────────────────────────────────────────────
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

    # ── 过滤不完整 single_cell ────────────────────────────────
    detections, skipped_count = filter_incomplete_detections(
        detections, class_names, image_shape, border_threshold
    )

    # ── conf 筛选（仅用于 annotation 可视化）─────────────────
    detections_conf = detections[detections.confidence >= conf]

    # ── 初始化统计计数器 ──────────────────────────────────────
    class_counts      = defaultdict(int)
    class_counts_conf = defaultdict(int)
    small_sc_count    = 0
    small_sc_conf     = 0

    # ── 初始化 JSON 模板 ──────────────────────────────────────
    desc_all  = f"Detected by YOLOv12 model (all detections, incomplete single_cell removed): {model_path}"
    desc_conf = f"Detected by YOLOv12 model (conf>={conf}, incomplete single_cell removed): {model_path}"
    json_data_all  = make_json_template(image_name, image_ext, image_shape, model_path, desc_all)
    json_data_conf = make_json_template(image_name, image_ext, image_shape, model_path, desc_conf)

    # ── 连续编号计数器 ────────────────────────────────────────
    # json_all_i  : _all.json  中 shape 的连续编号（非小细胞，所有 conf）
    # crop_conf_i : _conf.json 中 shape 的连续编号，与裁剪文件名完全对齐
    #               （非小细胞 且 confidence >= conf，两者过滤条件完全相同）
    json_all_i  = 0
    crop_conf_i = 0

    # ── 统计 + JSON 构建 + 裁剪保存（单次遍历）───────────────
    for xyxy, class_id, confidence in zip(
        detections.xyxy, detections.class_id, detections.confidence
    ):
        x1, y1, x2, y2 = map(float, xyxy)
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        w          = x2 - x1
        h          = y2 - y1
        class_name = class_names[class_id]
        is_small   = is_small_single_cell(class_name, w, h, small_cell_size)

        # ── 统计（小细胞也统计）──────────────────────────────
        class_counts[class_name] += 1
        if is_small:
            small_sc_count += 1
        if confidence >= conf:
            class_counts_conf[class_name] += 1
            if is_small:
                small_sc_conf += 1

        # ── 小细胞：只统计，不写 JSON，不裁剪保存 ────────────
        if is_small:
            continue

        # ── padding 坐标计算一次，JSON 和裁剪保存共用 ─────────
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

        # ── 写入 _all.json ────────────────────────────────────
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

        # ── conf 不足：只写 _all.json，不写 _conf.json，不裁剪
        if confidence < conf:
            continue

        # ── 写入 _conf.json ───────────────────────────────────
        shape_conf = {
            **shape_base,
            "attributes": {
                "confidence":   float(confidence),
                "class_id":     int(class_id),
                "detection_id": crop_conf_i,
            }
        }
        json_data_conf["shapes"].append(shape_conf)

        # ── 裁剪保存（只保存 single_cell 和 cluster 两类）─────
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

    # ── 保存 JSON ─────────────────────────────────────────────
    if save_json:
        with open(os.path.join(sample_json_dir, f"{image_name}_all.json"), "w", encoding="utf-8") as f:
            json.dump(json_data_all, f, indent=2, ensure_ascii=False)
        with open(os.path.join(sample_json_dir, f"{image_name}_conf{conf}.json"), "w", encoding="utf-8") as f:
            json.dump(json_data_conf, f, indent=2, ensure_ascii=False)

    # ── 保存 annotation（默认关闭）───────────────────────────
    if save_annotation:
        # 全量可视化
        ann_all    = image.copy()
        labels_all = [f"{class_names[c]} {cf:.2f}" for c, cf in zip(detections.class_id, detections.confidence)]
        ann_all    = box_annotator.annotate(scene=ann_all, detections=detections)
        ann_all    = label_annotator.annotate(scene=ann_all, detections=detections, labels=labels_all)
        cv2.imwrite(os.path.join(sample_annotation_dir, f"{image_name}_all_annotated.png"), ann_all)

        # conf 可视化
        if len(detections_conf) > 0:
            ann_conf    = image.copy()
            labels_conf = [f"{class_names[c]} {cf:.2f}" for c, cf in zip(detections_conf.class_id, detections_conf.confidence)]
            ann_conf    = box_annotator.annotate(scene=ann_conf, detections=detections_conf)
            ann_conf    = label_annotator.annotate(scene=ann_conf, detections=detections_conf, labels=labels_conf)
        else:
            ann_conf = image.copy()
        cv2.imwrite(os.path.join(sample_annotation_dir, f"{image_name}_conf{conf}_annotated.png"), ann_conf)

    # ── 更新统计记录 ──────────────────────────────────────────
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



# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()

    svs_path          = args.input_file
    sample_name       = os.path.splitext(os.path.basename(svs_path))[0]
    sample_output_dir = os.path.join(args.output_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)

    print("============================================================")
    print("  Step1+Step2 合并流式推理启动（胸腹水）")
    print(f"  SVS文件        : {svs_path}")
    print(f"  样本名         : {sample_name}")
    print(f"  输出目录       : {sample_output_dir}")
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

    # ── 读取 slide 尺寸（只打开一次取 metadata）──────────────
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

    # ── 加载 YOLO ─────────────────────────────────────────────
    model           = YOLO(args.model_path)
    class_names     = model.names
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    stats_records      = []
    skipped_bg         = 0
    batch_items        = []

    # ── flush_batch：将当前 batch 送入 YOLO 推理并处理结果 ───
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

    # ── 多线程读取 patch + 主线程 batch 推理 ─────────────────
    with ThreadPoolExecutor(max_workers=args.read_workers) as executor:
        # 按顺序提交，保留 future 列表（有序）
        futures = [
            executor.submit(read_patch_worker, svs_path, coord, args.size)
            for coord in coords
        ]

        # ✅ 按提交顺序迭代，而不是 as_completed
        for future in tqdm(futures, total=len(futures), desc="OpenSlide读取+YOLO推理"):
            try:
                item = future.result()   # 会等待该 future 完成
            except Exception as e:
                print(f"  ⚠️ 读取patch失败: {e}")
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

    # ── flush 剩余 batch ──────────────────────────────────────
    flush_batch(batch_items)

    # ── 保存 patch 级统计（字段与原胸腹水 step2 一致）────────
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

        print("\n检测统计信息:")
        print(f"- 处理patch数                              : {total_images}")
        print(f"- 跳过背景patch数                          : {skipped_bg}")
        print(f"- 总检测数                                 : {total_det}")
        print(f"- conf >= {args.conf} 的检测数              : {total_det_conf}"
              + (f" ({total_det_conf/total_det*100:.1f}%)" if total_det > 0 else ""))
        print(f"- 删除的不完整 single_cell 数              : {total_skipped}")
        print(f"- 小尺寸 single_cell 数量                  : {total_small}")
        print(f"- conf >= {args.conf} 的小尺寸 single_cell : {total_small_c}")
        print("\n各类别统计:")
        for cn in sorted(class_names.values()):
            t = stats_df[f"total_{cn}"].sum()
            c = stats_df[f"{cn}_conf"].sum()
            print(f"  - {cn}: 总数 {t}，conf >= {args.conf} 的数量 {c}")
        print(f"\n- patch级统计保存到: {stats_path}")

    elapsed = time.time() - total_start
    print("\n处理完成!")
    print(f"结果已保存到: {sample_output_dir}")
    print(f"总耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
