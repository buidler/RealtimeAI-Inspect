import os
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

import matplotlib

# 使用非交互式后端，确保在无显示环境下也能保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import convert_image_dtype, to_pil_image

import sys

# 将项目根目录加入 sys.path，方便在 tools 中导入 src 模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig
from src.misc import dist_utils


def _ensure_dir(path: Path) -> None:
    # 若目录不存在则递归创建，避免保存图片时报错
    path.mkdir(parents=True, exist_ok=True)


def _load_log(log_path: Path):
    epochs = []
    train_loss = []
    test_map = []
    test_map50 = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = obj.get("epoch")
            if epoch is None:
                continue
            epochs.append(epoch)
            if "train_loss" in obj:
                train_loss.append(obj["train_loss"])
            else:
                vals = [v for k, v in obj.items() if k.startswith("train_loss")]
                train_loss.append(float(vals[0]) if vals else float("nan"))
            coco = obj.get("test_coco_eval_bbox")
            if isinstance(coco, (list, tuple)) and len(coco) > 0:
                test_map.append(coco[0])
                if len(coco) > 1:
                    test_map50.append(coco[1])
                else:
                    test_map50.append(float("nan"))
            else:
                test_map.append(float("nan"))
                test_map50.append(float("nan"))
    return {
        "epoch": np.array(epochs, dtype=float) if epochs else np.zeros(0, dtype=float),
        "train_loss": np.array(train_loss, dtype=float) if train_loss else np.zeros(0, dtype=float),
        "test_map": np.array(test_map, dtype=float) if test_map else np.zeros(0, dtype=float),
        "test_map50": np.array(test_map50, dtype=float) if test_map50 else np.zeros(0, dtype=float),
    }


def plot_results(log_path: Path, output_dir: Path) -> None:
    # 从 log.txt 中解析训练 loss 和 mAP，并画出训练过程曲线
    stats = _load_log(log_path)
    epochs = stats["epoch"]
    if epochs.size == 0:
        return
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax1 = axes[0, 0]
    ax1.plot(epochs, stats["train_loss"], marker="o")
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax2 = axes[0, 1]
    ax2.plot(epochs, stats["test_map"], marker="o", label="mAP 0.5:0.95")
    if not np.all(np.isnan(stats["test_map50"])):
        ax2.plot(epochs, stats["test_map50"], marker="s", label="mAP 0.5")
    ax2.set_title("mAP")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    ax3 = axes[1, 0]
    if "train_lr" in stats:
        ax3.plot(epochs, stats["train_lr"], marker="o")
        ax3.set_title("Learning Rate")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("LR")
        ax3.grid(True, linestyle="--", alpha=0.5)
    else:
        ax3.axis("off")
    ax4 = axes[1, 1]
    ax4.axis("off")
    fig.tight_layout()
    save_path = output_dir / "results.png"
    fig.savefig(str(save_path), dpi=300)
    plt.close(fig)


def collect_label_stats(train_loader, max_images: int = 2000):
    # 统计训练集中标注框的分布，用于类似 YOLO 的 label 分析
    ds = train_loader.dataset
    n = len(ds)
    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[: min(max_images, n)]
    xs = []
    ys = []
    ws = []
    hs = []
    classes = []
    for idx in indices:
        try:
            img, target = ds[idx]
        except Exception:
            continue
        if "boxes" not in target or "labels" not in target:
            continue
        boxes = target["boxes"]
        labels = target["labels"]
        if boxes is None or labels is None:
            continue
        if isinstance(boxes, torch.Tensor):
            b = boxes.clone().detach()
        else:
            b = torch.as_tensor(boxes)
        if b.numel() == 0:
            continue
        x1 = b[:, 0]
        y1 = b[:, 1]
        x2 = b[:, 2]
        y2 = b[:, 3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        if "orig_size" in target:
            ow = float(target["orig_size"][0])
            oh = float(target["orig_size"][1])
        else:
            if hasattr(img, "size"):
                ow, oh = img.size
            else:
                oh = img.shape[-2]
                ow = img.shape[-1]
        ow = max(ow, 1.0)
        oh = max(oh, 1.0)
        xs.extend((cx / ow).cpu().numpy().tolist())
        ys.extend((cy / oh).cpu().numpy().tolist())
        ws.extend((w / ow).cpu().numpy().tolist())
        hs.extend((h / oh).cpu().numpy().tolist())
        classes.extend(labels.cpu().numpy().tolist())
    if not xs:
        return None
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    ws = np.array(ws, dtype=float)
    hs = np.array(hs, dtype=float)
    classes = np.array(classes, dtype=int)
    ar = ws / np.maximum(hs, 1e-6)
    return {
        "x": xs,
        "y": ys,
        "w": ws,
        "h": hs,
        "ar": ar,
        "cls": classes,
    }


def plot_labels_and_correlogram(label_stats, output_dir: Path) -> None:
    # 绘制类别直方图、框位置/尺寸分布以及特征相关性矩阵
    if label_stats is None:
        return
    _ensure_dir(output_dir)
    xs = label_stats["x"]
    ys = label_stats["y"]
    ws = label_stats["w"]
    hs = label_stats["h"]
    ar = label_stats["ar"]
    cls = label_stats["cls"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax1 = axes[0, 0]
    bins = np.arange(cls.min(), cls.max() + 2) - 0.5
    ax1.hist(cls, bins=bins, edgecolor="black")
    ax1.set_title("Class Distribution")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax2 = axes[0, 1]
    ax2.scatter(xs, ys, s=2, alpha=0.3)
    ax2.set_title("Box Centers")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax3 = axes[1, 0]
    ax3.hist(ws, bins=50, alpha=0.6, label="w")
    ax3.hist(hs, bins=50, alpha=0.6, label="h")
    ax3.set_title("Width/Height")
    ax3.set_xlabel("Normalized size")
    ax3.set_ylabel("Count")
    ax3.legend()
    ax4 = axes[1, 1]
    ax4.hist(ar, bins=50, edgecolor="black")
    ax4.set_title("Aspect Ratio w/h")
    ax4.set_xlabel("w/h")
    ax4.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(str(output_dir / "labels.png"), dpi=300)
    plt.close(fig)
    features = np.stack([xs, ys, ws, hs, ar], axis=1)
    corr = np.corrcoef(features, rowvar=False)
    fig2, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    names = ["x", "y", "w", "h", "ar"]
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    # 在相关性矩阵的每个色块中绘制对应的相关系数数值
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr[i, j]
            text_color = "white" if im.norm(val) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    fig2.savefig(str(output_dir / "labels_correlogram.png"), dpi=300)
    plt.close(fig2)


def collect_detection_stats(model, postprocessor, dataloader, device, iou_threshold: float = 0.5):
    # 在验证集上收集检测结果，计算 P/R/F1 曲线和混淆矩阵
    all_scores = []
    all_matches = []
    total_gt = 0
    confusion = None
    max_class = -1
    for samples, targets in dataloader:
        samples = samples.to(device)
        processed_targets = []
        for t in targets:
            new_t = {}
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    new_t[k] = v.to(device)
                else:
                    new_t[k] = v
            processed_targets.append(new_t)
        # 关闭梯度，前向推理并通过 postprocessor 获得最终检测结果
        with torch.no_grad():
            outputs = model(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in processed_targets], dim=0)
            results = postprocessor(outputs, orig_sizes)
        for res, tgt in zip(results, processed_targets):
            boxes_p = res["boxes"]
            scores_p = res["scores"]
            labels_p = res["labels"]
            boxes_g = tgt.get("boxes")
            labels_g = tgt.get("labels")
            if boxes_g is None or labels_g is None:
                continue
            if isinstance(boxes_p, torch.Tensor):
                boxes_p = boxes_p.detach().cpu()
            else:
                boxes_p = torch.as_tensor(boxes_p)
            if isinstance(scores_p, torch.Tensor):
                scores_p = scores_p.detach().cpu()
            else:
                scores_p = torch.as_tensor(scores_p)
            if isinstance(labels_p, torch.Tensor):
                labels_p = labels_p.detach().cpu().to(torch.int64)
            else:
                labels_p = torch.as_tensor(labels_p, dtype=torch.int64)
            if isinstance(boxes_g, torch.Tensor):
                boxes_g = boxes_g.detach().cpu()
            else:
                boxes_g = torch.as_tensor(boxes_g)
            if isinstance(labels_g, torch.Tensor):
                labels_g = labels_g.detach().cpu().to(torch.int64)
            else:
                labels_g = torch.as_tensor(labels_g, dtype=torch.int64)
            if boxes_g.numel() == 0:
                if scores_p.numel() > 0:
                    all_scores.extend(scores_p.numpy().tolist())
                    all_matches.extend([0] * int(scores_p.numel()))
                continue
            total_gt += int(labels_g.numel())
            if scores_p.numel() == 0:
                continue
            scores_np = scores_p.numpy()
            labels_p_np = labels_p.numpy()
            labels_g_np = labels_g.numpy()
            max_class = int(max(max_class, labels_p_np.max(initial=-1), labels_g_np.max(initial=-1)))
            # 动态扩展混淆矩阵大小，支持数据集中实际出现的最大类别 id
            if confusion is None:
                if max_class >= 0:
                    size = max_class + 1
                    confusion = np.zeros((size, size), dtype=np.int64)
            else:
                if max_class + 1 > confusion.shape[0]:
                    new_size = max_class + 1
                    new_conf = np.zeros((new_size, new_size), dtype=np.int64)
                    h, w = confusion.shape
                    new_conf[:h, :w] = confusion
                    confusion = new_conf
            from torchvision.ops import box_iou

            # 计算预测框和 GT 框的 IoU，用于按 IoU 阈值进行匹配
            ious = box_iou(boxes_p, boxes_g).numpy()
            order = np.argsort(-scores_np)
            used_gt = set()
            is_tp = np.zeros(len(scores_np), dtype=bool)
            # 按置信度从高到低尝试为每个预测匹配一个 GT
            for idx in order:
                if ious.size == 0:
                    break
                iou_row = ious[idx]
                best_gt = int(iou_row.argmax())
                best_iou = float(iou_row[best_gt])
                if best_iou >= iou_threshold and best_gt not in used_gt:
                    gt_cls = int(labels_g_np[best_gt])
                    pred_cls = int(labels_p_np[idx])
                    # 在混淆矩阵中累计 GT 类别与预测类别的对应次数
                    if confusion is not None and gt_cls < confusion.shape[0] and pred_cls < confusion.shape[1]:
                        confusion[gt_cls, pred_cls] += 1
                    if pred_cls == gt_cls:
                        is_tp[idx] = True
                    used_gt.add(best_gt)
            all_scores.extend(scores_np.tolist())
            all_matches.extend(is_tp.astype(int).tolist())
    if total_gt == 0 or not all_scores:
        return None
    # 根据所有预测结果计算 Precision / Recall / F1 曲线
    scores_arr = np.array(all_scores, dtype=float)
    matches_arr = np.array(all_matches, dtype=float)
    order = np.argsort(-scores_arr)
    scores_sorted = scores_arr[order]
    matches_sorted = matches_arr[order]
    tps = np.cumsum(matches_sorted)
    fps = np.cumsum(1.0 - matches_sorted)
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / float(total_gt)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "thresholds": scores_sorted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": confusion,
    }


def plot_pr_curves(stats, output_dir: Path) -> None:
    # 根据统计到的阈值、Precision、Recall、F1 绘制多种曲线
    if stats is None:
        return
    _ensure_dir(output_dir)
    thr = stats["thresholds"]
    precision = stats["precision"]
    recall = stats["recall"]
    f1 = stats["f1"]
    if thr.size > 400:
        idx = np.linspace(0, thr.size - 1, 400).astype(int)
        thr = thr[idx]
        precision = precision[idx]
        recall = recall[idx]
        f1 = f1[idx]
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(thr, f1)
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("F1")
    ax1.set_title("F1 Curve")
    ax1.grid(True, linestyle="--", alpha=0.5)
    fig1.tight_layout()
    fig1.savefig(str(output_dir / "F1_curve.png"), dpi=300)
    plt.close(fig1)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(thr, precision)
    ax2.set_xlabel("Confidence threshold")
    ax2.set_ylabel("Precision")
    ax2.set_title("P Curve")
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(str(output_dir / "P_curve.png"), dpi=300)
    plt.close(fig2)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(thr, recall)
    ax3.set_xlabel("Confidence threshold")
    ax3.set_ylabel("Recall")
    ax3.set_title("R Curve")
    ax3.grid(True, linestyle="--", alpha=0.5)
    fig3.tight_layout()
    fig3.savefig(str(output_dir / "R_curve.png"), dpi=300)
    plt.close(fig3)
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(recall, precision)
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.set_title("PR Curve")
    ax4.grid(True, linestyle="--", alpha=0.5)
    fig4.tight_layout()
    fig4.savefig(str(output_dir / "PR_curve.png"), dpi=300)
    plt.close(fig4)


def plot_confusion_matrices(stats, output_dir: Path) -> None:
    # 可视化原始混淆矩阵与按行归一化后的混淆矩阵
    confusion = stats.get("confusion")
    if confusion is None:
        return
    _ensure_dir(output_dir)
    num_classes = confusion.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(6, num_classes)))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    # 在原始混淆矩阵的每个非零色块中绘制整数计数
    for i in range(num_classes):
        for j in range(num_classes):
            val = int(confusion[i, j])
            if val == 0:
                continue
            text_color = "white" if im.norm(val) > 0.5 else "black"
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(output_dir / "confusion_matrix.png"), dpi=300)
    plt.close(fig)
    row_sums = confusion.sum(axis=1, keepdims=True)
    norm_conf = confusion.astype(float) / np.maximum(row_sums, 1.0)
    fig2, ax2 = plt.subplots(figsize=(max(6, num_classes), max(6, num_classes)))
    im2 = ax2.imshow(norm_conf, cmap="Blues", vmin=0.0, vmax=1.0)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Ground Truth")
    ax2.set_title("Confusion Matrix (Normalized)")
    # 在归一化混淆矩阵的每个较大数值色块中绘制比例
    for i in range(num_classes):
        for j in range(num_classes):
            val = norm_conf[i, j]
            if val < 1e-4:
                continue
            text_color = "white" if im2.norm(val) > 0.5 else "black"
            ax2.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    fig2.savefig(str(output_dir / "confusion_matrix_normalized.png"), dpi=300)
    plt.close(fig2)


def save_random_batches(dataloader, output_dir: Path, prefix: str, num_batches: int = 5) -> None:
    # 随机保存若干个 batch 中的第一张图像，并在上面画出标注框
    _ensure_dir(output_dir)
    count = 0
    for batch_idx, (images, targets) in enumerate(dataloader):
        if count >= num_batches:
            break
        if not isinstance(images, torch.Tensor):
            continue
        images_cpu = images.detach().cpu()
        tgt_list = targets
        if not isinstance(tgt_list, (list, tuple)):
            continue
        if images_cpu.shape[0] == 0:
            continue
        img = images_cpu[0]
        img_uint8 = convert_image_dtype(img, torch.uint8)
        t = tgt_list[0]
        boxes = t.get("boxes", None)
        if boxes is None or boxes.numel() == 0:
            annotated = img_uint8
        else:
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.as_tensor(boxes)
            boxes_cpu = boxes.detach().cpu()
            annotated = draw_bounding_boxes(img_uint8, boxes_cpu, colors="yellow", width=2)
        pil_img = to_pil_image(annotated)
        filename = f"{prefix}_batch_{batch_idx}.png"
        pil_img.save(str(output_dir / filename))
        count += 1


def generate_all_visualizations(config_path: str, output_dir: str = None, device: str = None) -> None:
    # 根据配置文件构建模型、数据加载器，并统一调用所有可视化函数
    cfg = YAMLConfig(config_path)
    if device is not None:
        device_str = device
    else:
        if getattr(cfg, "device", None):
            device_str = cfg.device
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device_str)
    out_dir = Path(output_dir) if output_dir is not None else Path(cfg.output_dir)
    _ensure_dir(out_dir)
    vis_dir = out_dir / "vis_results"
    _ensure_dir(vis_dir)
    # 构建模型与后处理器，并尽量加载 best.pth 中的 EMA 或模型权重
    model = cfg.model.to(dev)
    model.eval()
    postprocessor = cfg.postprocessor.to(dev)
    best_path = out_dir / "best.pth"
    if best_path.exists():
        try:
            state = torch.load(str(best_path), map_location=dev)
            if isinstance(state, dict):
                if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
                    model.load_state_dict(state["ema"]["module"], strict=False)
                elif "model" in state and isinstance(state["model"], dict):
                    model.load_state_dict(state["model"], strict=False)
        except Exception as e:
            print(f"load best.pth failed: {e}")
    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    log_path = out_dir / "log.txt"
    if log_path.exists():
        try:
            plot_results(log_path, vis_dir)
        except Exception as e:
            print(f"plot_results failed: {e}")
    # 标签统计与标签相关性可视化
    try:
        label_stats = collect_label_stats(train_loader)
        plot_labels_and_correlogram(label_stats, vis_dir)
    except Exception as e:
        print(f"label statistics failed: {e}")
    # 检测性能统计：P/R/F1 曲线与混淆矩阵
    try:
        det_stats = collect_detection_stats(model, postprocessor, val_loader, dev)
        if det_stats is not None:
            plot_pr_curves(det_stats, vis_dir)
            plot_confusion_matrices(det_stats, vis_dir)
    except Exception as e:
        print(f"detection statistics failed: {e}")
    # 随机保存部分训练样本
    try:
        save_random_batches(train_loader, vis_dir, prefix="train")
    except Exception as e:
        print(f"save train batches failed: {e}")
    # 随机保存部分验证样本
    try:
        save_random_batches(val_loader, vis_dir, prefix="val")
    except Exception as e:
        print(f"save val batches failed: {e}")


def main():
    import argparse

    # 提供命令行入口，方便直接对某个实验结果做可视化
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    # 只在主进程上执行可视化，避免多进程重复绘图
    if not dist_utils.is_main_process():
        return
    generate_all_visualizations(args.config, args.output_dir, args.device)


if __name__ == "__main__":
    main()
