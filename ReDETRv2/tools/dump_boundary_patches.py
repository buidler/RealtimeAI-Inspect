import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

# 将 ReDETRv2 根目录加入 sys.path，确保可导入 references.deploy 等模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from references.deploy.rtdetrv2_hqsam_infer import (
    _build_rtdetr_model,
    _contours_from_box,
    _filter_traditional_contours_with_hqsam,
    _filter_long_thin_contours,
    _select_best_hqsam_contour,
)
from segment_anything_hq import sam_model_registry, SamPredictor


def _list_images(image_dir):
    # 递归枚举目录中的常见图像文件，并按路径排序，保证采样顺序可复现
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_dir = Path(image_dir)
    files = []
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def _build_sam(hq_sam_checkpoint, hq_sam_model_type, device):
    # 可选构建 HQ-SAM 预测器：未提供权重时返回 None，调用方按无 HQ 分支处理
    if hq_sam_checkpoint is None:
        return None
    sam = sam_model_registry[hq_sam_model_type](checkpoint=hq_sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def _rasterize_polygon_to_mask(pts, size):
    # 将轮廓点序列栅格化为整图二值掩膜，作为 patch 的传统轮廓通道
    w, h = size
    mask_im = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_im)
    draw.polygon([(float(p[0]), float(p[1])) for p in pts], outline=1, fill=1)
    return np.array(mask_im, dtype=np.uint8)


def dump_patches(
    config_path,
    ckpt_path,
    image_dir,
    save_dir,
    device="cuda",
    score_thr=0.6,
    use_hq_sam=True,
    hq_sam_checkpoint=None,
    hq_sam_model_type="vit_l",
    max_patches_per_image=64,
):
    # 主流程：
    # 1) RT-DETRv2 检测目标框；2) HQ-SAM/传统算法生成轮廓；
    # 3) 沿轮廓采样局部 patch；4) 保存为 npz 并写入 csv 元信息
    device = torch.device(device)
    model = _build_rtdetr_model(config_path, ckpt_path, device)
    model.eval()

    predictor = _build_sam(hq_sam_checkpoint, hq_sam_model_type, device) if use_hq_sam else None

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    image_paths = _list_images(image_dir)
    save_dir = Path(save_dir)
    patch_dir = save_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "patches.csv"

    with csv_path.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "patch_path",
                "image_path",
                "box_index",
                "contour_index",
                "point_index",
                "x",
                "y",
                "label",
            ]
        )

        patch_idx_global = 0

        for img_idx, img_path in enumerate(image_paths):
            # 对每张图先做检测，再基于每个目标框提取轮廓采样点
            im_pil = Image.open(img_path).convert("RGB")
            w, h = im_pil.size
            image_np = np.array(im_pil)
            orig_size = torch.tensor([w, h])[None].to(device)
            im_data = transforms(im_pil)[None].to(device)

            with torch.no_grad():
                labels, boxes, scores = model(im_data, orig_size)

            scr = scores[0]
            lab = labels[0]
            box = boxes[0]
            keep = scr > score_thr
            scr = scr[keep]
            lab = lab[keep]
            box = box[keep]

            if use_hq_sam and predictor is not None:
                predictor.set_image(image_np)

            patches_this_image = 0

            for j, b in enumerate(box):
                # 先尝试拿到 HQ-SAM 轮廓，再用其约束传统轮廓的搜索区域
                b_np = b.detach().cpu().numpy()
                hq_pts = None
                hq_mask = None
                trad_contours = []

                if use_hq_sam and predictor is not None:
                    masks, _, _ = predictor.predict(
                        box=b_np,
                        multimask_output=True,
                    )
                    hq_pts, hq_mask = _select_best_hqsam_contour(
                        masks, b_np, conservative_boundary=False
                    )

                if hq_mask is not None:
                    ys_mask, xs_mask = np.nonzero(hq_mask)
                    if ys_mask.size > 0:
                        hx0, hx1 = int(xs_mask.min()), int(xs_mask.max()) + 1
                        hy0, hy1 = int(ys_mask.min()), int(ys_mask.max()) + 1
                        box_for_trad = np.array([hx0, hy0, hx1, hy1], dtype=b_np.dtype)
                    else:
                        box_for_trad = b_np
                else:
                    box_for_trad = b_np

                trad_contours = _contours_from_box(im_pil, box_for_trad, mask_hint=hq_mask)
                trad_contours = _filter_traditional_contours_with_hqsam(
                    trad_contours, hq_mask, im_pil.size
                )
                trad_contours = _filter_long_thin_contours(trad_contours, box_for_trad)

                if not trad_contours:
                    continue

                # 构建传统掩膜，用于 head 输入
                trad_mask_full = _rasterize_polygon_to_mask(trad_contours[0], im_pil.size)
                hq_mask_full = (
                    hq_mask.astype(np.uint8) if hq_mask is not None else np.zeros((h, w), dtype=np.uint8)
                )

                pts_arr = np.asarray(trad_contours[0], dtype=np.float32)
                num_samples = min(32, len(pts_arr))
                idx = np.linspace(0, len(pts_arr) - 1, num_samples).astype(int)
                samples = pts_arr[idx]

                patch_size = 64
                half = patch_size // 2

                for k, (x, y) in enumerate(samples):
                    # 在轮廓点周围截取 64x64 patch，边缘区域按 0 填充
                    if patches_this_image >= max_patches_per_image:
                        break
                    x_i, y_i = int(x), int(y)
                    x0 = max(0, x_i - half)
                    y0 = max(0, y_i - half)
                    x1 = min(w, x_i + half)
                    y1 = min(h, y_i + half)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    rgb = image_np[y0:y1, x0:x1, :]
                    hq_local = hq_mask_full[y0:y1, x0:x1]
                    trad_local = trad_mask_full[y0:y1, x0:x1]

                    ph = y1 - y0
                    pw = x1 - x0
                    pad_rgb = np.zeros((patch_size, patch_size, 3), dtype=rgb.dtype)
                    pad_hq = np.zeros((patch_size, patch_size), dtype=hq_local.dtype)
                    pad_trad = np.zeros((patch_size, patch_size), dtype=trad_local.dtype)
                    pad_rgb[:ph, :pw, :] = rgb
                    pad_hq[:ph, :pw] = hq_local
                    pad_trad[:ph, :pw] = trad_local

                    patch = np.concatenate(
                        [
                            pad_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0,
                            pad_hq[None, ...].astype(np.float32),
                            pad_trad[None, ...].astype(np.float32),
                        ],
                        axis=0,
                    )

                    patch_name = f"img{img_idx:04d}_box{j:02d}_pt{k:03d}.npz"
                    patch_path = patch_dir / patch_name
                    np.savez_compressed(patch_path, patch=patch)

                    # 初始 label 设为 -1，后续人工标注为 0/1
                    writer.writerow(
                        [
                            str(patch_path.relative_to(save_dir)),
                            str(img_path),
                            j,
                            0,
                            k,
                            float(x),
                            float(y),
                            -1,
                        ]
                    )
                    patch_idx_global += 1
                    patches_this_image += 1

            print(f"[dump] {img_idx + 1}/{len(image_paths)}: {img_path}, patches={patches_this_image}")


def main():
    parser = argparse.ArgumentParser(
        description="从 RT-DETRv2 + HQ-SAM 推理结果中采样局部轮廓 patch，用于 BoundaryQualityHead 训练。"
    )
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="boundary_patches")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thr", type=float, default=0.6)
    parser.add_argument("--use-hq-sam", action="store_true")
    parser.add_argument("--hq-sam-checkpoint", type=str, default=None)
    parser.add_argument("--hq-sam-model-type", type=str, default="vit_l")
    parser.add_argument("--max-patches-per-image", type=int, default=64)

    args = parser.parse_args()

    dump_patches(
        config_path=args.config,
        ckpt_path=args.resume,
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        device=args.device,
        score_thr=args.score_thr,
        use_hq_sam=args.use_hq_sam,
        hq_sam_checkpoint=args.hq_sam_checkpoint,
        hq_sam_model_type=args.hq_sam_model_type,
        max_patches_per_image=args.max_patches_per_image,
    )


if __name__ == "__main__":
    main()
