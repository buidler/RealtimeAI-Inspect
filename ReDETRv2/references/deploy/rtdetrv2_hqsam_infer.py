import os
import sys

# 把 ReDETRv2 工程根目录加入 sys.path，方便直接导入 src.core 等模块
# 当前文件在 ReDETRv2/references/deploy 下，因此需要返回两级目录
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from scipy import ndimage

# HQ-SAM 相关导入：sam_model_registry 用于按 backbone 类型构建模型，SamPredictor 封装 HQ-SAM 推理流程
from segment_anything_hq import sam_model_registry, SamPredictor

# ReDETRv2 的配置封装类，通过 YAMLConfig 构建模型和后处理器
from src.core import YAMLConfig


def _binary_mask_from_crop(crop):
    # 传统轮廓逻辑使用的辅助函数：
    # 对检测框裁剪出的局部图像做灰度 + Otsu 大津阈值 + 形态学操作，得到前景二值掩膜
    arr = np.asarray(crop.convert("L"))
    hist, _ = np.histogram(arr, bins=256, range=(0, 255))
    total = arr.size
    sum_total = np.dot(hist, np.arange(256))
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    mask = arr >= threshold
    inv_mask = ~mask
    if inv_mask.mean() > mask.mean():
        mask = inv_mask
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    return mask


def _contours_from_box(im, box):
    # 传统轮廓逻辑：在单个检测框内，根据二值掩膜提取连通域，
    # 只保留面积最大且靠近框中心的区域，然后提取其边界作为轮廓
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = im.size
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return []
    crop = im.crop((x0, y0, x1, y1))
    mask = _binary_mask_from_crop(crop)
    labeled, num = ndimage.label(mask)
    contours = []
    if num == 0:
        return contours
    h_crop, w_crop = mask.shape
    crop_center = np.array([0.5 * (w_crop - 1), 0.5 * (h_crop - 1)], dtype=np.float32)
    best_pts = None
    best_area = 0.0
    best_dist2 = None
    for label_id in range(1, num + 1):
        region = labeled == label_id
        eroded = ndimage.binary_erosion(region, structure=np.ones((3, 3)))
        boundary = region & ~eroded
        ys, xs = np.nonzero(boundary)
        if ys.size < 20:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        area = float(region.sum())
        center = pts.mean(axis=0)
        dist2 = float(((center - crop_center) ** 2).sum())
        if area > best_area or (area == best_area and (best_dist2 is None or dist2 < best_dist2)):
            best_area = area
            best_dist2 = dist2
            best_pts = pts
    if best_pts is None:
        return contours
    pts = best_pts
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]
    ordered[:, 0] += x0
    ordered[:, 1] += y0
    pts_arr = ordered
    dist_edge = np.minimum.reduce(
        [
            pts_arr[:, 0] - x0,
            pts_arr[:, 1] - y0,
            x1 - pts_arr[:, 0],
            y1 - pts_arr[:, 1],
        ]
    )
    if (dist_edge < 3.0).mean() > 0.7:
        return contours
    contours.append(pts_arr.tolist())
    return contours


def _contour_from_hqsam_mask(mask, box):
    # 将 HQ-SAM 输出的 mask 转换为一条外接轮廓曲线，
    # 并过滤掉明显贴着检测框边缘的伪轮廓
    region = mask.astype(bool)
    eroded = ndimage.binary_erosion(region, structure=np.ones((3, 3)))
    boundary = region & ~eroded
    ys, xs = np.nonzero(boundary)
    if ys.size < 20:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]
    x0, y0, x1, y1 = box
    pts_arr = ordered
    dist_edge = np.minimum.reduce(
        [
            pts_arr[:, 0] - x0,
            pts_arr[:, 1] - y0,
            x1 - pts_arr[:, 0],
            y1 - pts_arr[:, 1],
        ]
    )
    if (dist_edge < 3.0).mean() > 0.7:
        return None
    return pts_arr


def _build_rtdetr_model(config_path, ckpt_path, device):
    # 根据 YAML 配置和训练权重构建部署版 RT-DETRv2 模型（含后处理）
    cfg = YAMLConfig(config_path, resume=ckpt_path)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
        cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    return model


def _run_inference(
    config_path,
    ckpt_path,
    image_path,
    device,
    score_thr,
    use_hq_sam,
    hq_sam_checkpoint,
    hq_sam_model_type,
    use_traditional_contours,
    save_dir,
):
    # 1. 构建 RT-DETRv2 部署模型
    device = torch.device(device)
    model = _build_rtdetr_model(config_path, ckpt_path, device)
    # 2. 读入待推理图像，并转换为张量
    im_pil = Image.open(image_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)
    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil)[None].to(device)
    # 3. 前向推理，得到标签、框和得分
    labels, boxes, scores = model(im_data, orig_size)
    os.makedirs(save_dir, exist_ok=True)
    drawer = ImageDraw.Draw(im_pil)
    # 4. 只保留置信度大于阈值的检测框
    scr = scores[0]
    lab = labels[0]
    box = boxes[0]
    keep = scr > score_thr
    scr = scr[keep]
    lab = lab[keep]
    box = box[keep]
    predictor = None
    # 5. 如果启用 HQ-SAM，则加载 HQ-SAM 模型并对整张图像进行编码
    if use_hq_sam:
        if hq_sam_checkpoint is None:
            raise ValueError("hq_sam_checkpoint is required when use_hq_sam is True")
        sam = sam_model_registry[hq_sam_model_type](checkpoint=hq_sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        image_np = np.array(im_pil)
        predictor.set_image(image_np)
    # 6. 遍历每个检测框，画框和文本，并根据开关选择 HQ-SAM 或传统轮廓
    for j, b in enumerate(box):
        b_np = b.detach().cpu().numpy()
        drawer.rectangle(list(b_np), outline="red")
        drawer.text(
            (float(b_np[0]), float(b_np[1])),
            text=f"{lab[j].item()} {round(scr[j].item(), 2)}",
            fill="blue",
        )
        if use_hq_sam and predictor is not None:
            masks, _, _ = predictor.predict(box=b_np, multimask_output=False)
            m = masks[0]
            pts_arr = _contour_from_hqsam_mask(m, b_np)
            if pts_arr is None:
                continue
            pts_seq = [(float(p[0]), float(p[1])) for p in pts_arr]
            drawer.line(pts_seq + [pts_seq[0]], fill="yellow", width=2)
        elif use_traditional_contours:
            contour_list = _contours_from_box(im_pil, b_np)
            for pts in contour_list:
                if len(pts) < 3:
                    continue
                pts_seq = [(float(p[0]), float(p[1])) for p in pts]
                drawer.line(pts_seq + [pts_seq[0]], fill="yellow", width=2)
    # 7. 按模式组合输出文件名并保存结果图
    stem, _ = os.path.splitext(os.path.basename(image_path))
    if use_hq_sam:
        out_name = f"{stem}_hqsam_contours.jpg"
    elif use_traditional_contours:
        out_name = f"{stem}_contours.jpg"
    else:
        out_name = f"{stem}_det.jpg"
    out_path = os.path.join(save_dir, out_name)
    im_pil.save(out_path)
    return out_path


def main():
    import argparse

    # 命令行参数：兼容原有 RT-DETRv2 推理接口，并增加 HQ-SAM 相关开关
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-f", "--im-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thr", type=float, default=0.6)
    parser.add_argument("--save-dir", type=str, default="output_hqsam")
    parser.add_argument("--use-hq-sam", action="store_true")
    parser.add_argument("--hq-sam-checkpoint", type=str, default=None)
    parser.add_argument("--hq-sam-model-type", type=str, default="vit_l")
    parser.add_argument("--use-traditional-contours", action="store_true")

    args = parser.parse_args()

    out_path = _run_inference(
        config_path=args.config,
        ckpt_path=args.resume,
        image_path=args.im_file,
        device=args.device,
        score_thr=args.score_thr,
        use_hq_sam=args.use_hq_sam,
        hq_sam_checkpoint=args.hq_sam_checkpoint,
        hq_sam_model_type=args.hq_sam_model_type,
        use_traditional_contours=args.use_traditional_contours,
        save_dir=args.save_dir,
    )
    print(f"saved result to {out_path}")


if __name__ == "__main__":
    main()

