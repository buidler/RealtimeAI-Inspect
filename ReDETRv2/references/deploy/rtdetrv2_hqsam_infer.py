"""本脚本用于：
1）使用 ReDETRv2 对纤维进行目标检测（得到矩形框）；
2）将检测框作为 HQ-SAM 的 box prompt，生成高质量分割 mask；
3）根据 HQ-SAM 的 mask 或传统逻辑生成纤维轮廓并可视化。

支持的轮廓模式：
- 仅 HQ-SAM（--use-hq-sam）：依赖 HQ-SAM 的分割结果生成轮廓；
- 仅传统轮廓（--use-traditional-contours）：使用二值化+连通域的经典图像处理；
- 混合模式（同时指定两者）：能用传统算法的地方优先用传统轮廓细化凹陷，失败时退回 HQ-SAM。
"""

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


def _contours_from_box(im, box, mask_hint=None):
    # 传统轮廓逻辑：在单个检测框内，根据二值掩膜提取连通域，
    # 只保留面积最大且靠近框中心的区域，然后提取其边界作为轮廓。
    # mask_hint 可选：为整张图像上的 HQ-SAM 掩膜。
    # 若提供，则在裁剪区域内只保留靠近 HQ-SAM 边界的一圈窄带区域，
    # 让传统算法只在 HQ 边缘附近做细化，而不在整个前景内部重新分割，避免出现大面积色块。
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
    if mask_hint is not None:
        hint_crop = mask_hint[y0:y1, x0:x1]
        # 只在 HQ-SAM 掩膜内部、且靠近其边界的一圈窄带内保留传统分割结果
        if hint_crop.shape == mask.shape:
            # 计算 HQ 掩膜的内部窄带：hq_band = hq_mask & ~erode(hq_mask)
            # 这里 iterations=1 表示只向内收缩 1 个像素，得到相对“宽”的边缘带，
            # 既能限制传统算法只在边缘附近工作，又尽量避免轮廓被截断得太短。
            band = hint_crop.copy().astype(bool)
            eroded = ndimage.binary_erosion(band, iterations=1)
            band = band & ~eroded
            if not band.any():
                # 如果窄带被完全侵蚀掉，则退回到原始掩膜区域
                band = hint_crop.astype(bool)
            mask = mask & band
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


def _contour_from_hqsam_mask(mask, box, conservative_boundary=False):
    # 将 HQ-SAM 输出的单个 mask 转换为一条外接轮廓曲线，
    # 并过滤掉明显贴着检测框边缘的伪轮廓。
    # conservative_boundary=True 时不会做腐蚀操作，尽量保留细小凹陷；
    # False 时会先腐蚀再取边界，能弱化毛刺但也可能轻微抹平小凹陷。
    region = mask.astype(bool)
    if conservative_boundary:
        boundary = region ^ ndimage.binary_erosion(region, structure=np.ones((3, 3)))
    else:
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


def _select_best_hqsam_contour(masks, box, conservative_boundary=False):
    # 在 HQ-SAM 返回的多张候选 mask 中，选择“边界更复杂”的一张：
    # 这里用轮廓点数量作为复杂度近似，点越多说明轮廓越曲折、细节越丰富。
    # 同时返回对应的二值掩膜，用于后续约束传统轮廓（只保留与 HQ 区域重合度高的传统轮廓）。
    best_pts = None
    best_mask = None
    best_len = 0
    for m in masks:
        pts_arr = _contour_from_hqsam_mask(m, box, conservative_boundary=conservative_boundary)
        if pts_arr is None:
            continue
        if len(pts_arr) > best_len:
            best_len = len(pts_arr)
            best_pts = pts_arr
            best_mask = m.astype(bool)
    return best_pts, best_mask


def _filter_traditional_contours_with_hqsam(trad_contours, hq_mask, image_size, min_overlap=0.10):
    # 使用 HQ-SAM 的 mask 过滤传统轮廓：
    # 思路：把传统轮廓先 rasterize 成一个多边形区域 poly，再与 HQ 掩膜做交集，
    # 计算 overlap = (poly ∧ hq_mask) / poly，只有 overlap 足够大才认为传统轮廓可信。
    # 这样可以避免出现大块色块或者严重偏离 HQ 结果的传统伪轮廓。
    if hq_mask is None or not trad_contours:
        return trad_contours
    w, h = image_size
    filtered = []
    for pts in trad_contours:
        if pts is None or len(pts) < 3:
            continue
        mask_im = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_im)
        draw.polygon([(float(p[0]), float(p[1])) for p in pts], outline=1, fill=1)
        poly = np.array(mask_im, dtype=bool)
        inter = poly & hq_mask
        denom = poly.sum()
        if denom == 0:
            continue
        overlap = inter.sum() / float(denom)
        if overlap >= min_overlap:
            filtered.append(pts)
    return filtered


def _polyline_length(pts):
    # 计算一条多边形折线（首尾相连）的大致周长，用于评估轮廓覆盖程度。
    if pts is None or len(pts) < 2:
        return 0.0
    arr = np.asarray(pts, dtype=np.float32)
    diffs = np.diff(np.vstack([arr, arr[0]]), axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    return float(seg_len.sum())


def _build_rtdetr_model(config_path, ckpt_path, device):
    # 根据 YAML 配置和训练权重构建部署版 RT-DETRv2 模型（含后处理），
    # 输出已经过后处理的 (labels, boxes, scores)，坐标均在原图尺度。
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
    conservative_boundary,
):
    # 1. 构建 RT-DETRv2 部署模型（只做前向推理，不参与训练和反向传播）
    device = torch.device(device)
    model = _build_rtdetr_model(config_path, ckpt_path, device)
    # 2. 读入待推理图像，并转换为张量（缩放到 640x640 送入 RT-DETRv2）
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
    # 3. 前向推理，得到标签、框和得分；框坐标已经映射回原图尺度
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
    # 5. 如果启用 HQ-SAM，则加载 HQ-SAM 模型并对整张图像进行编码，
    #    后续每个检测框直接作为 box prompt 调用 predictor.predict。
    if use_hq_sam:
        if hq_sam_checkpoint is None:
            raise ValueError("hq_sam_checkpoint is required when use_hq_sam is True")
        sam = sam_model_registry[hq_sam_model_type](checkpoint=hq_sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        image_np = np.array(im_pil)
        predictor.set_image(image_np)
    # 6. 遍历每个检测框，画框和文本，并根据开关选择 HQ-SAM / 传统轮廓 / 混合模式
    for j, b in enumerate(box):
        b_np = b.detach().cpu().numpy()
        drawer.rectangle(list(b_np), outline="red")
        drawer.text(
            (float(b_np[0]), float(b_np[1])),
            text=f"{lab[j].item()} {round(scr[j].item(), 2)}",
            fill="blue",
        )
        hq_pts = None
        hq_mask = None
        trad_contours = []
        # 先计算 HQ-SAM 轮廓（如果启用），保证每个检测框都有一条“整体轮廓”候选
        if use_hq_sam and predictor is not None:
            # HQ-SAM 推理：仅使用检测框作为 box 提示，启用多掩码候选
            masks, _, _ = predictor.predict(
                box=b_np,
                multimask_output=True,
            )
            # 从多候选 mask 中挑选轮廓更复杂的一张，尽量贴合凹陷边缘
            hq_pts, hq_mask = _select_best_hqsam_contour(masks, b_np, conservative_boundary=conservative_boundary)
        # 再计算传统轮廓（如果启用），用于对 HQ-SAM 结果做凹陷细节的补充
        if use_traditional_contours:
            # 以 HQ-SAM 掩膜的外接矩形作为传统算法的 ROI，避免直接使用检测框导致区域过大
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
            # 使用 HQ-SAM 掩膜过滤掉与 HQ 结果明显不一致的传统轮廓，避免出现大块色块和贴框伪轮廓
            trad_contours = _filter_traditional_contours_with_hqsam(trad_contours, hq_mask, im_pil.size)

        # 根据模式选择最终要绘制的轮廓：
        # 1）仅 HQ-SAM：只绘制 HQ-SAM 轮廓（适合无传统处理的快速模式）；
        # 2）仅传统：只绘制传统轮廓（等价于老版 rtdetrv2_torch 的轮廓行为）；
        # 3）混合（两者都启用）：优先使用传统轮廓刻画局部凹陷，
        #    但仍要求传统轮廓长度不能太短；若只是一小段扇形，则退回使用 HQ-SAM 的整体轮廓。
        final_contours = []
        if use_hq_sam and use_traditional_contours:
            if trad_contours and hq_pts is not None:
                # 传统轮廓存在且与 HQ 掩膜重合度已通过筛选时，
                # 进一步要求其长度至少覆盖 HQ 轮廓的约 30%，
                # 以避免只保留极短的一小段扇形导致轮廓明显缺失。
                trad_len = _polyline_length(trad_contours[0])
                hq_len = _polyline_length(hq_pts)
                if hq_len > 0 and trad_len / hq_len >= 0.1:
                    final_contours = trad_contours
                else:
                    final_contours = [hq_pts]
            elif trad_contours:
                final_contours = trad_contours
            elif hq_pts is not None:
                final_contours = [hq_pts]
        elif use_traditional_contours:
            final_contours = trad_contours
        elif use_hq_sam and hq_pts is not None:
            final_contours = [hq_pts]

        for pts in final_contours:
            if pts is None or len(pts) < 3:
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

    # 命令行参数：兼容原有 RT-DETRv2 推理接口，并增加 HQ-SAM / 混合轮廓相关开关
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
    parser.add_argument(
        "--conservative-boundary",
        action="store_true",
        help="启用保守边界模式：HQ-SAM 轮廓提取时不做腐蚀，尽量保留细小凹陷。",
    )

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
        conservative_boundary=args.conservative_boundary,
    )
    print(f"saved result to {out_path}")


if __name__ == "__main__":
    main()

