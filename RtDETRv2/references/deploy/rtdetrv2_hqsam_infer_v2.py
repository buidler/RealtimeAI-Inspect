import argparse
import os
from pathlib import Path
import sys
from datetime import datetime
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
 # 线程限制用于降低 OpenCV/Torch 在某些环境下的并行不稳定风险。

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig
from tools.interactive_dump_boundary_patches import _build_sam
from references.deploy.rtdetrv2_hqsam_infer import (
    _contours_from_box,
    _filter_long_thin_contours,
    _filter_traditional_contours_with_hqsam,
    _select_best_hqsam_contour,
)

# V2 增强版流程：
# - 增加 OpenCV 预处理 + Soft-NMS + 可选后精修。
# - 提供追踪日志钩子，用于定位静默失败等原生运行时问题。
# - 可通过参数对“精度-面积-稳定性”做可控折中。

def _emit_trace(enabled, log_path, msg):
    # 统一追踪输出：
    # - 控制台实时打印
    # - 可选写入日志文件，便于定位在哪个阶段停止
    if not enabled:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_path:
        with Path(log_path).open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _resolve_path(raw_path, field_name, allow_missing=False):
    # 与 V1 一致的路径解析策略，避免因 cwd 变化导致找不到文件。
    p = Path(raw_path).expanduser()
    candidates = [p]
    if not p.is_absolute():
        candidates.append(PROJECT_ROOT / p)
        candidates.append(Path.cwd() / p)
    seen = set()
    ordered = []
    for c in candidates:
        k = str(c)
        if k not in seen:
            seen.add(k)
            ordered.append(c)
    for c in ordered:
        if c.exists():
            return c.resolve()
    if allow_missing:
        return ordered[0].resolve()
    tried = ", ".join(str(c.resolve()) for c in ordered)
    raise FileNotFoundError(f"{field_name} not found: {raw_path}. tried: {tried}")


def _list_images(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_dir = Path(image_dir).resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    return sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])


def _polyline_length(pts):
    if pts is None or len(pts) < 2:
        return 0.0
    arr = np.asarray(pts, dtype=np.float32)
    diffs = np.diff(np.vstack([arr, arr[0]]), axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def _is_edge_hugging(pts, box, thr=3.0, frac=0.85):
    if pts is None or len(pts) < 3:
        return False
    arr = np.asarray(pts, dtype=np.float32)
    x0, y0, x1, y1 = [float(v) for v in box]
    dist_edge = np.minimum.reduce(
        [
            arr[:, 0] - x0,
            arr[:, 1] - y0,
            x1 - arr[:, 0],
            y1 - arr[:, 1],
        ]
    )
    return float((dist_edge < thr).mean()) > frac


def _polygon_overlap_with_mask(pts, hq_mask, image_size):
    if hq_mask is None or pts is None or len(pts) < 3:
        return 0.0
    w, h = image_size
    mask_im = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_im)
    draw.polygon([(float(p[0]), float(p[1])) for p in pts], outline=1, fill=1)
    poly = np.array(mask_im, dtype=bool)
    denom = poly.sum()
    if denom == 0:
        return 0.0
    inter = poly & hq_mask.astype(bool)
    return float(inter.sum()) / float(denom)


def _box_iou(a, b):
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = max(1e-6, aa + bb - inter)
    return float(inter / union)


def _opencv_preprocess(
    image_np,
    clahe_clip=2.0,
    clahe_grid=8,
    bilateral_d=5,
    bilateral_sigma=35,
    sharpen=0.2,
    trace_enabled=False,
    trace_log_path=None,
):
    # 预处理链路（可关闭）：
    # RGB->LAB(局部对比度增强)->双边滤波(抑噪保边)->轻微锐化。
    # 目的：提高传统轮廓提取阶段的边界可见性与连贯性。
    _emit_trace(trace_enabled, trace_log_path, f"opencv_preprocess:start shape={tuple(image_np.shape)} dtype={image_np.dtype}")
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_grid), int(clahe_grid)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    bgr3 = cv2.bilateralFilter(bgr2, int(bilateral_d), float(bilateral_sigma), float(bilateral_sigma))
    if sharpen > 0:
        blur = cv2.GaussianBlur(bgr3, (0, 0), 1.0)
        bgr3 = cv2.addWeighted(bgr3, 1.0 + float(sharpen), blur, -float(sharpen), 0)
    out = cv2.cvtColor(bgr3, cv2.COLOR_BGR2RGB)
    _emit_trace(trace_enabled, trace_log_path, f"opencv_preprocess:done shape={tuple(out.shape)}")
    return out


def _soft_nms_gaussian(boxes_np, scores_np, sigma=0.5, score_thr=0.2):
    # Soft-NMS（高斯衰减）：
    # 与硬阈值 NMS 不同，这里保留相邻目标，只降低其分数，减少漏检。
    boxes = boxes_np.astype(np.float32).copy()
    scores = scores_np.astype(np.float32).copy()
    idxs = np.arange(len(scores))
    kept = []
    kept_scores = []
    while len(scores) > 0:
        i = int(np.argmax(scores))
        bi = boxes[i]
        si = float(scores[i])
        idi = int(idxs[i])
        if si < score_thr:
            break
        kept.append(idi)
        kept_scores.append(si)
        ious = np.array([_box_iou(bi, b) for b in boxes], dtype=np.float32)
        scores = scores * np.exp(-(ious * ious) / max(1e-6, float(sigma)))
        boxes = np.delete(boxes, i, axis=0)
        scores = np.delete(scores, i, axis=0)
        idxs = np.delete(idxs, i, axis=0)
    return kept, kept_scores


def _shape_metrics(contour, box):
    # 形状质量指标：
    # - solidity：凹陷程度/噪声敏感性
    # - circularity：边界平滑程度
    # - area_ratio：轮廓面积相对检测框占比
    c = np.asarray(contour, dtype=np.float32)
    if c.shape[0] < 3:
        return {"solidity": 0.0, "circularity": 0.0, "area_ratio": 0.0}
    area = float(abs(cv2.contourArea(c)))
    peri = float(cv2.arcLength(c, True))
    hull = cv2.convexHull(c)
    hull_area = float(abs(cv2.contourArea(hull)))
    solidity = area / max(1e-6, hull_area)
    circularity = 4.0 * np.pi * area / max(1e-6, peri * peri)
    x0, y0, x1, y1 = [float(v) for v in box]
    box_area = max(1e-6, (x1 - x0) * (y1 - y0))
    area_ratio = area / box_area
    return {"solidity": float(solidity), "circularity": float(circularity), "area_ratio": float(area_ratio)}


def _contour_inclusion_ratio(inner_contour, outer_contour, sample_points=80):
    # 估计 inner 是否位于 outer 内部：返回采样点落在 outer 内部的比例。
    inner = np.asarray(inner_contour, dtype=np.float32)
    outer = np.asarray(outer_contour, dtype=np.float32)
    if inner.shape[0] < 3 or outer.shape[0] < 3:
        return 0.0
    step = max(1, inner.shape[0] // max(1, int(sample_points)))
    pts = inner[::step]
    inside = 0
    for p in pts:
        if cv2.pointPolygonTest(outer, (float(p[0]), float(p[1])), False) >= 0:
            inside += 1
    return float(inside) / float(max(1, pts.shape[0]))


def _prefer_outer_contours(candidates, inclusion_thr=0.88, iou_gate=0.2):
    # 去除“同一目标双轮廓（内外圈）”：
    # 若小轮廓大部分点位于大轮廓内部，则只保留外侧轮廓。
    if not candidates:
        return candidates

    def _area(c):
        return float(abs(cv2.contourArea(np.asarray(c["contour"], dtype=np.float32))))

    ordered = sorted(candidates, key=_area, reverse=True)
    kept = []
    for cand in ordered:
        drop = False
        for k in kept:
            if _box_iou(cand["bbox"], k["bbox"]) < float(iou_gate):
                continue
            ratio = _contour_inclusion_ratio(cand["contour"], k["contour"])
            if ratio >= float(inclusion_thr):
                drop = True
                break
        if not drop:
            kept.append(cand)
    return kept


def _contour_to_mask_local(contour, bbox, image_shape):
    # 将全局轮廓投影到 bbox 局部 mask，供形态学和边缘对齐使用。
    h, w = image_shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return None, (x0, y0, x1, y1)
    roi_w = x1 - x0
    roi_h = y1 - y0
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    c = np.asarray(contour, dtype=np.float32).copy()
    c[:, 0] -= x0
    c[:, 1] -= y0
    cv2.fillPoly(mask, [c.astype(np.int32)], 255)
    return mask, (x0, y0, x1, y1)


def _largest_contour_from_mask(mask):
    # 仅保留最大连通轮廓，减少碎片和噪点干扰。
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cnt.shape[0] < 3:
        return None
    return cnt[:, 0, :].astype(np.float32)


def _active_contour_like(points, edge_map, iters=12, alpha=0.2, beta=0.35, search_radius=3):
    # 轻量“类主动轮廓”：
    # - alpha 控制平滑（邻点拉回）
    # - beta 控制向局部边缘点吸附
    # 目标是在不过度收缩的前提下，让边界更贴近真实边缘。
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.shape[0] < 6:
        return pts
    h, w = edge_map.shape[:2]
    ys, xs = np.where(edge_map > 0)
    edge_points = np.stack([xs, ys], axis=1).astype(np.float32) if xs.size > 0 else None
    for _ in range(int(iters)):
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
        pts = pts + float(alpha) * (prev + nxt - 2.0 * pts)
        if edge_points is not None:
            for i in range(pts.shape[0]):
                px, py = pts[i]
                x0 = max(0, int(px) - int(search_radius))
                x1 = min(w, int(px) + int(search_radius) + 1)
                y0 = max(0, int(py) - int(search_radius))
                y1 = min(h, int(py) + int(search_radius) + 1)
                patch = edge_map[y0:y1, x0:x1]
                if patch.size == 0 or patch.max() == 0:
                    continue
                yy, xx = np.where(patch > 0)
                gx = xx + x0
                gy = yy + y0
                d2 = (gx - px) ** 2 + (gy - py) ** 2
                j = int(np.argmin(d2))
                target = np.array([gx[j], gy[j]], dtype=np.float32)
                pts[i] = pts[i] + float(beta) * (target - pts[i])
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def _refine_contour_post(
    contour,
    box,
    image_np,
    morph_kernel=3,
    close_iter=1,
    open_iter=1,
    smooth_eps=0.002,
    trace_enabled=False,
    trace_log_path=None,
):
    # 后精修：mask 形态学 + 边缘吸附 + 多边形平滑。
    # 注意：open_iter 越大，轮廓可能越“收”；面积偏小时可降低该值。
    _emit_trace(trace_enabled, trace_log_path, "post_refine:start")
    mask, bbox = _contour_to_mask_local(contour, box, image_np.shape)
    if mask is None:
        _emit_trace(trace_enabled, trace_log_path, "post_refine:skip mask_none")
        return contour
    x0, y0, x1, y1 = bbox
    roi = image_np[y0:y1, x0:x1]
    if roi.size == 0:
        _emit_trace(trace_enabled, trace_log_path, "post_refine:skip roi_empty")
        return contour
    k = max(1, int(morph_kernel))
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=max(0, int(close_iter)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, iterations=max(0, int(open_iter)))
    c0 = _largest_contour_from_mask(m)
    if c0 is None:
        _emit_trace(trace_enabled, trace_log_path, "post_refine:skip largest_contour_none")
        return contour
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 40, 120)
    c1 = _active_contour_like(c0, edge, iters=10, alpha=0.2, beta=0.3, search_radius=3)
    eps = max(0.5, float(smooth_eps) * cv2.arcLength(c1.astype(np.float32), True))
    c2 = cv2.approxPolyDP(c1.astype(np.float32), eps, True)[:, 0, :].astype(np.float32)
    c2[:, 0] += x0
    c2[:, 1] += y0
    if c2.shape[0] < 3:
        _emit_trace(trace_enabled, trace_log_path, "post_refine:skip refined_too_short")
        return contour
    _emit_trace(trace_enabled, trace_log_path, f"post_refine:done pts={int(c2.shape[0])}")
    return c2


def _prepare_candidates_v3(
    im_pil_trad,
    image_np_orig,
    boxes,
    soft_scores,
    predictor,
    use_hq_sam,
    trad_overlap_thr_low=0.28,
    trad_overlap_thr_high=0.45,
    trad_min_len_ratio_low=0.2,
    trad_min_len_ratio_high=0.35,
    edge_dist_thr=3.0,
    edge_frac_thr=0.85,
    overlap_iou_thr=0.35,
    min_solidity=0.62,
    min_area_ratio=0.08,
    score_low_thr=0.62,
    post_refine_open_iter=1,
    enable_post_refine=True,
    trace_enabled=False,
    trace_log_path=None,
):
    # 相比 V1 使用更严格的形状门控来生成最终轮廓：
    # HQ-SAM/传统轮廓混合选择 + 对不规则轮廓执行可选后精修。
    # 关键差异：
    # - V1 主要看 overlap/len/风险；
    # - V2 在此基础上再加入 solidity/area_ratio 等形状质量约束。
    candidates = []
    if use_hq_sam and predictor is not None:
        predictor.set_image(image_np_orig)
    boxes_np_all = [b.detach().cpu().numpy() for b in boxes]
    for j, b in enumerate(boxes):
        _emit_trace(trace_enabled, trace_log_path, f"prepare:box_start idx={j}")
        b_np = boxes_np_all[j]
        s_j = float(soft_scores[j]) if j < len(soft_scores) else 0.0
        hq_mask = None
        hq_contour = None
        if use_hq_sam and predictor is not None:
            masks, _, _ = predictor.predict(box=b_np, multimask_output=True)
            hq_contour, hq_mask = _select_best_hqsam_contour(masks, b_np, conservative_boundary=False)
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

        trad_contours = _contours_from_box(im_pil_trad, box_for_trad, mask_hint=hq_mask)
        trad_contours = _filter_traditional_contours_with_hqsam(trad_contours, hq_mask, im_pil_trad.size)
        trad_contours = _filter_long_thin_contours(trad_contours, box_for_trad)

        trad_contour = None
        if trad_contours:
            trad_contour = np.asarray(trad_contours[0], dtype=np.float32)
            if trad_contour.shape[0] < 3:
                trad_contour = None
        if hq_contour is not None:
            hq_contour = np.asarray(hq_contour, dtype=np.float32)
            if hq_contour.shape[0] < 3:
                hq_contour = None

        neigh_iou = 0.0
        for k, bk in enumerate(boxes_np_all):
            if k == j:
                continue
            neigh_iou = max(neigh_iou, _box_iou(b_np, bk))

        final_contour = None
        if trad_contour is not None and hq_contour is not None:
            trad_edge = _is_edge_hugging(trad_contour, b_np, thr=edge_dist_thr, frac=edge_frac_thr)
            overlap = _polygon_overlap_with_mask(trad_contour, hq_mask, im_pil_trad.size)
            trad_len = _polyline_length(trad_contour)
            hq_len = _polyline_length(hq_contour)
            len_ratio = trad_len / max(1e-6, hq_len)
            shape = _shape_metrics(trad_contour, b_np)
            shape_good = shape["solidity"] >= min_solidity and shape["area_ratio"] >= min_area_ratio
            trad_high = (not trad_edge) and overlap >= trad_overlap_thr_high and len_ratio >= trad_min_len_ratio_high and shape_good
            trad_mid = overlap >= trad_overlap_thr_low and len_ratio >= trad_min_len_ratio_low and shape_good
            risky = neigh_iou >= overlap_iou_thr
            score_low = s_j < float(score_low_thr)
            # 分支优先级：
            # 1) trad_high：质量高时优先保留传统轮廓细节；
            # 2) trad_mid：中等质量时更保守，避免在拥挤区域误选；
            # 3) 兜底回退到 HQ-SAM，保证稳定性。
            if trad_high and not (risky and score_low):
                final_contour = trad_contour
            elif trad_mid and not risky and not score_low:
                final_contour = trad_contour
            else:
                final_contour = hq_contour
        elif trad_contour is not None:
            shape = _shape_metrics(trad_contour, b_np)
            trad_edge = _is_edge_hugging(trad_contour, b_np, thr=edge_dist_thr, frac=edge_frac_thr)
            if shape["solidity"] >= min_solidity and shape["area_ratio"] >= min_area_ratio and not trad_edge:
                final_contour = trad_contour
        elif hq_contour is not None:
            final_contour = hq_contour

        if final_contour is None:
            _emit_trace(trace_enabled, trace_log_path, f"prepare:box_drop idx={j} reason=no_final_contour")
            continue
        irregular = _shape_metrics(final_contour, b_np)["circularity"] < 0.45 or _is_edge_hugging(final_contour, b_np, thr=edge_dist_thr, frac=0.9)
        # 只对“不规则”轮廓做后精修，避免对本来质量好的轮廓过度加工。
        if enable_post_refine and irregular:
            final_contour = _refine_contour_post(
                final_contour,
                b_np,
                image_np_orig,
                open_iter=post_refine_open_iter,
                trace_enabled=trace_enabled,
                trace_log_path=trace_log_path,
            )
        x0, y0 = final_contour.min(axis=0)
        x1, y1 = final_contour.max(axis=0)
        candidates.append(
            {
                "box_idx": j,
                "contour": final_contour,
                "bbox": (int(x0), int(y0), int(x1), int(y1)),
                "hq_mask": hq_mask.astype(np.uint8) if hq_mask is not None else None,
            }
        )
        _emit_trace(trace_enabled, trace_log_path, f"prepare:box_done idx={j} bbox=({int(x0)},{int(y0)},{int(x1)},{int(y1)})")
    return candidates


def _build_rtdetr_model(config_path, ckpt_path, device):
    # 与 V1 一致：优先 EMA 权重，统一 deploy 路径。
    cfg = YAMLConfig(str(config_path), resume=str(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    return Model().to(device).eval()


def _draw_and_save(im_pil, boxes_np, candidates, save_dir, stem):
    # 红框是检测框，黑线是最终轮廓，便于快速肉眼评估“框与轮廓”一致性。
    drawer = ImageDraw.Draw(im_pil)
    for b in boxes_np:
        drawer.rectangle(list(b), outline="red")
    for c in candidates:
        pts = c["contour"]
        if pts is None or len(pts) < 3:
            continue
        pts_seq = [(float(p[0]), float(p[1])) for p in pts]
        drawer.line(pts_seq + [pts_seq[0]], fill="black", width=2)
    out_name = f"{stem}_hqsam_contours.jpg"
    out_path = save_dir / out_name
    im_pil.save(out_path)
    return out_path


def run_single(
    model,
    predictor,
    image_path,
    device,
    score_thr,
    save_dir,
    trad_overlap_thr_low,
    trad_overlap_thr_high,
    trad_min_len_ratio_low,
    trad_min_len_ratio_high,
    edge_dist_thr,
    edge_frac_thr,
    overlap_iou_thr,
    soft_nms_sigma,
    soft_nms_score_thr,
    clahe_clip,
    clahe_grid,
    bilateral_d,
    bilateral_sigma,
    sharpen,
    disable_opencv_preproc,
    disable_post_refine,
    min_solidity,
    min_area_ratio,
    score_low_thr,
    post_refine_open_iter,
    debug_trace=False,
    debug_log_path=None,
):
    # 单图推理（支持可选诊断）：
    # 读图 -> 可选预处理 -> 检测 -> soft-nms -> 轮廓选择/精修 -> 保存。
    # 若出现静默失败，可开启 --debug-trace 并查看最后一个阶段日志。
    _emit_trace(debug_trace, debug_log_path, f"run_single:start image={image_path}")
    im_pil = Image.open(image_path).convert("RGB")
    image_np_orig = np.array(im_pil)
    _emit_trace(debug_trace, debug_log_path, f"run_single:image_loaded shape={tuple(image_np_orig.shape)}")
    if disable_opencv_preproc:
        _emit_trace(debug_trace, debug_log_path, "run_single:opencv_preproc_disabled")
        image_np_proc = image_np_orig
    else:
        image_np_proc = _opencv_preprocess(
            image_np_orig,
            clahe_clip=clahe_clip,
            clahe_grid=clahe_grid,
            bilateral_d=bilateral_d,
            bilateral_sigma=bilateral_sigma,
            sharpen=sharpen,
            trace_enabled=debug_trace,
            trace_log_path=debug_log_path,
        )
    im_pil_proc = Image.fromarray(image_np_proc)
    w, h = im_pil.size
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    im_data = transforms(im_pil_proc)[None].to(device)
    orig_size = torch.tensor([w, h])[None].to(device)
    _emit_trace(debug_trace, debug_log_path, "run_single:model_infer:start")
    with torch.no_grad():
        _, boxes, scores = model(im_data, orig_size)
    _emit_trace(debug_trace, debug_log_path, "run_single:model_infer:done")
    scr = scores[0].detach().cpu().numpy().astype(np.float32)
    box = boxes[0]
    box_np = box.detach().cpu().numpy().astype(np.float32)
    keep0 = scr > float(score_thr)
    box_np = box_np[keep0]
    scr = scr[keep0]
    if box_np.shape[0] == 0:
        _emit_trace(debug_trace, debug_log_path, "run_single:no_boxes_after_thr")
        return _draw_and_save(im_pil, np.zeros((0, 4), dtype=np.float32), [], save_dir, Path(image_path).stem)
    _emit_trace(debug_trace, debug_log_path, f"run_single:soft_nms:start boxes={int(box_np.shape[0])}")
    keep_ids, keep_scores = _soft_nms_gaussian(
        box_np,
        scr,
        sigma=soft_nms_sigma,
        score_thr=soft_nms_score_thr,
    )
    if len(keep_ids) == 0:
        _emit_trace(debug_trace, debug_log_path, "run_single:no_boxes_after_softnms")
        return _draw_and_save(im_pil, np.zeros((0, 4), dtype=np.float32), [], save_dir, Path(image_path).stem)
    box_np = box_np[keep_ids]
    soft_scores = np.asarray(keep_scores, dtype=np.float32)
    box_t = torch.from_numpy(box_np).to(device=device, dtype=box.dtype)
    candidates = _prepare_candidates_v3(
        im_pil_proc,
        image_np_orig,
        box_t,
        soft_scores,
        predictor,
        predictor is not None,
        trad_overlap_thr_low=trad_overlap_thr_low,
        trad_overlap_thr_high=trad_overlap_thr_high,
        trad_min_len_ratio_low=trad_min_len_ratio_low,
        trad_min_len_ratio_high=trad_min_len_ratio_high,
        edge_dist_thr=edge_dist_thr,
        edge_frac_thr=edge_frac_thr,
        overlap_iou_thr=overlap_iou_thr,
        min_solidity=min_solidity,
        min_area_ratio=min_area_ratio,
        score_low_thr=score_low_thr,
        post_refine_open_iter=post_refine_open_iter,
        enable_post_refine=not disable_post_refine,
        trace_enabled=debug_trace,
        trace_log_path=debug_log_path,
    )
    candidates = _prefer_outer_contours(candidates)
    _emit_trace(debug_trace, debug_log_path, f"run_single:draw_save:start candidates={len(candidates)}")
    out = _draw_and_save(im_pil, box_np, candidates, save_dir, Path(image_path).stem)
    _emit_trace(debug_trace, debug_log_path, f"run_single:done out={out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    # 基础输入/输出与模型配置
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-f", "--im-file", type=str, default="")
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thr", type=float, default=0.6)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--use-hq-sam", action="store_true")
    parser.add_argument("--hq-sam-checkpoint", type=str, default=None)
    parser.add_argument("--hq-sam-model-type", type=str, default="vit_l")
    # 轮廓融合核心阈值
    # Light 参考参数：
    # trad-overlap-thr-low=0.28, trad-overlap-thr-high=0.45
    # trad-min-len-ratio-low=0.20, trad-min-len-ratio-high=0.35
    # edge-dist-thr=3.0, edge-frac-thr=0.85, overlap-iou-thr=0.35
    parser.add_argument("--trad-overlap-thr-low", type=float, default=0.28)
    parser.add_argument("--trad-overlap-thr-high", type=float, default=0.45)
    parser.add_argument("--trad-min-len-ratio-low", type=float, default=0.20)
    parser.add_argument("--trad-min-len-ratio-high", type=float, default=0.35)
    parser.add_argument("--edge-dist-thr", type=float, default=3.0)
    parser.add_argument("--edge-frac-thr", type=float, default=0.85)
    parser.add_argument("--overlap-iou-thr", type=float, default=0.35)
    # 检测后处理（框级）
    parser.add_argument("--soft-nms-sigma", type=float, default=0.5)
    # Light 参考参数：soft-nms-score-thr=0.16
    parser.add_argument("--soft-nms-score-thr", type=float, default=0.16)
    # 图像预处理（可选）
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-grid", type=int, default=8)
    parser.add_argument("--bilateral-d", type=int, default=5)
    parser.add_argument("--bilateral-sigma", type=float, default=35.0)
    parser.add_argument("--sharpen", type=float, default=0.2)
    parser.add_argument("--disable-opencv-preproc", action="store_true")
    # 轮廓后精修与形状门控
    # Light 参考参数：min-solidity=0.58, min-area-ratio=0.05, score-low-thr=0.56
    parser.add_argument("--disable-post-refine", action="store_true")
    parser.add_argument("--enable-post-refine", action="store_true")
    parser.add_argument("--min-solidity", type=float, default=0.58)
    parser.add_argument("--min-area-ratio", type=float, default=0.05)
    parser.add_argument("--score-low-thr", type=float, default=0.56)
    parser.add_argument("--post-refine-open-iter", type=int, default=1)
    # 调试定位
    parser.add_argument("--debug-trace", action="store_true")
    parser.add_argument("--debug-log-path", type=str, default="")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    # 默认开启后精修；仅在显式传 --disable-post-refine 时关闭。
    if args.enable_post_refine:
        args.disable_post_refine = False

    config_path = _resolve_path(args.config, "config")
    ckpt_path = _resolve_path(args.resume, "resume")
    save_dir = _resolve_path(args.save_dir, "save_dir", allow_missing=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = _build_rtdetr_model(config_path, ckpt_path, device)
    predictor = None
    if args.use_hq_sam:
        predictor = _build_sam(args.hq_sam_checkpoint, args.hq_sam_model_type, device)
    image_paths = []
    if args.im_file:
        image_paths.append(_resolve_path(args.im_file, "im_file"))
    if args.image_dir:
        image_paths.extend(_list_images(_resolve_path(args.image_dir, "image_dir")))
    if not image_paths:
        raise RuntimeError("im_file and image_dir are both empty")
    uniq = []
    seen = set()
    for p in image_paths:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(p.resolve())
    for i, p in enumerate(uniq, start=1):
        out_file = save_dir / f"{p.stem}_hqsam_contours.jpg"
        if args.skip_existing and out_file.exists():
            print(f"[{i}/{len(uniq)}] skip existing {out_file}")
            continue
        try:
            if args.debug_trace:
                _emit_trace(True, args.debug_log_path, f"main:image_start idx={i}/{len(uniq)} path={p}")
            out = run_single(
                model=model,
                predictor=predictor,
                image_path=p,
                device=device,
                score_thr=args.score_thr,
                save_dir=save_dir,
                trad_overlap_thr_low=args.trad_overlap_thr_low,
                trad_overlap_thr_high=args.trad_overlap_thr_high,
                trad_min_len_ratio_low=args.trad_min_len_ratio_low,
                trad_min_len_ratio_high=args.trad_min_len_ratio_high,
                edge_dist_thr=args.edge_dist_thr,
                edge_frac_thr=args.edge_frac_thr,
                overlap_iou_thr=args.overlap_iou_thr,
                soft_nms_sigma=args.soft_nms_sigma,
                soft_nms_score_thr=args.soft_nms_score_thr,
                clahe_clip=args.clahe_clip,
                clahe_grid=args.clahe_grid,
                bilateral_d=args.bilateral_d,
                bilateral_sigma=args.bilateral_sigma,
                sharpen=args.sharpen,
                disable_opencv_preproc=args.disable_opencv_preproc,
                disable_post_refine=args.disable_post_refine,
                min_solidity=args.min_solidity,
                min_area_ratio=args.min_area_ratio,
                score_low_thr=args.score_low_thr,
                post_refine_open_iter=args.post_refine_open_iter,
                debug_trace=args.debug_trace,
                debug_log_path=args.debug_log_path,
            )
            print(f"[{i}/{len(uniq)}] saved result to {out}")
            if args.debug_trace:
                _emit_trace(True, args.debug_log_path, f"main:image_done idx={i}/{len(uniq)} path={p}")
        except Exception as ex:
            im_fallback = Image.open(p).convert("RGB")
            im_fallback.save(out_file)
            print(f"[{i}/{len(uniq)}] fallback save {out_file} due to {ex}")
            if args.debug_trace:
                _emit_trace(True, args.debug_log_path, f"main:image_exception idx={i}/{len(uniq)} path={p} err={ex}")
                _emit_trace(True, args.debug_log_path, traceback.format_exc().rstrip())


if __name__ == "__main__":
    main()
