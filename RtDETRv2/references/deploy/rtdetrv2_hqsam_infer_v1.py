import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

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


def _resolve_path(raw_path, field_name, allow_missing=False):
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


def _prepare_candidates_v2(
    im_pil,
    image_np,
    boxes,
    scores,
    predictor,
    use_hq_sam,
    trad_overlap_thr_low=0.28,
    trad_overlap_thr_high=0.42,
    trad_min_len_ratio_low=0.18,
    trad_min_len_ratio_high=0.35,
    edge_dist_thr=3.0,
    edge_frac_thr=0.85,
    overlap_iou_thr=0.35,
):
    candidates = []
    if use_hq_sam and predictor is not None:
        predictor.set_image(image_np)
    boxes_np_all = [b.detach().cpu().numpy() for b in boxes]
    for j, b in enumerate(boxes):
        b_np = boxes_np_all[j]
        s_j = float(scores[j].item()) if j < len(scores) else 0.0
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

        trad_contours = _contours_from_box(im_pil, box_for_trad, mask_hint=hq_mask)
        trad_contours = _filter_traditional_contours_with_hqsam(trad_contours, hq_mask, im_pil.size)
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
            hq_edge = _is_edge_hugging(hq_contour, b_np, thr=edge_dist_thr, frac=0.95)
            overlap = _polygon_overlap_with_mask(trad_contour, hq_mask, im_pil.size)
            trad_len = _polyline_length(trad_contour)
            hq_len = _polyline_length(hq_contour)
            len_ratio = trad_len / max(1e-6, hq_len)
            trad_high = (not trad_edge) and overlap >= trad_overlap_thr_high and len_ratio >= trad_min_len_ratio_high
            trad_mid = (not trad_edge or hq_edge) and overlap >= trad_overlap_thr_low and len_ratio >= trad_min_len_ratio_low
            risky_overlap = neigh_iou >= overlap_iou_thr
            low_score = s_j < 0.72
            if trad_high and not (risky_overlap and low_score):
                final_contour = trad_contour
            elif trad_mid and not (risky_overlap or low_score):
                final_contour = trad_contour
            else:
                final_contour = hq_contour
        elif trad_contour is not None:
            trad_edge = _is_edge_hugging(trad_contour, b_np, thr=edge_dist_thr, frac=edge_frac_thr)
            if not trad_edge and neigh_iou < overlap_iou_thr and s_j >= 0.72:
                final_contour = trad_contour
        elif hq_contour is not None:
            final_contour = hq_contour

        if final_contour is None:
            continue
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
    return candidates


def _build_rtdetr_model(config_path, ckpt_path, device):
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
    drawer = ImageDraw.Draw(im_pil)
    for b in boxes_np:
        drawer.rectangle(list(b), outline="red")
    for c in candidates:
        pts = c["contour"]
        if pts is None or len(pts) < 3:
            continue
        pts_seq = [(float(p[0]), float(p[1])) for p in pts]
        drawer.line(pts_seq + [pts_seq[0]], fill="yellow", width=2)
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
):
    im_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(im_pil)
    w, h = im_pil.size
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    im_data = transforms(im_pil)[None].to(device)
    orig_size = torch.tensor([w, h])[None].to(device)
    with torch.no_grad():
        _, boxes, scores = model(im_data, orig_size)
    scr = scores[0]
    box = boxes[0]
    keep = scr > score_thr
    box = box[keep]
    scr = scr[keep]
    candidates = _prepare_candidates_v2(
        im_pil,
        image_np,
        box,
        scr,
        predictor,
        predictor is not None,
        trad_overlap_thr_low=trad_overlap_thr_low,
        trad_overlap_thr_high=trad_overlap_thr_high,
        trad_min_len_ratio_low=trad_min_len_ratio_low,
        trad_min_len_ratio_high=trad_min_len_ratio_high,
        edge_dist_thr=edge_dist_thr,
        edge_frac_thr=edge_frac_thr,
        overlap_iou_thr=overlap_iou_thr,
    )
    boxes_np = box.detach().cpu().numpy()
    return _draw_and_save(im_pil, boxes_np, candidates, save_dir, Path(image_path).stem)


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--trad-overlap-thr-low", type=float, default=0.28)
    parser.add_argument("--trad-overlap-thr-high", type=float, default=0.42)
    parser.add_argument("--trad-min-len-ratio-low", type=float, default=0.18)
    parser.add_argument("--trad-min-len-ratio-high", type=float, default=0.35)
    parser.add_argument("--edge-dist-thr", type=float, default=3.0)
    parser.add_argument("--edge-frac-thr", type=float, default=0.85)
    parser.add_argument("--overlap-iou-thr", type=float, default=0.35)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

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
            )
            print(f"[{i}/{len(uniq)}] saved result to {out}")
        except Exception as ex:
            im_fallback = Image.open(p).convert("RGB")
            im_fallback.save(out_file)
            print(f"[{i}/{len(uniq)}] fallback save {out_file} due to {ex}")


if __name__ == "__main__":
    main()
