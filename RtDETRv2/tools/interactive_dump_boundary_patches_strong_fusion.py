import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interactive_dump_boundary_patches import (
    _build_sam,
    _interactive_pick,
    _list_images,
    _rasterize_polygon_to_mask,
    _resolve_path,
)
from references.deploy.rtdetrv2_hqsam_infer import (
    _build_rtdetr_model,
    _contours_from_box,
    _filter_long_thin_contours,
    _filter_traditional_contours_with_hqsam,
    _select_best_hqsam_contour,
)


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
    return (dist_edge < thr).mean() > frac


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


def _prepare_candidates_strong(
    im_pil,
    image_np,
    boxes,
    predictor,
    use_hq_sam,
    trad_overlap_thr=0.3,
    trad_min_len_ratio=0.2,
    edge_dist_thr=3.0,
    edge_frac_thr=0.85,
):
    candidates = []
    if use_hq_sam and predictor is not None:
        predictor.set_image(image_np)
    for j, b in enumerate(boxes):
        b_np = b.detach().cpu().numpy()
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

        final_contour = None
        if trad_contour is not None and hq_contour is not None:
            trad_edge = _is_edge_hugging(trad_contour, b_np, thr=edge_dist_thr, frac=edge_frac_thr)
            hq_edge = _is_edge_hugging(hq_contour, b_np, thr=edge_dist_thr, frac=0.95)
            overlap = _polygon_overlap_with_mask(trad_contour, hq_mask, im_pil.size)
            trad_len = _polyline_length(trad_contour)
            hq_len = _polyline_length(hq_contour)
            len_ratio = trad_len / max(1e-6, hq_len)
            trad_good = (not trad_edge or hq_edge) and overlap >= trad_overlap_thr and len_ratio >= trad_min_len_ratio
            final_contour = trad_contour if trad_good else hq_contour
        elif trad_contour is not None:
            trad_edge = _is_edge_hugging(trad_contour, b_np, thr=edge_dist_thr, frac=edge_frac_thr)
            if not trad_edge:
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


def interactive_dump_patches(
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
    view_max_width=1600,
    view_max_height=900,
    state_path=None,
    disable_auto_resume=False,
    trad_overlap_thr=0.3,
    trad_min_len_ratio=0.2,
    edge_dist_thr=3.0,
    edge_frac_thr=0.85,
):
    config_path = _resolve_path(config_path, "config")
    ckpt_path = _resolve_path(ckpt_path, "resume")
    image_dir = _resolve_path(image_dir, "image_dir")
    save_dir = _resolve_path(save_dir, "save_dir", allow_missing=True)
    device = torch.device(device)
    model = _build_rtdetr_model(config_path, ckpt_path, device)
    model.eval()
    predictor = _build_sam(hq_sam_checkpoint, hq_sam_model_type, device) if use_hq_sam else None

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    image_paths = _list_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"no images found under: {image_dir}")

    save_dir = Path(save_dir).resolve()
    patch_dir = save_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "patches.csv"
    state_path = Path(state_path).resolve() if state_path else (save_dir / "interactive_dump_state.json")
    start_img_idx = 0
    if not disable_auto_resume and state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                state_data = json.load(f)
            if isinstance(state_data, dict) and isinstance(state_data.get("next_image_idx"), int):
                start_img_idx = max(0, min(len(image_paths), int(state_data["next_image_idx"])))
        except Exception:
            start_img_idx = 0

    def save_state(next_image_idx):
        if disable_auto_resume:
            return
        n = max(0, min(len(image_paths), int(next_image_idx)))
        with state_path.open("w", encoding="utf-8") as f:
            json.dump({"next_image_idx": n}, f, ensure_ascii=False)

    csv_mode = "a" if (start_img_idx > 0 and csv_path.exists()) else "w"
    with csv_path.open(csv_mode, newline="") as f_csv:
        writer = csv.writer(f_csv)
        if csv_mode == "w":
            writer.writerow(["patch_path", "image_path", "box_index", "contour_index", "point_index", "x", "y", "label"])

        stop_all = False
        for img_idx, img_path in enumerate(image_paths[start_img_idx:], start=start_img_idx):
            if stop_all:
                break
            im_pil = Image.open(img_path).convert("RGB")
            w, h = im_pil.size
            image_np = np.array(im_pil)
            orig_size = torch.tensor([w, h])[None].to(device)
            im_data = transforms(im_pil)[None].to(device)
            with torch.no_grad():
                _, boxes, scores = model(im_data, orig_size)
            scr = scores[0]
            box = boxes[0]
            keep = scr > score_thr
            box = box[keep]
            candidates = _prepare_candidates_strong(
                im_pil,
                image_np,
                box,
                predictor,
                use_hq_sam,
                trad_overlap_thr=trad_overlap_thr,
                trad_min_len_ratio=trad_min_len_ratio,
                edge_dist_thr=edge_dist_thr,
                edge_frac_thr=edge_frac_thr,
            )
            if not candidates:
                print(f"[pick] {img_idx + 1}/{len(image_paths)}: {img_path}, no candidates")
                save_state(img_idx + 1)
                continue

            selected_indices, stop_flag = _interactive_pick(
                image_np,
                candidates,
                f"Fiber Picker StrongFusion {img_idx + 1}/{len(image_paths)}",
                view_max_w=view_max_width,
                view_max_h=view_max_height,
            )
            if stop_flag:
                stop_all = True
            if not selected_indices:
                print(f"[pick] {img_idx + 1}/{len(image_paths)}: {img_path}, selected=0")
                save_state(img_idx + 1)
                continue

            patches_this_image = 0
            for sel_i in selected_indices:
                c = candidates[sel_i]
                contour = c["contour"]
                trad_mask_full = _rasterize_polygon_to_mask(contour, im_pil.size)
                if c["hq_mask"] is not None:
                    hq_mask_full = c["hq_mask"]
                else:
                    hq_mask_full = np.zeros((h, w), dtype=np.uint8)

                num_samples = min(32, len(contour))
                idx = np.linspace(0, len(contour) - 1, num_samples).astype(int)
                samples = contour[idx]
                patch_size = 64
                half = patch_size // 2
                for k, (x, y) in enumerate(samples):
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
                    patch_name = f"img{img_idx:04d}_box{c['box_idx']:02d}_sel{sel_i:02d}_pt{k:03d}.npz"
                    patch_path = patch_dir / patch_name
                    np.savez_compressed(patch_path, patch=patch)
                    writer.writerow(
                        [
                            str(patch_path.relative_to(save_dir)),
                            str(img_path),
                            c["box_idx"],
                            0,
                            k,
                            float(x),
                            float(y),
                            -1,
                        ]
                    )
                    patches_this_image += 1
            print(
                f"[dump] {img_idx + 1}/{len(image_paths)}: {img_path}, selected={len(selected_indices)}, patches={patches_this_image}"
            )
            save_state(img_idx + 1)

    if not disable_auto_resume and start_img_idx >= len(image_paths):
        save_state(len(image_paths))

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="强判别融合：HQ-SAM兜底，传统细节增强。")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thr", type=float, default=0.6)
    parser.add_argument("--use-hq-sam", action="store_true")
    parser.add_argument("--hq-sam-checkpoint", type=str, default=None)
    parser.add_argument("--hq-sam-model-type", type=str, default="vit_l")
    parser.add_argument("--max-patches-per-image", type=int, default=64)
    parser.add_argument("--view-max-width", type=int, default=1600)
    parser.add_argument("--view-max-height", type=int, default=900)
    parser.add_argument("--state-path", type=str, default="")
    parser.add_argument("--disable-auto-resume", action="store_true")
    parser.add_argument("--trad-overlap-thr", type=float, default=0.3)
    parser.add_argument("--trad-min-len-ratio", type=float, default=0.2)
    parser.add_argument("--edge-dist-thr", type=float, default=3.0)
    parser.add_argument("--edge-frac-thr", type=float, default=0.85)
    args = parser.parse_args()

    interactive_dump_patches(
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
        view_max_width=args.view_max_width,
        view_max_height=args.view_max_height,
        state_path=args.state_path,
        disable_auto_resume=args.disable_auto_resume,
        trad_overlap_thr=args.trad_overlap_thr,
        trad_min_len_ratio=args.trad_min_len_ratio,
        edge_dist_thr=args.edge_dist_thr,
        edge_frac_thr=args.edge_frac_thr,
    )


if __name__ == "__main__":
    main()
