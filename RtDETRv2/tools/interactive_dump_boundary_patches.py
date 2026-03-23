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

from references.deploy.rtdetrv2_hqsam_infer import (
    _build_rtdetr_model,
    _contours_from_box,
    _filter_long_thin_contours,
    _filter_traditional_contours_with_hqsam,
    _select_best_hqsam_contour,
)
from segment_anything_hq import SamPredictor, sam_model_registry


def _resolve_path(raw_path, field_name, allow_missing=False):
    p = Path(raw_path).expanduser()
    candidates = [p]
    if not p.is_absolute():
        candidates.append(PROJECT_ROOT / p)
        candidates.append(Path.cwd() / p)
    seen = set()
    ordered = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
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
    files = []
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def _build_sam(hq_sam_checkpoint, hq_sam_model_type, device):
    if hq_sam_checkpoint is None:
        return None
    hq_sam_checkpoint = _resolve_path(hq_sam_checkpoint, "hq_sam_checkpoint")
    sam = sam_model_registry[hq_sam_model_type](checkpoint=hq_sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def _rasterize_polygon_to_mask(pts, size):
    w, h = size
    mask_im = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_im)
    draw.polygon([(float(p[0]), float(p[1])) for p in pts], outline=1, fill=1)
    return np.array(mask_im, dtype=np.uint8)


def _prepare_candidates(im_pil, image_np, boxes, predictor, use_hq_sam):
    candidates = []
    if use_hq_sam and predictor is not None:
        predictor.set_image(image_np)
    for j, b in enumerate(boxes):
        b_np = b.detach().cpu().numpy()
        hq_mask = None
        if use_hq_sam and predictor is not None:
            masks, _, _ = predictor.predict(box=b_np, multimask_output=True)
            _, hq_mask = _select_best_hqsam_contour(masks, b_np, conservative_boundary=False)
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
        if not trad_contours:
            continue
        contour = np.asarray(trad_contours[0], dtype=np.float32)
        if contour.shape[0] < 3:
            continue
        x0, y0 = contour.min(axis=0)
        x1, y1 = contour.max(axis=0)
        candidates.append(
            {
                "box_idx": j,
                "contour": contour,
                "bbox": (int(x0), int(y0), int(x1), int(y1)),
                "hq_mask": hq_mask.astype(np.uint8) if hq_mask is not None else None,
            }
        )
    return candidates


def _interactive_pick(image_np, candidates, window_name, view_max_w=1600, view_max_h=900):
    selected = set()
    h, w = image_np.shape[:2]
    view_w = max(1, int(view_max_w))
    view_h = max(1, int(view_max_h))
    pan_x_max = max(0, w - view_w)
    pan_y_max = max(0, h - view_h)
    ui = {"selected": selected, "dragging": False, "last_x": 0, "last_y": 0, "pan_x": 0, "pan_y": 0}

    def clamp_pan():
        ui["pan_x"] = max(0, min(pan_x_max, int(ui["pan_x"])))
        ui["pan_y"] = max(0, min(pan_y_max, int(ui["pan_y"])))

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            param["dragging"] = True
            param["last_x"] = x
            param["last_y"] = y
            return
        if event == cv2.EVENT_RBUTTONUP:
            param["dragging"] = False
            return
        if event == cv2.EVENT_MOUSEMOVE and param["dragging"]:
            dx = x - param["last_x"]
            dy = y - param["last_y"]
            param["pan_x"] -= dx
            param["pan_y"] -= dy
            clamp_pan()
            param["last_x"] = x
            param["last_y"] = y
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        img_x = int(x + param["pan_x"])
        img_y = int(y + param["pan_y"])
        if img_x < 0 or img_x >= w or img_y < 0 or img_y >= h:
            return
        best_i = None
        best_area = None
        for i, c in enumerate(candidates):
            x0, y0, x1, y1 = c["bbox"]
            if x0 <= img_x <= x1 and y0 <= img_y <= y1:
                area = max(1, (x1 - x0) * (y1 - y0))
                if best_area is None or area < best_area:
                    best_area = area
                    best_i = i
        if best_i is None:
            return
        if best_i in param["selected"]:
            param["selected"].remove(best_i)
        else:
            param["selected"].add(best_i)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, view_w, view_h)
    cv2.setMouseCallback(window_name, on_mouse, ui)

    while True:
        clamp_pan()
        x0v = ui["pan_x"]
        y0v = ui["pan_y"]
        x1v = min(w, x0v + view_w)
        y1v = min(h, y0v + view_h)
        crop = image_np[y0v:y1v, x0v:x1v]
        canvas = np.zeros((view_h, view_w, 3), dtype=image_np.dtype)
        canvas[: crop.shape[0], : crop.shape[1]] = crop
        shift = np.array([[[x0v, y0v]]], dtype=np.int32)
        for i, c in enumerate(candidates):
            contour_i = c["contour"].astype(np.int32).reshape(-1, 1, 2) - shift
            color = (0, 255, 0) if i in selected else (255, 180, 80)
            cv2.polylines(canvas, [contour_i], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            x0, y0, x1, y1 = c["bbox"]
            dx0, dy0 = x0 - x0v, y0 - y0v
            dx1, dy1 = x1 - x0v, y1 - y0v
            cv2.rectangle(canvas, (dx0, dy0), (dx1, dy1), color, 1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, f"{i}", (dx0, max(12, dy0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        info = (
            f"selected={len(selected)}/{len(candidates)}  "
            f"left=toggle  right-drag=pan  arrows/WASD=pan  [a]=all [c]=clear [n]=next [q]=quit"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (20, 20, 20), -1)
        cv2.putText(canvas, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)

        cv2.imshow(window_name, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        key = cv2.waitKeyEx(30)
        if key == ord("a"):
            selected.clear()
            selected.update(range(len(candidates)))
        elif key == ord("c"):
            selected.clear()
        elif key in (2424832, ord("a"), ord("A")):
            ui["pan_x"] -= 120
        elif key in (2555904, ord("d"), ord("D")):
            ui["pan_x"] += 120
        elif key in (2490368, ord("w"), ord("W")):
            ui["pan_y"] -= 120
        elif key in (2621440, ord("s"), ord("S")):
            ui["pan_y"] += 120
        elif key == ord("n"):
            return sorted(selected), False
        elif key in (ord("q"), 27):
            return sorted(selected), True


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
                labels, boxes, scores = model(im_data, orig_size)
            scr = scores[0]
            box = boxes[0]
            keep = scr > score_thr
            box = box[keep]
            candidates = _prepare_candidates(im_pil, image_np, box, predictor, use_hq_sam)
            if not candidates:
                print(f"[pick] {img_idx + 1}/{len(image_paths)}: {img_path}, no candidates")
                continue

            selected_indices, stop_flag = _interactive_pick(
                image_np,
                candidates,
                f"Fiber Picker {img_idx + 1}/{len(image_paths)}",
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
    parser = argparse.ArgumentParser(
        description="先交互选择纤维，再仅对选中纤维采样 boundary patches。"
    )
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
    )


if __name__ == "__main__":
    main()
