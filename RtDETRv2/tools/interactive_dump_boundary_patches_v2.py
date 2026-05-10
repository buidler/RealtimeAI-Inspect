import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.interactive_dump_boundary_patches import (
    _build_sam,
    _interactive_pick,
    _list_images,
    _rasterize_polygon_to_mask,
    _resolve_path,
)
from references.deploy.rtdetrv2_hqsam_infer_v2 import (
    _build_rtdetr_model,
    _opencv_preprocess,
    _prefer_outer_contours,
    _prepare_candidates_v3,
    _soft_nms_gaussian,
)


def _count_unique_fibers_from_csv(csv_path):
    if not csv_path.exists():
        return 0, 0
    fibers = set()
    rows = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            img_p = row.get("image_path", "")
            box_i = row.get("box_index", "")
            if img_p == "" or box_i == "":
                continue
            fibers.add((img_p, box_i))
    return len(fibers), rows


def interactive_dump_patches_v2(
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
    auto_select_all=False,
    trad_overlap_thr_low=0.28,
    trad_overlap_thr_high=0.45,
    trad_min_len_ratio_low=0.20,
    trad_min_len_ratio_high=0.35,
    edge_dist_thr=3.0,
    edge_frac_thr=0.85,
    overlap_iou_thr=0.35,
    soft_nms_sigma=0.5,
    soft_nms_score_thr=0.16,
    clahe_clip=2.0,
    clahe_grid=8,
    bilateral_d=5,
    bilateral_sigma=35.0,
    sharpen=0.2,
    disable_opencv_preproc=False,
    disable_post_refine=False,
    min_solidity=0.58,
    min_area_ratio=0.05,
    score_low_thr=0.56,
    post_refine_open_iter=1,
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

    # 从已有 csv 恢复“每张图已经选过的 box_idx”，重启时高亮显示。
    selected_by_image = {}
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f_csv_read:
                reader = csv.DictReader(f_csv_read)
                for row in reader:
                    img_k = str(Path(row.get("image_path", "")).resolve())
                    try:
                        box_i = int(row.get("box_index", -1))
                    except Exception:
                        box_i = -1
                    if box_i < 0:
                        continue
                    selected_by_image.setdefault(img_k, set()).add(box_i)
        except Exception:
            selected_by_image = {}
    hist_fibers, hist_rows = _count_unique_fibers_from_csv(csv_path)
    print(
        f"[resume] start_image={start_img_idx + 1}/{len(image_paths)}, "
        f"historical_fibers={hist_fibers}, historical_patches={hist_rows}"
    )

    def save_state(next_image_idx):
        if disable_auto_resume:
            return
        n = max(0, min(len(image_paths), int(next_image_idx)))
        with state_path.open("w", encoding="utf-8") as f:
            json.dump({"next_image_idx": n}, f, ensure_ascii=False)

    csv_mode = "a" if (start_img_idx > 0 and csv_path.exists()) else "w"
    with csv_path.open(csv_mode, newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if csv_mode == "w":
            writer.writerow(["patch_path", "image_path", "box_index", "contour_index", "point_index", "x", "y", "label"])

        stop_all = False
        for img_idx, img_path in enumerate(image_paths[start_img_idx:], start=start_img_idx):
            if stop_all:
                break

            im_pil = Image.open(img_path).convert("RGB")
            w, h = im_pil.size
            image_np_orig = np.array(im_pil)

            if disable_opencv_preproc:
                image_np_proc = image_np_orig
            else:
                image_np_proc = _opencv_preprocess(
                    image_np_orig,
                    clahe_clip=clahe_clip,
                    clahe_grid=clahe_grid,
                    bilateral_d=bilateral_d,
                    bilateral_sigma=bilateral_sigma,
                    sharpen=sharpen,
                )
            im_pil_proc = Image.fromarray(image_np_proc)

            orig_size = torch.tensor([w, h])[None].to(device)
            im_data = transforms(im_pil_proc)[None].to(device)
            with torch.no_grad():
                _, boxes, scores = model(im_data, orig_size)

            scr = scores[0].detach().cpu().numpy().astype(np.float32)
            box = boxes[0]
            box_np = box.detach().cpu().numpy().astype(np.float32)
            keep0 = scr > float(score_thr)
            box_np = box_np[keep0]
            scr = scr[keep0]
            if box_np.shape[0] == 0:
                print(f"[pick] {img_idx + 1}/{len(image_paths)}: {img_path}, no boxes after score_thr")
                continue

            keep_ids, keep_scores = _soft_nms_gaussian(
                box_np,
                scr,
                sigma=soft_nms_sigma,
                score_thr=soft_nms_score_thr,
            )
            if len(keep_ids) == 0:
                print(f"[pick] {img_idx + 1}/{len(image_paths)}: {img_path}, no boxes after soft-nms")
                continue

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
                trace_enabled=False,
                trace_log_path=None,
            )
            candidates = _prefer_outer_contours(candidates)
            if not candidates:
                print(f"[pick] {img_idx + 1}/{len(image_paths)}: {img_path}, no candidates")
                continue

            if auto_select_all:
                selected_indices = list(range(len(candidates)))
                stop_flag = False
            else:
                preselected = set()
                img_k = str(Path(img_path).resolve())
                selected_box_ids = selected_by_image.get(img_k, set())
                if selected_box_ids:
                    for ci, cand in enumerate(candidates):
                        if int(cand.get("box_idx", -1)) in selected_box_ids:
                            preselected.add(ci)
                total_selected_base = sum(len(v) for v in selected_by_image.values()) - len(selected_box_ids)
                selected_indices, stop_flag = _interactive_pick(
                    image_np_orig,
                    candidates,
                    "Fiber Picker V2",
                    view_max_w=view_max_width,
                    view_max_h=view_max_height,
                    progress_text=f"image {img_idx + 1}/{len(image_paths)}: {img_path.name}",
                    initial_selected=preselected,
                    total_selected_base=total_selected_base,
                )
            if stop_flag:
                stop_all = True
            if not selected_indices:
                print(f"[pick] {img_idx + 1}/{len(image_paths)}: {img_path}, selected=0")
                continue

            patches_this_image = 0
            for sel_i in selected_indices:
                c = candidates[sel_i]
                contour = c["contour"]
                trad_mask_full = _rasterize_polygon_to_mask(contour, im_pil.size)
                hq_mask_full = c["hq_mask"] if c["hq_mask"] is not None else np.zeros((h, w), dtype=np.uint8)

                num_samples = min(32, len(contour))
                idx = np.linspace(0, len(contour) - 1, num_samples).astype(int)
                samples = contour[idx]
                patch_size = 64
                half = patch_size // 2
                for k, (x, y) in enumerate(samples):
                    if max_patches_per_image > 0 and patches_this_image >= max_patches_per_image:
                        break
                    x_i, y_i = int(x), int(y)
                    x0 = max(0, x_i - half)
                    y0 = max(0, y_i - half)
                    x1 = min(w, x_i + half)
                    y1 = min(h, y_i + half)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    rgb = image_np_orig[y0:y1, x0:x1, :]
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

                    patch_name = f"img{img_idx:04d}_box{c['box_idx']:02d}_pt{k:03d}.npz"
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
                selected_by_image.setdefault(str(Path(img_path).resolve()), set()).add(int(c["box_idx"]))

            print(
                f"[dump] {img_idx + 1}/{len(image_paths)}: {img_path}, selected={len(selected_indices)}, patches={patches_this_image}"
            )
            if patches_this_image > 0:
                # 仅在“实际写入标注 patch”时推进断点，且落在当前图片上。
                save_state(img_idx)
                cur_fibers, cur_rows = _count_unique_fibers_from_csv(csv_path)
                print(
                    f"[stats] cumulative_fibers={cur_fibers}, cumulative_patches={cur_rows}"
                )

    cv2.destroyAllWindows()
    final_fibers, final_rows = _count_unique_fibers_from_csv(csv_path)
    print(f"[final] fibers={final_fibers}, patches={final_rows}")


def main():
    parser = argparse.ArgumentParser(description="基于 infer_v2 流程交互采样 boundary patches（支持断点续采）。")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thr", type=float, default=0.6)
    parser.add_argument("--use-hq-sam", action="store_true")
    parser.add_argument("--hq-sam-checkpoint", type=str, default=None)
    parser.add_argument("--hq-sam-model-type", type=str, default="vit_l")
    parser.add_argument("--max-patches-per-image", type=int, default=-1)
    parser.add_argument("--view-max-width", type=int, default=1600)
    parser.add_argument("--view-max-height", type=int, default=900)
    parser.add_argument("--state-path", type=str, default="")
    parser.add_argument("--disable-auto-resume", action="store_true")
    parser.add_argument("--auto-select-all", action="store_true")

    parser.add_argument("--trad-overlap-thr-low", type=float, default=0.28)
    parser.add_argument("--trad-overlap-thr-high", type=float, default=0.45)
    parser.add_argument("--trad-min-len-ratio-low", type=float, default=0.20)
    parser.add_argument("--trad-min-len-ratio-high", type=float, default=0.35)
    parser.add_argument("--edge-dist-thr", type=float, default=3.0)
    parser.add_argument("--edge-frac-thr", type=float, default=0.85)
    parser.add_argument("--overlap-iou-thr", type=float, default=0.35)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.5)
    parser.add_argument("--soft-nms-score-thr", type=float, default=0.16)
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-grid", type=int, default=8)
    parser.add_argument("--bilateral-d", type=int, default=5)
    parser.add_argument("--bilateral-sigma", type=float, default=35.0)
    parser.add_argument("--sharpen", type=float, default=0.2)
    parser.add_argument("--disable-opencv-preproc", action="store_true")
    parser.add_argument("--disable-post-refine", action="store_true")
    parser.add_argument("--min-solidity", type=float, default=0.58)
    parser.add_argument("--min-area-ratio", type=float, default=0.05)
    parser.add_argument("--score-low-thr", type=float, default=0.56)
    parser.add_argument("--post-refine-open-iter", type=int, default=1)
    args = parser.parse_args()

    interactive_dump_patches_v2(
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
        auto_select_all=args.auto_select_all,
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
    )


if __name__ == "__main__":
    main()
