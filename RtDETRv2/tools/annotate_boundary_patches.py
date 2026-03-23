import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"csv not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []
    return headers, rows


def write_rows(csv_path: Path, headers, rows):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def to_int_label(v):
    try:
        return int(v)
    except Exception:
        return -1


def resolve_path(raw_path: str, csv_path: Path, workspace_root: Path):
    p = Path(raw_path)
    candidates = [p, csv_path.parent / p, workspace_root / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[1].resolve()


def load_patch_rgb(npz_path: Path):
    if not npz_path.exists():
        return np.zeros((256, 256, 3), dtype=np.uint8)
    data = np.load(npz_path)["patch"].astype(np.float32)
    if data.ndim != 3 or data.shape[0] < 3:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    rgb = data[:3].transpose(1, 2, 0)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    return rgb


def fit_height(image: np.ndarray, target_h: int):
    h, w = image.shape[:2]
    if h == target_h:
        return image
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)


def fit_to_box(image: np.ndarray, max_w: int, max_h: int):
    h, w = image.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 1.0:
        return image, 1.0
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def crop_local_view(image_bgr: np.ndarray, x: int, y: int, local_window_size: int, local_zoom_size: int):
    h, w = image_bgr.shape[:2]
    half = max(1, local_window_size // 2)
    x0 = max(0, x - half)
    y0 = max(0, y - half)
    x1 = min(w, x + half)
    y1 = min(h, y + half)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((local_window_size, local_window_size, 3), dtype=np.uint8)
    if crop.shape[0] != local_window_size or crop.shape[1] != local_window_size:
        canvas = np.zeros((local_window_size, local_window_size, 3), dtype=np.uint8)
        ch, cw = crop.shape[:2]
        sy = max(0, (local_window_size - ch) // 2)
        sx = max(0, (local_window_size - cw) // 2)
        canvas[sy:sy + ch, sx:sx + cw] = crop
        crop = canvas
    return cv2.resize(crop, (local_zoom_size, local_zoom_size), interpolation=cv2.INTER_NEAREST)


def draw_zoom_buttons(canvas: np.ndarray):
    h, w = canvas.shape[:2]
    y1 = 10
    y2 = min(50, h - 1)
    bw = 80
    gap = 10
    x_start = max(10, w - (bw * 4 + gap * 3 + 20))
    labels = [("L-", "left_minus"), ("L+", "left_plus"), ("R-", "right_minus"), ("R+", "right_plus")]
    buttons = {}
    for i, (text, key) in enumerate(labels):
        x1 = x_start + i * (bw + gap)
        x2 = x1 + bw
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), 2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, text, (x1 + 22, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        buttons[key] = (x1, y1, x2, y2)
    return buttons


def build_panel(
    row,
    row_index,
    total_rows,
    csv_path: Path,
    image_rgb: np.ndarray,
    patch_rgb: np.ndarray,
    max_image_h: int,
    patch_preview_size: int,
    view_max_w: int,
    view_max_h: int,
    left_view: str,
    right_view: str,
    local_window_size: int,
    local_zoom_size: int,
    left_less_zoom_times: float,
    left_zoom_mult: float,
    right_zoom_mult: float,
):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

    x = int(float(row["x"]))
    y = int(float(row["y"]))
    cv2.circle(image_bgr, (x, y), 22, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    cv2.circle(image_bgr, (x, y), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    right_window_size = max(1, int(local_window_size / max(1e-6, right_zoom_mult)))
    local_bgr = crop_local_view(image_bgr, x, y, right_window_size, local_zoom_size)
    left_local_window_size = max(1, int(local_window_size * max(1.0, left_less_zoom_times) / max(1e-6, left_zoom_mult)))
    left_local_bgr = crop_local_view(image_bgr, x, y, left_local_window_size, local_zoom_size)

    if left_view == "local":
        image_bgr = left_local_bgr
    elif image_bgr.shape[0] > max_image_h:
        image_bgr = fit_height(image_bgr, max_image_h)

    if right_view == "local":
        right_bgr = local_bgr
    else:
        right_bgr = cv2.resize(patch_bgr, (patch_preview_size, patch_preview_size), interpolation=cv2.INTER_AREA)
    right_bgr = fit_height(right_bgr, image_bgr.shape[0])

    canvas = np.hstack([image_bgr, right_bgr])
    bar_h = 130
    bar = np.full((bar_h, canvas.shape[1], 3), 20, dtype=np.uint8)
    canvas = np.vstack([bar, canvas])

    info_1 = f"{csv_path.name}  row={row_index + 1}/{total_rows}  label={row.get('label', '-1')}"
    info_2 = f"img={Path(row['image_path']).name}  box={row['box_index']}  pt={row['point_index']}  xy=({row['x']}, {row['y']})"
    info_3 = "keys: [1]=good  [0]=bad  [u]=-1  [s]=skip  [b]=back  [q]=save&quit"
    info_4 = f"left_zoom={left_zoom_mult:.3f}x  right_zoom={right_zoom_mult:.3f}x"

    cv2.putText(canvas, info_1, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, info_2, (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 210, 210), 2, cv2.LINE_AA)
    cv2.putText(canvas, info_3, (16, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (180, 220, 255), 2, cv2.LINE_AA)
    info_4_x = max(700, canvas.shape[1] - 880)
    cv2.putText(canvas, info_4, (info_4_x, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (180, 255, 200), 2, cv2.LINE_AA)
    buttons = draw_zoom_buttons(canvas)

    canvas, scale = fit_to_box(canvas, view_max_w, view_max_h)
    if scale != 1.0:
        scaled = {}
        for k, (x1, y1, x2, y2) in buttons.items():
            scaled[k] = (
                int(x1 * scale),
                int(y1 * scale),
                int(x2 * scale),
                int(y2 * scale),
            )
        buttons = scaled
    return canvas, buttons


def summarize_labels(rows):
    n_pos = 0
    n_neg = 0
    n_unlabeled = 0
    for r in rows:
        l = to_int_label(r.get("label", -1))
        if l == 1:
            n_pos += 1
        elif l == 0:
            n_neg += 1
        else:
            n_unlabeled += 1

    ratio_text = "N/A"
    if n_neg > 0:
        ratio_text = f"{n_pos / n_neg:.3f}"
    elif n_pos > 0:
        ratio_text = "inf"

    if n_pos == 0 and n_neg == 0:
        advice = "还没有 0/1 标注；下次先按 1:1 目标各标一批。"
    elif n_pos == 0:
        advice = "当前只有 0 类；下次优先补 1（真实外边界）。"
    elif n_neg == 0:
        advice = "当前只有 1 类；下次优先补 0（刀痕/伪轮廓）。"
    else:
        ratio = n_pos / n_neg
        if ratio > 2.0:
            advice = "1 类偏多；下次优先补 0，重点找刀痕与内部纹理。"
        elif ratio < 0.5:
            advice = "0 类偏多；下次优先补 1，重点找稳定外轮廓。"
        else:
            advice = "比例基本均衡；下次保持当前标准继续标注。"

    return n_pos, n_neg, n_unlabeled, ratio_text, advice


def main():
    parser = argparse.ArgumentParser(description="边界patch交互标注器")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--workspace-root", type=str, default=".")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--autosave-every", type=int, default=50)
    parser.add_argument("--all", action="store_true", help="标注所有行；默认只看label=-1")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--window-name", type=str, default="Boundary Patch Annotator")
    parser.add_argument("--max-image-height", type=int, default=700)
    parser.add_argument("--patch-preview-size", type=int, default=320)
    parser.add_argument("--view-max-width", type=int, default=1600)
    parser.add_argument("--view-max-height", type=int, default=900)
    parser.add_argument("--left-view", type=str, choices=["full", "local"], default="local")
    parser.add_argument("--right-view", type=str, choices=["patch", "local"], default="local")
    parser.add_argument("--local-window-size", type=int, default=220)
    parser.add_argument("--local-zoom-size", type=int, default=1000)
    parser.add_argument("--left-less-zoom-times", type=float, default=5.0)
    parser.add_argument("--state-path", type=str, default="")
    parser.add_argument("--disable-auto-resume", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    headers, rows = read_rows(csv_path)
    state_path = Path(args.state_path).resolve() if args.state_path else (csv_path.parent / "annotate_state.json")
    resume_row = None
    if not args.disable_auto_resume and state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("next_row"), int):
                resume_row = data["next_row"]
        except Exception:
            resume_row = None

    def save_state(next_row: int):
        if args.disable_auto_resume:
            return
        row = max(0, min(len(rows), int(next_row)))
        with state_path.open("w", encoding="utf-8") as f:
            json.dump({"next_row": row}, f, ensure_ascii=False)

    required = {"patch_path", "image_path", "x", "y", "label", "box_index", "point_index"}
    missing = required.difference(set(headers))
    if missing:
        raise RuntimeError(f"csv缺少字段: {sorted(missing)}")

    if not args.no_backup:
        bak_path = csv_path.with_name(csv_path.name + ".bak")
        if not bak_path.exists():
            shutil.copy2(csv_path, bak_path)

    only_unlabeled = not args.all
    targets = []
    for i, row in enumerate(rows):
        if only_unlabeled:
            if to_int_label(row.get("label", -1)) == -1:
                targets.append(i)
        else:
            targets.append(i)
    if not targets:
        print("没有可标注的行。")
        save_state(len(rows))
        n_pos, n_neg, n_unlabeled, ratio_text, advice = summarize_labels(rows)
        print(f"saved: {csv_path}")
        print(f"label_counts: {{1: {n_pos}, 0: {n_neg}, -1: {n_unlabeled}}}")
        print(f"ratio_1_to_0: {ratio_text}")
        print(f"next_focus: {advice}")
        return

    pos = 0
    effective_start_row = args.start_row if args.start_row > 0 else (resume_row if resume_row is not None else 0)
    if effective_start_row > 0:
        found = None
        for j, idx in enumerate(targets):
            if idx >= effective_start_row:
                found = j
                break
        if found is not None:
            pos = found
    if pos < len(targets):
        save_state(targets[pos])

    zoom_step = 2.5
    ui_state = {
        "buttons": {},
        "pending_action": None,
        "left_zoom_mult": 1.0,
        "right_zoom_mult": 1.0,
    }

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for action, (x1, y1, x2, y2) in param["buttons"].items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                param["pending_action"] = action
                return

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(args.window_name, on_mouse, ui_state)

    changed = 0
    while 0 <= pos < len(targets):
        row_idx = targets[pos]
        row = rows[row_idx]

        img_path = resolve_path(row["image_path"], csv_path, workspace_root)
        patch_path = resolve_path(row["patch_path"], csv_path, workspace_root)

        if img_path.exists():
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                image_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)

        patch_rgb = load_patch_rgb(patch_path)
        while True:
            panel, buttons = build_panel(
                row,
                row_idx,
                len(rows),
                csv_path,
                image_rgb,
                patch_rgb,
                max_image_h=args.max_image_height,
                patch_preview_size=args.patch_preview_size,
                view_max_w=args.view_max_width,
                view_max_h=args.view_max_height,
                left_view=args.left_view,
                right_view=args.right_view,
                local_window_size=args.local_window_size,
                local_zoom_size=args.local_zoom_size,
                left_less_zoom_times=args.left_less_zoom_times,
                left_zoom_mult=ui_state["left_zoom_mult"],
                right_zoom_mult=ui_state["right_zoom_mult"],
            )
            ui_state["buttons"] = buttons
            cv2.imshow(args.window_name, panel)
            key = cv2.waitKeyEx(30)

            action = ui_state["pending_action"]
            ui_state["pending_action"] = None
            if action == "left_plus":
                ui_state["left_zoom_mult"] *= zoom_step
                continue
            if action == "left_minus":
                ui_state["left_zoom_mult"] = max(1.0 / (zoom_step ** 6), ui_state["left_zoom_mult"] / zoom_step)
                continue
            if action == "right_plus":
                ui_state["right_zoom_mult"] *= zoom_step
                continue
            if action == "right_minus":
                ui_state["right_zoom_mult"] = max(1.0 / (zoom_step ** 6), ui_state["right_zoom_mult"] / zoom_step)
                continue

            if key in (ord("1"), ord("0"), ord("u")):
                new_label = 1 if key == ord("1") else 0 if key == ord("0") else -1
                if to_int_label(row.get("label", -1)) != new_label:
                    rows[row_idx]["label"] = str(new_label)
                    changed += 1
                pos += 1
                save_state(targets[pos] if pos < len(targets) else len(rows))
                if changed >= max(1, args.autosave_every):
                    write_rows(csv_path, headers, rows)
                    changed = 0
                break
            if key == ord("s"):
                pos += 1
                save_state(targets[pos] if pos < len(targets) else len(rows))
                break
            if key == ord("b"):
                pos = max(0, pos - 1)
                save_state(targets[pos] if pos < len(targets) else len(rows))
                break
            if key in (ord("q"), 27):
                save_state(row_idx)
                pos = len(targets)
                break

    if changed > 0:
        write_rows(csv_path, headers, rows)
    cv2.destroyAllWindows()

    n_pos, n_neg, n_unlabeled, ratio_text, advice = summarize_labels(rows)
    print(f"saved: {csv_path}")
    print(f"label_counts: {{1: {n_pos}, 0: {n_neg}, -1: {n_unlabeled}}}")
    print(f"ratio_1_to_0: {ratio_text}")
    print(f"next_focus: {advice}")


if __name__ == "__main__":
    main()
