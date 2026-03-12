import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_rows(csv_path: Path):
    # 读取整份 CSV 到内存，便于交互式修改后统一回写
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []
    return headers, rows


def write_rows(csv_path: Path, headers, rows):
    # 将更新后的标注结果覆盖写回原 CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def to_int_label(v):
    # 容错解析标签，非法值统一按 -1 处理
    try:
        return int(v)
    except Exception:
        return -1


def resolve_path(raw_path: str, csv_path: Path, workspace_root: Path):
    # 兼容绝对路径、相对 CSV 路径、相对工作区路径三种写法
    p = Path(raw_path)
    candidates = [p, csv_path.parent / p, workspace_root / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[1].resolve()


def load_patch_rgb(npz_path: Path):
    # 从 npz 提取前三个通道作为 RGB patch 预览
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
    # 按高度等比缩放，便于原图与 patch 拼接展示
    h, w = image.shape[:2]
    if h == target_h:
        return image
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)


def build_panel(row, row_index, total_rows, csv_path: Path, image_rgb: np.ndarray, patch_rgb: np.ndarray):
    # 构建可视化面板：左侧原图点位，右侧 patch，顶部状态栏
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

    x = int(float(row["x"]))
    y = int(float(row["y"]))
    cv2.circle(image_bgr, (x, y), 22, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    cv2.circle(image_bgr, (x, y), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    max_h = 900
    if image_bgr.shape[0] > max_h:
        image_bgr = fit_height(image_bgr, max_h)
    patch_bgr = cv2.resize(patch_bgr, (384, 384), interpolation=cv2.INTER_AREA)
    patch_bgr = fit_height(patch_bgr, image_bgr.shape[0])

    canvas = np.hstack([image_bgr, patch_bgr])
    h = canvas.shape[0]
    bar_h = 130
    bar = np.full((bar_h, canvas.shape[1], 3), 20, dtype=np.uint8)
    canvas = np.vstack([bar, canvas])

    info_1 = f"{csv_path.name}  row={row_index + 1}/{total_rows}  label={row.get('label', '-1')}"
    info_2 = f"img={Path(row['image_path']).name}  box={row['box_index']}  pt={row['point_index']}  xy=({row['x']}, {row['y']})"
    info_3 = "keys: [1]=good  [0]=bad  [u]=-1  [s]=skip  [b]=back  [q]=save&quit"

    cv2.putText(canvas, info_1, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, info_2, (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (210, 210, 210), 2, cv2.LINE_AA)
    cv2.putText(canvas, info_3, (16, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (180, 220, 255), 2, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="边界patch交互标注器")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--workspace-root", type=str, default=".")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--autosave-every", type=int, default=50)
    parser.add_argument("--all", action="store_true", help="标注所有行；默认只看label=-1")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--window-name", type=str, default="Boundary Patch Annotator")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    headers, rows = read_rows(csv_path)
    required = {"patch_path", "image_path", "x", "y", "label", "box_index", "point_index"}
    missing = required.difference(set(headers))
    if missing:
        raise RuntimeError(f"csv缺少字段: {sorted(missing)}")

    if not args.no_backup:
        # 首次运行自动备份，避免误操作导致原标注丢失
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
        return

    pos = 0
    if args.start_row > 0:
        found = None
        for j, idx in enumerate(targets):
            if idx >= args.start_row:
                found = j
                break
        if found is not None:
            pos = found

    changed = 0
    while 0 <= pos < len(targets):
        row_idx = targets[pos]
        row = rows[row_idx]

        img_path = resolve_path(row["image_path"], csv_path, workspace_root)
        patch_path = (csv_path.parent / row["patch_path"]).resolve()

        if img_path.exists():
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                image_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)

        patch_rgb = load_patch_rgb(patch_path)
        panel = build_panel(row, row_idx, len(rows), csv_path, image_rgb, patch_rgb)
        cv2.imshow(args.window_name, panel)
        key = cv2.waitKeyEx(0)

        if key in (ord("1"), ord("0"), ord("u")):
            # 1/0/u 直接写标签并前进
            new_label = 1 if key == ord("1") else 0 if key == ord("0") else -1
            if to_int_label(row.get("label", -1)) != new_label:
                rows[row_idx]["label"] = str(new_label)
                changed += 1
            pos += 1
            if changed >= max(1, args.autosave_every):
                # 达到自动保存阈值后落盘，降低意外中断损失
                write_rows(csv_path, headers, rows)
                changed = 0
        elif key == ord("s"):
            # 跳过当前样本
            pos += 1
        elif key == ord("b"):
            # 回到上一条，便于修正刚才的标注
            pos = max(0, pos - 1)
        elif key in (ord("q"), 27):
            # 退出前会在循环外统一保存未落盘修改
            break

    if changed > 0:
        write_rows(csv_path, headers, rows)
    cv2.destroyAllWindows()

    counts = {}
    for r in rows:
        l = to_int_label(r.get("label", -1))
        counts[l] = counts.get(l, 0) + 1
    print(f"saved: {csv_path}")
    print(f"label_counts: {counts}")


if __name__ == "__main__":
    main()
