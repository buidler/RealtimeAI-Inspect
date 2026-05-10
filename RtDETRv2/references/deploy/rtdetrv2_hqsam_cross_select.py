import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def _resolve_path(raw_path, field_name, allow_missing=False):
    p = Path(raw_path).expanduser()
    candidates = [p]
    if not p.is_absolute():
        cwd = Path.cwd()
        candidates.append(cwd / p)
    seen = set()
    ordered = []
    for c in candidates:
        k = str(c.resolve()) if c.exists() else str(c)
        if k not in seen:
            seen.add(k)
            ordered.append(c)
    for c in ordered:
        if c.exists():
            return c.resolve()
    if allow_missing:
        return ordered[0].resolve()
    raise FileNotFoundError(f"{field_name} not found: {raw_path}")


def _strip_known_suffix(stem):
    for suf in ("_entity_mask", "_hqsam_contours", "_mask", "_contour"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _index_by_stem(dir_path, pattern):
    m = {}
    for p in sorted(dir_path.glob(pattern)):
        key = _strip_known_suffix(p.stem)
        m[key] = p
    return m


def _load_mask_area(mask_path):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return int((m > 0).sum())


def _pick_by_score(area_a, area_b, area_ref, strategy):
    if strategy == "area_max":
        return "A" if area_a >= area_b else "B", 0.0, 0.0
    if strategy == "area_min":
        return "A" if area_a <= area_b else "B", 0.0, 0.0
    # area_to_ref
    denom = max(1, int(area_ref))
    err_a = abs(int(area_a) - int(area_ref)) / float(denom)
    err_b = abs(int(area_b) - int(area_ref)) / float(denom)
    return ("A" if err_a <= err_b else "B"), err_a, err_b


def main():
    parser = argparse.ArgumentParser(description="基于已生成文件做 A/B 交叉择优（同名图 2 选 1）")
    parser.add_argument("--a-result-dir", type=str, required=True, help="A 组轮廓结果目录（181 张）")
    parser.add_argument("--b-result-dir", type=str, required=True, help="B 组轮廓结果目录（181 张）")
    parser.add_argument("--a-mask-dir", type=str, required=True, help="A 组实体掩膜目录")
    parser.add_argument("--b-mask-dir", type=str, required=True, help="B 组实体掩膜目录")
    parser.add_argument("--ref-mask-dir", type=str, default="", help="参考掩膜目录（推荐 HQSAM 参考）")
    parser.add_argument("--save-dir", type=str, required=True, help="交叉择优后的最终结果目录")
    parser.add_argument("--save-mask-dir", type=str, default="", help="可选：保存择优后的实体掩膜目录")
    parser.add_argument("--expected-count", type=int, default=181, help="期望图片数量（默认 181）")
    parser.add_argument("--result-pattern", type=str, default="*_hqsam_contours.jpg")
    parser.add_argument("--mask-pattern", type=str, default="*_entity_mask.png")
    parser.add_argument(
        "--strategy",
        type=str,
        default="area_to_ref",
        choices=["area_to_ref", "area_max", "area_min"],
        help="area_to_ref: 与参考面积差最小；area_max: 面积更大；area_min: 面积更小",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    a_result_dir = _resolve_path(args.a_result_dir, "a_result_dir")
    b_result_dir = _resolve_path(args.b_result_dir, "b_result_dir")
    a_mask_dir = _resolve_path(args.a_mask_dir, "a_mask_dir")
    b_mask_dir = _resolve_path(args.b_mask_dir, "b_mask_dir")
    ref_mask_dir = _resolve_path(args.ref_mask_dir, "ref_mask_dir") if args.ref_mask_dir else None
    save_dir = _resolve_path(args.save_dir, "save_dir", allow_missing=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_mask_dir = _resolve_path(args.save_mask_dir, "save_mask_dir", allow_missing=True) if args.save_mask_dir else None
    if save_mask_dir is not None:
        save_mask_dir.mkdir(parents=True, exist_ok=True)

    a_res = _index_by_stem(a_result_dir, args.result_pattern)
    b_res = _index_by_stem(b_result_dir, args.result_pattern)
    a_mask = _index_by_stem(a_mask_dir, args.mask_pattern)
    b_mask = _index_by_stem(b_mask_dir, args.mask_pattern)
    ref_mask = _index_by_stem(ref_mask_dir, args.mask_pattern) if ref_mask_dir else {}

    common = sorted(set(a_res.keys()) & set(b_res.keys()) & set(a_mask.keys()) & set(b_mask.keys()))
    if len(common) == 0:
        raise RuntimeError("A/B 公共样本为空，请检查结果目录和掩膜目录命名是否一致。")
    if args.expected_count > 0 and len(common) < args.expected_count:
        print(f"warning: common={len(common)} < expected_count={args.expected_count}")

    picked_a = 0
    picked_b = 0
    no_ref = 0
    for i, stem in enumerate(common, start=1):
        out_name = a_res[stem].name
        out_file = save_dir / out_name
        if args.skip_existing and out_file.exists():
            print(f"[{i}/{len(common)}] skip existing {out_file}")
            continue

        area_a = _load_mask_area(a_mask[stem])
        area_b = _load_mask_area(b_mask[stem])
        if area_a is None or area_b is None:
            print(f"[{i}/{len(common)}] skip {stem} (mask read failed)")
            continue

        if args.strategy == "area_to_ref":
            ref_path = ref_mask.get(stem)
            if ref_path is None:
                # 无参考时回退到 area_max，避免中断批处理。
                no_ref += 1
                pick, err_a, err_b = _pick_by_score(area_a, area_b, 0, "area_max")
            else:
                area_ref = _load_mask_area(ref_path)
                if area_ref is None:
                    no_ref += 1
                    pick, err_a, err_b = _pick_by_score(area_a, area_b, 0, "area_max")
                else:
                    pick, err_a, err_b = _pick_by_score(area_a, area_b, area_ref, "area_to_ref")
        else:
            pick, err_a, err_b = _pick_by_score(area_a, area_b, 0, args.strategy)

        if pick == "A":
            picked_a += 1
            src_img = a_res[stem]
            src_mask = a_mask[stem]
        else:
            picked_b += 1
            src_img = b_res[stem]
            src_mask = b_mask[stem]

        shutil.copy2(src_img, out_file)
        if save_mask_dir is not None:
            shutil.copy2(src_mask, save_mask_dir / src_mask.name)

        print(
            f"[{i}/{len(common)}] pick={pick} stem={stem} "
            f"area_a={area_a} area_b={area_b} err_a={err_a:.4f} err_b={err_b:.4f}"
        )

    print(
        f"done. common={len(common)} picked_A={picked_a} picked_B={picked_b} "
        f"no_ref_fallback={no_ref} save_dir={save_dir}"
    )


if __name__ == "__main__":
    main()
