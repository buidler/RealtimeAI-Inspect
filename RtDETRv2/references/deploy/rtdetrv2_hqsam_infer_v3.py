import runpy
import sys
from pathlib import Path


# v3 说明：
# - 复用 v2 全部流程与实现，避免代码分叉。
# - 默认把 score_low_thr 调到 0.8，以减小亮暗差异导致的分支敏感性。
# - 其他参数保持 v2 默认；若命令行显式传入 --score-low-thr，则以用户传入为准。


def _has_arg(argv, name):
    for i, a in enumerate(argv):
        if a == name:
            return True
        if a.startswith(name + "="):
            return True
        # 兼容简写场景（尽管本参数通常是长选项）
        if i + 1 < len(argv) and a in ("-slt",) and name == "--score-low-thr":
            return True
    return False


def main():
    argv = list(sys.argv)
    if not _has_arg(argv, "--score-low-thr"):
        argv.extend(["--score-low-thr", "0.8"])
    sys.argv = argv

    v2_path = Path(__file__).resolve().with_name("rtdetrv2_hqsam_infer_v2.py")
    runpy.run_path(str(v2_path), run_name="__main__")


if __name__ == "__main__":
    main()
