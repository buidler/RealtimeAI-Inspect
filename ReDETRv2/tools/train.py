"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os

import sys

# 将项目根目录加入 sys.path，方便在 tools 中导入 src 下的模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 允许重复加载libiomp5md.dll，风险较低

import argparse

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


def main(args, ) -> None:
    """main
    """
    # 初始化分布式环境与随机种子（在单卡时也会统一入口）
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    # resume 与 tuning 不能同时开启，只允许一种方式
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # 解析命令行中的 update 字段，将其转成字典形式
    update_dict = yaml_utils.parse_cli(args.update)
    # 用命令行参数覆盖 / 更新配置文件中的对应字段
    update_dict.update({k: v for k, v in args.__dict__.items()
        if k not in ['update', ] and v is not None})

    # 从 YAML 配置文件和命令行更新项构建配置对象
    cfg = YAMLConfig(args.config, **update_dict)

    # 如果 output_dir 已经存在，则自动追加递增后缀，避免覆盖历史结果
    base_output_dir = args.output_dir if args.output_dir is not None else cfg.output_dir
    if base_output_dir is not None:
        # 先使用配置（或命令行）中给定的基础路径
        output_dir = base_output_dir
        # 若该目录已存在，则在同级目录下寻找「名称_1、名称_2...」
        if os.path.exists(output_dir):
            base_dir, name = os.path.split(output_dir)
            # 处理 output_dir 末尾带斜杠等情况，保证 name 不为空
            if not name:
                base_dir, name = os.path.split(base_dir)
            prefix = os.path.join(base_dir, name)
            idx = 1
            # 依次尝试 prefix_1、prefix_2...，直到找到一个不存在的目录名
            while True:
                candidate = f"{prefix}_{idx}"
                if not os.path.exists(candidate):
                    output_dir = candidate
                    break
                idx += 1
        # 将最终选择的目录写回 cfg，供后续训练和可视化使用
        cfg.output_dir = output_dir

    print('cfg: ', cfg.__dict__)

    # 根据任务类型（det、seg 等）构建对应的 Solver
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        # 仅验证模式：只跑验证流程
        solver.val()
    else:
        # 正常训练模式：先进行训练
        solver.fit()
        # 只在主进程上做可视化，避免多进程重复绘图
        if dist_utils.is_main_process():
            try:
                # 导入训练结果可视化脚本
                import visualize_training_results

                # 优先使用命令行传入的输出目录，否则用配置中的 output_dir
                out_dir = args.output_dir if args.output_dir is not None else cfg.output_dir
                # 调用可视化入口，生成 loss 曲线、混淆矩阵、PR 曲线等图像
                visualize_training_results.generate_all_visualizations(
                    args.config,
                    out_dir,
                    args.device,
                )
            except Exception as e:
                # 避免可视化出错导致整体训练脚本崩溃
                print(f"post-training visualization failed: {e}")

    

if __name__ == '__main__':

    # 命令行参数解析器
    parser = argparse.ArgumentParser()
    
    # priority 0：基础运行配置
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1：通过 -u 覆盖 / 更新 YAML 配置中的字段
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env：分布式与打印相关的环境控制参数
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
