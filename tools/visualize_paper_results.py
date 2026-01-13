
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFont, ImageDraw
from pycocotools.coco import COCO
from collections import defaultdict
import sys
from pathlib import Path

# 将 src 目录添加到路径中，以便导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

def get_color_map(num_classes):
    """
    为可视化获取一组不同的颜色。
    参数:
        num_classes (int): 类别数量
    返回:
        colors (list): 颜色列表
    """
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(num_classes)]
    return colors

def visualize_dataset_stats(json_file, output_dir):
    """
    可视化数据集统计信息：类别分布。
    生成一个柱状图，显示每个类别的实例数量。
    
    参数:
        json_file (str): COCO 格式的标注文件路径
        output_dir (str): 保存图表的目录
    """
    print(f"正在加载 COCO 标注文件: {json_file}...")
    coco = COCO(json_file)
    cats = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in cats]
    cat_ids = [cat['id'] for cat in cats]
    
    # 统计每个类别的实例数
    counts = []
    for cid in cat_ids:
        ann_ids = coco.getAnnIds(catIds=[cid])
        counts.append(len(ann_ids))
        
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(cat_names, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Category (类别)', fontsize=12)
    plt.ylabel('Number of Instances (实例数量)', fontsize=12)
    plt.title('Class Distribution (类别分布)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱状图上方添加数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05*max(counts), int(yval), ha='center', va='bottom')
        
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"类别分布图已保存至: {save_path}")

def visualize_ground_truth(json_file, img_dir, output_dir, num_samples=5):
    """
    可视化 Ground Truth (真实标注) 样本。
    随机选择几张图片，并画出标注框和类别名称。
    
    参数:
        json_file (str): COCO 格式的标注文件路径
        img_dir (str): 图片所在目录
        output_dir (str): 保存结果图片的目录
        num_samples (int): 要可视化的样本数量
    """
    coco = COCO(json_file)
    img_ids = coco.getImgIds()
    cats = coco.loadCats(coco.getCatIds())
    # 为每个类别 ID 生成对应的颜色
    id2color = {cat['id']: plt.cm.tab10(i % 10) for i, cat in enumerate(cats)}
    
    # 随机选择样本
    samples = np.random.choice(img_ids, min(len(img_ids), num_samples), replace=False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_id in enumerate(samples):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"图片未找到: {img_path}")
            continue
            
        image = Image.open(img_path).convert('RGB')
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()
        
        for ann in anns:
            bbox = ann['bbox'] # COCO bbox 格式为 [x, y, w, h]
            cat_id = ann['category_id']
            cat_name = next(c['name'] for c in cats if c['id'] == cat_id)
            color = id2color[cat_id]
            
            # 绘制矩形框
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # 添加类别标签背景和文字
            ax.text(bbox[0], bbox[1] - 2, cat_name, fontsize=10, color='white', 
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1))
            
        plt.axis('off') # 关闭坐标轴
        save_path = os.path.join(output_dir, f'gt_sample_{i}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"GT 样本已保存至: {save_path}")

def visualize_inference(config_path, checkpoint, img_dir, output_dir, device='cuda', threshold=0.5):
    """
    可视化模型推理结果。
    加载模型和权重，对指定目录下的图片进行推理，并绘制检测框。
    
    参数:
        config_path (str): 模型配置文件路径 (.yml)
        checkpoint (str): 模型权重文件路径 (.pth)
        img_dir (str): 测试图片目录
        output_dir (str): 保存结果的目录
        device (str): 运行设备 ('cuda' 或 'cpu')
        threshold (float): 置信度阈值，低于此值的检测框将被过滤
    """
    print(f"正在从 {config_path} 加载模型配置...")
    cfg = YAMLConfig(config_path, resume=checkpoint)
    
    if checkpoint:
        print(f"正在加载权重: {checkpoint}")
        checkpoint_state = torch.load(checkpoint, map_location='cpu')
        if 'ema' in checkpoint_state:
            state_dict = checkpoint_state['ema']['module']
        elif 'model' in checkpoint_state:
            state_dict = checkpoint_state['model']
        else:
            state_dict = checkpoint_state
            
        cfg.model.load_state_dict(state_dict)
    
    model = cfg.model.to(device)
    model.eval()
    
    # 预处理变换：调整大小并转换为 Tensor
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    # 获取目录下所有 jpg 和 png 图片
    image_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
    if not image_paths:
        print(f"在 {img_dir} 中未找到图片")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在对 {len(image_paths)} 张图片进行推理 (仅展示前10张)...")
    
    for i, img_path in enumerate(image_paths):
        if i >= 10: break # 为了演示，限制处理前10张
        
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)
        
        im_data = transforms(im_pil)[None].to(device)
        
        with torch.no_grad():
            output = model(im_data, orig_target_sizes=orig_size)
            
        labels, boxes, scores = output
        
        # 开始绘图
        plt.figure(figsize=(10, 10))
        plt.imshow(im_pil)
        ax = plt.gca()
        
        scr = scores[0]
        # 根据阈值过滤结果
        keep = scr > threshold
        lab = labels[0][keep]
        box = boxes[0][keep]
        scrs = scores[0][keep]
        
        for j, b in enumerate(box):
            xmin, ymin, xmax, ymax = b.cpu().numpy()
            score = scrs[j].item()
            label_idx = lab[j].item()
            
            # 计算宽高
            bw = xmax - xmin
            bh = ymax - ymin
            
            color = plt.cm.tab10(label_idx % 10)
            
            # 绘制检测框
            rect = patches.Rectangle((xmin, ymin), bw, bh, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # 添加标签文本 (类别 ID + 置信度)
            label_text = f"Class {label_idx}: {score:.2f}"
            ax.text(xmin, ymin - 2, label_text, fontsize=10, color='white',
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1))
            
        plt.axis('off')
        save_path = os.path.join(output_dir, f'pred_{img_path.name}')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"预测结果已保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为论文生成可视化结果 (数据集统计/模型推理)")
    parser.add_argument("--mode", type=str, required=True, choices=['dataset', 'inference'], help="可视化模式: 'dataset' (数据集统计) 或 'inference' (模型推理)")
    
    # 数据集模式参数
    parser.add_argument("--json-file", type=str, help="COCO json 标注文件路径 (仅用于 dataset 模式)")
    parser.add_argument("--img-dir", type=str, help="图片文件夹路径")
    
    # 推理模式参数
    parser.add_argument("--config", type=str, help="模型配置文件路径 (仅用于 inference 模式)")
    parser.add_argument("--checkpoint", type=str, help="模型权重文件路径 (仅用于 inference 模式)")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值 (默认: 0.5)")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="paper_results", help="结果保存目录 (默认: paper_results)")
    
    args = parser.parse_args()
    
    if args.mode == 'dataset':
        if not args.json_file:
            print("错误: dataset 模式需要提供 --json-file 参数")
        else:
            print("正在可视化数据集统计信息...")
            visualize_dataset_stats(args.json_file, args.output_dir)
            if args.img_dir:
                print("正在可视化 Ground Truth 样本...")
                visualize_ground_truth(args.json_file, args.img_dir, args.output_dir)
                
    elif args.mode == 'inference':
        if not args.config or not args.checkpoint or not args.img_dir:
            print("错误: inference 模式需要提供 --config, --checkpoint 和 --img-dir 参数")
        else:
            visualize_inference(args.config, args.checkpoint, args.img_dir, args.output_dir, threshold=args.threshold)
