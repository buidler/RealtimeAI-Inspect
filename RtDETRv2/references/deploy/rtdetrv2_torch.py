"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

本脚本用于：
1）从 YAML 配置和权重文件构建 RTDETR 推理模型；
2）对单张输入图像进行目标检测（矩形框）；
3）基于检测框做传统图像处理，提取纤维的不规则轮廓并可视化。

本脚本只使用传统图像处理生成轮廓，不依赖 HQ-SAM。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw
from scipy import ndimage, spatial

from src.core import YAMLConfig


def _binary_mask_from_crop(crop):
    # 对单个检测框裁剪出的局部图像做灰度+自适应阈值，生成前景二值掩膜
    arr = np.asarray(crop.convert("L"))
    hist, _ = np.histogram(arr, bins=256, range=(0, 255))
    total = arr.size
    sum_total = np.dot(hist, np.arange(256))
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    mask = arr >= threshold
    inv_mask = ~mask
    if inv_mask.mean() > mask.mean():
        mask = inv_mask
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    return mask


def _contours_from_box(im, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = im.size
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return []
    crop = im.crop((x0, y0, x1, y1))
    mask = _binary_mask_from_crop(crop)
    labeled, num = ndimage.label(mask)
    contours = []
    if num == 0:
        return contours
    h_crop, w_crop = mask.shape
    crop_center = np.array([0.5 * (w_crop - 1), 0.5 * (h_crop - 1)], dtype=np.float32)
    best_pts = None
    best_area = 0.0
    best_dist2 = None
    for label_id in range(1, num + 1):
        region = labeled == label_id
        eroded = ndimage.binary_erosion(region, structure=np.ones((3, 3)))
        boundary = region & ~eroded
        ys, xs = np.nonzero(boundary)
        if ys.size < 20:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        area = float(region.sum())
        center = pts.mean(axis=0)
        dist2 = float(((center - crop_center) ** 2).sum())
        if area > best_area or (area == best_area and (best_dist2 is None or dist2 < best_dist2)):
            best_area = area
            best_dist2 = dist2
            best_pts = pts
    if best_pts is None:
        return contours
    pts = best_pts
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]
    ordered[:, 0] += x0
    ordered[:, 1] += y0
    pts_arr = ordered
    dist_edge = np.minimum.reduce([
        pts_arr[:, 0] - x0,
        pts_arr[:, 1] - y0,
        x1 - pts_arr[:, 0],
        y1 - pts_arr[:, 1],
    ])
    if (dist_edge < 3.0).mean() > 0.7:
        return contours
    contours.append(pts_arr.tolist())
    return contours


def draw(images, labels, boxes, scores, thrh=0.6, draw_contours=False, save_dir=None, im_name=None):
    # 遍历每张图，先根据置信度阈值画红色矩形框，可选叠加黄色不规则轮廓
    # save_dir 指定结果图保存目录；im_name 用于根据原图文件名生成输出文件名
    for i, im in enumerate(images):
        drawer = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j, b in enumerate(box):
            drawer.rectangle(list(b), outline="red")
            drawer.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill="blue")

            if draw_contours:
                contour_list = _contours_from_box(im, b)
                for pts in contour_list:
                    if len(pts) < 3:
                        continue
                    pts_seq = [(float(p[0]), float(p[1])) for p in pts]
                    drawer.line(pts_seq + [pts_seq[0]], fill="yellow", width=2)

        # 如果未显式指定保存目录，则默认使用当前目录
        if save_dir is None:
            save_dir = "."
        if im_name is not None and len(images) == 1:
            stem, _ = os.path.splitext(os.path.basename(im_name))
            if draw_contours:
                out_name = f"{stem}_contours.jpg"
            else:
                out_name = f"{stem}_det.jpg"
        else:
            out_name = f"results_{i}.jpg"
        out_path = os.path.join(save_dir, out_name)
        im.save(out_path)


def main(args, ):
    """main
    使用 YAML 配置和训练权重构建部署模型，对单张图像进行推理与可视化。

    这里不直接操作 PyTorch 的底层细节，而是通过 YAMLConfig 这个封装类：
    - cfg.model 是根据配置文件构建好的 RTDETR 网络结构（nn.Module 子类）；
    - cfg.postprocessor 包含解码框、阈值过滤、NMS、缩放回原图尺寸等后处理逻辑。
    """
    # 通过 YAMLConfig 读取配置文件，并在其中记录权重路径（resume）
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        # 从磁盘加载 checkpoint（state_dict），map_location='cpu' 表示先加载到 CPU
        checkpoint = torch.load(args.resume, map_location='cpu') 
        # 训练时如果启用了 EMA，这里优先使用 ema 的权重；否则使用普通 model 权重
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        # 当前脚本只支持从权重文件加载，不支持“随机初始化”推理
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE 加载训练阶段权重，并转换为推理部署模式
    # cfg.model 是一个普通的 PyTorch 模型（nn.Module），这里把参数字典填进去
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            # 部署版 RTDETR 模型和后处理（含 NMS、尺度还原等）
            # deploy() 一般会做一些结构上的简化，例如去掉训练专用分支等
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            # images: 形状为 [1, 3, H, W] 的张量（BCHW），值范围为 0~1
            # orig_target_sizes: 原图宽高，用于把预测框从 640x640 缩放回原图尺寸
            outputs = self.model(images)
            # postprocessor 内部会完成分类得分、边框解码、NMS 等，并返回 numpy 风格结果
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    # 把自定义的 Model 包装成一个普通的 PyTorch 模型，并移动到指定设备（CPU/GPU）
    model = Model().to(args.device)

    # 读入待推理图像，并转换成 RGB（PyTorch 一般假设输入是 3 通道）
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    # orig_size 形状为 [1, 2]，内部是 [width, height]，放到与模型相同的 device 上
    orig_size = torch.tensor([w, h])[None].to(args.device)

    # torchvision.transforms 用于把 PIL 图像转成张量，并缩放到模型期望的大小
    transforms = T.Compose([
        T.Resize((640, 640)),  # 统一缩放到 640x640
        T.ToTensor(),          # 转成 [C, H, W]，像素值从 0~255 归一化到 0~1 的 float32
    ])
    # im_data 形状为 [1, 3, 640, 640]，在最前面增加 batch 维度
    im_data = transforms(im_pil)[None].to(args.device)

    # 前向推理，返回的是 (labels, boxes, scores)
    output = model(im_data, orig_size)
    labels, boxes, scores = output

    # 确定结果图保存路径：
    # 1）优先使用命令行传入的 --save-dir；
    # 2）否则在权重文件同级目录下创建 contour_results 子目录；
    # 3）如果没有权重路径（理论上不会发生），则退回到当前工作目录下的 contour_results。
    if args.save_dir:
        save_dir = args.save_dir
    else:
        if args.resume:
            ckpt_dir = os.path.dirname(args.resume)
            save_dir = os.path.join(ckpt_dir, "contour_results")
        else:
            save_dir = "contour_results"
    os.makedirs(save_dir, exist_ok=True)

    # 把 PIL 原图和推理结果送入可视化函数，生成带矩形框/轮廓的结果图
    draw([im_pil], labels, boxes, scores, thrh=args.score_thr, draw_contours=args.contours, save_dir=save_dir, im_name=args.im_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # -c: 配置文件路径（例如 configs/rtdetrv2/rtdetrv2_fiber.yml）
    parser.add_argument('-c', '--config', type=str)
    # -r: 训练好的模型权重路径（.pth）
    parser.add_argument('-r', '--resume', type=str)
    # -f: 单张输入图像路径
    parser.add_argument('-f', '--im-file', type=str)
    # -d: 使用的设备，cpu 或 cuda:0 等
    parser.add_argument('-d', '--device', type=str, default='cpu')
    # --score-thr: 置信度阈值，控制画哪些检测框
    parser.add_argument('--score-thr', type=float, default=0.6)
    # --contours: 是否启用基于传统图像处理的不规则轮廓绘制
    parser.add_argument('--contours', action='store_true')
    # --save-dir: 结果图保存目录，默认自动放到权重同级目录的 contour_results 中
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()
    main(args)
