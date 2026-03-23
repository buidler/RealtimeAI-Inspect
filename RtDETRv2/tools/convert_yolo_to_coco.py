import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import glob

def yolo_to_coco(data_dir, output_dir=None):
    """
    将 YOLO 格式数据集转换为 COCO JSON 格式。
    
    参数:
        data_dir: 数据集根目录，应包含 train, val, test 子目录，
                  每个子目录下应有 images 和 labels 文件夹。
        output_dir: 输出 JSON 文件的目录。默认为 data_dir/annotations
    """
    data_path = Path(data_dir)
    if output_dir is None:
        output_dir = data_path / 'annotations'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义子集
    subsets = ['train', 'val', 'test']
    
    # 扫描所有标签文件以获取类别 ID 集合
    print("正在扫描所有标签以确定类别列表...")
    all_class_ids = set()
    
    # 预先扫描一遍获取所有类别
    for subset in subsets:
        labels_dir = data_path / subset / 'labels'
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.rglob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_id = int(parts[0])
                        all_class_ids.add(class_id)
                    except ValueError:
                        pass

    # 生成类别信息
    # 尝试从 data_dir 读取 classes.txt
    classes_file = data_path / 'classes.txt'
    class_names = {}
    if classes_file.exists():
        print(f"Loading class names from {classes_file}")
        with open(classes_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            for idx, name in enumerate(lines):
                class_names[idx] = name
    else:
        print("Warning: classes.txt not found. Using generic class names.")

    sorted_ids = sorted(list(all_class_ids))
    categories = []
    for cid in sorted_ids:
        name = class_names.get(cid, f"class_{cid}")
        categories.append({
            "id": cid, # 保持与 YOLO class_id 一致
            "name": name,
            "supercategory": "object"
        })
    
    print(f"检测到的类别 ID: {sorted_ids}")
    print(f"生成的类别信息: {categories}")
    
    # 开始转换每个子集
    for subset in subsets:
        subset_dir = data_path / subset
        images_dir = subset_dir / 'images'
        labels_dir = subset_dir / 'labels'
        
        if not images_dir.exists():
            print(f"跳过 {subset}: 找不到 images 目录")
            continue
            
        print(f"\n正在转换 {subset} 数据集...")
        
        coco_data = {
            "info": {
                "description": f"COCO format dataset converted from YOLO ({subset})",
                "year": 2024,
                "version": "1.0",
                "contributor": ""
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        # 支持的图片扩展名
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in images_dir.rglob('*') if f.suffix.lower() in valid_extensions]
        
        annotation_id = 1
        
        for img_path in tqdm(image_files, desc=f"Processing {subset}"):
            # 1. 处理图片信息
            try:
                # 获取图片宽高
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"无法读取图片 {img_path}: {e}")
                continue
                
            # 使用文件名作为 image_id (去除非数字字符，或者使用简单的自增 ID)
            # 这里为了简单和唯一性，使用文件名 stem 的哈希或者尝试解析数字
            # 为了兼容性，建议使用纯数字 ID。尝试从文件名解析数字，如果失败则使用 hash
            image_id = None
            try:
                # 尝试提取文件名中的数字 (e.g., 00001.jpg -> 1)
                image_id = int(''.join(filter(str.isdigit, img_path.stem)))
            except ValueError:
                # 如果文件名没有数字，使用 path 的 hash (取模以防过大)
                image_id = abs(hash(img_path.stem)) % 100000000
            
            # 添加图片记录
            img_info = {
                "id": image_id,
                "file_name": img_path.name, # 只保留文件名
                "width": width,
                "height": height,
                "date_captured": ""
            }
            coco_data["images"].append(img_info)
            
            # 2. 处理标签信息
            # 寻找对应的 label 文件
            # 假设结构对应：images/xxx.jpg -> labels/xxx.txt
            label_file = None
            
            # 尝试直接对应
            potential_label = labels_dir / (img_path.stem + '.txt')
            if potential_label.exists():
                label_file = potential_label
            else:
                # 尝试递归查找或处理子目录结构
                # 简单起见，如果上面没找到，这里就不深入了，除非用户有特殊结构
                pass
                
            if label_file and label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                        
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    
                    category_id = class_id # 直接映射
                    
                    bbox = []
                    segmentation = []
                    area = 0
                    
                    if len(coords) == 4:
                        # YOLO 格式: x_center, y_center, w, h (归一化)
                        cx, cy, w, h = coords
                        
                        # 转换为 COCO 格式: x_min, y_min, w, h (绝对像素值)
                        abs_w = w * width
                        abs_h = h * height
                        abs_x = (cx * width) - (abs_w / 2)
                        abs_y = (cy * height) - (abs_h / 2)
                        
                        bbox = [abs_x, abs_y, abs_w, abs_h]
                        area = abs_w * abs_h
                        segmentation = [] # BBox 模式没有分割点
                        
                    elif len(coords) > 4:
                        # YOLO Segmentation 格式: x1, y1, x2, y2, ... (归一化)
                        # 转换为绝对坐标
                        points = []
                        for i in range(0, len(coords), 2):
                            px = coords[i] * width
                            py = coords[i+1] * height
                            points.append(px)
                            points.append(py)
                        
                        segmentation = [points]
                        
                        # 从多边形计算 bbox
                        xs = points[0::2]
                        ys = points[1::2]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        
                        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                        # 简单估算面积，或者使用多边形面积公式
                        # 这里简单使用 bbox 面积作为近似，或者 0.5 * abs(sum(x_i*y_{i+1} - x_{i+1}*y_i))
                        area = (max_x - min_x) * (max_y - min_y) 
                    
                    ann_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "segmentation": segmentation,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(ann_info)
                    annotation_id += 1

        # 保存 JSON 文件
        output_file = output_dir / f'instances_{subset}.json'
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        print(f"成功转换 {subset} 集: 包含 {len(coco_data['images'])} 张图片, {len(coco_data['annotations'])} 个标注")
        print(f"JSON 文件已保存至: {output_file.absolute()}") # 重点：打印绝对路径

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 YOLO 数据集转换为 COCO JSON 格式")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出 JSON 的目录 (默认: data_dir/annotations)")
    
    args = parser.parse_args()
    
    yolo_to_coco(args.data_dir, args.output_dir)
