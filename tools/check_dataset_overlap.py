import os
import argparse
from pathlib import Path
from tqdm import tqdm

def check_dataset_overlap(data_dir, auto_fix=False):
    """
    检查 train, val, test 数据集之间是否存在重叠（重复图片）。
    基于文件名（不含后缀）进行比较。
    如果启用 auto_fix，将按照 Test > Val > Train 的优先级保留文件（移除低优先级的文件）。
    """
    data_path = Path(data_dir)
    subsets = ['train', 'val', 'test']
    
    # 存储每个子集的图片文件名（stem）
    subset_files = {
        'train': set(),
        'val': set(),
        'test': set()
    }
    
    # 存储 stem 到完整路径的映射: {'subset': {'stem': Path}}
    subset_paths = {
        'train': {},
        'val': {},
        'test': {}
    }

    print(f"正在扫描数据集目录: {data_path} ...")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 扫描每个子集
    for subset in subsets:
        subset_images_dir = data_path / subset / 'images'
        if not subset_images_dir.exists():
            print(f"警告: 目录 {subset_images_dir} 不存在，跳过扫描该子集。")
            continue
            
        files = [f for f in subset_images_dir.rglob('*') if f.suffix.lower() in valid_extensions]
        print(f"  - {subset}: 发现 {len(files)} 张图片")
        
        for f in files:
            stem = f.stem
            # 记录文件名，检查同个子集内的重复（虽然文件系统通常不允许同名文件在同一目录下，但如果是递归扫描可能有）
            if stem in subset_files[subset]:
                print(f"    [警告] 在 {subset} 内部发现重复文件名的文件 (可能在不同子目录): {stem}")
            
            subset_files[subset].add(stem)
            subset_paths[subset][stem] = f

    print("\n开始检查跨子集重叠...")
    has_overlap = False
    
    # 定义检查对和处理逻辑 (High Priority, Low Priority)
    # 如果重叠，从 Low Priority 中删除
    priority_pairs = [
        ('test', 'train'), # Keep Test, Remove Train
        ('test', 'val'),   # Keep Test, Remove Val
        ('val', 'train')   # Keep Val, Remove Train
    ]
    
    files_to_remove = [] # List of (path, reason)
    
    for high_p, low_p in priority_pairs:
        intersection = subset_files[high_p].intersection(subset_files[low_p])
        
        if intersection:
            has_overlap = True
            count = len(intersection)
            print(f"\n[发现重叠] {high_p} (保留) 和 {low_p} (待移除) 共有 {count} 个重复文件。")
            
            for stem in intersection:
                # 获取低优先级子集中的文件路径
                file_to_del = subset_paths[low_p][stem]
                files_to_remove.append((file_to_del, low_p, stem))
                
                if count <= 10: # 只打印前10个
                    print(f"  - 重复: {stem}")
            
            if count > 10:
                print(f"  ... 以及其他 {count - 10} 个文件")
        else:
            print(f"√ {high_p} 和 {low_p} 没有重叠。")
            
    if not has_overlap:
        print("\n恭喜！Train, Val, Test 数据集之间没有发现重叠。")
        return

    if not auto_fix:
        print(f"\n[建议] 发现 {len(files_to_remove)} 个重复文件实例。")
        print("请使用 --auto-fix 参数运行此脚本以自动清理（保留优先级: Test > Val > Train）。")
        return
        
    print(f"\n[Auto-Fix] 准备移除 {len(files_to_remove)} 个重复文件...")
    
    deleted_count = 0
    for img_path, subset, stem in tqdm(files_to_remove, desc="Removing duplicates"):
        try:
            # 1. 删除图片
            if img_path.exists():
                os.remove(img_path)
                deleted_count += 1
            
            # 2. 删除对应的标签文件
            # 假设 labels 目录与 images 同级
            # img_path: .../subset/images/xxx.jpg
            # label_path: .../subset/labels/xxx.txt
            
            # 尝试推断 labels 路径
            parents = list(img_path.parents)
            # parents[0] is images dir (or subdir), parents[1] might be 'subset'
            
            # 简单的路径替换: .../images/... -> .../labels/...
            # 这种方法比较通用，只要结构是标准的
            img_path_str = str(img_path)
            if 'images' in img_path_str:
                label_path_str = img_path_str.replace('images', 'labels', 1)
                label_path = Path(label_path_str).with_suffix('.txt')
                
                if label_path.exists():
                    os.remove(label_path)
                    # print(f"Deleted label: {label_path}")
            
        except Exception as e:
            print(f"删除失败 {img_path}: {e}")
            
    print(f"\n成功移除了 {deleted_count} 个重复图片及其对应标签。")
    print("数据集现在应该是干净的。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查数据集 train/val/test 之间的重叠情况")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集根目录 (包含 train, val, test 文件夹)")
    parser.add_argument("--auto-fix", action="store_true", help="自动修复重叠：保留 Test > Val > Train")
    
    args = parser.parse_args()
    
    check_dataset_overlap(args.data_dir, auto_fix=args.auto_fix)
