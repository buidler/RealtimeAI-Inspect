import os
import argparse
from pathlib import Path
from tqdm import tqdm

def clean_augmented_images(data_dir, dry_run=True, auto_yes=False):
    """
    删除数据集中的增强图片及其对应的标签文件。
    通常增强图片会有特定的后缀，例如 _aug, _rotate, _flip, _noise 等，
    或者是通过文件名长度/模式识别。
    
    这里假设增强图片包含以下常见关键词后缀。
    """
    
    # 定义常见的增强后缀关键词
    aug_suffixes = [
        '_aug', '_rotated', '_flip', '_blur', '_noise', 
        '_crop', '_resized', '_enhanced', '_bright', 
        '_dark', '_contrast', '_mosaic', '_mixup',
        # 根据用户数据集发现的特定后缀
        '_turn_light', '_turn_dark', '_sp', '_gaussian'
    ]
    
    data_path = Path(data_dir)
    
    # 自动检测子目录结构 (train, val, test)
    subsets = ['train', 'val', 'test']
    target_dirs = []
    
    # 检查根目录下是否有 images
    if (data_path / 'images').exists():
        target_dirs.append(data_path)
    
    # 检查子目录
    for subset in subsets:
        subset_path = data_path / subset
        if (subset_path / 'images').exists():
            target_dirs.append(subset_path)
            
    if not target_dirs:
        print(f"Warning: No standard 'images' directory found in {data_path} or its subdirectories (train/val/test).")
        print("Scanning root directory recursively...")
        target_dirs.append(data_path)
        
    files_to_delete = []
    
    for current_root in target_dirs:
        images_dir = current_root / 'images'
        labels_dir = current_root / 'labels'
        
        # 如果是 fallback 到 root 且没有 images 目录，则直接搜索 root
        if not images_dir.exists() and current_root == data_path:
            images_dir = current_root
            # 如果没有 labels 目录，则假设 label 和 image 在一起或者没有 label
            if not labels_dir.exists():
                labels_dir = current_root 

        print(f"Scanning {images_dir}...")
        
        # 收集所有图片文件
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in images_dir.rglob('*') if f.suffix.lower() in valid_extensions]
        
        print(f"Found {len(image_files)} images in {images_dir}")
        
        for img_file in image_files:
            stem = img_file.stem
            
            # 检查文件名是否包含增强后缀
            is_augmented = False
            for suffix in aug_suffixes:
                if suffix in stem:
                    is_augmented = True
                    break
            
            if is_augmented:
                files_to_delete.append(img_file)
                
                # 同时标记对应的标签文件（如果存在）
                # 尝试在 labels 目录下寻找同名 txt 文件
                # 假设 labels 文件名与 images 文件名对应（除了后缀）
                
                # 1. 尝试直接对应 (flat structure)
                label_file = labels_dir / (stem + '.txt')
                if label_file.exists():
                    files_to_delete.append(label_file)
                else:
                     # 2. 如果 images 在子目录中 (e.g. images/class1/x.jpg), label 可能在 labels/class1/x.txt
                     # 获取相对于 images_dir 的相对路径
                     try:
                        rel_path = img_file.relative_to(images_dir)
                        # 将后缀改为 .txt
                        label_rel_path = rel_path.with_suffix('.txt')
                        label_file_recursive = labels_dir / label_rel_path
                        if label_file_recursive.exists():
                            files_to_delete.append(label_file_recursive)
                     except ValueError:
                        pass
    
    if not files_to_delete:
        print("未发现符合常见增强后缀（如 _aug, _rotated 等）的文件。")
        return

    print(f"\n发现 {len(files_to_delete)} 个疑似增强文件:")
    for f in files_to_delete[:10]:
        print(f"  - {f.name}")
    if len(files_to_delete) > 10:
        print(f"  ... 以及其他 {len(files_to_delete) - 10} 个文件")
        
    if dry_run:
        print("\n[DRY RUN] 这是一个试运行。没有文件被实际删除。")
        print("请检查上述文件列表。如果确认要删除，请使用 --no-dry-run 参数运行此脚本。")
        print("或者，您可以编辑脚本中的 `aug_suffixes` 列表以匹配您的文件名特征。")
    else:
        if auto_yes:
            confirm = 'y'
        else:
            confirm = input(f"\n警告: 即将永久删除 {len(files_to_delete)} 个文件。确定吗？(y/n): ")
            
        if confirm.lower() == 'y':
            deleted_count = 0
            for f in tqdm(files_to_delete, desc="Deleting"):
                try:
                    os.remove(f)
                    deleted_count += 1
                except Exception as e:
                    print(f"删除 {f} 失败: {e}")
            print(f"\n成功删除了 {deleted_count} 个文件。")
        else:
            print("操作已取消。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理数据集中的手动增强图片")
    parser.add_argument("--data_dir", type=str, required=True, help="数据集根目录 (包含 images 和 labels 文件夹)")
    parser.add_argument("--no-dry-run", action="store_true", help="实际执行删除操作 (默认只列出文件)")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认提示直接删除")
    
    args = parser.parse_args()
    
    clean_augmented_images(args.data_dir, dry_run=not args.no_dry_run, auto_yes=args.yes)
