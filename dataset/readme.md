## 使用自定义 YOLO 数据集

如果您的数据集是 YOLO 格式（txt 标签），请按照以下步骤操作：

1. **准备数据**:
   按照以下结构组织您的数据（示例）：
   ```
   dataset/my_dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── classes.txt  # 每行一个类别名称
   ```

2. **转换为 COCO 格式**:
   为训练集和验证集运行提供的转换脚本：
   ```bash
   # 转换训练集
   python tools/convert_yolo_to_coco.py --data_dir dataset/my_dataset/train --save_file dataset/my_dataset/annotations/train.json --class_file dataset/my_dataset/classes.txt

   # 转换验证集
   python tools/convert_yolo_to_coco.py --data_dir dataset/my_dataset/val --save_file dataset/my_dataset/annotations/val.json --class_file dataset/my_dataset/classes.txt
   ```

3. **更新配置**:
   编辑 `configs/dataset/my_dataset.yml`：
   - 将 `num_classes` 设置为您的类别数量。
   - 如果 `img_folder` 和 `ann_file` 路径与上述不同，请更新它们。
