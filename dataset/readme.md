
## Use Custom YOLO Dataset

If your dataset is in YOLO format (txt labels), follow these steps:

1. **Prepare your data**:
   Organize your data like this (example):
   ```
   dataset/my_dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── classes.txt  # One class name per line
   ```

2. **Convert to COCO format**:
   Run the provided conversion script for both train and val sets:
   ```bash
   # Convert Train
   python tools/convert_yolo_to_coco.py --data_dir dataset/my_dataset/train --save_file dataset/my_dataset/annotations/train.json --class_file dataset/my_dataset/classes.txt

   # Convert Val
   python tools/convert_yolo_to_coco.py --data_dir dataset/my_dataset/val --save_file dataset/my_dataset/annotations/val.json --class_file dataset/my_dataset/classes.txt
   ```

3. **Update Config**:
   Edit `configs/dataset/my_dataset.yml`:
   - Set `num_classes` to your number of classes.
   - Update `img_folder` and `ann_file` paths if they are different from above.
