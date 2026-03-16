import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.misc.boundary_head import BoundaryQualityHead


class BoundaryPatchDataset(Dataset):
    """
    简单的 patch 数据集：
    - csv 中每一行记录一个 patch 文件路径和对应标签（0/1）
    - patch 文件为 npz，键名为 'patch'，形状 [C, H, W]
    """

    def __init__(self, csv_path: str):
        super().__init__()
        self.records = []
        base_dir = Path(csv_path).parent
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row["label"])
                if label not in (0, 1):
                    continue
                patch_rel = row["patch_path"]
                patch_path = (base_dir / patch_rel).resolve()
                self.records.append((patch_path, label))
        if not self.records:
            raise RuntimeError(f"no valid labeled patches (label in {{0,1}}) found in {csv_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        data = np.load(path)["patch"].astype(np.float32)
        x = torch.from_numpy(data)
        y = torch.tensor([float(label)], dtype=torch.float32)
        return x, y


def train_boundary_head(
    csv_path: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda",
    out_ckpt: str = "boundary_head.pth",
):
    device = torch.device(device)
    dataset = BoundaryPatchDataset(csv_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = BoundaryQualityHead().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            running_loss += loss.item() * bs
            n_samples += bs
        avg_loss = running_loss / max(1, n_samples)
        print(f"[train] epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}, samples={n_samples}")

    out_path = Path(out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved boundary head checkpoint to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 dump_boundary_patches 采集的 patch 数据训练 BoundaryQualityHead。"
    )
    parser.add_argument("--csv-path", type=str, required=True, help="dump_boundary_patches 生成的 patches.csv，且 label 已人工修改为 0/1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-ckpt", type=str, default="boundary_head.pth")

    args = parser.parse_args()
    train_boundary_head(
        csv_path=args.csv_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        out_ckpt=args.out_ckpt,
    )


if __name__ == "__main__":
    main()
