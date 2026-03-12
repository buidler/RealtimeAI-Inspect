import torch
import torch.nn as nn


class BoundaryQualityHead(nn.Module):
    """
    用于对局部轮廓片段打分的小型分类头：
    - 场景：给定某个检测框内的一小段轮廓，希望判断它更像“真实纤维边界”还是“刀痕/伪轮廓”
    - 输入：从原图裁剪出的局部 patch，通道为 RGB + HQ 掩膜 + 传统掩膜（共 5 通道）
    - 输出：该 patch 中的轮廓片段为“真实纤维外边界”的置信度（logit，外部再接 sigmoid）
    """

    def __init__(self, in_channels: int = 5, base_channels: int = 32):
        super().__init__()
        # 三层卷积 + 下采样 + 自适应全局池化，提取局部上下文特征
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(base_channels * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: Tensor，形状为 [B, C, H, W]，其中 C 一般为 5（RGB + HQ 掩膜 + 传统掩膜）
        返回:
            logit: Tensor，形状为 [B, 1]，未经过 sigmoid 的二分类 logit
        """
        feat = self.features(x).flatten(1)
        logit = self.classifier(feat)
        return logit


def load_boundary_head(ckpt_path: str, device: torch.device) -> BoundaryQualityHead:
    """
    从权重文件中加载 BoundaryQualityHead，并移动到指定 device。
    约定：ckpt 中直接保存的是 model.state_dict()（而不是包含其它字段的 dict）。
    """
    model = BoundaryQualityHead()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device=device)
    model.eval()
    return model

