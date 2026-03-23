import torch
import torchvision

print("----- 安装状态检查 -----")
print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {torchvision.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"检测到显卡: {torch.cuda.get_device_name(0)}")
    # 测试一下显卡计算
    a = torch.randn(100, 100).cuda()
    b = torch.randn(100, 100).cuda()
    c = torch.matmul(a, b)
    print("显卡计算测试: 成功！")
else:
    print("显卡计算测试: 失败，请检查驱动。")