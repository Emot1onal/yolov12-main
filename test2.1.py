#
#import torch
#print(f"PyTorch 版本: {torch.__version__}")
#print(f"CUDA 是否可用: {torch.cuda.is_available()}")
#print(f"GPU 数量: {torch.cuda.device_count()}")
#if torch.cuda.is_available():
#    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
#    print(f"CUDA 版本: {torch.version.cuda}")

import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda version:", torch.version.cuda)
    print("gpu count:", torch.cuda.device_count())
    print("gpu name:", torch.cuda.get_device_name(0))


x = torch.randn(2000, 2000, device="cuda")
y = torch.matmul(x, x)
print("GPU computation OK:", y.shape)
