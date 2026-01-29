import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("using device:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("using device: CPU")

# 간단한 연산으로 실제 GPU 사용 확인
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = x @ y
print("result device:", z.device)
