import torch, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
print("cuda is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

