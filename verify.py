import torch

print("CUDA Detection Status:", torch.cuda.is_available())
print("Active GPU Core:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
