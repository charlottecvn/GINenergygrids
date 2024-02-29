import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")

print("Is CUDA enabled?", torch.cuda.is_available())
print("Is MPS enabled?", torch.backends.mps.is_available())
print("Is CPU enabled?", torch.cpu.is_available())
print("Using device:", device)