import torch
t = torch.tensor([[1, 2]])
try:
    print(t[0, 1, 2])
except Exception as e:
    print("Error:", e)
