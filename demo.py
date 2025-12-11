import torch
from sv_core import SVCore

# Example of basic SV-Core usage
model = SVCore(dim=512)

x = torch.randn(1, 512)     # Input vector
goal = torch.randn(1, 512)  # Goal vector

output = model(x, goal)

print("SV-Core output:")
print(output)
