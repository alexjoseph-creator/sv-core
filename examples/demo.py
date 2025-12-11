import torch
from sv_core import SVCore

# Exemple minimal d'utilisation de SV-Core
if __name__ == "__main__":
    model = SVCore(dim=512)

    x = torch.randn(1, 512)
    goal = torch.randn(1, 512)

    output = model(x, goal)
    print("SV-Core output vector:")
    print(output)
