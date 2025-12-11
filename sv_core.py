"""
SV-Core : Implémentation minimale exécutable
Architecture cognitive téléologique pour LLM agentifs
Auteur : Alexandre Vinas (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# =====================================================
# Φ* — Presence (orientation initiale)
# =====================================================
class Presence(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.orientation = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.orientation(x)



# =====================================================
# μ-TEL — Mémoire téléologique
# =====================================================
class MuTEL(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.z = nn.Linear(dim * 2, dim)
        self.r = nn.Linear(dim * 2, dim)
        self.m_tilde = nn.Linear(dim * 2, dim)

    def forward(self, h, m, g):
        z = torch.sigmoid(self.z(torch.cat([h, g], dim=-1)))
        r = torch.sigmoid(self.r(torch.cat([m, g], dim=-1)))
        m_candidate = torch.tanh(self.m_tilde(torch.cat([h, r * m], dim=-1)))
        return (1 - z) * m + z * m_candidate



# =====================================================
# Λ — Structure
# =====================================================
class LambdaStruct(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.extract = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.extract(x))



# =====================================================
# Ω* — Orientation Haute
# =====================================================
class OmegaStar(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.orient = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.tanh(self.orient(x))



# =====================================================
# ⦿ — Noyau Unitaire
# =====================================================
class UnitCore(nn.Module):
    def forward(self, x):
        return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)



# =====================================================
# CΩ — Cohérence interne
# =====================================================
class Coherence(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return x + 0.1 * torch.tanh(self.linear(x))



# =====================================================
# PTOr — Transition de Phase
# =====================================================
class PTOr(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.shift = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.shift(x))



# =====================================================
# SV-Core — Pipeline téléologique complet
# =====================================================
class SVCore(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.phi = Presence(dim)
        self.mu = MuTEL(dim)
        self.lam = LambdaStruct(dim)
        self.omega = OmegaStar(dim)
        self.unit = UnitCore()
        self.coh = Coherence(dim)
        self.pt = PTOr(dim)

        # mémoire interne persistante (suit CPU/GPU automatiquement)
        self.register_buffer("memory", torch.zeros(1, dim))

    def forward(self, x, goal):
        h = self.phi(x)
        self.memory = self.mu(h, self.memory, goal)
        s = self.lam(self.memory)
        o = self.omega(s)
        u = self.unit(o)
        c = self.coh(u)
        t = self.pt(c)
        return t

    def reset_memory(self):
        """Réinitialise la mémoire entre deux sessions agentiques."""
        self.memory.zero_()



# =====================================================
# Exemple d'utilisation
# =====================================================
if __name__ == "__main__":
    dim = 512
    model = SVCore(dim)

    x = torch.randn(1, dim)
    g = torch.randn(1, dim)

    out = model(x, g)

    print("Sortie SV-Core :")
    print(out)
