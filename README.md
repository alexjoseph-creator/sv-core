# SV-Core — Teleological Cognitive Architecture for Agentic LLMs

**Author:** Alexandre Vinas (2025)  
**Scientific Publication (Zenodo):** https://doi.org/10.5281/zenodo.17901727

---

## 🧠 Overview

SV-Core is a minimal, modular teleological architecture designed to add goal-oriented cognitive structure to Large Language Models (LLMs) without modifying model weights.

This prototype provides:

- goal-conditioned memory (μ-TEL)  
- structural semantic transformation (Λ)  
- orientation shaping (Ω*)  
- internal stabilization (⦿, CΩ)  
- phase-transition dynamics (PTOr)

SV-Core is a **fully executable subset** of a larger cognitive framework (the Living System, SV), published as a standalone research prototype.

---

## 🧪 Features

- Clean, ~200-line PyTorch implementation  
- No retraining or fine-tuning required  
- Integrates with any transformer hidden states (LLaMA, Mistral, Phi, Qwen…)  
- Reproducible, interpretable, modular  
- Backed by a peer-review-ready scientific paper

---

## 🔧 Installation

```bash
git clone https://github.com/alexjoseph-creator/sv-core
cd sv-core
pip install -r requirements.txt

---

import torch
from sv_core import SVCore

model = SVCore(dim=512)

x = torch.randn(1, 512)
goal = torch.randn(1, 512)

output = model(x, goal)
print(output)

---

## 📄 Scientific Paper

The full theoretical and mathematical details are available in:

**Vinas, Alexandre (2025). SV-Core: A Teleological Cognitive Architecture for Agentic Large Language Models.**  
📌 Zenodo: https://doi.org/10.5281/zenodo.17901727

---

sv-core/
├── sv_core.py                # Core implementation
├── sv_core_llama.py          # LLaMA integration template
├── examples/
│   └── demo.py
├── paper/
│   └── SV-Core_LaTeX.tex     # LaTeX source of the paper
├── LICENSE
└── README.md

---

## 📄 License

This project is released under the **MIT License**.

---

## 📬 Contact

For research collaboration or access to extended SV architecture modules:  
📧 cine4ever66@gmail.com

---
