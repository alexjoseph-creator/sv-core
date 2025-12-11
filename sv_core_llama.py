"""
SV-Core + LLaMA Integration Template
Author: Alexandre Vinas (2025)

This file shows how to connect SV-Core to a LLaMA-like transformer model.
It is a minimal template meant to be extended.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from sv_core import SVCore

# Hidden dimensions for known models
MODEL_DIMS = {
    "NousResearch/Llama-2-7b-chat-hf": 4096,
    "mistralai/Mistral-7B-v0.1": 4096,
    "microsoft/phi-3-mini-4k-instruct": 3072,
}


class SVLlamaAgent:
    def __init__(self, model_path="NousResearch/Llama-2-7b-chat-hf"):
        # Determine hidden dimension based on model
        self.dim = MODEL_DIMS.get(model_path, 4096)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load LLM backbone (AutoModel → access hidden states)
        self.llm = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Instantiate SV-Core with matching hidden dimension
        self.sv = SVCore(self.dim)

    def encode(self, text):
        """Encode text into a hidden state vector."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.llm.device)

        with torch.no_grad():
            outputs = self.llm(**inputs, output_hidden_states=True)

        # Extract last hidden state of last token
        hidden = outputs.last_hidden_state[:, -1, :]  # shape: [1, dim]
        return hidden

    def forward(self, prompt, goal_text):
        """Process a prompt and a goal using the SV-Core teleological pipeline."""
        llama_state = self.encode(prompt)
        goal_vector = self.encode(goal_text)

        # SV-Core processing
        sv_state = self.sv(llama_state, goal_vector)

        # Simple fusion: enrich LLM with SV-Core output
        fused = llama_state + sv_state

        return fused

    def reset(self):
        """Reset SV-Core memory between sessions."""
        self.sv.reset_memory()


# =====================================================
# Example usage
# =====================================================
if __name__ == "__main__":
    print("SV-Core + LLaMA Template Loaded")
    print("Install dependencies: pip install transformers accelerate")
    print()
    print("Example:")
    print("  agent = SVLlamaAgent()")
    print("  out = agent.forward('Bonjour', 'Réserver un vol')")
