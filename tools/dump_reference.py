"""
dump_reference.py

Runs TinyLlama through the reference PyTorch/transformers implementation and
saves every intermediate tensor to disk as .npy files. These files are the
ground truth that the C++ engine's output gets diffed against, layer by
layer, during development.

Not part of the engine itself. This is a one-off tool, run manually whenever
a new reference dump is needed; the engine never calls into Python at runtime.
"""

import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# HuggingFace repo name for the model weights (safetensors format).
# Distinct from the GGUF file used by the C++ engine; transformers cannot
# read GGUF directly, so the same model is downloaded twice, once per format.
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Fixed prompt used for every reference dump. Keeping this constant means
# the .npy files here and the engine's own output are always comparable.
PROMPT = "The capital of France is"

# Output directory for the .npy files. Kept separate from the GGUF model
# file and the venv; this directory is pure generated output.
OUT_DIR = "reference_tensors"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading tokenizer and model...")

# Tokenizer converts text to token IDs and back. Must be TinyLlama's own
# tokenizer, since vocabulary and merge rules are model-specific.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the model with float32 weights explicitly. Checkpoints are often
# stored in float16; forcing float32 here keeps the reference dump at a
# known, consistent precision regardless of the source checkpoint's dtype.
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

# eval() disables training-only behaviour (e.g. dropout), giving
# deterministic output that matches what an inference-only engine should produce.
model.eval()

# Tokenize the prompt. return_tensors="pt" returns a PyTorch tensor rather
# than a plain Python list.
inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"]

print(f"Prompt: {PROMPT!r}")
print(f"Token IDs: {input_ids.tolist()}")


def save(name: str, tensor: torch.Tensor) -> None:
    """Convert a tensor to float32 numpy and write it to OUT_DIR/{name}.npy."""
    # detach() removes gradient-tracking metadata (unused here, and not
    # convertible to numpy directly). to(float32) normalises precision.
    arr = tensor.detach().to(torch.float32).numpy()
    path = os.path.join(OUT_DIR, f"{name}.npy")
    np.save(path, arr)
    print(f"Saved {name}: shape {arr.shape}")


# --- Capture intermediate layer outputs via forward hooks ---
#
# By default a forward pass only exposes the final output. Hooks are
# functions PyTorch calls automatically when a given submodule finishes
# computing, which is how the output of each individual decoder layer is
# captured here without modifying the model's own code.

# Collects one entry per layer: {layer_index: hidden_state_tensor}
layer_outputs: dict[int, torch.Tensor] = {}

# New: collects the RMSNorm output specifically for each layer, captured
# by hooking input_layernorm directly (the RMSNorm applied before
# self-attention), since output_hidden_states only exposes whole-layer
# outputs, not this intermediate step.
rmsnorm_outputs: dict[int, torch.Tensor] = {}


def make_hook(layer_idx: int):
    """
    Returns a hook function bound to a specific layer index via closure.
    A factory is needed because the hook signature (module, input, output)
    gives no direct way to know which layer it was called from.
    """

    def hook(module, input, output):
        # For this model family, a decoder layer returns a tuple whose
        # first element is the hidden state passed to the next layer.
        layer_outputs[layer_idx] = output[0]

    return hook


def make_rmsnorm_hook(layer_idx: int):
    """
    Same idea as make_hook, but for input_layernorm specifically.
    input_layernorm takes one tensor and returns one tensor directly
    (not a tuple), unlike the full decoder layer.
    """

    def hook(module, input, output):
        rmsnorm_outputs[layer_idx] = output

    return hook


# Attach the hook to every decoder layer.
hooks = []
for i, layer in enumerate(model.model.layers):
    handle = layer.register_forward_hook(make_hook(i))
    hooks.append(handle)
    # New: also hook this layer's RMSNorm submodule directly.
    rmsnorm_handle = layer.input_layernorm.register_forward_hook(make_rmsnorm_hook(i))
    hooks.append(rmsnorm_handle)

# Run the forward pass.
# no_grad() disables gradient tracking, unnecessary for inference and
# wasteful to compute.
# output_hidden_states=True additionally returns the hidden state at every
# stage, including immediately after the embedding layer (index 0).
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)

# Detach hooks now that the forward pass is complete, so they don't keep
# firing (and leaking memory) if this model object is reused.
for handle in hooks:
    handle.remove()

# --- Save everything ---

# Token IDs, so the C++ tokenizer's output can be checked against this
# before any numerical computation is compared.
save("input_ids", input_ids)

# hidden_states[0] is the embedding output, before any transformer layer
# has run. Per-layer outputs are already captured separately via the hooks
# above, so only index 0 is needed here.
save("embeddings", outputs.hidden_states[0])

# Each layer's captured output, in order.
for i, tensor in layer_outputs.items():
    save(f"layer_{i}_output", tensor)

# Each layer's RMSNorm output specifically, in order.
for i, tensor in rmsnorm_outputs.items():
    save(f"layer_{i}_rmsnorm_output", tensor)

# The RMSNorm weight itself for layer 0, saved directly so the C++ test
# has something to compare against without re-reading the GGUF file.
save("layer_0_attn_norm_weight", model.model.layers[0].input_layernorm.weight)

# Final logits: unnormalised scores over the vocabulary for the next token,
# at every position in the sequence, before any sampling is applied.
save("logits", outputs.logits)

print("Done.")