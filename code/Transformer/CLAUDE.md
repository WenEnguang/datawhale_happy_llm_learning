# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch educational implementation of the core Transformer building blocks in PyTorch. It is part of a learning notebook (companion docs live in the git root: `Transformer架构.md`, `NLP概念.md`). There is no training loop, dataset, or entrypoint yet — only the module definitions and a config dict.

The git repository root is `E:/datawhale`; this code lives in `code/Transformer/`.

## Architecture

Each Transformer component is implemented as a standalone `nn.Module` in its own file, and all modules share one config dict passed by reference.

- `config.py` — defines `model_config`, a single dict of hyperparameters. **This dict is the contract between all modules**; every module's `__init__(self, config)` reads keys from it (e.g. `config['emb_dim']`, `config['n_heads']`).
- `Attention.py` — `MHA`, multi-head self-attention. Separate `q_proj`/`k_proj`/`v_proj`/`out_proj` linears (not fused QKV). Pre-allocates a causal mask buffer of size `[context_length, context_length]` when `is_causal` is set, then slices `[:,:,:l,:l]` to the actual sequence length at runtime.
- `FFN.py` — `FFN`, a two-layer feed-forward network with ReLU (note: classic Transformer style, not SwiGLU/GELU).
- `Norm.py` — both `LayerNorm` (with learnable bias) and `RMSNorm` (no bias), selectable by the user.

There is no assembled `TransformerBlock` or top-level model class yet — the layers have not been wired together.

## Known gotcha: inconsistent config keys

The config dict is not consumed consistently across modules, which is the main thing to watch when editing:

- `config.py` defines `'drop_rate'` (used by `FFN` and nowhere else reads `'dropout'`).
- `Attention.py` reads `config['dropout']` — **this key does not exist in `model_config`**, so instantiating `MHA(model_config)` raises `KeyError` today. If you touch either file, reconcile the key name (`drop_rate` vs `dropout`) rather than papering over it.

There is also no formal test suite, linter, or build step. Verify changes by importing the module and instantiating it with `model_config`, e.g. `python -c "import config, Attention; Attention.MHA(config.model_config)"` (this will surface the `KeyError` above).

## Conventions

- Modules take `config` (a dict) in the constructor — follow this signature when adding new components.
- Tensor layout throughout is `[batch_size, seq_len, hidden_dim]`; attention reshapes to `[batch, n_heads, seq_len, head_dim]`.
- Comments in the code are in Chinese; match the surrounding language when editing.
