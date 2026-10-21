---
title: OpenPeerLLM Demo
emoji: 🕸️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
models:
  - OpenPeerAI/OpenPeerLLM
short_description: Generate and score text with OpenPeerLLM on ZeroGPU
---

# OpenPeerLLM Demo

Text generation and perplexity scoring with
[OpenPeerAI/OpenPeerLLM](https://huggingface.co/OpenPeerAI/OpenPeerLLM), a
decentralised-training language model by Andrew Magdy Kamal Nassief
(DOI [10.57967/hf/6469](https://doi.org/10.57967/hf/6469)).

The space loads OpenPeerLLM's published `model.safetensors` (163M parameters)
tensor for tensor into a reimplementation of its architecture that adds the
causal attention mask the reference forward pass omits. OpenPeerLLM is the
starting rung of the
[My Name is Riemann](https://huggingface.co/OpenPeerAI/My-Name-is-Riemann-Prerelease)
growth ladder.

Maintained by OpenPeer AI. OpenPeerLLM is released under OPNL, MIT and
CC BY 4.0.
