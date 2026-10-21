from __future__ import annotations
import torch
import gradio as gr
import spaces
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from configuration_riemann import RiemannConfig
from modeling_riemann import RiemannForCausalLM
from analysis import (
    token_surprisal,
    surprisal_html,
    greedy_continue,
    contrast_rows,
    md_table,
    WIKI_FEWSHOT,
    CONTRAST_SET,
)

OPENPEER_REPO = "OpenPeerAI/OpenPeerLLM"
RIEMANN_REPO = "OpenPeerAI/My-Name-is-Riemann-Prerelease"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = RiemannConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=1024,
)
openpeer = RiemannForCausalLM(config)
_state = load_file(hf_hub_download(OPENPEER_REPO, "model.safetensors"))
_missing, _unexpected = openpeer.load_state_dict(_state, strict=False)
openpeer.eval()
LOAD_REPORT = f"{len(_state)} tensors loaded from {OPENPEER_REPO}, {len(_missing)} missing, {len(_unexpected)} unexpected."
N_OPENPEER = sum(p.numel() for p in openpeer.parameters())

_riemann = {"model": None}


def riemann():
    if _riemann["model"] is None:
        m = AutoModelForCausalLM.from_pretrained(
            RIEMANN_REPO,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        m.eval()
        _riemann["model"] = m
    return _riemann["model"]


def _gpu(m):
    if m.device.type != "cuda":
        m.to("cuda")


@spaces.GPU(duration=60)
def complete(subject, opening, max_new, compare):
    prompt_tail = f"{subject.strip()} {opening.strip()}".strip()
    _gpu(openpeer)
    a = greedy_continue(
        openpeer,
        tokenizer,
        WIKI_FEWSHOT + prompt_tail,
        int(max_new),
        repetition_penalty=1.3,
    )
    out = (
        f"### OpenPeerLLM ({N_OPENPEER/1e6:.0f}M, the starting rung)\n{prompt_tail}{a}"
    )
    if compare:
        r = riemann()
        _gpu(r)
        b = greedy_continue(
            r,
            tokenizer,
            WIKI_FEWSHOT + prompt_tail,
            int(max_new),
            repetition_penalty=1.25,
            max_context=2048,
        )
        out += f"\n\n### My Name is Riemann Prerelease (13.1B, six rungs later)\n{prompt_tail}{b}"
    return out


@spaces.GPU(duration=60)
def analyse(text, compare):
    _gpu(openpeer)
    a = token_surprisal(openpeer, tokenizer, text)
    out = surprisal_html(a, f"OpenPeerLLM ({N_OPENPEER/1e6:.0f}M):")
    if compare:
        r = riemann()
        _gpu(r)
        b = token_surprisal(r, tokenizer, text, max_length=2048)
        out += "<hr/>" + surprisal_html(b, "My Name is Riemann Prerelease (13.1B):")
        out += f"<p><b>Perplexity {a['ppl']:.1f} to {b['ppl']:.1f}</b> across the growth ladder on this text.</p>"
    return out


@spaces.GPU(duration=90)
def contrast(compare):
    _gpu(openpeer)
    rows_a = contrast_rows(openpeer, tokenizer, CONTRAST_SET)
    if not compare:
        return md_table([[r[0], r[1], r[2]] for r in rows_a])
    r = riemann()
    _gpu(r)
    rows_b = contrast_rows(r, tokenizer, CONTRAST_SET)
    header = ("Text", "OpenPeerLLM 163M", "Riemann Prerelease 13.1B")
    rows = [[ra[0], ra[1], rb[1]] for ra, rb in zip(rows_a, rows_b)]
    return (
        md_table(rows, header)
        + "\n\nPerplexity on each string. Lower is better on fluent text; the gap between fluent text and noise "
        "should widen as the model improves. OpenPeerLLM is the 163M checkpoint that every later rung inherits from."
    )


DESCRIPTION = f"""
# OpenPeerLLM Demo

[OpenPeerAI/OpenPeerLLM](https://huggingface.co/{OPENPEER_REPO}) is a decentralised-training language model
by Andrew Magdy Kamal Nassief (DOI [10.57967/hf/6469](https://doi.org/10.57967/hf/6469)). This space loads its
published `model.safetensors` ({N_OPENPEER/1e6:.0f}M parameters) into a faithful reimplementation of its
architecture, with the causal attention mask the reference forward pass omits. {LOAD_REPORT}

OpenPeerLLM is the **starting rung** of the
[My Name is Riemann](https://huggingface.co/{RIEMANN_REPO}) growth ladder. Every tab lets you put the
13.1B Prerelease next to it on the same input, so the effect of six rungs of growth and training is visible
directly. Both are base models: they complete and score text, they do not chat.
"""

with gr.Blocks(title="OpenPeerLLM Demo") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab("Complete an article"):
        gr.Markdown(
            "Two short example articles are prepended so the model knows the register."
        )
        with gr.Row():
            subject = gr.Textbox(label="Subject", value="The Nile")
            opening = gr.Textbox(
                label="Opening clause",
                value="is a major north-flowing river in northeastern Africa. It",
            )
        with gr.Row():
            max_new = gr.Slider(16, 120, value=64, step=8, label="Max new tokens")
            cmp1 = gr.Checkbox(
                value=True, label="Also run My Name is Riemann Prerelease (13.1B)"
            )
        out1 = gr.Markdown()
        gr.Button("Complete", variant="primary").click(
            complete, [subject, opening, max_new, cmp1], out1
        )
        gr.Examples(
            [
                ["Mount Kilimanjaro", "is a dormant volcano in Tanzania. It"],
                [
                    "The Rosetta Stone",
                    "is a stele inscribed with three versions of a decree issued in",
                ],
            ],
            inputs=[subject, opening],
        )
    with gr.Tab("Analyse text"):
        gr.Markdown(
            "Each token is coloured by how surprised the model was to see it; green is expected, red is surprising."
        )
        a_text = gr.Textbox(
            label="Text",
            lines=4,
            value="The Great Pyramid of Giza is the oldest of the Seven Wonders of the Ancient World.",
        )
        cmp2 = gr.Checkbox(
            value=True, label="Also analyse with My Name is Riemann Prerelease (13.1B)"
        )
        out2 = gr.HTML()
        gr.Button("Analyse", variant="primary").click(analyse, [a_text, cmp2], out2)
    with gr.Tab("Fluency test"):
        gr.Markdown(
            "Six fixed strings from fluent English down to random words, scored by perplexity."
        )
        cmp3 = gr.Checkbox(
            value=True, label="Compare with My Name is Riemann Prerelease (13.1B)"
        )
        out3 = gr.Markdown()
        gr.Button("Run fluency test", variant="primary").click(contrast, cmp3, out3)
    gr.Markdown(
        "OpenPeerLLM is released under OPNL, MIT and CC BY 4.0. Maintained by OpenPeer AI."
    )

demo.queue(max_size=16).launch()
