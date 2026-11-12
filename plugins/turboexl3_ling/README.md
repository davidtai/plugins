# turboexl3_ling

Ling-3.1-flash EXL3 packs on mlx-serve's own Ling model (`bailing_hybrid`), as a quant plugin. mlx-serve runs the
forward (KDA, MLA, the grouped MoE, the head). This plugin does two things:

- It runs the trunk's EXL3 linears on turboexl3's Metal kernels.
- It streams the routed experts of layers 4 to 71 from the pack's `experts.bin` into slot rows on the GPU, over the
  plugin kit's expert stream.

It needs an M5-class Mac (the prompt kernels run on the GPU's matrix units).

## Packs

`turboderp/Ling-3.1-flash-exl3` at 2.60bpw or 2.10bpw, repacked for streaming by
`qwen38-exl3-mlx-serve-converter/scripts/ling_exl3_to_streaming.py`. The repack keeps the release's `config.json`, so
mlx-serve knows it by its EXL3 `quantization_config` beside `experts.bin`. The plugin refuses by name a codebook other
than mul1, a pack without its bank files (`experts.bin`, `expert-manifest-v2.json`, `expert-manifest.json`), and
malformed SwiGLU limit lists.

## Install

    scripts/install.sh turboexl3_ling
    mlx-serve --model <pack dir> --ctx-size 32768

The load log names the bank: `[turboexl3_ling] streamed experts: <prompt> prompt / <decode> decode slot rows per layer`.

## What runs where

| Part | Where |
|---|---|
| Trunk linears (KDA, MLA, dense layers 0-3, shared experts, head) | this plugin, turboexl3 `linear` |
| Routed experts, layers 4-71 | this plugin's bank (`open_bank`, src/ling_bank.zig over `kit.streamed`), turboexl3's Ling SwiGLU; Ling-3.1's clamp after the SiLU on layers 65-71 |
| Shared expert's clamp (layers 64-71) | mlx-serve (`share_expert_swiglu_limit_list`) |
| Attention, routing, norms, sampling | mlx-serve's `bailing_hybrid` |

| MTP head's routed experts (layer 72, resident) | this plugin's `swiglu` over format banks, the clamp after the SiLU the host passes (`Swiglu.limit_out`) |

Each routed layer's clamp comes from the host (`BankGeometry.limits_out`) where it passes one, checked against the pack's
`expert_swiglu_limit_list`; on an older host, from the list alone.

## Memory

The bank opens at the room mlx-serve gives it (one 16K + 1K prompt past the loaded weights). The host bills a streamed
model's prompt at 0, so the bank checks the host's real headroom (its KV included) at each phase and refuses by name:
`LingPromptUnderBase` when a prompt pass has less than one slot row of headroom, `LingDecodeUnderBase` when a decode plan
falls more than the 12 rows 4 GiB of free RAM covers under the base (within them it warns and stays at its base).

## Tests

`zig build test` runs the MLX-free tests: the claim and the pack checks (config.zig), the clamp lists and the host's
limits against them, the bank's rows and its memory plans, PF_DFIN's rule and the bank's manifest reader.
