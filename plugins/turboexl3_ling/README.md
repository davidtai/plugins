# turboexl3_ling

Ling-3.1-flash EXL3 packs on mlx-serve's own Ling model (`bailing_hybrid`), as a quant plugin. mlx-serve runs the
forward (KDA, MLA, the grouped MoE, the head). This plugin does two things:

- It runs the trunk's EXL3 linears on turboexl3's Metal kernels.
- It streams the routed experts of layers 4 to 71 from the pack's `experts.bin` into slot rows on the GPU.

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
| Routed experts, layers 4-71 | this plugin's bank (`open_bank`), turboexl3's Ling SwiGLU; Ling-3.1's clamp after the SiLU on layers 65-71 |
| Shared expert's clamp (layers 64-71) | mlx-serve (`share_expert_swiglu_limit_list`) |
| Attention, routing, norms, sampling | mlx-serve's `bailing_hybrid` |

No MTP head yet: mlx-serve does not load Ling's NextN layer.

## Tests

`zig build test` runs the MLX-free tests: the claim and the pack checks (config.zig), the clamp lists, the bank's rows
and PF_DFIN's rule.
