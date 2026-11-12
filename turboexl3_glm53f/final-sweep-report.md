# GLM-5.3-Flash final context sweep: 1K to 256K, exact and int8 prefill

The final sweep on the frozen configurations, 2026-11-12: 9 context sizes (1K to 256K, doubling) x 6 configurations x 2 runs = 108 runs. All four windows were VALID and every run served its prompt at its exact size.

## Setup

- Machine: ai-server, Apple M5 Max, 128 GB.
- Model: GLM-5.3-Flash, the streaming EXL3 repack sc_3.10bpw (`GLM-5.3-Flash-streaming-repack-exl3-sc-3.10bpw`).
- Server: mlx-serve kit5main (binary 80ef5168ca33) with the outside plugin turboexl3_glm53f at ac87fe1 (dylib a137149cab54).
- Configurations (frozen 2026-11-11):
  - exact: the 64 accepted plugin switches + `TURBOEXL3_GLM53F_DSA_HOLD=1` + `TURBOEXL3_GLM53F_PF_TAIL_EVAL=1`.
  - int8=N (N = 1, 2, 3, 4): exact + `TURBOEXL3_GLM53F_PF_INT8=N`.
- Workload: the code-ctx prompt family at exact token counts (1,024 to 262,144), `max_tokens` 1,024. Every run decoded to the cap.
- Sampling: greedy (temperature 0) for every configuration, plus exact with the model card's sampling (temperature 1.0, top_p 0.95), "exact card".
- Each run: a fresh server (`--ctx-size` = prompt + 1,024), the pack's files evicted from the page cache, the 40 C cool gate, one streamed request after the model has loaded. Seeds 1234 and 4321. One unmeasured pre-warm run per configuration per window.
- Columns: prefill and decode tok/s from the server's own timings; TTFT and wall time from the client; spec hit % is the server's MTP per-draft acceptance; peak GiB is the server's peak memory footprint.

| window | sizes | runs | start (UTC) | end (UTC) | verdict | guard peak (of 117 GiB) |
|---|---|---|---|---|---|---|
| g53fsw1 | 1K, 2K, 4K | 36 | 2026-11-12T01:37:14Z | 2026-11-12T02:29:44Z | VALID | 109.9 GiB |
| g53fsw2 | 8K, 16K, 32K | 36 | 2026-11-12T03:48:22Z | 2026-11-12T04:52:32Z | VALID | 110.3 GiB |
| g53fsw3 | 64K, 128K | 24 | 2026-11-12T08:47:16Z | 2026-11-12T10:05:49Z | VALID | 108.2 GiB |
| g53fsw4 | 256K | 12 | 2026-11-12T10:08:30Z | 2026-11-12T11:24:43Z | VALID | 108.0 GiB |

Data: `box/decode-lane/i8card/results/g53fsw1..4/` (each window's logs, summary.txt, summary.json and per-run memory phases); `results/final-sweep/all_runs.tsv`, `means.tsv`, `memory.tsv`. On the box: `~/glm53/sweep/g53fsw<N>-1112-*/`.

## Arm means per size

Each cell is the mean of the configuration's 2 runs at that size.

### Prefill tok/s

| size | exact greedy | exact card | int8=1 greedy | int8=2 greedy | int8=3 greedy | int8=4 greedy |
|---|---|---|---|---|---|---|
| 1K | 285.3 | 284.2 | 287.3 | 289.2 | 247.1 | 288.1 |
| 2K | 524.3 | 523.9 | 543.0 | 547.5 | 543.1 | 545.5 |
| 4K | 884.6 | 884.7 | 942.0 | 959.5 | 947.7 | 958.5 |
| 8K | 1,224.5 | 1,222.6 | 1,359.9 | 1,398.3 | 1,435.7 | 1,488.6 |
| 16K | 1,235.5 | 1,236.2 | 1,381.0 | 1,423.2 | 1,545.7 | 1,668.7 |
| 32K | 1,170.5 | 1,171.9 | 1,302.6 | 1,344.1 | 1,460.9 | 1,572.8 |
| 64K | 1,123.1 | 1,122.7 | 1,242.3 | 1,277.7 | 1,391.2 | 1,495.5 |
| 128K | 1,076.5 | 1,073.9 | 1,181.6 | 1,214.6 | 1,319.0 | 1,413.8 |
| 256K | 1,012.4 | 1,012.3 | 1,101.5 | 1,129.4 | 1,220.4 | 1,302.0 |

### Decode tok/s

| size | exact greedy | exact card | int8=1 greedy | int8=2 greedy | int8=3 greedy | int8=4 greedy |
|---|---|---|---|---|---|---|
| 1K | 44.10 | 42.10 | 44.70 | 44.10 | 43.30 | 43.80 |
| 2K | 41.90 | 41.70 | 41.85 | 42.90 | 41.85 | 43.05 |
| 4K | 43.55 | 42.60 | 42.40 | 43.85 | 42.90 | 42.15 |
| 8K | 42.10 | 43.30 | 40.70 | 43.05 | 40.45 | 41.85 |
| 16K | 40.90 | 41.40 | 43.45 | 43.00 | 42.90 | 40.85 |
| 32K | 42.10 | 41.10 | 40.45 | 40.20 | 40.90 | 41.80 |
| 64K | 39.95 | 38.95 | 39.25 | 38.80 | 39.55 | 40.00 |
| 128K | 36.70 | 36.20 | 35.80 | 35.55 | 35.55 | 35.55 |
| 256K | 30.70 | 29.80 | 30.20 | 30.25 | 30.30 | 30.70 |

### TTFT s

| size | exact greedy | exact card | int8=1 greedy | int8=2 greedy | int8=3 greedy | int8=4 greedy |
|---|---|---|---|---|---|---|
| 1K | 3.60 | 3.62 | 3.57 | 3.55 | 4.26 | 3.56 |
| 2K | 3.92 | 3.92 | 3.78 | 3.75 | 3.78 | 3.76 |
| 4K | 4.64 | 4.63 | 4.36 | 4.28 | 4.33 | 4.29 |
| 8K | 6.70 | 6.71 | 6.04 | 5.87 | 5.71 | 5.51 |
| 16K | 13.27 | 13.27 | 11.88 | 11.52 | 10.61 | 9.83 |
| 32K | 28.01 | 27.98 | 25.17 | 24.39 | 22.45 | 20.85 |
| 64K | 58.38 | 58.39 | 52.77 | 51.31 | 47.12 | 43.84 |
| 128K | 121.79 | 122.09 | 110.95 | 107.94 | 99.40 | 92.74 |
| 256K | 258.99 | 259.02 | 238.05 | 232.16 | 214.85 | 201.39 |

### Wall time s

| size | exact greedy | exact card | int8=1 greedy | int8=2 greedy | int8=3 greedy | int8=4 greedy |
|---|---|---|---|---|---|---|
| 1K | 26.91 | 28.05 | 26.55 | 26.82 | 27.98 | 27.01 |
| 2K | 28.40 | 28.54 | 28.34 | 27.69 | 28.34 | 27.63 |
| 4K | 28.21 | 28.74 | 28.55 | 27.69 | 28.27 | 28.65 |
| 8K | 31.09 | 30.44 | 31.27 | 29.73 | 31.09 | 30.02 |
| 16K | 38.37 | 38.05 | 35.50 | 35.41 | 34.55 | 34.96 |
| 32K | 52.38 | 52.98 | 50.55 | 49.92 | 47.56 | 45.41 |
| 64K | 84.09 | 84.78 | 78.97 | 77.79 | 73.12 | 69.55 |
| 128K | 149.81 | 150.46 | 139.63 | 136.85 | 128.34 | 121.68 |
| 256K | 292.49 | 293.50 | 272.11 | 266.17 | 248.81 | 234.86 |

### Spec hit %

| size | exact greedy | exact card | int8=1 greedy | int8=2 greedy | int8=3 greedy | int8=4 greedy |
|---|---|---|---|---|---|---|
| 1K | 80.3 | 79.8 | 81.5 | 75.4 | 71.9 | 79.4 |
| 2K | 77.3 | 74.5 | 78.8 | 74.3 | 75.1 | 77.2 |
| 4K | 81.5 | 79.0 | 73.8 | 81.4 | 75.3 | 73.2 |
| 8K | 73.8 | 79.1 | 70.7 | 81.5 | 69.5 | 75.1 |
| 16K | 68.9 | 77.8 | 81.5 | 79.5 | 78.4 | 69.1 |
| 32K | 79.8 | 78.5 | 72.1 | 74.4 | 74.9 | 80.7 |
| 64K | 81.8 | 75.5 | 79.4 | 71.0 | 79.9 | 81.8 |
| 128K | 79.4 | 77.7 | 76.9 | 73.6 | 76.9 | 75.5 |
| 256K | 78.4 | 76.9 | 80.4 | 73.1 | 78.4 | 76.5 |

### Peak GiB (server)

| size | exact greedy | exact card | int8=1 greedy | int8=2 greedy | int8=3 greedy | int8=4 greedy |
|---|---|---|---|---|---|---|
| 1K | 98.1 | 98.1 | 97.9 | 98.2 | 98.2 | 98.1 |
| 2K | 98.2 | 98.2 | 98.2 | 98.2 | 98.2 | 98.2 |
| 4K | 98.3 | 98.2 | 98.2 | 98.2 | 98.3 | 98.0 |
| 8K | 98.3 | 98.3 | 98.3 | 98.3 | 98.3 | 98.3 |
| 16K | 98.5 | 98.3 | 98.4 | 98.3 | 98.5 | 98.5 |
| 32K | 98.3 | 98.3 | 98.4 | 98.3 | 98.3 | 98.3 |
| 64K | 98.6 | 98.5 | 98.3 | 98.7 | 98.7 | 98.3 |
| 128K | 97.2 | 97.2 | 97.0 | 97.2 | 97.2 | 97.2 |
| 256K | 94.2 | 94.2 | 94.2 | 94.2 | 94.2 | 94.2 |

## Every run

| size | config | seed | prefill tok/s | decode tok/s | TTFT s | wall s | spec hit % | peak GiB | text sha |
|---|---|---|---|---|---|---|---|---|---|
| 1K | exact greedy | 1234 | 285.3 | 43.30 | 3.60 | 27.33 | 80.3 | 98.1 | `87b8a21b6733` |
| 1K | exact greedy | 4321 | 285.2 | 44.90 | 3.60 | 26.49 | 80.3 | 98.1 | `87b8a21b6733` |
| 1K | exact card | 1234 | 283.8 | 40.30 | 3.62 | 29.11 | 77.3 | 98.1 | `94b56db8ffd3` |
| 1K | exact card | 4321 | 284.6 | 43.90 | 3.61 | 27.00 | 82.2 | 98.1 | `c1e46126b239` |
| 1K | int8=1 greedy | 1234 | 287.4 | 44.60 | 3.57 | 26.60 | 81.5 | 97.7 | `43db5f41f040` |
| 1K | int8=1 greedy | 4321 | 287.2 | 44.80 | 3.57 | 26.51 | 81.5 | 98.1 | `43db5f41f040` |
| 1K | int8=2 greedy | 1234 | 289.3 | 44.10 | 3.55 | 26.83 | 75.4 | 98.2 | `1b777c646abe` |
| 1K | int8=2 greedy | 4321 | 289.2 | 44.10 | 3.55 | 26.82 | 75.4 | 98.2 | `1b777c646abe` |
| 1K | int8=3 greedy | 1234 | 287.9 | 43.10 | 3.56 | 27.40 | 71.9 | 98.2 | `6a4413cda0e7` |
| 1K | int8=3 greedy | 4321 | 206.4 | 43.50 | 4.97 | 28.55 | 71.9 | 98.2 | `6a4413cda0e7` |
| 1K | int8=4 greedy | 1234 | 288.1 | 43.80 | 3.56 | 27.01 | 79.4 | 98.1 | `b49ae08410e9` |
| 1K | int8=4 greedy | 4321 | 288.1 | 43.80 | 3.56 | 27.00 | 79.4 | 98.1 | `b49ae08410e9` |
| 2K | exact greedy | 1234 | 524.5 | 42.70 | 3.91 | 27.93 | 77.3 | 98.2 | `31b038e1e2ee` |
| 2K | exact greedy | 4321 | 524.1 | 41.10 | 3.92 | 28.87 | 77.3 | 98.2 | `31b038e1e2ee` |
| 2K | exact card | 1234 | 523.7 | 41.10 | 3.92 | 28.87 | 69.0 | 98.2 | `36f59411d8e9` |
| 2K | exact card | 4321 | 524.1 | 42.30 | 3.92 | 28.20 | 79.9 | 98.2 | `af6252f0e7f3` |
| 2K | int8=1 greedy | 1234 | 544.1 | 43.20 | 3.77 | 27.54 | 78.8 | 98.2 | `1542af67cc9a` |
| 2K | int8=1 greedy | 4321 | 541.9 | 40.50 | 3.79 | 29.15 | 78.8 | 98.2 | `1542af67cc9a` |
| 2K | int8=2 greedy | 1234 | 547.4 | 42.90 | 3.75 | 27.68 | 74.3 | 98.2 | `5c479415a46d` |
| 2K | int8=2 greedy | 4321 | 547.6 | 42.90 | 3.75 | 27.69 | 74.3 | 98.2 | `5c479415a46d` |
| 2K | int8=3 greedy | 1234 | 542.7 | 43.00 | 3.78 | 27.64 | 75.1 | 98.2 | `e8bbcbea02e9` |
| 2K | int8=3 greedy | 4321 | 543.5 | 40.70 | 3.78 | 29.03 | 75.1 | 98.2 | `e8bbcbea02e9` |
| 2K | int8=4 greedy | 1234 | 545.0 | 43.00 | 3.77 | 27.65 | 77.2 | 98.2 | `9da60f542994` |
| 2K | int8=4 greedy | 4321 | 546.0 | 43.10 | 3.76 | 27.61 | 77.2 | 98.2 | `9da60f542994` |
| 4K | exact greedy | 1234 | 885.0 | 43.50 | 4.64 | 28.23 | 81.5 | 98.3 | `dea1545da9e4` |
| 4K | exact greedy | 4321 | 884.1 | 43.60 | 4.64 | 28.20 | 81.5 | 98.3 | `dea1545da9e4` |
| 4K | exact card | 1234 | 885.2 | 42.70 | 4.63 | 28.65 | 78.5 | 98.3 | `412f3501f8dc` |
| 4K | exact card | 4321 | 884.1 | 42.50 | 4.64 | 28.83 | 79.4 | 98.2 | `7fa1e4b44a77` |
| 4K | int8=1 greedy | 1234 | 941.4 | 42.40 | 4.36 | 28.55 | 73.8 | 98.2 | `98e3385a7292` |
| 4K | int8=1 greedy | 4321 | 942.7 | 42.40 | 4.35 | 28.56 | 73.8 | 98.2 | `98e3385a7292` |
| 4K | int8=2 greedy | 1234 | 961.1 | 43.90 | 4.27 | 27.66 | 81.4 | 98.2 | `9b64ebcd3d04` |
| 4K | int8=2 greedy | 4321 | 957.9 | 43.80 | 4.28 | 27.71 | 81.4 | 98.2 | `9b64ebcd3d04` |
| 4K | int8=3 greedy | 1234 | 946.4 | 42.90 | 4.34 | 28.25 | 75.3 | 98.3 | `247afef91361` |
| 4K | int8=3 greedy | 4321 | 949.0 | 42.90 | 4.32 | 28.28 | 75.3 | 98.3 | `247afef91361` |
| 4K | int8=4 greedy | 1234 | 957.7 | 42.10 | 4.29 | 28.69 | 73.2 | 97.8 | `36eb4410072e` |
| 4K | int8=4 greedy | 4321 | 959.3 | 42.20 | 4.28 | 28.61 | 73.2 | 98.2 | `36eb4410072e` |
| 8K | exact greedy | 1234 | 1,227.1 | 42.10 | 6.69 | 31.09 | 73.8 | 98.3 | `830e1d68412d` |
| 8K | exact greedy | 4321 | 1,222.0 | 42.10 | 6.71 | 31.10 | 73.8 | 98.3 | `830e1d68412d` |
| 8K | exact card | 1234 | 1,221.8 | 42.90 | 6.71 | 30.67 | 78.6 | 98.3 | `d158587cc10a` |
| 8K | exact card | 4321 | 1,223.4 | 43.70 | 6.70 | 30.21 | 79.6 | 98.3 | `e0599eb7837f` |
| 8K | int8=1 greedy | 1234 | 1,359.2 | 41.50 | 6.04 | 30.81 | 70.7 | 98.3 | `d35fa319fbc9` |
| 8K | int8=1 greedy | 4321 | 1,360.6 | 39.90 | 6.03 | 31.73 | 70.7 | 98.3 | `d35fa319fbc9` |
| 8K | int8=2 greedy | 1234 | 1,398.2 | 43.10 | 5.87 | 29.72 | 81.5 | 98.3 | `c8da95ceb783` |
| 8K | int8=2 greedy | 4321 | 1,398.4 | 43.00 | 5.87 | 29.73 | 81.5 | 98.3 | `c8da95ceb783` |
| 8K | int8=3 greedy | 1234 | 1,434.7 | 40.40 | 5.72 | 31.11 | 69.5 | 98.3 | `477e914fac98` |
| 8K | int8=3 greedy | 4321 | 1,436.7 | 40.50 | 5.71 | 31.06 | 69.5 | 98.3 | `477e914fac98` |
| 8K | int8=4 greedy | 1234 | 1,490.3 | 41.90 | 5.51 | 29.98 | 75.1 | 98.3 | `8d2532c1eb0a` |
| 8K | int8=4 greedy | 4321 | 1,487.0 | 41.80 | 5.52 | 30.06 | 75.1 | 98.3 | `8d2532c1eb0a` |
| 16K | exact greedy | 1234 | 1,235.1 | 40.90 | 13.28 | 38.38 | 68.9 | 98.5 | `5af2588141a8` |
| 16K | exact greedy | 4321 | 1,236.0 | 40.90 | 13.27 | 38.35 | 68.9 | 98.5 | `5af2588141a8` |
| 16K | exact card | 1234 | 1,236.7 | 41.50 | 13.26 | 37.98 | 77.6 | 98.5 | `0e840955c7dd` |
| 16K | exact card | 4321 | 1,235.8 | 41.30 | 13.27 | 38.11 | 78.1 | 98.1 | `5226052b7073` |
| 16K | int8=1 greedy | 1234 | 1,382.3 | 43.40 | 11.86 | 35.51 | 81.5 | 98.4 | `4df948a3328d` |
| 16K | int8=1 greedy | 4321 | 1,379.8 | 43.50 | 11.89 | 35.49 | 81.5 | 98.4 | `4df948a3328d` |
| 16K | int8=2 greedy | 1234 | 1,421.9 | 42.90 | 11.53 | 35.48 | 79.5 | 98.1 | `bc5176b4a950` |
| 16K | int8=2 greedy | 4321 | 1,424.6 | 43.10 | 11.51 | 35.33 | 79.5 | 98.4 | `bc5176b4a950` |
| 16K | int8=3 greedy | 1234 | 1,538.5 | 42.90 | 10.66 | 34.59 | 78.4 | 98.5 | `4f599dfa0e11` |
| 16K | int8=3 greedy | 4321 | 1,552.8 | 42.90 | 10.56 | 34.51 | 78.4 | 98.5 | `4f599dfa0e11` |
| 16K | int8=4 greedy | 1234 | 1,668.4 | 40.90 | 9.83 | 34.95 | 69.1 | 98.4 | `3540089ae6a0` |
| 16K | int8=4 greedy | 4321 | 1,668.9 | 40.80 | 9.83 | 34.97 | 69.1 | 98.5 | `3540089ae6a0` |
| 32K | exact greedy | 1234 | 1,168.7 | 42.10 | 28.05 | 52.42 | 79.8 | 98.3 | `8ff84e8f1b7f` |
| 32K | exact greedy | 4321 | 1,172.4 | 42.10 | 27.96 | 52.33 | 79.8 | 98.3 | `8ff84e8f1b7f` |
| 32K | exact card | 1234 | 1,172.8 | 40.70 | 27.95 | 53.21 | 76.4 | 98.3 | `b3afe91a7d62` |
| 32K | exact card | 4321 | 1,171.1 | 41.50 | 28.00 | 52.75 | 80.5 | 98.3 | `857201438328` |
| 32K | int8=1 greedy | 1234 | 1,302.5 | 40.40 | 25.17 | 50.57 | 72.1 | 98.4 | `99236745861f` |
| 32K | int8=1 greedy | 4321 | 1,302.7 | 40.50 | 25.17 | 50.53 | 72.1 | 98.4 | `99236745861f` |
| 32K | int8=2 greedy | 1234 | 1,343.3 | 40.20 | 24.41 | 49.94 | 74.4 | 98.3 | `b64823f0af3b` |
| 32K | int8=2 greedy | 4321 | 1,345.0 | 40.20 | 24.38 | 49.91 | 74.4 | 98.3 | `b64823f0af3b` |
| 32K | int8=3 greedy | 1234 | 1,462.0 | 40.90 | 22.43 | 47.55 | 74.9 | 98.3 | `016d384c53f2` |
| 32K | int8=3 greedy | 4321 | 1,459.7 | 40.90 | 22.46 | 47.58 | 74.9 | 98.3 | `016d384c53f2` |
| 32K | int8=4 greedy | 1234 | 1,572.1 | 41.80 | 20.86 | 45.42 | 80.7 | 98.3 | `d7ccfe5753d1` |
| 32K | int8=4 greedy | 4321 | 1,573.5 | 41.80 | 20.84 | 45.41 | 80.7 | 98.3 | `d7ccfe5753d1` |
| 64K | exact greedy | 1234 | 1,122.6 | 40.00 | 58.40 | 84.11 | 81.8 | 98.6 | `a9605c66e4b1` |
| 64K | exact greedy | 4321 | 1,123.6 | 39.90 | 58.35 | 84.08 | 81.8 | 98.5 | `a9605c66e4b1` |
| 64K | exact card | 1234 | 1,123.1 | 39.40 | 58.37 | 84.46 | 75.1 | 98.6 | `150d97e90039` |
| 64K | exact card | 4321 | 1,122.3 | 38.50 | 58.42 | 85.11 | 76.0 | 98.3 | `87e315c9409f` |
| 64K | int8=1 greedy | 1234 | 1,242.0 | 39.30 | 52.78 | 78.97 | 79.4 | 98.3 | `913731e0003e` |
| 64K | int8=1 greedy | 4321 | 1,242.6 | 39.20 | 52.76 | 78.98 | 79.4 | 98.3 | `913731e0003e` |
| 64K | int8=2 greedy | 1234 | 1,276.7 | 38.80 | 51.35 | 77.85 | 71.0 | 98.7 | `1202a29231b6` |
| 64K | int8=2 greedy | 4321 | 1,278.6 | 38.80 | 51.28 | 77.74 | 71.0 | 98.7 | `1202a29231b6` |
| 64K | int8=3 greedy | 1234 | 1,390.8 | 39.50 | 47.14 | 73.14 | 79.9 | 98.7 | `ed298452964b` |
| 64K | int8=3 greedy | 4321 | 1,391.6 | 39.60 | 47.11 | 73.09 | 79.9 | 98.7 | `ed298452964b` |
| 64K | int8=4 greedy | 1234 | 1,495.6 | 40.00 | 43.84 | 69.55 | 81.8 | 98.3 | `ff9e5f0b05e7` |
| 64K | int8=4 greedy | 4321 | 1,495.3 | 40.00 | 43.85 | 69.54 | 81.8 | 98.3 | `ff9e5f0b05e7` |
| 128K | exact greedy | 1234 | 1,076.7 | 36.70 | 121.77 | 149.79 | 79.4 | 97.2 | `c9de460617c5` |
| 128K | exact greedy | 4321 | 1,076.3 | 36.70 | 121.81 | 149.83 | 79.4 | 97.1 | `c9de460617c5` |
| 128K | exact card | 1234 | 1,071.0 | 36.40 | 122.41 | 150.66 | 78.6 | 97.2 | `f78644b77738` |
| 128K | exact card | 4321 | 1,076.8 | 36.00 | 121.76 | 150.26 | 76.8 | 97.2 | `615775458b76` |
| 128K | int8=1 greedy | 1234 | 1,181.6 | 35.80 | 110.96 | 139.65 | 76.9 | 97.0 | `4c3a343a7422` |
| 128K | int8=1 greedy | 4321 | 1,181.7 | 35.80 | 110.95 | 139.61 | 76.9 | 97.0 | `4c3a343a7422` |
| 128K | int8=2 greedy | 1234 | 1,215.0 | 35.60 | 107.91 | 136.79 | 73.6 | 97.2 | `1c5a794d7cd5` |
| 128K | int8=2 greedy | 4321 | 1,214.2 | 35.50 | 107.98 | 136.92 | 73.6 | 97.2 | `1c5a794d7cd5` |
| 128K | int8=3 greedy | 1234 | 1,319.8 | 36.40 | 99.34 | 127.56 | 76.9 | 97.2 | `1a772cd35a75` |
| 128K | int8=3 greedy | 4321 | 1,318.2 | 34.70 | 99.46 | 129.11 | 76.9 | 97.2 | `1a772cd35a75` |
| 128K | int8=4 greedy | 1234 | 1,413.4 | 36.10 | 92.76 | 121.26 | 75.5 | 97.2 | `cf20c314ed97` |
| 128K | int8=4 greedy | 4321 | 1,414.1 | 35.00 | 92.72 | 122.09 | 75.5 | 97.2 | `cf20c314ed97` |
| 256K | exact greedy | 1234 | 1,012.6 | 30.70 | 258.94 | 292.44 | 78.4 | 94.2 | `67627cb802fd` |
| 256K | exact greedy | 4321 | 1,012.2 | 30.70 | 259.03 | 292.54 | 78.4 | 94.2 | `67627cb802fd` |
| 256K | exact card | 1234 | 1,012.6 | 29.70 | 258.93 | 293.53 | 74.7 | 94.2 | `6053b9fae5ca` |
| 256K | exact card | 4321 | 1,011.9 | 29.90 | 259.12 | 293.47 | 79.1 | 94.2 | `9a7338428180` |
| 256K | int8=1 greedy | 1234 | 1,101.0 | 30.20 | 238.15 | 272.24 | 80.4 | 94.2 | `2a70223b5f03` |
| 256K | int8=1 greedy | 4321 | 1,102.0 | 30.20 | 237.94 | 271.97 | 80.4 | 94.2 | `2a70223b5f03` |
| 256K | int8=2 greedy | 1234 | 1,129.6 | 30.30 | 232.13 | 266.09 | 73.1 | 94.2 | `44d25208c545` |
| 256K | int8=2 greedy | 4321 | 1,129.3 | 30.20 | 232.19 | 266.26 | 73.1 | 94.2 | `44d25208c545` |
| 256K | int8=3 greedy | 1234 | 1,221.0 | 30.30 | 214.74 | 248.71 | 78.4 | 94.2 | `4c8c1219ca1f` |
| 256K | int8=3 greedy | 4321 | 1,219.8 | 30.30 | 214.96 | 248.90 | 78.4 | 94.2 | `4c8c1219ca1f` |
| 256K | int8=4 greedy | 1234 | 1,301.6 | 30.60 | 201.45 | 235.05 | 76.5 | 94.2 | `baed3cd9beb7` |
| 256K | int8=4 greedy | 4321 | 1,302.4 | 30.80 | 201.33 | 234.67 | 76.5 | 94.2 | `baed3cd9beb7` |

## Memory: the plan against the peaks

The plugin's memory plan (BILL_FIT) sizes the streamed-expert rows for the server's context so the box's non-file memory peaks at the 110 GiB ceiling. Per size, over all 12 runs: the plan's rows and its growth past the 16K + 1K cell, the memory in use outside the server at load (FIT_INUSE), the server's peak footprint, and the box's highest use (wired + anonymous - purgeable + compressor) in the prompt and decode phases, with the room left under the ceiling.

| size | ctx | rows prompt/decode | plan growth prompt/decode MiB | outside at load GiB | server peak GiB | box prompt-phase GiB | box decode-phase GiB | room prompt / decode GiB |
|---|---|---|---|---|---|---|---|---|
| 1K | 2,048 | 185/221 (11 runs), 185/220 (1 run) | 0 / 0 | 7.93-8.16 | 98.2 | 97.9 | 108.8 | 12.1 / 1.2 |
| 2K | 3,072 | 185/221 | 0 / 0 | 7.94-7.95 | 98.2 | 98.8 | 109.1 | 11.2 / 0.9 |
| 4K | 5,120 | 185/221 (11 runs), 185/220 (1 run) | 0 / 0 | 7.94-8.16 | 98.3 | 99.7 | 108.9 | 10.3 / 1.1 |
| 8K | 9,216 | 185/221 | 0 / 0 | 7.98-8.01 | 98.3 | 102.0 | 109.2 | 8.0 / 0.8 |
| 16K | 17,408 | 185/221 (10 runs), 184/220 (2 runs) | 0 / 0 | 7.99-8.22 | 98.5 | 107.3 | 109.5 | 2.7 / 0.5 |
| 32K | 33,792 | 181/219 | 1,495 / 835 | 8.00-8.01 | 98.4 | 105.9 | 109.6 | 4.1 / 0.4 |
| 64K | 66,560 | 174/215 | 4,487 / 2,507 | 7.72-7.78 | 98.7 | 104.4 | 106.9 | 5.6 / 3.1 |
| 128K | 132,096 | 158/206 | 10,471 / 5,851 | 7.76-8.00 | 97.2 | 100.0 | 105.4 | 10.0 / 4.6 |
| 256K | 263,168 | 127/189 | 22,263 / 12,363 | 7.78-7.81 | 94.2 | 92.1 | 102.5 | 17.9 / 7.5 |

The peak is in the decode phase at every size. Up to 32K the decode phase came within 0.4-1.2 GiB of the ceiling. From 64K the plan leaves room unused: in the decode phase 3.1 GiB at 64K, 4.6 at 128K and 7.5 at 256K; in the prompt phase 10.0 GiB at 128K and 17.9 at 256K. Its indexer term reserves more than the DSA_GLM kernels use (the prompt phase), and the decode phase's reserve also grows faster than its use; the cause of the latter is not yet identified.

## Notes

- One outlier: int8=3 at 1K, seed 4321, prefilled at 206.4 tok/s (TTFT 4.97 s); its seed-1234 twin ran 287.9 tok/s (3.56 s) with the same text. Both runs read the same 49.245 GB of experts (5,160 misses, 10,318 preadv calls); the slow run's reads took 5,116 ms of wall time against 3,665 ms. That is the SSD reading at about 9.6 GB/s instead of 13.4 GB/s; the 1.45 s difference is the TTFT gap. The 1K int8=3 means include it.
- 4 runs got one row fewer than the others at their size, because the memory in use outside the server at load was higher and FIT_INUSE planned for it: 1K int8=1 greedy seed 1234 (185/220; 8.16 GiB outside against the size's median 7.94); 4K int8=4 greedy seed 1234 (185/220; 8.16 GiB outside against the size's median 7.94); 16K exact card seed 4321 (184/220; 8.22 GiB outside against the size's median 7.99); 16K int8=2 greedy seed 1234 (184/220; 8.22 GiB outside against the size's median 7.99).
- Every greedy configuration decoded the same text with both seeds. The exact greedy and int8=4 texts at 32K-256K match the calibration window g53cal2's (2026-11-12 00:05Z). Each exact card run decoded its own text.
- int8=2 (per-token scales) was served for the first time in g53fsw1 and printed its engage line in all 18 of its runs.
- Every run decoded to the 1,024-token cap (finish `length`).
