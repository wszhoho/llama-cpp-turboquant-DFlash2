# llama-cpp-turboquant-DFlash2

LLM inference in C/C++ - a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) focused on **extreme KV-cache compression (TurboQuant)** and **fast speculative decoding (DFlash / DFlash2 / DSpark)** for DeepSeek-V4 class models.

[English](README.md) | [简体中文](README.zh-CN.md)

---

## Highlights

### 1. TurboQuant - KV-cache quantization down to 2-bit

KV cache is usually the memory bottleneck for long-context inference. TurboQuant compresses it aggressively with a **WHT rotation + PolarQuant** scheme (based on [arXiv 2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026), adding three new GGUF types:

| Type | Bits | GGML enum | CLI name |
|------|------|-----------|----------|
| TURBO2_0 | 2-bit | `GGML_TYPE_TURBO2_0` (43) | `turbo2` |
| TURBO3_0 | 3-bit | `GGML_TYPE_TURBO3_0` (44) | `turbo3` |
| TURBO4_0 | 4-bit | `GGML_TYPE_TURBO4_0` (45) | `turbo4` |

Key implementation points (CUDA, `ggml/src/ggml-cuda/`):

- **WHT rotation** - Fast Walsh-Hadamard Transform (128/64 elements) with two seed-42 sign arrays, spreads outliers before quantization (`turbo-wht.cu/h`)
- **Lloyd-Max centroids** - 2/3/4-bit codebooks optimized for the `N(0, 1/128)` distribution (`turbo-quant.cuh`)
- **InnerQ channel equalization** (`TURBO_INNERQ=N` env var, N = calibration tokens) - per-channel variance equalization that preserves the dot product `<Q/s, s*K> = <Q, K>` (`turbo-innerq.cu/h`)
- **Fused Flash Attention** - `turbo2/3/4` KV types supported directly in the FA kernels, with 21 generated template instances combing turbo/f16/q8_0 pairs (`template-instances/`)
- **CPU support** - scalar vec-dot and the WHT op (`ggml_compute_forward_turbo_wht`) so the types also run without a GPU

Enable via the standard cache-type flags:

```sh
llama-cli -m model.gguf -n 512 --cache-type-k turbo3 --cache-type-v turbo3

# optionally calibrate InnerQ channel scales with a few thousand tokens
TURBO_INNERQ=4096 llama-cli -m model.gguf -n 512 --cache-type-k turbo3 --cache-type-v turbo3
```

### 2. DFlash / DFlash2 / DSpark - speculative decoding drafts

DFlash is a *draft* model architecture that predicts the next tokens of a target model (e.g. DeepSeek-V4) from a small number of extracted target hidden states, enabling fast speculative decoding with a cheap draft pass.

- **DFlash** - encoder-decoder draft: a feature-fusion layer (`fc` + norm) maps extracted target-layer features into the draft embedding space (`src/models/dflash.cpp`)
- **DFlash2** - adds a temporal **convolution** + **selector** head (`conv_kernel_size`, `conv_group_size`, `selector_rank`, `selector_top_k` in GGUF metadata) for higher acceptance rates
- **DSpark** - DFlash + a semi-autoregressive **Markov head** and a **confidence head**, optionally running full DeepSeek-V4 blocks with a uniform sliding-window draft KV ring (`llama-kv-cache-dsv4.cpp`)
- **Reduced-vocab drafts** - a `d2t` mapping lets drafts use a small vocab and map results back to target token ids

Converting a DFlash draft requires the target model directory (to reuse its tokenizer/vocab):

```sh
python convert_hf_to_gguf.py <draft-model-dir> --target-model-dir <target-model-dir> --outfile dflash-draft.gguf
```

### 3. DeepSeek-V4 ecosystem support

- Model graphs: `deepseek.cpp`, `deepseek2.cpp`, `deepseek2ocr.cpp`, `deepseek32.cpp`, `deepseek4.cpp`, `dflash.cpp`
- HF conversion (`conversion/deepseek.py`): DeepSeek OCR / V2 / V3.2 / V4 (incl. FP8 dequant) / V4-DSpark
- Dedicated KV-cache variants: `llama-kv-cache-dsv4.cpp` (draft ring), `llama-kv-cache-iswa.cpp` (interleaved sliding window), `llama-kv-cache-dsa.cpp`, `llama-kv-cache-msa.cpp`
- Chat template helpers for V3.2/V4, incl. thinking-retention and tool-result ordering (`common/chat.cpp`)

---

On top of the fork features, the codebase tracks upstream llama.cpp closely, so the entire model zoo (Qwen3.5, Gemma4, Kimi-K3, ...) and all backends (CUDA, Vulkan, SYCL, OpenCL, Metal, WebGPU, Hexagon HTP, CANN, OpenVINO) remain available.

## Run as a systemd service (Linux)

Create `/etc/systemd/system/llama-server.service` (adjust user, paths, model and flags to your setup):

```ini
[Unit]
Description=llama-server (TurboQuant fork)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=llama
Group=llama
WorkingDirectory=/opt/llama-cpp-turboquant
# point ExecStart at the llama-server binary built from this fork
ExecStart=/opt/llama-cpp-turboquant/build/bin/llama-server \
    --model /models/your-model.gguf \
    --host 127.0.0.1 --port 8080 \
    --cache-type-k turbo3 --cache-type-v turbo3 \
    --n-gpu-layers 999
# optional: calibrate InnerQ channel scales, N = calibration tokens
Environment=TURBO_INNERQ=4096
Restart=on-failure
RestartSec=5
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
sudo systemctl status llama-server
journalctl -u llama-server -f     # follow server logs
```

Common operations:

```sh
sudo systemctl stop llama-server
sudo systemctl restart llama-server
```

Notes:
- The service binds to `127.0.0.1:8080` by default. If you need remote access, put a reverse proxy (nginx/Caddy) in front - do **not** put API keys or tokens in the unit file; use `EnvironmentFile=` (root-only readable) if the server itself needs secrets, or better keep them client-side.
- For multi-GPU setups you can pin devices via `Environment=CUDA_VISIBLE_DEVICES=0,1`.
- Every `ExecStart` flag maps 1:1 to a `llama-server` CLI option (`--cache-type-k turbo3` etc.), see `llama-server --help`.

## Build

See the upstream [build guide](docs/build.md) and the [server docs](tools/server/README.md). CUDA is the primary backend for the TurboQuant kernels:

```sh
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
```

## Repository layout

```
ggml/src/ggml-cuda/     TurboQuant kernels (turbo-quant, turbo-innerq, turbo-wht, set-rows, fattn)
src/models/dflash.cpp   DFlash / DFlash2 / DSpark graph implementation
src/models/deepseek*.cpp DeepSeek V2/V3.2/V4 (+OCR) graphs
src/llama-kv-cache-*.cpp Dedicated KV-cache variants (dsv4 / iswa / dsa / msa)
conversion/deepseek.py  DeepSeek family HF -> GGUF converters
conversion/qwen.py      DFlash draft converter (register: DFlashDraftModel / DFlash2DraftModel)
common/chat.cpp         DeepSeek chat-template helpers
```

## License

MIT - see [LICENSE](LICENSE). This project is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).