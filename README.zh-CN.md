# llama-cpp-turboquant-DFlash2

C/C++ 大语言模型推理框架 —— 基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 fork，核心聚焦于**极致 KV 缓存压缩（TurboQuant）**与**快速投机解码（DFlash / DFlash2 / DSpark）**，面向 DeepSeek-V4 等模型的推理加速。

[English](README.md) | [简体中文](README.zh-CN.md)

---

## 核心特性

### 1. TurboQuant —— KV 缓存量化低至 2-bit

长上下文推理时，KV 缓存往往是显存瓶颈。TurboQuant 采用 **WHT 旋转 + PolarQuant** 方案（基于 [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)，ICLR 2026），将 KV 缓存极致压缩，新增三种 GGUF 类型：

| 类型 | 位宽 | GGML 枚举 | CLI 名称 |
|------|------|-----------|----------|
| TURBO2_0 | 2-bit | `GGML_TYPE_TURBO2_0` (43) | `turbo2` |
| TURBO3_0 | 3-bit | `GGML_TYPE_TURBO3_0` (44) | `turbo3` |
| TURBO4_0 | 4-bit | `GGML_TYPE_TURBO4_0` (45) | `turbo4` |

核心实现要点（CUDA，`ggml/src/ggml-cuda/`）：

- **WHT 旋转**：快速沃尔什-哈达玛变换（128/64 元素）+ 两组 seed-42 符号数组，量化前将离群值扩散（`turbo-wht.cu/h`）
- **Lloyd-Max 质心**：针对 `N(0, 1/128)` 分布优化的 2/3/4-bit 码本（`turbo-quant.cuh`）
- **InnerQ 通道均衡**（环境变量 `TURBO_INNERQ=N`，N 为校准 token 数）：按通道均衡方差，保持点积不变 `<Q/s, s*K> = <Q, K>`（`turbo-innerq.cu/h`）
- **Flash Attention 融合**：`turbo2/3/4` 类型直接嵌入 FA 内核，并生成 21 个 turbo/f16/q8_0 组合模板实例（`template-instances/`）
- **CPU 支持**：标量 vec-dot 与 WHT 算子（`ggml_compute_forward_turbo_wht`），无 GPU 也可运行

通过标准缓存类型参数启用：

```sh
llama-cli -m model.gguf -n 512 --cache-type-k turbo3 --cache-type-v turbo3

# 可选：用数千 token 校准 InnerQ 通道缩放
TURBO_INNERQ=4096 llama-cli -m model.gguf -n 512 --cache-type-k turbo3 --cache-type-v turbo3
```

### 2. DFlash / DFlash2 / DSpark —— 投机解码草案模型

DFlash 是一种 *draft*（草案）模型架构：从目标模型（如 DeepSeek-V4）少量抽取的隐层特征预测下一 token，以极低的 draft 开销实现快速投机解码。

- **DFlash**：编码器-解码器草案 —— 特征融合层（`fc` + norm）将抽取的目标层特征映射到草案嵌入空间（`src/models/dflash.cpp`）
- **DFlash2**：新增时序**卷积** + **选择器**头（GGUF 元数据含 `conv_kernel_size`、`conv_group_size`、`selector_rank`、`selector_top_k`），提升接受率
- **DSpark**：DFlash + 半自回归**马尔可夫头** + **置信度头**；可选运行完整 DeepSeek-V4 块，配合均匀滑窗草案 KV 环形缓存（`llama-kv-cache-dsv4.cpp`）
- **缩减词表草案**：通过 `d2t` 映射，草案可只使用小词表，结果再映射回目标 token

转换 DFlash 草案需要目标模型目录（复用其 tokenizer/词表）：

```sh
python convert_hf_to_gguf.py <草案模型目录> --target-model-dir <目标模型目录> --outfile dflash-draft.gguf
```

### 3. DeepSeek-V4 生态支持

- 模型图：`deepseek.cpp`、`deepseek2.cpp`、`deepseek2ocr.cpp`、`deepseek32.cpp`、`deepseek4.cpp`、`dflash.cpp`
- HF 转换（`conversion/deepseek.py`）：DeepSeek OCR / V2 / V3.2 / V4（含 FP8 反量化）/ V4-DSpark
- 专用 KV 缓存变体：`llama-kv-cache-dsv4.cpp`（草案环形）、`llama-kv-cache-iswa.cpp`（交错滑窗）、`llama-kv-cache-dsa.cpp`、`llama-kv-cache-msa.cpp`
- V3.2/V4 聊天模板辅助，含思考保留与工具结果排序（`common/chat.cpp`）

---

除 fork 特性外，代码库与上游 llama.cpp 保持同步，因此完整模型集合（Qwen3.5、Gemma4、Kimi-K3 等）与全部后端（CUDA、Vulkan、SYCL、OpenCL、Metal、WebGPU、Hexagon HTP、CANN、OpenVINO）均可用。

## 以 systemd 服务运行（Linux）

创建 `/etc/systemd/system/llama-server.service`（按你的环境调整用户、路径、模型与参数）：

```ini
[Unit]
Description=llama-server（TurboQuant fork）
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=llama
Group=llama
WorkingDirectory=/opt/llama-cpp-turboquant
# 指向本 fork 构建出的 llama-server 可执行文件
ExecStart=/opt/llama-cpp-turboquant/build/bin/llama-server \
    --model /models/your-model.gguf \
    --host 127.0.0.1 --port 8080 \
    --cache-type-k turbo3 --cache-type-v turbo3 \
    --n-gpu-layers 999
# 可选：校准 InnerQ 通道缩放，N 为校准 token 数
Environment=TURBO_INNERQ=4096
Restart=on-failure
RestartSec=5
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

然后启用并启动：

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
sudo systemctl status llama-server
journalctl -u llama-server -f     # 持续查看服务日志
```

常用操作：

```sh
sudo systemctl stop llama-server
sudo systemctl restart llama-server
```

注意事项：
- 服务默认监听 `127.0.0.1:8080`。如需远程访问，请在前面加反向代理（nginx/Caddy）——**不要把 API 密钥或 Token 写进 unit 文件**；若服务确实需要密钥，用仅管理员可读的 `EnvironmentFile=`，或更推荐放在客户端侧。
- 多 GPU 环境可用 `Environment=CUDA_VISIBLE_DEVICES=0,1` 指定设备。
- `ExecStart` 中的每个参数与 `llama-server` 命令行选项一一对应（如 `--cache-type-k turbo3`），详见 `llama-server --help`。

## 构建

参考上游[构建指南](docs/build.md)与[服务器文档](tools/server/README.md)。TurboQuant 内核的主要后端为 CUDA：

```sh
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
```

## 仓库结构

```
ggml/src/ggml-cuda/       TurboQuant 内核（turbo-quant、turbo-innerq、turbo-wht、set-rows、fattn）
src/models/dflash.cpp     DFlash / DFlash2 / DSpark 图实现
src/models/deepseek*.cpp  DeepSeek V2/V3.2/V4（+OCR）模型图
src/llama-kv-cache-*.cpp  专用 KV 缓存变体（dsv4 / iswa / dsa / msa）
conversion/deepseek.py    DeepSeek 系列 HF -> GGUF 转换器
conversion/qwen.py        DFlash 草案转换器（注册：DFlashDraftModel / DFlash2DraftModel）
common/chat.cpp           DeepSeek 聊天模板辅助
```

## 许可证

MIT —— 见 [LICENSE](LICENSE)。本项目是 [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT）的 fork。