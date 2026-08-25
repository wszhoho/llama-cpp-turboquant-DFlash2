# DFlash2 + turbo4 集成处理记录 (2026-08-24, aiserver1)

## 背景与目标
- 机器: aiserver1, 2x NVIDIA RTX 2080 Ti (11GB x2), CUDA 12.4, 驱动 595.84
- 目标: 在 DFlash2 投机解码(草稿模型)下获得 >=64k 上下文, 替换旧 turboquant+MTP 服务(70k, ~15 t/s)
- 模型: 主 Qwen3.8-27B (65层, GQA 6:1: n_head=24/n_head_kv=4), draft Qwen3.8-27B-DFlash2-Q4_K_M (1.9B, block=8, n_extract=5)

## 关键步骤与结论

### 1. DFlash2 支持 (llama.cpp PR#27342)
- DFlash2 草稿是块扩散(block diffusion)+候选选择器(candidate selector); 需要 llama.cpp PR#27342 (spec: add DFlash2 support)
- 在 llama-cpp-master 快照(上游 8/21)上应用 git-apply 组装后的 PR diff (16 文件, 部分跳过 conversion/gguf-py 亦可运行)
- 关键新 API/机制: llama_set_embeddings_nextn / llama_get_embeddings_nextn / set_embeddings_layer_inp / 
  llama_model_target_layer_ids(_n) / llama_model_dflash_selector_top_k, 以及 cparams 的 embeddings_nextn 体系

### 2. turbo4 KV cache 反移植 (turboquant -> master)
- 背景: master 上游删除了 CUDA split buffer(改走 NCCL comm), 仅剩 -sm layer(整层分配, 双卡显存不均)
  且无 turbo4 类型; turboquant fork 保留 turbo2/3/4 (WHT + PolarQuant KV 量化) 与层自适应
- 移植内容:
  - ggml.h: GGML_TYPE_TURBO2_0=43/TURBO3_0=44/TURBO4_0=45 + GGML_OP_TURBO_WHT
  - ggml.c: type_traits(blck_size/quant 函数) + TURBO_WHT 处理
  - ggml-cuda: turbo-quant.cuh / turbo-innerq.cu/.cuh / turbo-wht.cu/.cuh + op 分发
  - llama-kv-cache.cpp: k_is_turbo/v_is_turbo + 层自适应(Boundary V: 边界层 turbo4, 其余 turbo2; TURBO_AUTO_ASYMMETRIC)
  - fattn: fattn-common.cuh 内嵌 turbo KQ dot/V dequant; fattn-vec.cuh 增加 turbo 类型分支;
    template-instances/ 按 (K,V) 组合建立 fattn-vec-instance-<K>-<V>.cu, **必须显式加入 ggml/src/ggml-cuda/CMakeLists.txt**
    (else 分支只列 4 个基础组合, 不加 turbo 会链接失败 undefined reference ggml_cuda_flash_attn_ext_vec_case)
  - arg.cpp: cache-type 列表加 turbo2/turbo3/turbo4
- 设计意图: K 用 q8_0/turbo3 保持注意力质量, V 用 turbo2/turbo4 省显存(V 量化误差被加权平均吸收)

### 3. 显存与上下文博弈 (2x11GB 硬约束)
- Q4_K_M 主模型 17.1G + draft 1.1G 后, KV/计算缓冲余量 <1G: 64k 崩溃边缘, 70k 不稳, 128k 必然 OOM
- 换 Qwen3.8-27B-UD-IQ4_XS (13.27G, 质量近 Q4_K_M) -> 腾出 ~4G
- K=turbo3 + V=turbo2(层自适应) + TURBO_AUTO_ASYMMETRIC=0 (禁 K 自动升级 q8, 省 1.4G@70k)
- **速度规律(实证)**: decode 速度随"关键路径 CPU 层数"剧烈下降:
  - 全 GPU (-ngl 99): 70k=53.5 t/s, 96k=46.9 t/s
  - 5 层 CPU (-ngl 60): 128k=19 t/s; -ngl 65(≠全GPU, output 组件在 CPU)=25.6 t/s
  - 结论: 65 层模型必须 -ngl>=99(全 GPU) 才快; 128k 需牺牲速度(5层CPU)或更多显存
- 最终生产: -c 98304 -ngl 99 -b 512 -ub 512, K=turbo3 V=turbo2, draft GPU -> **96k @ 46.9 t/s** 稳定

### 4. 性能全景 (DFlash2 投机) — 实测含长上下文修正
**重要警告: 短输入测速不反映长上下文真实性能。K=turbo3(3bit) 在长输入下接受率崩盘(0.7->0.045), 实际掉到 ~12 t/s。**
**K=q8_0 是长上下文(干活)的必要条件。**
| 配置 | 输入长度 | 接受率 | eval 吞吐 |
|---|---|---|---|
| K=turbo3+96k, 全GPU | 短(<1k) | 0.70 | 46.9 t/s |
| K=turbo3+96k, 全GPU | 10-19k | 0.05-0.34 | 11.8-15 t/s |
| K=turbo3+128k, 5层CPU | 短 | - | ~19 t/s |
| K=q8_0+70k, 全GPU | 12k | 0.49 | 35 t/s |
| K=q8_0+70k, 全GPU | 38k(重复文本) | 0.80 | 53.8 t/s |
| **K=q8_0+96k, 全GPU (生产定稿)** | **12k 业务型** | **0.33** | **30.4 t/s** |
| K=q8_0+70k, 全GPU | 20k 真实业务 | 0.25 | 24 t/s |
- 需求优先级: 上下文长度优先, 速度>=20 t/s 即可 -> 定稿 96k + K=q8_0 全 GPU
- 长上下文规律: 3bit K 注意力误差随序列累积 -> draft 被全拒; q8_0 恢复 0.5-0.8 接受率
- 速度上限: CPU 层决定关键路径(见上节), 全 GPU + q8_0 K 是真实生产形态
- DFlash2 接受率: 数学题 0.66-0.70, 块长 5.1-5.9; 对比旧 MTP 服务 ~15.2 t/s: 提速 3 倍+
- 可选: K=turbo3 3bit 有轻微注意力质量影响(数学/短链无明显差异, 长链待业务验证)

## 还原/恢复
- 构建: cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON && cmake --build build -j 24
- 启动: sudo bash /home/sdzn/llama-server.sh  (写入/重启 systemd llama-server.service)
- 模型: /home/sdzn/models/{Qwen3.8-27B-UD-IQ4_XS.gguf, Qwen3.8-27B-DFlash2-Q4_K_M.gguf, mmproj-F16.gguf}
- turboquant 旧生产(main 分支, MTP 70k): /home/sdzn/llama-cpp-turboquant (feature/turboquant-kv-cache, 已重置干净)
