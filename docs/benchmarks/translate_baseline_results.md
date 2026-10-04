# 翻译与语言检测基线（translate_bench）

- 日期：2026-10-04
- 命令：`cargo bench --features onnx --bench translate_bench`
- 主机：AMD Ryzen 9 9950X（16 核 32 线程，bench 进程 ORT intra 线程取 `available_parallelism`） / 64 GiB RAM / WSL2
- 构建：bench profile（`opt-level = 3`）；ORT 1.28.2 dylib（`3rdparty/onnxruntime/libonnxruntime.so`，CPU EP）
- 模型：Xenova/opus-mt-en-zh、Xenova/opus-mt-en-es（fp32 encoder + decoder_model_merged，贪心解码、缓存分支禁用）
- 方法：criterion（典型值取中位）；langdetect 组缺省 100 样本；translate 组 `sample_size(10)`、`warm_up_time(5s)`（单次调用为毫秒~秒级，小样本防跑时爆炸）

## 结果

| 基准 | 输入（字符） | 时间（中位） | 说明 |
|---|---|---|---|
| langdetect/short | 26 | **28.4 µs** | whatlang 检测（`vecboost::detect_language`，None allowlist） |
| langdetect/max-10k | 7 979 字节（10 000 字符截取） | **106.5 µs** | 上限长度文本亚毫秒，`detect` 端点不走 spawn_blocking 的决策据此成立 |
| translate/en-zh/short | 26 | **106.6 ms** | "The weather is nice today." → zh |
| translate/en-zh/medium | 74 | **326.6 ms** | 市场/晚餐句 → zh |
| translate/en-zh/long | 202 | **591.1 ms** | 气候研究长句 → zh（O(L²) 全量重算，长度↑耗时超线性） |
| translate/en-es/short | 26 | **214.6 ms** | 同 short 句 → es |
| translate/en-zh/count_tokens | 74 | **9.8 µs** | tokenizer 精确计数（服务层预算预判成本口径） |

## 观察

- 翻译单请求毫秒~百毫秒级（相对 46 token 句 ~1.5s 的 debug 构建历史数据，release 级优化显著）；长句超线性增长与「无缓存分支 O(L²)」已知限制一致。
- 语言检测 µs 级且 10k 上限文本仍 <0.2 ms，`DetectService` 直连（不走 spawn_blocking）设计实测成立。
- `count_tokens`（9.8 µs）相对解码（百 ms 级）可忽略，双层 token 预算的重复 tokenize（评审 PERF-5 显性保留项）实测代价极小。

复跑口径变化（模型/ORT/主机）时更新本文件并标注新日期。
