# VecBoost 示例

本目录是独立的 workspace member crate `vecboost-examples`(`publish = false`)，包含 VecBoost 向量嵌入服务的公共 API 示例代码，按功能分类组织为 13 个分类、30 个可执行二进制。

所有示例专注于展示公共 API 和实际应用场景，不包含内部实现细节。

## 运行示例

`vecboost-examples` 作为 workspace member，通过 `-p vecboost-examples --bin <name>` 运行：

```bash
# 基础示例（默认 features）
cargo run -p vecboost-examples --bin embed

# Library SDK 示例
cargo run -p vecboost-examples --bin library_usage

# Matryoshka 维度约简
cargo run -p vecboost-examples --bin matryoshka

# ONNX 引擎示例（需要 ONNX Runtime）
cargo run -p vecboost-examples --bin onnx --features onnx
```

## Features 透传

`vecboost-examples` 通过自身 features 透传引擎特性到主 crate：

| Feature | 启用 | 说明 |
|---------|------|------|
| `onnx` | `vecboost/onnx` | ONNX Runtime 引擎 |
| `metal` | `vecboost/metal` | Apple Silicon GPU(macOS) |

默认 features 为空(`default = []`)，绝大多数示例无需额外 feature 即可运行。

## 示例分类

| 分类 | 目录 | 示例数 | 说明 |
|------|------|--------|------|
| basic | `basic/` | 6 | 基础用法：嵌入、批量、相似度、重排序、Matryoshka 维度约简、输入验证 |
| engine | `engine/` | 3 | 引擎抽象：Candle/ONNX 引擎初始化与运行时切换 |
| http | `http/` | 4 | HTTP API：REST 接口调用（reqwest） |
| cli | `cli/` | 3 | 命令行工具：CLI 子命令调用 |
| auth | `auth/` | 3 | 认证授权：JWT 令牌、CSRF 防护、token 刷新 |
| cache | `cache/` | 2 | 缓存：oxcache 后端配置与 TTL 验证 |
| rate-limiting | `rate-limiting/` | 2 | 限流：limiteron 多维度限流配置 |
| monitoring | `monitoring/` | 2 | 监控：Prometheus 指标与性能采集 |
| library | `library/` | 1 | Library SDK：VecBoostLibrary 嵌入式集成 |
| semantic-cache | `semantic-cache/` | 1 | 语义缓存：trigram Jaccard 三级查询 |
| security | `security/` | 1 | 安全：密钥管理、盐值生成、敏感数据脱敏 |
| audit | `audit/` | 1 | 审计日志：异步批量写入与文件轮转 |

另外 `download_model.rs` 位于根目录，是模型下载工具。

## 示例清单

### basic — 基础用法
- `embed` — 单文本嵌入，演示 `EmbeddingService` + `api::embed`
- `batch` — 批量嵌入，演示 `api::embed_batch` 处理多条文本
- `similarity` — 余弦相似度，演示 `api::compute_similarity`
- `rerank` — 重排序，演示 `RerankService` 文档相关性排序
- `matryoshka` — Matryoshka 维度约简：任务自适应截断、信息保留率、L2 归一化
- `validation` — 输入验证：文本长度/批量大小/路径遍历防护/HuggingFace Repo ID 校验

### engine — 引擎抽象
- `candle` — Candle 引擎初始化与推理
- `onnx` — ONNX 引擎初始化（需要 `onnx` feature）
- `switch` — 运行时 `EmbeddingService::switch_model`

### http — HTTP API
- `embed_api` — HTTP POST `/api/v1/embed`
- `batch_api` — HTTP POST `/api/v1/embed/batch`
- `similarity_api` — HTTP POST `/api/v1/similarity`
- `rerank_api` — HTTP POST `/api/v1/rerank`

### cli — 命令行工具
- `embed_cli` — 调用 `vecboost embed --text "hello"`
- `batch_cli` — 调用 `vecboost batch --input texts.txt`
- `rerank_cli` — 调用 `vecboost rerank --query "..." --documents docs.txt`

### auth — 认证授权
- `jwt_auth` — JWT token 生成与验证
- `csrf` — CSRF token 获取与使用
- `refresh` — token 刷新流程

### cache — 缓存
- `cache_config` — oxcache 配置与缓存命中验证
- `ttl` — TTL 过期验证

### rate-limiting — 限流
- `rate_limit` — limiteron 限流配置
- `multi_dimension` — 全局+IP+用户多维度限流

### monitoring — 监控
- `metrics` — Prometheus 指标暴露
- `performance` — 推理性能监控

### library — Library SDK
- `library_usage` — VecBoostLibrary 嵌入式集成（异步+同步 API、嵌入+重排序）

### semantic-cache — 语义缓存
- `semantic_cache_demo` — SemanticCache 三级查询策略（精确匹配→trigram 语义搜索→模型推理回填）

### security — 安全
- `security_demo` — 密钥管理、盐值生成、敏感数据脱敏

### audit — 审计日志
- `audit_demo` — AuditLogger 异步批量写入与文件轮转

### 工具
- `download_model` — 下载预训练模型
