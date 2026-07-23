# Changelog

All notable changes to VecBoost are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-24

### Added

- **sdforge 统一接口生成**：HTTP/gRPC/MCP/CLI 四种协议通过 `#[forge(...)]` 宏从 `src/api/embedding.rs` 单一源定义生成，消除手写协议代码。
- **gRPC 迁移到 sdforge**：手写 tonic gRPC 实现替换为 `#[forge(grpc_method = "vecboost.*")]` 宏封装，支持 9 个 gRPC 方法（embed/embed_batch/compute_similarity/embed_file/model_switch/get_current_model/get_model_info/list_models/health_check）。
- **gRPC 配置项**：新增 `grpc_max_connections`、`grpc_timeout_seconds`、`grpc_require_auth`、`grpc_allowed_roots` 配置，支持 JWT 认证和速率限制。
- **Bert/XlmRoberta 模型架构支持**：通过 `ModelArchitecture` 枚举自动检测模型类型。
- **Matryoshka 截断重归一化**：截断维度后自动调用 `normalize_l2`，保证余弦相似度正确。
- **vuln-0009 安全加固**：HF Hub `repo_id` 格式校验统一在 `src/utils/hf_hub.rs`（`is_valid_hf_repo_id` + `build_hf_repo`），覆盖所有远程下载入口。
- **协议无关 handler**：提取 `*_handler` 函数消除 gRPC/HTTP/CLI 间的代码重复。
- **批量大小校验**：`validate_batch_size` 使用 `EmbeddingConfig.max_batch_size` 限制。

### Changed

- **依赖升级**：hf-hub 0.4→1.0、prometheus 0.13→0.14、tokio 1.52→1.53、aes-gcm 0.10→0.11。
- **default feature 简化**：`default = ["http"]`（原 `["http", "oxcache", "limiteron"]`）。
- **必选依赖**：`confers`、`inklog`、`oxcache`、`limiteron`、`trait-kit` 改为始终启用，无需 feature 开启。
- **sdforge 本地依赖**：临时使用 `path = "../sdforge"`（v0.4.7，含 tokio 1.53 支持），待 sdforge 0.4.7 发布到 crates.io 后切回。

### Removed

- **手写 gRPC 实现**：删除 `src/grpc/`、`proto/` 目录。
- **手写 HTTP 路由**：删除 `src/routes/` 目录。
- **手写 CLI**：删除 `src/cli/` 目录。
- **CandleEngine.tensor_pool**：移除存在 mask 全零 bug 的张量池逻辑。
- **MemoryPoolManager.tensor_pool 死代码**：清理无消费者的张量池字段和方法。
- **tensorrt/openvino stub 特性**：移除未实现的引擎 stub。

### Fixed

- **Matryoshka 截断未重归一化**：截断后向量非单位向量导致余弦相似度计算错误。
- **attention_mask 张量池 bug**：原张量池路径 TODO 未回填数据导致批量推理 mask 全零。
- **多字节字符切片 panic**：`sanitize_secret`、`sanitize_jwt_secret`、`mask_value`、`text_preview` 使用 `floor_char_boundary`/`ceil_char_boundary` 确保 UTF-8 安全切片。
- **fallback/onnx/recovery 路径缺少 repo_id 校验**：统一通过 `build_hf_repo` 处理。

### Security

- **vuln-0009**：HF Hub `repo_id` 格式校验防止恶意配置注入和路径遍历攻击。
- **UTF-8 安全切片**：所有密钥脱敏和文本预览函数使用字符边界安全切片。
- **gRPC 路径校验**：`PathValidator` 从配置读取允许的根目录，默认拒绝 `/`、`/etc` 等敏感目录。

## [0.1.0] - 2025-12-15

Initial release of VecBoost.
