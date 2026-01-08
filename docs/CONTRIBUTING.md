<div align="center">

# 🤝 VecBoost 贡献指南

感谢您对 VecBoost 项目的兴趣！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 报告 Bug
- 💡 提出新功能建议
- 📝 完善文档
- 🔧 提交代码修复
- ✨ 添加新功能

</div>

---

## 📋 目录

- [🏁 快速开始](#快速开始)
- [🔧 开发环境设置](#开发环境设置)
- [📝 代码规范](#代码规范)
- [🧪 测试要求](#测试要求)
- [📤 提交Pull Request](#提交pull-request)
- [❓ 获取帮助](#获取帮助)

---

## 🏁 快速开始

### 1. Fork 仓库

点击 GitHub 页面右上角的 **Fork** 按钮，将仓库 Fork 到您的账户。

### 2. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/vecboost.git
cd vecboost
```

### 3. 创建功能分支

```bash
git checkout -b feature/your-feature-name
```

### 4. 安装依赖并验证构建

```bash
# 安装 Rust (如果尚未安装)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 构建项目
cargo build --release

# 运行测试
cargo test --all-features
```

---

## 🔧 开发环境设置

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+), macOS, Windows with WSL2
- **内存**: 至少 4GB
- **磁盘**: 至少 10GB 可用空间
- **Rust**: 1.70+ (建议使用 rustup 安装)

### 可选依赖

| 依赖 | 用途 | 安装方式 |
|------|------|---------|
| CUDA | NVIDIA GPU 加速 | [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) |
| ONNX Runtime | ONNX 模型支持 | 通过 cargo feature 启用 |
| Redis | 分布式限流 | `brew install redis` / `apt install redis-server` |

### 构建命令

```bash
# CPU 版本
cargo build --release

# CUDA 支持 (NVIDIA GPU)
cargo build --release --features cuda

# Metal 支持 (Apple Silicon)
cargo build --release --features metal

# ONNX Runtime 支持
cargo build --release --features onnx

# gRPC 支持
cargo build --release --features grpc

# 所有功能
cargo build --release --features cuda,metal,onnx,grpc,redis
```

---

## 📝 代码规范

### 代码格式化

```bash
# 格式化代码
cargo fmt

# 检查格式
cargo fmt --check
```

### 代码检查

```bash
# 运行 clippy (必须通过，无警告)
cargo clippy --all-features -- -D warnings
```

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块 | snake_case | `embedding_service` |
| 结构体/枚举 | PascalCase | `EmbeddingRequest` |
| 函数/变量 | snake_case | `calculate_similarity` |
| 常量 | SCREAMING_SNAKE_CASE | `MAX_BATCH_SIZE` |

### 导入顺序

```rust
use std::xxx;      // 标准库
use xxx::xxx;      // 外部 crate
use crate::xxx;    // 内部模块
```

### 文档要求

所有公开 API 必须添加文档注释：

```rust
/// 生成文本的向量嵌入
///
/// # 参数
///
/// * `text` - 输入文本
/// * `normalize` - 是否归一化向量
///
/// # 返回
///
/// 返回包含嵌入向量的结果
///
/// # 示例
///
/// ```
/// let embedding = service.embed("Hello world", true).await?;
/// ```
pub async fn embed(&self, text: &str, normalize: bool) -> Result<EmbedResponse> {
    // ...
}
```

---

## 🧪 测试要求

### 运行测试

```bash
# 所有测试
cargo test --all-features

# 单元测试
cargo test --lib

# 集成测试
cargo test --tests

# 性能测试
cargo test --features cuda,grpc --test performance_test

# 运行特定测试
cargo test test_name -- --nocapture
```

### 测试覆盖率

- 新功能必须包含对应的单元测试
- 集成测试覆盖主要功能路径
- 性能关键代码需要性能测试

### 测试规范

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_feature_name() {
        // 测试代码
    }
}
```

---

## 📤 提交 Pull Request

### 提交信息规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

| 类型 | 说明 |
|------|------|
| `feat:` | 新功能 |
| `fix:` | Bug 修复 |
| `docs:` | 文档更新 |
| `style:` | 代码格式（不影响功能）|
| `refactor:` | 重构代码 |
| `perf:` | 性能优化 |
| `test:` | 测试相关 |
| `chore:` | 构建/工具更新 |

示例：
```
feat(auth): 添加 CSRF 保护机制
fix(engine): 修复 CUDA 内存泄漏问题
docs(readme): 更新快速开始指南
```

### PR 检查清单

在提交 PR 前，请确认：

- [ ] 代码通过 `cargo fmt` 格式化
- [ ] 代码通过 `cargo clippy` 检查
- [ ] 所有测试通过
- [ ] 添加了必要的文档
- [ ] 更新了 CHANGELOG（如果需要）
- [ ] PR 描述清晰，包含变更说明

### PR 描述模板

```markdown
## 描述
简要说明本次变更的内容和目的。

## 变更类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 代码重构
- [ ] 性能优化

## 测试
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试通过

## 截图/示例（如果适用）
添加截图或代码示例来说明变更效果。

## 注意事项
需要评审者特别关注的地方。
```

---

## ❓ 获取帮助

### 常见问题

**Q: 构建失败怎么办？**
```bash
# 清理构建缓存
cargo clean

# 重新构建
cargo build --release
```

**Q: 测试超时怎么办？**
```bash
# 增加测试超时时间
cargo test --all-features -- --test-threads=1
```

**Q: GPU 功能不可用？**
确认已启用对应的 feature：
```bash
cargo build --release --features cuda
```

### 联系方式

- 📮 **Issue**: [GitHub Issues](https://github.com/Kirky-X/vecboost/issues)
- 💬 **讨论**: [GitHub Discussions](https://github.com/Kirky-X/vecboost/discussions)
- 📧 **邮件**: dev@example.com

---

## 📜 行为准则

请遵守我们的[行为准则](CODE_OF_CONDUCT.md)，保持友善和尊重的交流环境。

---

<div align="center">

**感谢您的贡献！让我们一起构建更好的 VecBoost。**

</div>
