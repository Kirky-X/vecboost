# 代码贡献指南（Contributing Guide）
**项目名称**: Rust 文本向量化模块  
**版本**: v1.0.0  
**创建日期**: 2025

---

## 1. 欢迎参与贡献

感谢您考虑为 VecBoost 项目贡献代码！本指南将帮助您了解贡献流程、代码规范和最佳实践。

### 1.1 贡献类型

我们欢迎多种形式的贡献：

| 贡献类型 | 描述 | 示例 |
|----------|------|------|
| **代码贡献** | 修复 bug、实现新功能 | 修复 GPU 内存泄漏 |
| **文档改进** | 完善文档、修复错误 | 补充 API 使用示例 |
| **问题反馈** | 报告 bug、提出建议 | 性能问题报告 |
| **代码审查** | 参与 PR 审查 | 代码质量审查 |
| **测试贡献** | 添加单元测试、集成测试 | 增加边界条件测试 |

### 1.2 行为准则

请遵守以下行为准则：

- **尊重他人**：友好沟通，避免人身攻击
- **开放协作**：欢迎新贡献者，耐心解答问题
- **质量优先**：关注代码质量和可维护性
- **持续改进**：积极反馈建设性意见

---

## 2. 开发环境设置

### 2.1 环境要求

在开始贡献之前，请确保您的开发环境满足以下要求：

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| Rust | 1.75.0+ | 编译器工具链 |
| Git | 2.0+ | 版本控制 |
| CMake | 3.16+ | 构建工具 |
| GitHub CLI | 2.0+ | （可选）简化 PR 操作 |

### 2.2 本地环境配置

```bash
# 克隆项目
git clone https://github.com/your-org/vecboost.git
cd vecboost

# 安装开发依赖
cargo fetch
cargo install cargo-watch cargo-audit cargo-spellcheck

# 运行测试验证环境
cargo test --lib
```

详细环境配置请参考 [开发环境设置指南](docs/getting-started.md)。

---

## 3. 开发流程

### 3.1 选择任务

1. **浏览 Issues**：查看标签为 `good first issue` 或 `help wanted` 的问题
2. **任务分配**：在 Issue 下留言，表明您要参与解决
3. **任务确认**：Maintainer 会将任务分配给您

### 3.2 创建分支

```bash
# 确保基于最新的 develop 分支
git checkout develop
git pull origin develop

# 创建功能分支
# 分支命名规范: <type>/<issue-id>-<short-description>

# 功能开发
git checkout -b feature/123-add-tokenizer-cache

# Bug 修复
git checkout -b fix/456-resolve-gpu-memory-leak

# 文档改进
git checkout -b docs/789-update-api-docs
```

### 3.3 分支类型说明

| 分支类型 | 前缀 | 示例 | 说明 |
|----------|------|------|------|
| 功能分支 | feature/ | feature/123-add-batch-processing | 新功能开发 |
| Bug 修复 | fix/ | fix/456-resolve-memory-leak | Bug 修复 |
| 文档分支 | docs/ | docs/789-update-readme | 文档改进 |
| 重构分支 | refactor/ | refactor/abc-optimize-inference | 代码重构 |
| 性能分支 | perf/ | perf/def-improve-qps | 性能优化 |

### 3.4 开发规范

```bash
# 编码过程中
cargo fmt          # 格式化代码
cargo clippy       # 运行 lint 检查
cargo test --lib   # 运行单元测试

# 提交前
git add <files>
git commit -m "feat: add tokenizer cache mechanism (closes #123)"
```

### 3.5 提交规范

**提交信息格式**：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 类型**：

| Type | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | `feat(inference): add batch processing support` |
| fix | Bug 修复 | `fix(device): resolve GPU memory leak` |
| docs | 文档改进 | `docs(readme): update installation guide` |
| style | 代码格式 | `style(format): run cargo fmt` |
| refactor | 重构 | `refactor(tokenizer): optimize encoding performance` |
| perf | 性能优化 | `perf(embedding): improve QPS by 30%` |
| test | 测试相关 | `test(unit): add tokenizer boundary tests` |
| chore | 维护任务 | `chore(deps): update candle to v0.8.0` |

**示例提交信息**：

```
feat(tokenizer): add LRU cache for tokenization results

- Implement token cache with configurable size limit
- Cache hit improves performance by 40% for repeated texts
- Add cache statistics in metrics endpoint

Closes #123
```

---

## 4. Pull Request 流程

### 4.1 创建 PR 前检查清单

在创建 Pull Request 之前，请确保完成以下检查：

- [ ] 代码符合项目编码规范
- [ ] 所有测试通过
- [ ] 添加了必要的测试用例
- [ ] 更新了相关文档
- [ ] 提交信息符合规范
- [ ] 分支已同步最新代码

```bash
# 最终检查命令
cargo fmt --check
cargo clippy --all-features --all-targets
cargo test --all
cargo doc --no-deps
```

### 4.2 创建 Pull Request

1. **推送分支**：
```bash
git push origin feature/123-add-tokenizer-cache
```

2. **创建 PR**：通过 GitHub 界面创建 PR

3. **PR 模板**：

```markdown
## 描述
<!-- 简要描述您的更改 -->

## 关联 Issue
<!-- 关联的 Issue 编号 -->
Closes #123

## 更改类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 破坏性更改
- [ ] 文档更新

## 测试验证
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试通过

## 性能影响
<!-- 性能变化说明（如果适用） -->
```

### 4.3 PR 审查流程

| 阶段 | 说明 | 期望时间 |
|------|------|----------|
| **自动化检查** | CI/CD 运行测试和 lint | 5-15 分钟 |
| **代码审查** | Maintainer 审查代码 | 1-3 天 |
| **修改完善** | 根据反馈修改 | 根据反馈 |
| **合并** | 代码合并到 develop | 审查通过后 |

### 4.4 响应审查反馈

- **积极响应**：及时回复审查意见
- **澄清说明**：对疑问提供清晰解释
- **讨论协商**：有分歧时寻求共识
- **耐心等待**：给审查者足够时间

---

## 5. 代码规范

### 5.1 Rust 编码规范

遵循以下 Rust 编码规范：

**命名规范**：
```rust
// 类/类型名：PascalCase
pub struct EmbeddingService;
pub enum DeviceType;

// 函数/方法名：camelCase，以动词开头
pub fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
pub fn is_gpu_available(&self) -> bool;

// 变量名：camelCase
let max_batch_size = 32;
let mut token_ids = Vec::new();

// 常量：UPPER_SNAKE_CASE
const MAX_SEQUENCE_LENGTH: usize = 512;
const DEFAULT_CACHE_SIZE_MB: usize = 100;

// 私有成员：以下划线开头
struct Tokenizer {
    inner: TokenizerWrapper,
    cache: LruCache<String, Vec<u32>>,
}
```

**代码组织**：
- 单一职责原则：每个模块/函数只负责一个明确的功能
- 函数长度限制：不超过 50 行（特殊情况需注释说明）
- 参数数量限制：不超过 3 个参数
- 嵌套深度限制：不超过 3 层

### 5.2 代码注释规范

```rust
/// Embedding 服务主接口
///
/// 提供文本向量化、文件处理和相似度计算等核心功能
/// 支持 CPU 和 GPU 双模式运行
pub struct EmbeddingService {
    engine: Box<dyn InferenceEngine>,
    tokenizer: TokenizerWrapper,
    config: ServiceConfig,
}

/// 初始化 Embedding 服务
///
/// # Arguments
///
/// * `config` - 服务配置，包含模型路径和推理参数
///
/// # Errors
///
/// 返回 `EmbeddingError` 当：
/// - 模型文件不存在或损坏
/// - 推理引擎初始化失败
///
/// # Examples
///
/// ```
/// let config = ServiceConfig::default();
/// let service = EmbeddingService::new(config)?;
/// let embedding = service.embed_text("Hello, world!")?;
/// ```
pub fn new(config: ServiceConfig) -> Result<Self, EmbeddingError> {
    // 实现代码
}
```

### 5.3 测试规范

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// 测试短文本向量化功能
    ///
    /// 验证：
    /// 1. 返回正确维度的向量
    /// 2. 向量值在 [-1, 1] 范围内
    /// 3. 与 Python 基准输出一致
    #[test]
    fn test_short_text_embedding() -> Result<(), EmbeddingError> {
        let service = create_test_service()?;
        
        let embedding = service.embed_text("人工智能")?;
        
        assert_eq!(embedding.len(), 1024);
        assert!(embedding.iter().all(|v| v.abs() <= 1.0));
        
        Ok(())
    }
}
```

---

## 6. Git 工作流

### 6.1 主分支策略

| 分支 | 保护状态 | 说明 |
|------|----------|------|
| `main` | 强制保护 | 生产版本，只接受合并 |
| `develop` | 强制保护 | 开发主分支，集成最新功能 |
| `feature/*` | 可选保护 | 功能开发分支 |
| `release/*` | 强制保护 | 发布准备分支 |

### 6.2 版本发布流程

```
1. 从 develop 创建发布分支
   git checkout -b release/v1.0.0 develop

2. 更新版本号和 CHANGELOG
   - 修改 Cargo.toml 中的版本
   - 更新 CHANGELOG.md

3. 最终测试和修复
   cargo test --all
   cargo doc --no-deps

4. 合并到 main 和 develop
   git checkout main
   git merge release/v1.0.0 --no-ff
   git tag -a v1.0.0 -m "Release v1.0.0"
   
   git checkout develop
   git merge release/v1.0.0 --no-ff

5. 删除发布分支
   git branch -d release/v1.0.0
```

---

## 7. 依赖管理

### 7.1 添加新依赖

```bash
# 添加生产依赖
cargo add <package-name>

# 添加开发依赖
cargo add --dev <package-name>

# 添加特定版本
cargo add <package-name>@<version>
```

### 7.2 更新依赖

```bash
# 检查可用的更新
cargo outdated

# 更新依赖（保持 SemVer 兼容）
cargo update

# 更新到最新版本（可能包含破坏性更改）
cargo upgrade
```

### 7.3 依赖安全检查

```bash
# 运行安全审计
cargo audit

# 检查依赖漏洞
cargo deny check
```

---

## 8. 文档贡献

### 8.1 文档类型

| 文档类型 | 位置 | 说明 |
|----------|------|------|
| API 文档 | 代码注释 | rustdoc 生成 |
| 用户指南 | docs/*.md | 用户使用文档 |
| 架构文档 | docs/architecture/ | 设计和架构文档 |
| 示例代码 | examples/ | 运行示例 |

### 8.2 文档编写规范

- 使用中文编写，术语保持中英文对照
- 代码示例必须有实际运行结果
- 文档更新与代码更改同步

---

## 9. 问题反馈

### 9.1 报告 Bug

请使用 Issue 模板并提供以下信息：

```markdown
## Bug 描述
<!-- 清晰描述遇到的问题 -->

## 重现步骤
1. <!-- 第一步 -->
2. <!-- 第二步 -->
3. <!-- ... -->

## 期望行为
<!-- 描述应该发生什么 -->

## 实际行为
<!-- 描述实际发生了什么 -->

## 环境信息
- 操作系统：
- Rust 版本：
- GPU 型号（如适用）：
- 其他相关信息：

## 日志输出
<!-- 相关的错误日志 -->
```

### 9.2 提出功能建议

```markdown
## 功能建议
<!-- 描述您想要的功能 -->

## 使用场景
<!-- 解释为什么需要这个功能 -->

## 期望实现方式
<!-- 建议的实现方案（如果有） -->

## 参考资料
<!-- 相关的链接或参考项目 -->
```

---

## 10. 常见问题

### Q1: 如何开始第一个贡献？

**A**: 推荐从以下任务开始：
1. 解决标记为 `good first issue` 的问题
2. 改进文档中的错别字或表述
3. 添加缺失的测试用例

### Q2: PR 审查需要多长时间？

**A**: 通常在 1-3 个工作日内。如果需要更长时间，Maintainer 会提前告知。

### Q3: 可以直接修改 main 分支吗？

**A**: 不可以。所有更改必须通过 PR 流程，main 分支受到保护。

### Q4: 如何处理大的功能改动？

**A**: 
1. 在 Issue 中讨论设计思路
2. 将大功能拆分为多个小 PR
3. 每个 PR 保持原子性，可独立合并

### Q5: 贡献代码需要签署 CLA 吗？

**A**: 目前不需要签署 CLA，但贡献即表示您同意代码采用项目许可证（MIT）开源。

---

## 11. 联系方式

| 渠道 | 说明 |
|------|------|
| **GitHub Issues** | Bug 报告、功能建议 |
| **GitHub Discussions** | 问答、讨论 |
| **邮件列表** | 重要通知 |
| **Maintainer** | 直接联系项目维护者 |

---

## 12. 致谢

感谢所有为 VecBoost 项目做出贡献的开发者！

您的贡献使这个项目变得更好！