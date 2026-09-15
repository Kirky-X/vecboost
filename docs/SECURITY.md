# 🔒 VecBoost 安全文档

VecBoost 将安全性作为核心设计目标。本文档介绍 VecBoost 的安全策略、漏洞报告流程、内置安全机制与安全最佳实践。

## 📋 目录

<details open>
<summary>📑 目录（点击展开）</summary>

- [支持版本](#-支持版本)
- [漏洞报告流程](#-漏洞报告流程)
- [安全设计概览](#️-安全设计概览)
- [供应链与安全门禁](#-供应链与安全门禁)
- [已知且已处置的风险](#-已知且已处置的风险)
- [安全最佳实践](#-安全最佳实践)

</details>

---

## 📌 支持版本

| 版本 | 状态 | 说明 |
|------|------|------|
| Unreleased（0.3.0-dev 工作区） | ✅ 开发中 | 安全默认值收敛、RBAC 接线、XFF 信任反转等破坏性加固（见 [CHANGELOG](CHANGELOG.md)） |
| 0.2.1 | ✅ 支持 | i18n、Rerank、语义缓存、AES-256-GCM 配置加密 |
| 0.2.0 | ⚠️ 仅关键修复 | sdforge 四协议、7 库生态接线 |
| 0.1.0 | ❌ 不再支持 | 初始发布 |

### 最低支持 Rust 版本（MSRV）

VecBoost 要求 **Rust 1.91+**（edition 2024，以 `Cargo.toml` 的 `rust-version` 字段为权威值）。请使用 `rustup update stable` 保持工具链更新以获得安全补丁。

### 依赖安全

```bash
# 运行安全审计（CI Security Audit job 同款命令）
cargo install cargo-audit --locked
cargo audit

# 依赖供应链门禁（advisories/licenses/bans）
cargo deny check
```

- `cargo audit`：RustSec 安全公告扫描，豁免配置在 `.cargo/audit.toml`（须附理由，与 `deny.toml` 处置策略一致）。
- `cargo deny check`：漏洞必须修复；unmaintained/yanked/unsound 为信息性警告，不阻断 CI，发布前人工复核并记录处置结论。
- 新增依赖须经审批：优先活跃维护、特性门控按需启用（Cargo.toml 中各依赖均 `default-features = false` 并显式列出 feature）。

---

## 🐛 漏洞报告流程

### 如何报告

**请勿通过公开 issue 报告安全漏洞。** 请发送邮件联系 maintainer：<kirky-x@outlook.com>，并在标题注明 `[security]`。确认时限与披露时间线：待补充（维护者收到报告后会尽快回复）。

### 报告内容

请尽量包含：

- 受影响版本（`vecboost --version` 或 git commit）
- 漏洞类型与影响面（认证绕过 / 路径遍历 / 注入 / DoS / 信息泄露等）
- 复现步骤或 PoC（配置片段、请求样例）
- 缓解建议（如有）

### 处置原则

- 漏洞（vulnerability）必须修复；不可达路径（transitive/未启用 feature）的 advisory 可记录豁免并附理由。
- 修复随语义化版本发布，并在 [CHANGELOG](CHANGELOG.md) 的 `### 安全` 段记录（先例：RUSTSEC-2026-0258 h2 DoS、vuln-0009 HF Hub repo_id 校验）。

---

## 🛡️ 安全设计概览

### 默认安全（Secure by Default）

- 出厂 `host = "127.0.0.1"` 仅回环绑定；`auth.enabled=false` 时绑定非回环地址**拒绝启动**，逃生阀 `VECBOOST_ALLOW_INSECURE=1` 会打 ERROR 告警（仅供受信网络容器）。
- 出厂 `use_gpu = false`；请求 GPU 但 feature 缺失时 WARN + CPU 回退，不静默。
- `--config` 显式路径不存在时 fail-fast 报错退出（码 2），不静默回退默认配置。
- CLI 未知子命令报错退出（码 2），不静默启动服务。

### 认证与授权（auth feature，garrison）

- JWT 认证（`protocol-jwt`）+ CSRF 保护（跟随 `auth.enabled`）+ Web CORS/防火墙系列（bruteforce / ratelimit / ddos / anomalous）。
- RBAC：`/api/1/model/*` 与 `/embed/file` 要求 **admin 角色**。
- 登录收敛：仅 `default_admin_username`（默认 admin）可登录；启用认证时必须配置 `VECBOOST_ADMIN_PASSWORD`（≥8 位，缺失拒绝启动）。
- Token 生命周期：`token_expiration_hours` 缺省 1 小时；支持 TOTP（`secure-totp`）、账号锁定（`account-lockout`）、凭证清零（`account-credential-zeroize`）、异常检测（`anomalous-detector-dual`）与安全告警。
- 会话存储：进程内存（oxcache DAO），**auth 开启时仅限单副本**；外置会话需 garrison db 后端（规划中）。

### 请求边界防护

- XFF 信任反转：`trusted_proxies` 为空时忽略 `X-Forwarded-For`，使用直连地址（防头伪造限流绕过）。
- 速率限制：limiteron 令牌桶，全局 / 每 IP / 每用户 / 每 API 密钥多维独立计数，支持封禁管理（ban-manager）与 GCRA。
- 输入限制：单文本最大字节长度（`max_text_length`，防资源耗尽）、批量大小校验（`validate_batch_size`）。

### 文件与路径安全

- `/embed/file` 必须显式配置 `[server] grpc_allowed_roots` 允许根（不再回退 cwd）；单文件上限 10 MiB；`text_preview` 仅 admin。
- `PathValidator` 默认拒绝 `/`、`/etc` 等敏感目录。
- HF Hub `repo_id` 格式校验（`src/utils/hf_hub.rs` 的 `is_valid_hf_repo_id` + `build_hf_repo`）统一覆盖所有远程下载入口，防恶意配置注入与路径遍历（vuln-0009）。

### 数据与密钥保护

- AES-256-GCM 配置加密（`src/config/encryption.rs`），`VECBOOST_ENCRYPTION_KEY` / `VECBOOST_REQUIRE_ENCRYPTION` / `VECBOOST_KEY_STORAGE_TYPE` / `VECBOOST_KEY_FILE_PATH` 控制密钥存储。
- `SecretKey` 零化（zeroize）；环境变量 keystore 只读。
- 所有密钥脱敏与文本预览函数使用 UTF-8 字符边界安全切片（防多字节切片 panic）。

### 日志与可观测

- 审计日志（audit 段 + inklog）：用户、操作、资源、IP、时间戳。
- 错误响应脱敏：`sanitize_error_message` 统一清理；错误码经 i18n 双语翻译，不泄露内部细节。

---

## ⛓️ 供应链与安全门禁

| 门禁 | 工具 | 配置 |
|------|------|------|
| RustSec 公告扫描 | `cargo audit`（CI + release 前置校验） | `.cargo/audit.toml` |
| 依赖许可/禁用/漏洞门禁 | `cargo deny check` | `deny.toml`（licenses 白名单 MIT/Apache-2.0/BSD/ISC 等） |
| 静态安全分析 | CodeQL（Rust，push/PR/每周） | `.github/workflows/codeql.yml` |
| 镜像漏洞扫描 | Trivy（CRITICAL 即失败） | `.github/workflows/docker.yml` |
| Dockerfile 扫描 | Checkov | `.github/workflows/health-check.yml` |
| 私密信息扫描 | gitleaks | `.gitleaks.toml`（docs/target 等排除项） |
| 提交前检查 | pre-commit（fmt/clippy/check/build + MIT 版权头） | `.pre-commit-config.yaml` → `scripts/pre-commit.sh` |
| panic 面门禁 | `clippy::unwrap_used` + `RUSTFLAGS="-D warnings"` | CI env；测试经 `clippy.toml` 豁免 |
| Python 脚本扫描 | bandit（生产代码 src/、scripts/ 全量；tests/ 惯用 assert 误报排除） | `pyproject.toml` |

---

## ✅ 已知且已处置的风险

| 风险 | 处置 |
|------|------|
| RUSTSEC-2023-0071（Marvin Attack，RSA） | 豁免并持续跟踪：rsa 仅由 garrison → jsonwebtoken 传递引入用于 JWT 签名/验证，不涉及 RSA 解密路径；上游无补丁版本，见 `deny.toml` / `.cargo/audit.toml` 条目，上游发布 constant-time 实现后撤销豁免并升级 |
| RUSTSEC-2026-0258（h2 DoS） | 已修复：h2 0.4.15 → 0.4.19（v0.2.1） |
| vuln-0009（HF Hub repo_id 注入） | 已修复：`is_valid_hf_repo_id` 统一校验全部远程下载入口 |
| admin 密码缺失时任意凭据可得 admin（P0） | 已修复（Unreleased）：启用认证时缺失 `VECBOOST_ADMIN_PASSWORD` 拒绝启动 |

---

## 🛡️ 安全最佳实践

**生产部署**

- 始终通过 HTTPS 暴露服务；反代部署显式配置 `trusted_proxies`。
- 启用 `auth.enabled = true`，`VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD` 经环境变量或密钥管理注入，不入库不入镜像。
- 按需评估 `token_expiration_hours`（默认 1 小时）；纯 Bearer API 可显式关闭 CSRF。
- CORS 勿用通配符 `"*"`（等同公开接口）；显式列出可信来源。
- auth 开启时为单副本部署（会话在进程内存）；水平扩展需先外置会话（待 garrison db 后端）。
- 容器等受信网络如需非回环绑定，显式设置 `VECBOOST_ALLOW_INSECURE=1` 并理解 ERROR 告警含义。

**密钥与配置**

- 定期轮换 JWT 密钥；`/embed/file` 允许根配置最小化（仅业务目录）。
- 使用 `cargo audit` / `cargo deny check` 保持依赖更新；新增豁免必须附理由。
- 客户端不要依赖 `/embed/file` 的 `text_preview`（已收敛为 admin-only）。

---

## 📚 相关文档

| 文档 | 说明 |
|:-----|:-----|
| [📖 用户指南](USER_GUIDE.md) | 认证配置与部署指引 |
| [📘 API 参考](API_REFERENCE.md) | 认证端点与错误响应格式 |
| [🏗️ 架构文档](ARCHITECTURE.md) | 安全架构与错误处理设计 |
| [🧪 测试场景矩阵](TEST_SCENARIOS.md) | 安全场景（R-auth-001~011）覆盖 |
| [📋 更新日志](CHANGELOG.md) | 安全修复的版本记录 |
