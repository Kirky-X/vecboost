# Spec — configuration

> Delta spec for change `vecboost-v0.2.0-ecosystem-refactor`. 覆盖 confers 配置管理能力域需求。

## Requirements

### R-configuration-001: confers derive 配置派生

`src/config/app_config.rs` 用 `#[derive(Config)]` 派生 `AppConfig`/`ServerConfig`/`ModelConfig`/`AuthConfig`/`CacheConfig`/`DatabaseConfig`/`LoggingConfig`/`FlowControlConfig`,从 TOML 文件 + 环境变量加载,支持嵌套段。

**验收标准:**
- `AppConfig::load("config_minimal.toml")` 返回 `Ok(AppConfig)` 含默认值
- 环境变量 `VECBOOST_JWT_SECRET` 覆盖 TOML 中 `[auth].jwt_secret`
- 环境变量 `VECBOOST_SERVER_PORT` 覆盖 `[server].port`
- 嵌套段 `[database]`/`[logging]`/`[flow_control]`/`[cache]` 正确解析

### R-configuration-002: 热重载订阅

`AppConfig::subscribe(callback)` 注册回调,配置文件变更时触发回调接收新配置。底层使用 confers `watch` feature。

**验收标准:**
- 修改 `config.toml` 后 5 秒内回调被调用
- 回调收到的新 `AppConfig` 反映文件变更
- 多个订阅者按注册顺序依次调用

### R-configuration-003: 配置加密存储(可选)

启用 `encryption` feature 时,`AppConfig::set_encrypted(key, value)` 用 XChaCha20-Poly1305 加密敏感字段(jwt_secret/db_password),`get_encrypted(key)` 解密读取。

**v0.2.0 实际实施状态(未实现,推迟到 v0.3.0)**:
- `Cargo.toml` **无 `encryption` feature**
- 无 `AppConfig::set_encrypted` / `get_encrypted` 方法
- 无 XChaCha20-Poly1305 加密集成

**验收标准(目标,v0.3.0 实现):**
- 加密字段在内存中为密文,`std::mem::inspect` 无法读取明文
- 密钥从环境变量 `VECBOOST_CONFIG_KEY` 读取(32 字节 base64)
- 错误密钥解密返回 `Err(ConfigError::DecryptionFailed)`

## Constraints

- confers 通过 `config` feature 启用,**v0.2.0 实际 `default = ["http", "oxcache", "limiteron"]`**(原 spec 写 `default = ["http", "config"]`,config 未在 default 中,偏离原因见 `design.md` D9)
- 旧 `src/config/app.rs` 保留一个版本周期,内部转为 confers 加载
- 配置文件路径优先级:CLI `--config` > `VECBOOST_CONFIG` 环境变量 > `./config.toml` > 默认值

## Out of Scope

- 不实现 YAML 配置(本轮仅 TOML)
- 不实现配置中心远程拉取(Consul/etcd 留待下个 change)
- 不实现配置 schema 验证(仅类型验证)
