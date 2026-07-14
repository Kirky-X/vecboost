# Spec — persistence

> Delta spec for change `vecboost-v0.2.0-ecosystem-refactor`. 覆盖 dbnexus 持久化能力域需求。

## Requirements

### R-persistence-001: DbPool 初始化

`src/db/mod.rs` 定义 `DbModule` 实现 `AutoBuilder`,`build()` 调用 `dbnexus::DbPoolBuilder::new().url(url).max_connections(10).build().await`。默认 SQLite,通过 `postgres` feature 切换 PostgreSQL。

**验收标准:**
- `DbModule::build()` with `sqlite::memory:` 返回 `Arc<DbPool>`
- `DbModule::build()` with `postgres://user:pass@localhost/db` 返回 `Arc<DbPool>`(需 postgres feature)
- 连接池 max_connections 从 `[database].max_connections` 读取,默认 10
- 连接失败返回 `Err(VecboostError::Config(...))`

### R-persistence-002: 用户存储持久化

`src/auth/user_store.rs` 重构为基于 `dbnexus::Session`。新增 `migrations/001_users.sql` 建表:`id INTEGER PK`/`username TEXT UNIQUE`/`password_hash TEXT`/`role TEXT`/`created_at TIMESTAMP`/`updated_at TIMESTAMP`。

**验收标准:**
- `UserStore::create(username, password)` 插入用户,返回 `User`
- `UserStore::get(username)` 查询用户,不存在返回 `None`
- `UserStore::verify_password(username, password)` 验证 argon2 哈希
- `UserStore::update_role(username, role)` 更新角色
- 重启服务后用户数据保留(SQLite 文件模式)

### R-persistence-003: 审计日志持久化

`src/audit/mod.rs` 重构为基于 `dbnexus::Session`。新增 `migrations/001_audit_logs.sql` 建表:`id INTEGER PK`/`user_id TEXT`/`action TEXT`/`resource TEXT`/`ip TEXT`/`timestamp INTEGER`/`metadata TEXT`。

**v0.2.0 实际实施状态(双轨实现 + 部分功能推迟)**:
- `src/audit/mod.rs` **同时支持文件日志和 DB 日志(双轨)**:
  - `AuditLogger::new(config)` 走文件日志路径(`tokio::fs::OpenOptions`,line 9)
  - `AuditLogger::new_with_db(config, db_pool)` 走 DB 日志路径(line 116)
  - 文件日志路径、轮转逻辑仍存在(line 90-93, 300-313, 451-455)
- `migrations/001_audit_logs.sql` 建表语句已存在 ✓
- `AuditLogger::log(user_id, action, resource, ip)` 写入 `audit_logs` 表 ✓
- 审计日志不可修改(无 update/delete API)✓

**推迟到 v0.3.0:**
- **`AuditLogger::query(filter)` 按条件筛选**:按 user_id/action/时间范围筛选,当前未实现
- **审计日志自动清理 90 天前记录**:可配置 `[audit].retention_days`,当前未实现
- **完整 DB-only 迁移**:移除文件日志路径,实现 DB-only 单轨

**验收标准(按实际代码):**
- `AuditLogger::log(user_id, action, resource, ip)` 写入 `audit_logs` 表
- 审计日志不可修改(无 update/delete API)

**推迟到 v0.3.0:**
- `AuditLogger::query(filter)` 支持按 user_id/action/时间范围筛选
- 审计日志自动清理 90 天前记录(可配置 `[audit].retention_days`)

### R-persistence-004: 权限控制

启用 dbnexus `permission` feature,基于角色的表级访问控制。`admin` 角色可 CRUD 所有表,`user` 角色仅可读 `audit_logs`(自己的记录),`service` 角色可写 `audit_logs` 但不可读。

**验收标准:**
- `admin` 用户 `session = pool.get_session("admin")` 可执行 `SELECT * FROM users`
- `user` 用户 `session = pool.get_session("user")` 执行 `SELECT * FROM users` 返回 `PermissionError`
- 权限规则从 `[database].permissions` 配置加载

### R-persistence-005: 嵌入元数据存储(可选)

启用 `metadata-persistence` feature 时,新增 `migrations/001_embeddings.sql` 建表:`id INTEGER PK`/`text TEXT`/`embedding BLOB`/`dimension INTEGER`/`model TEXT`/`created_at TIMESTAMP`/`text_hash TEXT UNIQUE`。

**v0.2.0 实际实施状态(未实现,推迟到 v0.3.0)**:
- `Cargo.toml` **无 `metadata-persistence` feature**
- 无 `EmbeddingStore` 结构
- 无 `migrations/001_embeddings.sql` 建表语句
- 无向量相似度搜索实现

**验收标准(目标,v0.3.0 实现):**
- `EmbeddingStore::put(text, embedding)` 存储,重复 text_hash 更新而非插入
- `EmbeddingStore::get(text_hash)` 查询,命中返回 `Option<Embedding>`
- `EmbeddingStore::search(embedding, top_k)` 向量相似度搜索(余弦相似度)
- 存储大小超过 `[metadata].max_size_mb` 时按 LRU 驱逐

## Constraints

- dbnexus 通过 `db` feature 启用(默认 SQLite),`postgres` feature 切换 PostgreSQL
- 迁移文件放在 `src/db/migrations/` 目录,启动时自动执行
- 连接池 RAII 管理,无需手动 close
- 不启用 dbnexus 数据分片(`sharding` feature 留待下个 change)

## Out of Scope

- 不启用 dbnexus 全局索引(`global-index` feature 不启用)
- 不启用 dbnexus 高级权限引擎(`permission-engine` feature 不启用)
- 不启用 dbnexus OpenTelemetry 追踪(`tracing` feature 留待下个 change)
- 不实现 MySQL 支持(本轮仅 SQLite + PostgreSQL)
