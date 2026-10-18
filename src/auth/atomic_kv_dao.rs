// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! `AtomicKvDao`：在 garrison `DbnexusKvDao` 之上把 `incr` 提升为单语句原子自增。
//!
//! garrison 的 `DbnexusKvDao::incr` 为 `live_row`（读）+ `upsert`（写）两步，
//! 多副本并发失败登录时可能少计。本包装仅覆写 `incr`：
//! `INSERT ... ON CONFLICT (k) DO UPDATE ... RETURNING v`（SQLite/PostgreSQL），
//! 使「读取当前值 + 递增 + 写回」在数据库内一次完成。
//!
//! 其余方法全部委托内层（锁定路径只消费 `incr`；`decr` 等无跨副本强一致消费者）。

use async_trait::async_trait;
use dbnexus::DbPool;
use dbnexus::sea_orm::{ConnectionTrait, DbBackend, Statement, Value};
use garrison::dao::{DbnexusKvDao, GarrisonDao};
use garrison::error::{GarrisonError, GarrisonResult};

/// 原子 `incr` 包装（仅 database 会话后端使用）。
pub struct AtomicKvDao {
    inner: DbnexusKvDao,
    pool: DbPool,
}

impl AtomicKvDao {
    pub fn new(inner: DbnexusKvDao, pool: DbPool) -> Self {
        Self { inner, pool }
    }

    /// 幂等建表（委托内层 `garrison_kv`）。
    pub async fn ensure_table(&self) -> GarrisonResult<()> {
        self.inner.ensure_table().await
    }

    fn now_secs() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or_default()
    }

    fn expire_at_from_ttl(ttl_seconds: u64) -> Option<i64> {
        if ttl_seconds == 0 {
            None
        } else {
            Some(Self::now_secs() + ttl_seconds as i64)
        }
    }

    async fn incr_sql(&self, key: &str, ttl_seconds: u64) -> GarrisonResult<u64> {
        let session = self
            .pool
            .get_session("admin")
            .await
            .map_err(|e| GarrisonError::Dao(format!("atomic-incr-session::{e}")))?;
        let conn = session
            .connection()
            .map_err(|e| GarrisonError::Dao(format!("atomic-incr-connection::{e}")))?;
        let backend = conn.get_database_backend();
        let now = Self::now_secs();
        let expire_at = Self::expire_at_from_ttl(ttl_seconds);
        let sql = incr_sql_for_backend(backend)?;

        let values = vec![
            Value::String(Some(key.to_string())),
            Value::BigInt(expire_at),
            Value::BigInt(Some(now)),
            Value::BigInt(Some(now)),
        ];
        let stmt = Statement::from_sql_and_values(backend, sql, values);
        let row = conn
            .query_one_raw(stmt)
            .await
            .map_err(|e| GarrisonError::Dao(format!("atomic-incr-execute::{e}")))?;
        let row = row.ok_or_else(|| GarrisonError::Dao("atomic-incr-no-row".to_string()))?;
        let value: String = row
            .try_get("", "v")
            .map_err(|e| GarrisonError::Dao(format!("atomic-incr-read::{e}")))?;
        value
            .parse::<u64>()
            .map_err(|e| GarrisonError::Dao(format!("atomic-incr-parse::{key}::{value}::{e}")))
    }

    /// SQLite 写-写并发返回 `database is locked`（code 5，SQLITE_BUSY）；
    /// dbnexus 未暴露 busy_timeout 配置面（池级修复属上游边界），此处
    /// 以有界退避重试收口：WAL 启用后写锁窗口为单语句，重试即收敛。
    async fn incr_with_busy_retry(&self, key: &str, ttl_seconds: u64) -> GarrisonResult<u64> {
        const MAX_ATTEMPTS: usize = 5;
        const BACKOFF_MS: u64 = 20;
        let mut attempt = 1;
        loop {
            match self.incr_sql(key, ttl_seconds).await {
                Ok(value) => return Ok(value),
                Err(e) if attempt < MAX_ATTEMPTS && Self::is_sqlite_busy(&e) => {
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(
                        BACKOFF_MS * attempt as u64,
                    ))
                    .await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn is_sqlite_busy(err: &GarrisonError) -> bool {
        let text = err.to_string();
        text.contains("code: 5") || text.contains("database is locked")
    }

    /// SQLite 文件库启用 WAL（库级持久属性，一次设置对全部连接生效）：
    /// 多副本下读不再阻塞写，缩小写锁窗口。非 sqlite 后端跳过。
    async fn enable_sqlite_wal(&self) -> GarrisonResult<()> {
        let session = self
            .pool
            .get_session("admin")
            .await
            .map_err(|e| GarrisonError::Dao(format!("atomic-wal-session::{e}")))?;
        if session
            .connection()
            .map_err(|e| GarrisonError::Dao(format!("atomic-wal-connection::{e}")))?
            .get_database_backend()
            != DbBackend::Sqlite
        {
            return Ok(());
        }
        session
            .execute_raw("PRAGMA journal_mode=WAL")
            .await
            .map_err(|e| GarrisonError::Dao(format!("atomic-wal-enable::{e}")))?;
        Ok(())
    }
}

#[async_trait]
impl GarrisonDao for AtomicKvDao {
    async fn get(&self, key: &str) -> GarrisonResult<Option<String>> {
        self.inner.get(key).await
    }

    async fn set(&self, key: &str, value: &str, ttl_seconds: u64) -> GarrisonResult<()> {
        self.inner.set(key, value, ttl_seconds).await
    }

    async fn update(&self, key: &str, value: &str) -> GarrisonResult<()> {
        self.inner.update(key, value).await
    }

    async fn expire(&self, key: &str, seconds: u64) -> GarrisonResult<()> {
        self.inner.expire(key, seconds).await
    }

    async fn delete(&self, key: &str) -> GarrisonResult<()> {
        self.inner.delete(key).await
    }

    async fn set_if_absent(
        &self,
        key: &str,
        value: &str,
        ttl_seconds: u64,
    ) -> GarrisonResult<bool> {
        self.inner.set_if_absent(key, value, ttl_seconds).await
    }

    async fn rename(&self, old_key: &str, new_key: &str) -> GarrisonResult<()> {
        self.inner.rename(old_key, new_key).await
    }

    async fn get_and_delete(&self, key: &str) -> GarrisonResult<Option<String>> {
        self.inner.get_and_delete(key).await
    }

    /// 原子自增（覆盖内层的读-改-写实现）。
    async fn incr(&self, key: &str, ttl_seconds: u64) -> GarrisonResult<u64> {
        self.incr_with_busy_retry(key, ttl_seconds).await
    }

    async fn decr(&self, key: &str) -> GarrisonResult<u64> {
        self.inner.decr(key).await
    }

    async fn compare_and_update_if_greater(
        &self,
        key: &str,
        new_value: u64,
        ttl_seconds: u64,
    ) -> GarrisonResult<bool> {
        self.inner
            .compare_and_update_if_greater(key, new_value, ttl_seconds)
            .await
    }

    async fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&str>,
        new_value: &str,
        ttl_seconds: u64,
    ) -> GarrisonResult<bool> {
        self.inner
            .compare_and_swap(key, expected, new_value, ttl_seconds)
            .await
    }

    async fn eval_lua(
        &self,
        script: &str,
        keys: Vec<String>,
        args: Vec<String>,
    ) -> GarrisonResult<Vec<String>> {
        self.inner.eval_lua(script, keys, args).await
    }
}

/// 方言 SQL 选择（单语句原子自增 + 返回新值）。
fn incr_sql_for_backend(backend: DbBackend) -> GarrisonResult<&'static str> {
    match backend {
        DbBackend::Sqlite => Ok(
            "INSERT INTO garrison_kv (k, v, expire_at) VALUES (?, '1', ?) \
             ON CONFLICT(k) DO UPDATE SET \
               v = CASE WHEN garrison_kv.expire_at IS NOT NULL AND garrison_kv.expire_at <= ? \
                        THEN '1' ELSE CAST(CAST(garrison_kv.v AS INTEGER) + 1 AS TEXT) END, \
               expire_at = CASE WHEN garrison_kv.expire_at IS NOT NULL AND garrison_kv.expire_at <= ? \
                                THEN excluded.expire_at ELSE garrison_kv.expire_at END \
             RETURNING v",
        ),
        DbBackend::Postgres => Ok(
            "INSERT INTO garrison_kv (k, v, expire_at) VALUES ($1, '1', $2) \
             ON CONFLICT (k) DO UPDATE SET \
               v = CASE WHEN garrison_kv.expire_at IS NOT NULL AND garrison_kv.expire_at <= $3 \
                        THEN '1' ELSE (CAST(garrison_kv.v AS BIGINT) + 1)::text END, \
               expire_at = CASE WHEN garrison_kv.expire_at IS NOT NULL AND garrison_kv.expire_at <= $4 \
                                THEN excluded.expire_at ELSE garrison_kv.expire_at END \
             RETURNING v",
        ),
        other => Err(GarrisonError::Dao(format!(
            "atomic-incr-unsupported-backend::{other:?}"
        ))),
    }
}

/// 供会话层使用的统一入口：database 会话后端返回原子 DAO。
pub async fn build_atomic_dao(pool: DbPool) -> GarrisonResult<AtomicKvDao> {
    let inner = DbnexusKvDao::new(pool.clone());
    let dao = AtomicKvDao::new(inner, pool);
    dao.ensure_table().await?;
    dao.enable_sqlite_wal().await?;
    Ok(dao)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incr_sql_covers_sqlite_and_postgres_dialects() {
        let sqlite = incr_sql_for_backend(DbBackend::Sqlite).expect("sqlite sql");
        assert!(sqlite.contains("ON CONFLICT(k)"));
        assert!(sqlite.contains("RETURNING v"));
        assert!(sqlite.contains("CAST(CAST(garrison_kv.v AS INTEGER) + 1 AS TEXT)"));

        let postgres = incr_sql_for_backend(DbBackend::Postgres).expect("postgres sql");
        assert!(postgres.contains("ON CONFLICT (k)"));
        assert!(postgres.contains("RETURNING v"));
        assert!(postgres.contains("::text"));
        assert!(postgres.contains("$1"));
    }

    #[test]
    fn incr_sql_rejects_unsupported_dialect() {
        let err = incr_sql_for_backend(DbBackend::MySql).expect_err("mysql must be rejected");
        assert!(err.to_string().contains("atomic-incr-unsupported-backend"));
    }
}
