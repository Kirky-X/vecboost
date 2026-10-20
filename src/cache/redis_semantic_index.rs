// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 跨副本共享的语义缓存 trigram 索引。
//!
//! 现网模糊层是 trigram Jaccard 文本相似度（不是 embedding KNN）。
//! 本模块把**同一套 trigram 语义**搬到 Redis：每个模型一份 Hash
//! （trigrams JSON）与一份 embedding Hash，查询时 HGETALL 后在进程内
//! 计算 Jaccard，命中后只 HGET 最佳条目的向量。行为与单机
//! `SemanticCache::find_similar` 完全一致，只是索引跨副本共享。
//!
//! 设计取舍：
//! - 不引入 RediSearch：现网算法是文本 trigram，不是向量 KNN；用通用
//!   Redis Hash 即可保持行为，避免 `redis` crate 的 unstable search API。
//! - 容量与 LRU：每个模型一个 ZSET（score = 最后访问秒），插入脚本
//!   原子裁剪到 `capacity`，避免多副本并发插入时索引无界增长。
//! - 键空间：模型 ID / 文本均经 SHA-256 摘要，避免超长/控制字符键。

#![cfg(feature = "distributed-redis")]

use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::time::{SystemTime, UNIX_EPOCH};

use redis::AsyncCommands;
use sha2::{Digest, Sha256};

use crate::error::VecboostError;

use super::semantic_cache::{pack_trigrams, trigram_jaccard_with_set};

/// 原子插入 + LRU 裁剪脚本。
const INSERT_SCRIPT: &str = r#"
local over = redis.call('ZCARD', KEYS[3]) + 1 - tonumber(ARGV[4])
redis.call('SADD', KEYS[4], ARGV[6])
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HSET', KEYS[2], ARGV[1], ARGV[3])
redis.call('ZADD', KEYS[3], tonumber(ARGV[5]), ARGV[1])
if over > 0 then
  local old = redis.call('ZRANGE', KEYS[3], 0, over - 1)
  for i = 1, #old do
    redis.call('HDEL', KEYS[1], old[i])
    redis.call('HDEL', KEYS[2], old[i])
    redis.call('ZREM', KEYS[3], old[i])
  end
end
return redis.call('HLEN', KEYS[1])
"#;

const MODELS_KEY: &str = "vecboost:semantic:models";

fn digest(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    hex::encode(hasher.finalize())
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn map_err(op: &str, err: impl Display) -> VecboostError {
    VecboostError::internal_error(format!("semantic-redis-{op}: {err}"))
}

fn as_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>()
}

fn from_bytes(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

/// 跨副本 trigram 索引（`distributed-redis`）。
pub(crate) struct RedisSemanticIndex {
    conn: redis::aio::MultiplexedConnection,
}

impl RedisSemanticIndex {
    /// 连接 Redis 并做 PING 能力校验；失败即启动失败（fail-closed）。
    pub(crate) async fn connect(url: &str) -> Result<Self, VecboostError> {
        let client = redis::Client::open(url)
            .map_err(|e| VecboostError::config_error(format!("semantic-redis-url-invalid: {e}")))?;
        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| {
                VecboostError::config_error(format!("semantic-redis-connect-failed: {e}"))
            })?;
        let _: String = redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .map_err(|e| VecboostError::config_error(format!("semantic-redis-ping-failed: {e}")))?;
        Ok(Self { conn })
    }

    fn keys_for_hash(model: &str) -> (String, String, String) {
        (
            format!("vecboost:semantic:{model}:tri"),
            format!("vecboost:semantic:{model}:emb"),
            format!("vecboost:semantic:{model}:lru"),
        )
    }

    fn keys(model_id: &str) -> (String, String, String) {
        Self::keys_for_hash(&digest(model_id))
    }

    /// 写入一条记录并执行容量裁剪（原子 Lua）。
    pub(crate) async fn insert(
        &self,
        model_id: &str,
        text: &str,
        embedding: &[f32],
        capacity: usize,
    ) -> Result<usize, VecboostError> {
        let (tri_key, emb_key, lru_key) = Self::keys(model_id);
        let field = digest(text);
        let mut trigrams: Vec<u32> = pack_trigrams(text.as_bytes()).into_iter().collect();
        trigrams.sort_unstable();
        trigrams.dedup();
        let trigrams_json =
            serde_json::to_string(&trigrams).map_err(|e| map_err("serialize-trigrams", e))?;
        let embedding_bytes = as_bytes(embedding);
        let capacity = capacity.max(1);

        let mut conn = self.conn.clone();
        let model_hash = digest(model_id);
        // 直接 EVAL（而非 EVALSHA+Script）：避免新脚本首次 NOSCRIPT 回退路径
        // 在 multiplexed connection 上留下悬挂应答；插入是低频路径，脚本体
        // 传输成本可接受。
        let inserted: usize = redis::cmd("EVAL")
            .arg(INSERT_SCRIPT)
            .arg(4)
            .arg(&tri_key)
            .arg(&emb_key)
            .arg(&lru_key)
            .arg(MODELS_KEY)
            .arg(&field)
            .arg(&trigrams_json)
            .arg(&embedding_bytes)
            .arg(capacity)
            .arg(now_secs())
            .arg(&model_hash)
            .query_async(&mut conn)
            .await
            .map_err(|e| map_err("insert", e))?;
        Ok(inserted)
    }

    /// 按 trigram Jaccard 查找最佳命中，返回对应 embedding。
    pub(crate) async fn find_similar(
        &self,
        model_id: &str,
        query: &str,
        fuzzy_threshold: f32,
    ) -> Result<Option<Vec<f32>>, VecboostError> {
        let (tri_key, emb_key, lru_key) = Self::keys(model_id);
        let mut conn = self.conn.clone();
        let entries: HashMap<String, String> = conn
            .hgetall(&tri_key)
            .await
            .map_err(|e| map_err("hgetall-trigrams", e))?;
        if entries.is_empty() {
            return Ok(None);
        }

        let query_trigrams = pack_trigrams(query.as_bytes());
        let query_len = query_trigrams.len();
        let mut best: Option<(f32, String)> = None;
        for (field, payload) in entries {
            let stored: Vec<u32> = match serde_json::from_str(&payload) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let stored_len = stored.len();
            let upper_bound =
                query_len.min(stored_len) as f32 / query_len.max(stored_len).max(1) as f32;
            if upper_bound < fuzzy_threshold {
                continue;
            }
            let stored_set: HashSet<u32> = stored.into_iter().collect();
            let similarity = trigram_jaccard_with_set(&query_trigrams, &stored_set);
            if similarity >= fuzzy_threshold
                && best
                    .as_ref()
                    .map(|(current, _)| similarity > *current)
                    .unwrap_or(true)
            {
                best = Some((similarity, field));
            }
        }

        let Some((_, field)) = best else {
            return Ok(None);
        };
        // 命中刷新 LRU（与本地热缓存的 last_access 语义一致）。
        let _: usize = conn
            .zadd(&lru_key, &field, now_secs())
            .await
            .map_err(|e| map_err("touch-lru", e))?;
        let bytes: Option<Vec<u8>> = conn
            .hget(&emb_key, &field)
            .await
            .map_err(|e| map_err("hget-embedding", e))?;
        match bytes {
            Some(bytes) => from_bytes(&bytes).map(Some).ok_or_else(|| {
                VecboostError::internal_error("semantic-redis-embedding-corrupt".to_string())
            }),
            None => Ok(None),
        }
    }

    /// 清空指定模型的共享索引。
    pub(crate) async fn clear(&self, model_id: &str) -> Result<(), VecboostError> {
        let (tri_key, emb_key, lru_key) = Self::keys(model_id);
        let mut conn = self.conn.clone();
        let _: usize = conn
            .del((tri_key, emb_key, lru_key))
            .await
            .map_err(|e| map_err("clear", e))?;
        let _: usize = conn
            .srem(MODELS_KEY, digest(model_id))
            .await
            .map_err(|e| map_err("clear-model-set", e))?;
        Ok(())
    }

    /// 清空全部模型的共享索引（模型卸载/全局 reset）。
    pub(crate) async fn clear_all(&self) -> Result<(), VecboostError> {
        let mut conn = self.conn.clone();
        let models: Vec<String> = conn
            .smembers(MODELS_KEY)
            .await
            .map_err(|e| map_err("list-models", e))?;
        for model in models {
            let (tri_key, emb_key, lru_key) = Self::keys_for_hash(&model);
            let _: usize = conn
                .del((tri_key, emb_key, lru_key))
                .await
                .map_err(|e| map_err("clear-all", e))?;
        }
        let _: usize = conn
            .del(MODELS_KEY)
            .await
            .map_err(|e| map_err("clear-all-models", e))?;
        Ok(())
    }
}
