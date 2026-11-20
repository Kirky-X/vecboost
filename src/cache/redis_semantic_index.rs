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

/// 键命名空间前缀（`keys_for_hash` 与 Lua 清理脚本共用的单一事实源）。
const KEY_NAMESPACE: &str = "vecboost:semantic:";

/// clear(model) 原子脚本：清理基础键 + SREM 基础登记 + 脚本内快照登记
/// 成员、按 `{model_hash}:` 前缀过滤 caller 分区并逐分区 DEL + SREM，
/// 一次 EVAL 完成。Redis 脚本执行原子（单线程串行），消除旧实现
/// 「SMEMBERS 快照 → 逐成员串行 DEL+SREM（2N+3 RTT）」与并发
/// insert_for_caller 之间的残留窗口（快照后落库的分区不被旧快照覆盖）。
/// KEYS[1]=MODELS_KEY；KEYS[2..4]=基础 tri/emb/lru；ARGV[1]=model_hash、
/// ARGV[2]=键命名空间（分区键在脚本内按成员拼接，与 [`keys_for_hash`]
/// 同构）。单实例语义：成员派生键无法预declared进 KEYS（与旧实现逐成员
/// DEL 的键空间假设一致）。
const CLEAR_MODEL_SCRIPT: &str = r#"
local set_key = KEYS[1]
local model_hash = ARGV[1]
local ns = ARGV[2]
local prefix = model_hash .. ':'
redis.call('DEL', KEYS[2], KEYS[3], KEYS[4])
redis.call('SREM', set_key, model_hash)
local members = redis.call('SMEMBERS', set_key)
for i = 1, #members do
  local m = members[i]
  if string.sub(m, 1, #prefix) == prefix then
    local base = ns .. m
    redis.call('DEL', base .. ':tri', base .. ':emb', base .. ':lru')
    redis.call('SREM', set_key, m)
  end
end
return 1
"#;

/// clear_all 原子脚本：逐登记成员 DEL 三件键 + DEL 登记集合一次完成
///（成员含基础 `model_hash` 与 caller 分区命名空间串，[`keys_for_hash`]
/// 对两者同构）。旧实现逐成员串行 DEL（N+1 RTT、非原子）收敛为单 EVAL。
const CLEAR_ALL_SCRIPT: &str = r#"
local set_key = KEYS[1]
local ns = ARGV[1]
local members = redis.call('SMEMBERS', set_key)
for i = 1, #members do
  local base = ns .. members[i]
  redis.call('DEL', base .. ':tri', base .. ':emb', base .. ':lru')
end
redis.call('DEL', set_key)
return #members
"#;

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
    VecboostError::internal_error(crate::i18n::tr_with_args(
        "cache-semantic-redis-op-failed",
        crate::i18n::tr_args(&[("op", op), ("detail", &err.to_string())]),
    ))
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
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
        let client = redis::Client::open(url).map_err(|e| {
            VecboostError::config_error(crate::i18n::tr_with_args(
                "cache-semantic-redis-url-invalid",
                crate::i18n::tr_args(&[("detail", &e.to_string())]),
            ))
        })?;
        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| {
                VecboostError::config_error(crate::i18n::tr_with_args(
                    "cache-semantic-redis-connect-failed",
                    crate::i18n::tr_args(&[("detail", &e.to_string())]),
                ))
            })?;
        let _: String = redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .map_err(|e| {
                VecboostError::config_error(crate::i18n::tr_with_args(
                    "cache-semantic-redis-ping-failed",
                    crate::i18n::tr_args(&[("detail", &e.to_string())]),
                ))
            })?;
        Ok(Self { conn })
    }

    fn keys_for_hash(model: &str) -> (String, String, String) {
        (
            format!("{KEY_NAMESPACE}{model}:tri"),
            format!("{KEY_NAMESPACE}{model}:emb"),
            format!("{KEY_NAMESPACE}{model}:lru"),
        )
    }

    fn keys(model_id: &str) -> (String, String, String) {
        Self::keys_for_hash(&digest(model_id))
    }

    /// 调用方分区键：user 隔离模式下 caller 非空，键名附加 caller 摘要，
    /// 使不同调用方落在独立 Redis 哈希命名空间，多副本共享层不成为隔离旁路；
    /// off 模式 caller 为空串，键与 [`Self::keys`] 完全一致（行为不变）。
    fn keys_with_caller(model_id: &str, caller_id: &str) -> (String, String, String) {
        if caller_id.is_empty() {
            Self::keys(model_id)
        } else {
            Self::keys_for_hash(&format!("{}:{}", digest(model_id), digest(caller_id)))
        }
    }

    /// 写入一条记录并执行容量裁剪（原子 Lua）。原签名，caller 为空分区。
    #[cfg(test)]
    pub(crate) async fn insert(
        &self,
        model_id: &str,
        text: &str,
        embedding: &[f32],
        capacity: usize,
    ) -> Result<usize, VecboostError> {
        self.insert_for_caller(model_id, "", text, embedding, capacity)
            .await
    }

    /// 带调用方分区的写入（user 隔离模式的真实入口）。
    ///
    /// caller 分区同样登记进 [`MODELS_KEY`]：成员为
    /// `{model_hash}:{caller_digest}`（与 [`Self::keys_with_caller`] 的命名
    /// 空间串一致），`clear`/`clear_all` 据此清空全部分区键——分区键无 TTL，
    /// 不登记即随模型切换/卸载残留。off 模式（caller 空串）成员仍为基础
    /// `model_hash`，登记与清键行为不变。
    pub(crate) async fn insert_for_caller(
        &self,
        model_id: &str,
        caller_id: &str,
        text: &str,
        embedding: &[f32],
        capacity: usize,
    ) -> Result<usize, VecboostError> {
        let (tri_key, emb_key, lru_key) = Self::keys_with_caller(model_id, caller_id);
        let field = digest(text);
        let mut trigrams: Vec<u32> = pack_trigrams(text.as_bytes()).into_iter().collect();
        trigrams.sort_unstable();
        trigrams.dedup();
        let trigrams_json =
            serde_json::to_string(&trigrams).map_err(|e| map_err("serialize-trigrams", e))?;
        let embedding_bytes = as_bytes(embedding);
        let capacity = capacity.max(1);

        let mut conn = self.conn.clone();
        let member = if caller_id.is_empty() {
            digest(model_id)
        } else {
            format!("{}:{}", digest(model_id), digest(caller_id))
        };
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
            .arg(&member)
            .query_async(&mut conn)
            .await
            .map_err(|e| map_err("insert", e))?;
        Ok(inserted)
    }

    /// 按 trigram Jaccard 查找最佳命中，返回对应 embedding。原签名，caller 为空分区。
    #[cfg(test)]
    pub(crate) async fn find_similar(
        &self,
        model_id: &str,
        query: &str,
        fuzzy_threshold: f32,
    ) -> Result<Option<Vec<f32>>, VecboostError> {
        self.find_similar_for_caller(model_id, "", query, fuzzy_threshold)
            .await
    }

    /// 带调用方分区的查找（user 隔离模式的真实入口）。
    pub(crate) async fn find_similar_for_caller(
        &self,
        model_id: &str,
        caller_id: &str,
        query: &str,
        fuzzy_threshold: f32,
    ) -> Result<Option<Vec<f32>>, VecboostError> {
        let (tri_key, emb_key, lru_key) = Self::keys_with_caller(model_id, caller_id);
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

    /// 清空指定模型的共享索引（基础命名空间 + 全部 caller 分区）。
    ///
    /// 单 EVAL 原子完成（架构 LOW-4）：清键集合与旧串行实现逐字节一致——
    /// 基础 tri/emb/lru + SREM 基础登记 + 按 `{model_hash}:` 前缀过滤出的
    /// caller 分区三件键与登记成员；差别仅在快照与删除合并进同一原子脚本，
    /// 并发 insert_for_caller 不再产生「快照后落库、漏清残留」的窗口。
    pub(crate) async fn clear(&self, model_id: &str) -> Result<(), VecboostError> {
        let (tri_key, emb_key, lru_key) = Self::keys(model_id);
        let model_hash = digest(model_id);
        let mut conn = self.conn.clone();
        let _: usize = redis::cmd("EVAL")
            .arg(CLEAR_MODEL_SCRIPT)
            .arg(4)
            .arg(MODELS_KEY)
            .arg(&tri_key)
            .arg(&emb_key)
            .arg(&lru_key)
            .arg(&model_hash)
            .arg(KEY_NAMESPACE)
            .query_async(&mut conn)
            .await
            .map_err(|e| map_err("clear", e))?;
        Ok(())
    }

    /// 清空全部模型的共享索引（模型卸载/全局 reset）。
    ///
    /// 成员既含基础 `model_hash`，也含 caller 分区命名空间串
    /// `{model_hash}:{caller_digest}`；脚本按成员同构清键，同时覆盖基础键
    /// 与全部分区键（off 模式成员只有前者，清键集合与旧实现一致）。
    pub(crate) async fn clear_all(&self) -> Result<(), VecboostError> {
        let mut conn = self.conn.clone();
        let _: usize = redis::cmd("EVAL")
            .arg(CLEAR_ALL_SCRIPT)
            .arg(1)
            .arg(MODELS_KEY)
            .arg(KEY_NAMESPACE)
            .query_async(&mut conn)
            .await
            .map_err(|e| map_err("clear-all", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 隔离键分区钉：user 模式下不同 caller 命中不同 Redis 键命名空间，
    /// off 模式（caller 空串）与原有 `keys()` 完全一致。
    #[test]
    fn test_keys_with_caller_partition() {
        let (tri_off, _, _) = RedisSemanticIndex::keys_with_caller("model-a", "");
        let (tri_base, _, _) = RedisSemanticIndex::keys("model-a");
        assert_eq!(tri_off, tri_base, "off 模式键必须与既有实现一致");

        let (tri_alice, _, _) = RedisSemanticIndex::keys_with_caller("model-a", "alice");
        let (tri_bob, _, _) = RedisSemanticIndex::keys_with_caller("model-a", "bob");
        assert_ne!(tri_alice, tri_bob, "不同调用方必须落在不同键分区");
        assert_ne!(tri_alice, tri_off, "user 分区不得与 off 键空间重合");
    }

    /// 每个用例占一个独立的逻辑库（DB 11..=15），避免与 `tests/*_redis.rs`
    /// （默认库）以及并发跑的自身用例互相干扰：本模块的键空间是全局
    /// `vecboost:semantic:models`，同库跑 `clear_all` 会掉光别的用例数据。
    const SLOT_A: u8 = 11;
    const SLOT_B: u8 = 12;
    const SLOT_C: u8 = 13;
    const SLOT_D: u8 = 14;
    const SLOT_E: u8 = 15;

    /// 未设 `VECBOOST_TEST_REDIS_URL` 时整组 SKIP（与仓内资产型测试同一语义）。
    fn base_url() -> Option<String> {
        match std::env::var("VECBOOST_TEST_REDIS_URL") {
            Ok(url) if !url.trim().is_empty() => Some(url.trim().trim_end_matches('/').to_string()),
            _ => {
                eprintln!("SKIP: VECBOOST_TEST_REDIS_URL not set");
                None
            }
        }
    }

    fn slot_url(slot: u8) -> Option<String> {
        base_url().map(|base| format!("{base}/{slot}"))
    }

    async fn connect_slot(slot: u8) -> Option<RedisSemanticIndex> {
        let url = slot_url(slot)?;
        Some(
            RedisSemanticIndex::connect(&url)
                .await
                .expect("本地/CI Redis 应可连接"),
        )
    }

    async fn raw_conn(slot: u8) -> redis::aio::MultiplexedConnection {
        let url = slot_url(slot).expect("slot_url 仅在 URL 存在时调用");
        redis::Client::open(url)
            .expect("client")
            .get_multiplexed_async_connection()
            .await
            .expect("raw connection")
    }

    /// 每个用例_own_ 的模型名：同一 slot 内并发用例也不会互看对方索引。
    fn unique_model(tag: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("model-{tag}-{}-{nanos}", std::process::id())
    }

    // ── 纯函数钉（不需要 Redis，永远跑） ──────────────────────────────

    #[test]
    fn key_layout_is_deterministic_and_digest_pinned() {
        // 同一模型两次推导必须同一组键（跨副本共享的前提）。
        let (tri1, emb1, lru1) = RedisSemanticIndex::keys("m-integration");
        let (tri2, emb2, lru2) = RedisSemanticIndex::keys("m-integration");
        assert_eq!(
            (tri1.clone(), emb1.clone(), lru1.clone()),
            (tri2, emb2, lru2)
        );
        assert!(tri1.starts_with("vecboost:semantic:"));
        assert!(tri1.ends_with(":tri") && emb1.ends_with(":emb") && lru1.ends_with(":lru"));
        // 模型名不得裸拼进键（超长/控制字符防护）：取的是 SHA-256 摘要。
        assert!(!tri1.contains("m-integration"));
        assert_eq!(digest("m-integration").len(), 64);
        assert_eq!(
            digest("same-input"),
            digest("same-input"),
            "digest 必须是纯函数"
        );
        assert_ne!(digest("a"), digest("b"));
        // keys_for_hash 与 keys 只在是否再摘要上差异：直接给摘要结果应一致。
        let (tri3, _, _) = RedisSemanticIndex::keys_for_hash(&digest("m-integration"));
        assert_eq!(tri1, tri3);
    }

    #[test]
    fn embedding_byte_codec_round_trip_and_guards() {
        let vec_in: Vec<f32> = vec![0.0, -1.5, f32::MIN, f32::MAX, 1.0 / 3.0];
        let bytes = as_bytes(&vec_in);
        assert_eq!(bytes.len(), vec_in.len() * 4);
        assert_eq!(from_bytes(&bytes).expect("往返可读"), vec_in);
        // 非 4 字节倍数必须是 None（上层据此报 corrupt，而不是静默截断）。
        assert!(from_bytes(&bytes[..bytes.len() - 1]).is_none());
        assert_eq!(
            from_bytes(&[]).expect("空切片是合法零维向量"),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn now_secs_and_map_err_shape() {
        crate::i18n::init();
        let first = now_secs();
        assert!(first > 1_700_000_000.0, "必须是真实 UNIX 秒：{first}");
        assert!(now_secs() >= first, "同一进程内不得回退");
        let err = map_err("insert", std::io::Error::other("boom"));
        let detail = err.error_detail();
        let expected = crate::i18n::tr_with_args(
            "cache-semantic-redis-op-failed",
            crate::i18n::tr_args(&[("op", "insert"), ("detail", "boom")]),
        );
        assert_eq!(
            detail,
            expected.as_str(),
            "错误需同时点名操作与原因：{detail}"
        );
    }

    // ── 连接与 fail-closed ──────────────────────────────────────────

    #[tokio::test]
    async fn connect_rejects_invalid_url_and_unreachable_endpoint() {
        crate::i18n::init();
        // 非法 URL：解析期即拒，不进入连接阶段。（不用 expect_err：
        // RedisSemanticIndex 持有 MultiplexedConnection 无法 derive Debug）
        let bad = match RedisSemanticIndex::connect("not-a-redis-url-at-all").await {
            Ok(_) => panic!("非法 URL 必须拒绝"),
            Err(e) => e,
        };
        let url_invalid_prefix = crate::i18n::tr_with_args(
            "cache-semantic-redis-url-invalid",
            crate::i18n::tr_args(&[("detail", "")]),
        );
        assert!(
            bad.error_detail().contains(&url_invalid_prefix),
            "实际：{}",
            bad.error_detail()
        );
        // 端点不可达：连接阶段即拒（启动期 fail-closed，不降级为无模糊层）。
        let unreachable = match RedisSemanticIndex::connect("redis://127.0.0.1:1/11").await {
            Ok(_) => panic!("不可达端点必须拒绝"),
            Err(e) => e,
        };
        let connect_failed_prefix = crate::i18n::tr_with_args(
            "cache-semantic-redis-connect-failed",
            crate::i18n::tr_args(&[("detail", "")]),
        );
        assert!(
            unreachable.error_detail().contains(&connect_failed_prefix),
            "实际：{}",
            unreachable.error_detail()
        );
    }

    // ── 插入 / 相似查询 ────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn insert_then_find_similar_returns_exact_vector_and_touches_lru() {
        let Some(index) = connect_slot(SLOT_A).await else {
            return;
        };
        let model = unique_model("roundtrip");
        let stored = vec![0.25f32, -0.5, 1.0 / 3.0, 12345.5];
        let inserted = index
            .insert(&model, "the quick brown fox", &stored, 8)
            .await
            .expect("insert 必须成功");
        assert_eq!(inserted, 1, "首次插入后 HLEN 应为 1");

        // 完全同文：Jaccard = 1.0。
        let hit = index
            .find_similar(&model, "the quick brown fox", 0.8)
            .await
            .expect("find_similar")
            .expect("同文必须命中");
        assert_eq!(hit, stored, "f32 字节往返必须逐位相等");

        // 小扰动仍应命中（trigram 相似度高于阈值）。
        index
            .insert(&model, "the quick brown fix", &[9.0f32; 4], 8)
            .await
            .expect("第二条插入");
        let near = index
            .find_similar(&model, "the quick brown fox", 0.5)
            .await
            .expect("find_similar near")
            .expect("阈值降到 0.5 仍应命中");
        assert_eq!(near, stored, "同文条目必须优于近似条目胜出");

        // 命中必须刷新 LRU score（与本机热缓存 last_access 语义对齐）。
        let (tri_key, _, lru_key) = RedisSemanticIndex::keys(&model);
        let field = digest("the quick brown fox");
        let mut raw = raw_conn(SLOT_A).await;
        assert_eq!(
            index
                .find_similar(&model, "the quick brown fox", 0.8)
                .await
                .expect("二次查询")
                .expect("二次查询必须仍命中"),
            stored
        );
        let refreshed: Option<f64> = redis::cmd("ZSCORE")
            .arg(&lru_key)
            .arg(&field)
            .query_async(&mut raw)
            .await
            .expect("ZSCORE");
        assert!(refreshed.is_some(), "命中后 LRU 条目必须存在");
        assert!(
            refreshed.expect("上面已断言 Some") <= now_secs(),
            "LRU score 不得写到未来"
        );

        index.clear(&model).await.expect("clear");
        assert_eq!(tri_key.len(), RedisSemanticIndex::keys(&model).0.len());
    }

    #[tokio::test]
    async fn find_similar_misses_on_empty_model_and_below_threshold() {
        let Some(index) = connect_slot(SLOT_B).await else {
            return;
        };
        let empty = unique_model("empty");
        assert_eq!(
            index
                .find_similar(&empty, "anything at all", 0.3)
                .await
                .expect("空模型查询不得报错"),
            None,
            "空索引必须直接 None"
        );

        let model = unique_model("threshold");
        index
            .insert(&model, "aaaaaaaaaaaaaaaaaaaa", &[1.0f32; 2], 4)
            .await
            .expect("insert");
        // 完全不共享 trigram 的查询：upper-bound 剪枝与阈值判定都要给 None。
        assert_eq!(
            index
                .find_similar(&model, "zzzzzzzzzzzzzzzzzzzz", 0.9)
                .await
                .expect("不相似查询不得报错"),
            None,
            "零交集输入不得返回命中"
        );
        // 有部分交集但低于高阈值：同样 None（阈值是行为契约，不是建议）。
        assert_eq!(
            index
                .find_similar(&model, "aaaaaaaaaaaaaaaaaaab", 0.999)
                .await
                .expect("近似查询不得报错"),
            None,
            "阈值 0.999 下近似文本也不得命中"
        );
        assert!(
            index
                .find_similar(&model, "aaaaaaaaaaaaaaaaaaab", 0.5)
                .await
                .expect("同一输入降阈值不得报错")
                .is_some(),
            "同一输入在低阈值下必须命中（证明上一行是阈值在起作用）"
        );
        index.clear(&model).await.expect("clear");
    }

    #[tokio::test]
    async fn capacity_eviction_is_atomic_and_zero_clamps_to_one() {
        let Some(index) = connect_slot(SLOT_C).await else {
            return;
        };
        // 三条互不共享 trigram 的文本，capacity=2 → 最旧一条必须被原子裁掉。
        let model = unique_model("evict");
        let texts = ["kkkkkkkkkkkk", "mmmmmmmmmmmm", "pppppppppppp"];
        for (i, text) in texts.iter().enumerate() {
            let len = index
                .insert(&model, text, &[i as f32; 2], 2)
                .await
                .expect("insert with capacity 2");
            assert!(len <= 2, "裁剪后 HLEN 不得超过 capacity，实际 {len}");
        }
        assert_eq!(
            index
                .find_similar(&model, texts[0], 0.9)
                .await
                .expect("查已淘汰条目不得报错"),
            None,
            "最旧条目必须已被淘汰"
        );
        assert!(
            index
                .find_similar(&model, texts[2], 0.9)
                .await
                .expect("查最新条目不得报错")
                .is_some(),
            "最新条目必须仍在索引里"
        );

        // capacity=0 必须被抬到 1（否则脚本会裁光整个索引）。
        let clamp = unique_model("clamp");
        index
            .insert(&clamp, "first entry text", &[1.0f32; 2], 0)
            .await
            .expect("capacity=0 插入");
        let second = index
            .insert(&clamp, "second entry text", &[2.0f32; 2], 0)
            .await
            .expect("capacity=0 二次插入");
        assert_eq!(second, 1, "capacity 夹到 1 后只能留一条");
        assert_eq!(
            index
                .find_similar(&clamp, "second entry text", 0.9)
                .await
                .expect("查存活条目")
                .expect("存活条目必须命中"),
            vec![2.0f32; 2]
        );
        index.clear(&model).await.expect("clear model");
        index.clear(&clamp).await.expect("clear clamp");
    }

    // ── caller 分区清理（user 模式 clear/clear_all 必须清掉分区键） ──

    /// clear(model) 必须同时清空该模型全部 caller 分区键与其 MODELS_KEY
    /// 登记：分区键无 TTL，模型切换/卸载后漏清即残留。
    #[tokio::test]
    async fn clear_model_removes_caller_partitions_and_registrations() {
        let Some(index) = connect_slot(SLOT_B).await else {
            return;
        };
        let model = unique_model("caller-clear");
        index
            .insert(&model, "base partition probe", &[3.0f32; 2], 4)
            .await
            .expect("基础分区插入");
        index
            .insert_for_caller(&model, "alice", "caller partition probe", &[1.0f32; 2], 4)
            .await
            .expect("alice 分区插入");
        index
            .insert_for_caller(&model, "bob", "caller partition probe", &[2.0f32; 2], 4)
            .await
            .expect("bob 分区插入");

        let (tri_alice, _, _) = RedisSemanticIndex::keys_with_caller(&model, "alice");
        let (tri_bob, _, _) = RedisSemanticIndex::keys_with_caller(&model, "bob");
        let mut raw = raw_conn(SLOT_B).await;
        for key in [&tri_alice, &tri_bob] {
            let exists: bool = redis::cmd("EXISTS")
                .arg(key)
                .query_async(&mut raw)
                .await
                .expect("EXISTS 前置");
            assert!(exists, "user 分区键必须已登记写入：{key}");
        }

        index.clear(&model).await.expect("clear");

        for key in [&tri_alice, &tri_bob] {
            let exists: bool = redis::cmd("EXISTS")
                .arg(key)
                .query_async(&mut raw)
                .await
                .expect("EXISTS 后置");
            assert!(!exists, "clear 后 caller 分区键必须清空：{key}");
        }
        let members: Vec<String> = raw.smembers(MODELS_KEY).await.expect("SMEMBERS");
        let prefix = format!("{}:", digest(&model));
        assert!(
            members.iter().all(|m| !m.starts_with(&prefix)),
            "clear 后不得残留分区登记成员：{members:?}"
        );
        assert!(
            !members.contains(&digest(&model)),
            "clear 后基础模型登记也必须移除"
        );
        // 行为验证：分区与基础索引都不可再命中
        assert_eq!(
            index
                .find_similar_for_caller(&model, "alice", "caller partition probe", 0.9)
                .await
                .expect("clear 后分区查询不得报错"),
            None,
            "clear 后 caller 分区不得再命中"
        );
        assert_eq!(
            index
                .find_similar(&model, "base partition probe", 0.9)
                .await
                .expect("clear 后基础查询不得报错"),
            None,
            "clear 后基础索引不得再命中"
        );
    }

    /// 并发窗口钉（架构 LOW-4）：clear(model) 与并发 insert_for_caller 竞争
    /// 时，登记成员与分区键必须保持一致（原子脚本消除「SMEMBERS 快照后
    /// 落库漏清」的残留窗口）。结束后双向核验——每个带前缀成员的三件键
    /// 必须存在（有登记必有键），SCAN 到的本模型键必须对应登记成员（有键
    /// 必有登记）；静默后二次 clear 必须全部清空（无残留）。
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn clear_model_is_atomic_against_concurrent_partition_inserts() {
        let index = std::sync::Arc::new(match connect_slot(SLOT_A).await {
            Some(index) => index,
            None => return,
        });
        let model = unique_model("clear-race");
        // 竞争前预置基础分区 + 若干 caller 分区，保证清理目标非空。
        index
            .insert(&model, "race base probe", &[0.0f32; 2], 64)
            .await
            .expect("基础插入");
        for i in 0..8 {
            index
                .insert_for_caller(
                    &model,
                    &format!("pre-callers-{i}"),
                    &format!("pre probe text {i}"),
                    &[i as f32; 2],
                    64,
                )
                .await
                .expect("预置分区插入");
        }

        // 20 个并发 caller 插入与 clear 同时起跑。
        let mut handles = Vec::new();
        for i in 0..20 {
            let index = std::sync::Arc::clone(&index);
            let model = model.clone();
            handles.push(tokio::spawn(async move {
                index
                    .insert_for_caller(
                        &model,
                        &format!("race-callers-{i}"),
                        &format!("race probe text {i}"),
                        &[i as f32; 2],
                        64,
                    )
                    .await
                    .expect("并发插入");
            }));
        }
        {
            let index = std::sync::Arc::clone(&index);
            let model = model.clone();
            handles.push(tokio::spawn(async move {
                index.clear(&model).await.expect("clear");
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }

        // 双向一致性核验（登记成员 ⇔ 分区键）。
        let mut raw = raw_conn(SLOT_A).await;
        let members: Vec<String> = raw.smembers(MODELS_KEY).await.expect("SMEMBERS");
        let prefix = format!("{}:", digest(&model));
        let mut registered = std::collections::HashSet::new();
        for member in members.iter().filter(|m| m.starts_with(&prefix)) {
            registered.insert(member.clone());
            for suffix in [":tri", ":emb", ":lru"] {
                let key = format!("{KEY_NAMESPACE}{member}{suffix}");
                let exists: bool = redis::cmd("EXISTS")
                    .arg(&key)
                    .query_async(&mut raw)
                    .await
                    .expect("EXISTS");
                assert!(
                    exists,
                    "登记成员 {member} 的键必须存在（不得残留空登记）：{key}"
                );
            }
        }
        let mut cursor: u64 = 0;
        loop {
            let (next, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(format!("{KEY_NAMESPACE}{}:*", digest(&model)))
                .arg("COUNT")
                .arg(500)
                .query_async(&mut raw)
                .await
                .expect("SCAN");
            for key in keys {
                let member = key
                    .strip_prefix(KEY_NAMESPACE)
                    .and_then(|rest| rest.rsplit_once(':'))
                    .map(|(m, _)| m.to_string())
                    .expect("键形如 ns{member}:{suffix}");
                assert!(
                    registered.contains(&member),
                    "SCAN 到的键 {key} 必须有登记成员（不得残留孤儿键）"
                );
            }
            cursor = next;
            if cursor == 0 {
                break;
            }
        }

        // 静默后二次 clear：登记成员与键均不残留。
        index.clear(&model).await.expect("二次 clear");
        let members_after: Vec<String> = raw.smembers(MODELS_KEY).await.expect("SMEMBERS after");
        assert!(
            members_after.iter().all(|m| !m.starts_with(&prefix)),
            "静默 clear 后不得残留分区登记：{members_after:?}"
        );
        assert!(
            !members_after.contains(&digest(&model)),
            "基础登记也必须移除"
        );
        let (tri_base, emb_base, lru_base) = RedisSemanticIndex::keys(&model);
        for key in [&tri_base, &emb_base, &lru_base] {
            let exists: bool = redis::cmd("EXISTS")
                .arg(key)
                .query_async(&mut raw)
                .await
                .expect("EXISTS base");
            assert!(!exists, "基础键必须清空：{key}");
        }
    }

    #[tokio::test]
    async fn corrupt_trigram_payload_is_skipped_and_corrupt_embedding_errors() {
        let Some(index) = connect_slot(SLOT_D).await else {
            return;
        };
        let model = unique_model("corrupt");
        let query = "corruption probe text";
        let (tri_key, emb_key, _) = RedisSemanticIndex::keys(&model);
        let mut raw = raw_conn(SLOT_D).await;

        // 1) trigram JSON 非法：该条目必须被跳过，而不是拖垮整个查询。
        let _: () = redis::cmd("HSET")
            .arg(&tri_key)
            .arg(digest("garbage row"))
            .arg("{not json")
            .query_async(&mut raw)
            .await
            .expect("注入非法 trigram");
        assert_eq!(
            index
                .find_similar(&model, query, 0.5)
                .await
                .expect("存在脏条目时查询仍不得报错"),
            None,
            "脏 trigram 条目应被跳过且无其他候选"
        );

        // 2) embedding 字节长度非 4 倍数：必须显式报 corrupt（不静默丢数据）。
        let trigrams: Vec<u32> = pack_trigrams(query.as_bytes()).into_iter().collect();
        let payload = serde_json::to_string(&trigrams).expect("serialize trigrams");
        let field = digest(query);
        let _: () = redis::cmd("HSET")
            .arg(&tri_key)
            .arg(&field)
            .arg(&payload)
            .query_async(&mut raw)
            .await
            .expect("注入合法 trigram");
        let _: () = redis::cmd("HSET")
            .arg(&emb_key)
            .arg(&field)
            .arg(vec![1u8, 2, 3])
            .query_async(&mut raw)
            .await
            .expect("注入残缺 embedding");
        let err = index
            .find_similar(&model, query, 0.5)
            .await
            .expect_err("残缺 embedding 必须显性失败");
        assert!(
            err.error_detail()
                .contains("semantic-redis-embedding-corrupt"),
            "实际：{}",
            err.error_detail()
        );

        index.clear(&model).await.expect("clear");
        assert_eq!(
            index
                .find_similar(&model, query, 0.5)
                .await
                .expect("clear 后查询不得报错"),
            None,
            "clear 必须同时移三件键与模型集成员"
        );
    }

    #[tokio::test]
    async fn clear_all_drops_every_registered_model() {
        let Some(index) = connect_slot(SLOT_E).await else {
            return;
        };
        let first = unique_model("all-1");
        let second = unique_model("all-2");
        index
            .insert(&first, "shared clear probe one", &[3.0f32; 2], 4)
            .await
            .expect("insert first");
        index
            .insert(&second, "shared clear probe two", &[4.0f32; 2], 4)
            .await
            .expect("insert second");
        // user 模式 caller 分区同样登记进 MODELS_KEY，clear_all 必须一并清空
        index
            .insert_for_caller(&first, "carol", "caller clear-all probe", &[5.0f32; 2], 4)
            .await
            .expect("caller 分区插入");
        let (tri_carol, _, _) = RedisSemanticIndex::keys_with_caller(&first, "carol");
        assert!(
            index
                .find_similar(&first, "shared clear probe one", 0.9)
                .await
                .expect("查第一个模型")
                .is_some()
        );
        let mut raw = raw_conn(SLOT_E).await;
        let before: usize = redis::cmd("SCARD")
            .arg(MODELS_KEY)
            .query_async(&mut raw)
            .await
            .expect("SCARD");
        assert!(before >= 2, "两个模型都应登记进集合，实际 {before}");
        let carol_exists: bool = redis::cmd("EXISTS")
            .arg(&tri_carol)
            .query_async(&mut raw)
            .await
            .expect("EXISTS carol 前置");
        assert!(carol_exists, "caller 分区键必须已登记写入");

        index.clear_all().await.expect("clear_all");

        let after: usize = redis::cmd("SCARD")
            .arg(MODELS_KEY)
            .query_async(&mut raw)
            .await
            .expect("SCARD after");
        assert_eq!(after, 0, "clear_all 后模型集合必须为空");
        let carol_after: bool = redis::cmd("EXISTS")
            .arg(&tri_carol)
            .query_async(&mut raw)
            .await
            .expect("EXISTS carol 后置");
        assert!(!carol_after, "clear_all 后 caller 分区键必须清空");
        assert_eq!(
            index
                .find_similar(&first, "shared clear probe one", 0.9)
                .await
                .expect("clear_all 后查询不得报错"),
            None,
            "第一个模型的索引必须已消失"
        );
        assert_eq!(
            index
                .find_similar(&second, "shared clear probe two", 0.9)
                .await
                .expect("clear_all 后第二个查询不得报错"),
            None,
            "第二个模型的索引也必须已消失"
        );
    }
}
