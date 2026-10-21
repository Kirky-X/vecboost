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

#[cfg(test)]
mod tests {
    use super::*;

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
        let first = now_secs();
        assert!(first > 1_700_000_000.0, "必须是真实 UNIX 秒：{first}");
        assert!(now_secs() >= first, "同一进程内不得回退");
        let err = map_err("insert", std::io::Error::other("boom"));
        let detail = err.error_detail();
        assert!(
            detail.contains("semantic-redis-insert") && detail.contains("boom"),
            "错误需同时点名操作与原因：{detail}"
        );
    }

    // ── 连接与 fail-closed ──────────────────────────────────────────

    #[tokio::test]
    async fn connect_rejects_invalid_url_and_unreachable_endpoint() {
        // 非法 URL：解析期即拒，不进入连接阶段。（不用 expect_err：
        // RedisSemanticIndex 持有 MultiplexedConnection 无法 derive Debug）
        let bad = match RedisSemanticIndex::connect("not-a-redis-url-at-all").await {
            Ok(_) => panic!("非法 URL 必须拒绝"),
            Err(e) => e,
        };
        assert!(
            bad.error_detail().contains("semantic-redis-url-invalid"),
            "实际：{}",
            bad.error_detail()
        );
        // 端点不可达：连接阶段即拒（启动期 fail-closed，不降级为无模糊层）。
        let unreachable = match RedisSemanticIndex::connect("redis://127.0.0.1:1/11").await {
            Ok(_) => panic!("不可达端点必须拒绝"),
            Err(e) => e,
        };
        assert!(
            unreachable
                .error_detail()
                .contains("semantic-redis-connect-failed"),
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

    // ── 损坏数据与清理 ────────────────────────────────────────────

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

        index.clear_all().await.expect("clear_all");

        let after: usize = redis::cmd("SCARD")
            .arg(MODELS_KEY)
            .query_async(&mut raw)
            .await
            .expect("SCARD after");
        assert_eq!(after, 0, "clear_all 后模型集合必须为空");
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
