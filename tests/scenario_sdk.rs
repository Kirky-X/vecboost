// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! SDK 场景集成测试 — 模型矩阵 + 正常/异常全场景（design.md M1 矩阵）
//!
//! 覆盖场景：
//! - M0 模型矩阵：4 模型 × 3 厂商 × 2 架构（Bert/XlmRoberta），
//!   各跑 embed / embed_batch / rerank + 维度断言（SM-M1…SM-M4）
//! - 正常：确定性、批量边界（1/32）、归一化开关、相似度语义、同步 API（SM-N01…）
//! - 异常：空文本/纯空白/超长/空批/超批/rerank 空 docs/top_k=0/空 query（SM-A01…）
//!
//! 模型缺失时对应用例打印 SKIP 并通过（下载失败不阻塞其余矩阵）。

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::RwLock;
use vecboost::config::model::{DeviceType, EngineType, ModelConfig, Precision};
use vecboost::domain::{EmbedRequest, SimilarityRequest};
use vecboost::engine::AnyEngine;
use vecboost::service::embedding::EmbeddingService;
use vecboost::{LibraryConfig, VecBoostLibrary, VecboostError};

const M1_DIR: &str = "BAAI-bge-small-en-v1.5";
const M1_DIM: usize = 384;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn model_dir(dirname: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(dirname);
    let weights_ok = p.join("model.safetensors").exists() || p.join("pytorch_model.bin").exists();
    (p.is_dir() && weights_ok && p.join("tokenizer.json").exists()).then_some(p)
}

fn model_config(dirname: &str, dim: usize) -> ModelConfig {
    ModelConfig {
        name: dirname.to_string(),
        engine_type: EngineType::Candle,
        model_path: model_dir(dirname).unwrap_or_else(|| PathBuf::from("models").join(dirname)),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: Some(dim),
        memory_limit_bytes: None,
        oom_fallback_enabled: true,
        model_sha256: None,
    }
}

async fn make_library(dirname: &str, dim: usize) -> Option<VecBoostLibrary> {
    model_dir(dirname)?;
    let config = LibraryConfig::from_model_config(model_config(dirname, dim));
    Some(VecBoostLibrary::new(config).await.expect("library init"))
}

fn assert_unit_norm(vec: &[f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-2,
        "归一化向量范数应为 1，实际 {norm}"
    );
}

fn assert_valid_embedding(vec: &[f32], dim: usize) {
    assert_eq!(vec.len(), dim, "维度应为 {dim}");
    assert!(vec.iter().all(|x| x.is_finite()), "向量分量应全部有限");
    assert_unit_norm(vec);
}

// ---------------------------------------------------------------------------
// M0 模型矩阵（SM-M1…SM-M4）
// ---------------------------------------------------------------------------

async fn model_matrix_case(label: &str, dirname: &str, dim: usize) {
    let Some(lib) = make_library(dirname, dim).await else {
        eprintln!("SKIP [{label}]: 模型目录缺失（下载失败），跳过该模型矩阵用例");
        return;
    };

    // embed 单条：维度 + 有限 + 单位范数
    let resp = lib
        .embed("The quick brown fox jumps over the lazy dog")
        .await
        .expect("embed");
    assert_valid_embedding(&resp.embedding, dim);

    // embed_batch：顺序保持、每条维度一致
    let texts = [
        "first sample text".to_string(),
        "机器学习模型训练数据".to_string(),
        "mixed 中英混合 text with emoji 🚀".to_string(),
    ];
    let batch = lib.embed_batch(&texts).await.expect("embed_batch");
    assert_eq!(batch.embeddings.len(), 3, "批量顺序与数量保持");
    for (i, item) in batch.embeddings.iter().enumerate() {
        assert_valid_embedding(&item.embedding, dim);
        assert!(
            !item.text_preview.is_empty(),
            "批量第 {i} 条应有 text_preview"
        );
    }

    // rerank：相关文档第一、分数降序
    let docs = [
        "Machine learning models are trained on large datasets".to_string(),
        "I had pasta with tomato sauce for lunch".to_string(),
        "Deep learning is a subset of machine learning".to_string(),
    ];
    let ranked = lib
        .rerank("What is machine learning?", &docs, None)
        .await
        .expect("rerank");
    assert_eq!(ranked.results.len(), 3, "top_k=None 返回全量");
    let scores: Vec<f32> = ranked.results.iter().map(|r| r.score).collect();
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(scores, sorted, "rerank 分数应降序: {scores:?}");
    // 语义断言：任一 ML 相关文档（index 0/2）排在无关文档（index 1，午餐）之前
    assert_ne!(
        ranked.results[0].index, 1,
        "无关文档不得排名第一: scores={scores:?} top={}",
        ranked.results[0].index
    );
    let idx_of_irrelevant = ranked
        .results
        .iter()
        .position(|r| r.index == 1)
        .expect("无关文档应在结果中");
    assert_eq!(idx_of_irrelevant, 2, "无关文档应排名最后: {scores:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_matrix_m1_bge_small_en() {
    model_matrix_case("m1-bge-small-en", M1_DIR, M1_DIM).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_matrix_m2_minilm_l6() {
    model_matrix_case("m2-minilm-l6", "all-MiniLM-L6-v2", 384).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_matrix_m3_bge_small_zh() {
    // 中文模型：中文 query + 中文文档相关性
    let Some(lib) = make_library("BAAI-bge-small-zh-v1.5", 512).await else {
        eprintln!("SKIP [m3-bge-small-zh]: 模型目录缺失");
        return;
    };
    let resp = lib.embed("今天天气真好").await.expect("zh embed");
    assert_valid_embedding(&resp.embedding, 512);

    let docs = [
        "机器学习是人工智能的一个分支".to_string(),
        "今天的午餐是意大利面".to_string(),
    ];
    // BAAI bge-zh 模型卡要求检索类 query 添加指令前缀（模型设计用法）
    let query = "为这个句子生成表示以用于检索相关文章：什么是机器学习";
    let ranked = lib.rerank(query, &docs, None).await.expect("zh rerank");
    // 中文语义断言：相关文档（index 0）得分高于无关文档（index 1）
    let zh_scores: Vec<f32> = ranked.results.iter().map(|r| r.score).collect();
    let idx_rel = ranked
        .results
        .iter()
        .position(|r| r.index == 0)
        .expect("结果含 index 0");
    let idx_irr = ranked
        .results
        .iter()
        .position(|r| r.index == 1)
        .expect("结果含 index 1");
    assert!(
        zh_scores[idx_rel] > zh_scores[idx_irr],
        "中文相关文档得分应高于无关文档: {zh_scores:?}"
    );

    let batch = lib
        .embed_batch(&["向量检索".to_string(), "语义匹配".to_string()])
        .await
        .expect("zh batch");
    assert_eq!(batch.embeddings.len(), 2);
    assert_eq!(batch.embeddings[0].embedding.len(), 512);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_matrix_m4_e5_small_xlmroberta() {
    // XlmRoberta 架构加载路径（区别于 M1-M3 的 Bert）
    let Some(lib) = make_library("multilingual-e5-small", 384).await else {
        eprintln!("SKIP [m4-e5-small]: 模型目录缺失");
        return;
    };
    let resp = lib.embed("Bonjour le monde").await.expect("e5 embed");
    assert_valid_embedding(&resp.embedding, 384);

    let batch = lib
        .embed_batch(&["multilingual text".to_string(), "多语言文本".to_string()])
        .await
        .expect("e5 batch");
    assert_eq!(batch.embeddings[1].embedding.len(), 384);

    let ranked = lib
        .rerank(
            "retrieval query",
            &[
                "relevant passage about retrieval".to_string(),
                "unrelated text".to_string(),
            ],
            Some(2),
        )
        .await
        .expect("e5 rerank");
    assert_eq!(ranked.results.len(), 2);
}

// ---------------------------------------------------------------------------
// 正常场景（M1；SM-N01…）
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_normal_determinism_and_batch_boundaries() {
    let lib = make_library(M1_DIR, M1_DIM)
        .await
        .expect("M1 缺失（基线模型应始终存在）");

    // 确定性：同输入两次调用向量一致
    let a = lib.embed("determinism probe").await.unwrap();
    let b = lib.embed("determinism probe").await.unwrap();
    assert_eq!(a.embedding, b.embedding, "同输入向量必须一致");

    // 边界：批量 1 条
    let one = lib.embed_batch(&["single".to_string()]).await.unwrap();
    assert_eq!(one.embeddings.len(), 1);

    // 边界：批量 32 条（默认 max_batch_size）顺序与确定性
    let texts: Vec<String> = (0..32).map(|i| format!("batch item {i}")).collect();
    let b1 = lib.embed_batch(&texts).await.unwrap();
    let b2 = lib.embed_batch(&texts).await.unwrap();
    assert_eq!(b1.embeddings.len(), 32);
    for (i, item) in b1.embeddings.iter().enumerate() {
        assert_eq!(
            item.embedding, b2.embeddings[i].embedding,
            "批量第 {i} 条两次结果一致"
        );
    }
    // 批量内顺序与单条一致
    let single = lib.embed(&texts[7]).await.unwrap();
    assert_eq!(
        b1.embeddings[7].embedding, single.embedding,
        "批量第 7 条应等于单条结果"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_normal_normalize_toggle() {
    if model_dir(M1_DIR).is_none() {
        eprintln!("SKIP: M1 缺失");
        return;
    }
    let cfg = model_config(M1_DIR, M1_DIM);
    let engine = AnyEngine::new(&cfg, EngineType::Candle, Precision::Fp32).expect("engine");
    let svc = EmbeddingService::new(Arc::new(RwLock::new(engine)), Some(cfg));

    // normalize=true（显式）→ 单位范数
    let on = svc
        .process_text(
            EmbedRequest {
                text: "normalize on".into(),
                normalize: Some(true),
            },
            None,
        )
        .await
        .unwrap();
    assert_unit_norm(&on.embedding);

    // normalize=false → 跳过 L2 归一化（范数不再强约束为 1）
    let off = svc
        .process_text(
            EmbedRequest {
                text: "normalize off".into(),
                normalize: Some(false),
            },
            None,
        )
        .await
        .unwrap();
    let norm: f32 = off.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm > 0.0, "未归一化向量范数应为正: {norm}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_normal_similarity_semantics() {
    if model_dir(M1_DIR).is_none() {
        eprintln!("SKIP: M1 缺失");
        return;
    }
    let cfg = model_config(M1_DIR, M1_DIM);
    let engine = AnyEngine::new(&cfg, EngineType::Candle, Precision::Fp32).expect("engine");
    let svc = EmbeddingService::new(Arc::new(RwLock::new(engine)), Some(cfg));

    // 同文本 = 1.0（SIM-002 语义回归：无关文本不得为 1.0）
    let same = svc
        .process_similarity(SimilarityRequest {
            source: "machine learning algorithms".into(),
            target: "machine learning algorithms".into(),
            metric: None,
        })
        .await
        .unwrap();
    assert!(
        (same.score - 1.0).abs() < 1e-3,
        "同文本相似度应为 1.0，实际 {}",
        same.score
    );

    let related = svc
        .process_similarity(SimilarityRequest {
            source: "machine learning algorithms".into(),
            target: "neural networks and deep learning models".into(),
            metric: None,
        })
        .await
        .unwrap();
    let unrelated = svc
        .process_similarity(SimilarityRequest {
            source: "machine learning algorithms".into(),
            target: "今天的午餐是面条".into(),
            metric: None,
        })
        .await
        .unwrap();
    assert!(
        related.score > unrelated.score,
        "相关对 {} 应高于无关对 {}",
        related.score,
        unrelated.score
    );
}

/// 同步 API（SM-N08）：必须在无 tokio runtime 的同步上下文调用
#[test]
fn sdk_sync_api_works_outside_runtime() {
    if model_dir(M1_DIR).is_none() {
        eprintln!("SKIP [sync]: M1 缺失");
        return;
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let lib = rt.block_on(async {
        VecBoostLibrary::new(LibraryConfig::from_model_config(model_config(
            M1_DIR, M1_DIM,
        )))
        .await
        .unwrap()
    });

    let emb = lib.embed_sync("sync api probe").expect("embed_sync");
    assert_eq!(emb.embedding.len(), M1_DIM);

    let batch = lib
        .embed_batch_sync(&["a".to_string(), "b".to_string()])
        .expect("embed_batch_sync");
    assert_eq!(batch.embeddings.len(), 2);

    let ranked = lib
        .rerank_sync(
            "query",
            &["doc one".to_string(), "doc two".to_string()],
            Some(1),
        )
        .expect("rerank_sync");
    assert_eq!(ranked.results.len(), 1, "top_k=1 截断");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_normal_rerank_topk_boundaries() {
    let lib = make_library(M1_DIR, M1_DIM).await.expect("M1 缺失");
    let docs: Vec<String> = (0..5)
        .map(|i| format!("document number {i} about storage"))
        .collect();

    // top_k=None → 全量；top_k=1 → 1 条；top_k=len → 全部
    let all = lib.rerank("storage", &docs, None).await.unwrap();
    assert_eq!(all.results.len(), 5);
    let one = lib.rerank("storage", &docs, Some(1)).await.unwrap();
    assert_eq!(one.results.len(), 1);
    let exact = lib.rerank("storage", &docs, Some(5)).await.unwrap();
    assert_eq!(exact.results.len(), 5);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_batch_matches_single_across_lengths() {
    // DEFECT-BATCH-001 回归钉：批内短序列不得被长序列 padding 污染
    // （Bert forward 参数错位修复的永久守护）
    if model_dir(M1_DIR).is_none() {
        eprintln!("SKIP: M1 缺失");
        return;
    }
    let lib = make_library(M1_DIR, M1_DIM).await.expect("M1 缺失");
    let short = "hello world";
    let long = "这是一段比较长的中文文本用来测试批量填充对短序列向量的影响";
    let single_short = lib.embed(short).await.unwrap().embedding;
    let single_long = lib.embed(long).await.unwrap().embedding;
    let batch = lib
        .embed_batch(&[short.to_string(), long.to_string()])
        .await
        .unwrap();
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    };
    let cos_short = cos(&single_short, &batch.embeddings[0].embedding);
    let cos_long = cos(&single_long, &batch.embeddings[1].embedding);
    assert!(
        cos_short > 0.9999,
        "批内短序列与单条推理应一致（DEFECT-BATCH-001）: cos={cos_short}"
    );
    assert!(
        cos_long > 0.9999,
        "批内长序列与单条推理应一致: cos={cos_long}"
    );
}

// ---------------------------------------------------------------------------
// 异常场景（SM-A01…；库层统一返回校验类错误，不得 panic / 500 语义）
// ---------------------------------------------------------------------------

fn assert_rejected(res: Result<impl Sized, VecboostError>, what: &str) {
    let Err(e) = res else {
        panic!("{what} 应被拒绝")
    };
    let ok = matches!(
        e,
        VecboostError::InvalidInput(_) | VecboostError::ValidationError(_)
    );
    assert!(ok, "{what} 应返回校验类错误，实际 {e:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_abnormal_inputs_rejected() {
    let lib = make_library(M1_DIR, M1_DIM).await.expect("M1 缺失");

    // SM-A01 空文本
    assert_rejected(lib.embed("").await, "空文本");
    // SM-A02 纯空白文本
    assert_rejected(lib.embed("   \n\t  ").await, "纯空白文本");
    // SM-A04 空批
    assert_rejected(lib.embed_batch(&[]).await, "空批");
    // SM-A03 超长文本（校验上限 10000 字符，取 2 万）
    let long = "x".repeat(20_000);
    assert_rejected(lib.embed(&long).await, "超长文本");
    // SM-A05 超批（校验上限 100 条，取 150）
    let big: Vec<String> = (0..150).map(|i| format!("item {i}")).collect();
    assert_rejected(lib.embed_batch(&big).await, "超批");

    // SM-A06 rerank 空 documents
    assert_rejected(lib.rerank("query", &[], None).await, "rerank 空 documents");
    // SM-A07 top_k=0
    let docs = vec!["a".to_string(), "b".to_string()];
    assert_rejected(lib.rerank("query", &docs, Some(0)).await, "rerank top_k=0");
    // SM-A08 空 query
    assert_rejected(lib.rerank("", &docs, None).await, "rerank 空 query");
}
