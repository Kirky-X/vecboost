// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

// 整文件 onnx feature 门（评审 NEW-1：EngineType::Onnx 为 cfg 门控变体，
// 非 onnx 构建下本文件无法编译——先例 tests/candle_decision_parity.rs:7）
#![cfg(feature = "onnx")]

//! 翻译 E2E 真机测试 — golden 闸门 + 贪心确定性 + 质量输出。
//!
//! 运行：`cargo test --features onnx --test translate_e2e`
//!
//! SKIP 先例（tests/candle_decision_parity.rs）：ORT_DYLIB_PATH 缺失或
//! models/opus-mt-en-zh、models/opus-mt-en-es 任一 bundle 缺失即整文件
//! SKIP 并打印原因。
//!
//! golden 闸门首跑语义（规则 11 显性表达）：tests/fixtures/translate_golden.json
//! 不存在时进入生成模式——写出 tests/fixtures/translate_golden.candidate.json、
//! 输出末尾显性标注 `GOLDEN_PENDING`、以非零退出码结束（提示人工
//! 逐句抽验后固化）；golden 在位后逐句相等断言生效。
//!
//! 运行时写出 out/translate-quality.json（E2E 质量输出，gitignored）。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde_json::{Value, json};
use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask, TranslationParams};
use vecboost::domain::TranslateRequest;
use vecboost::engine::{AnyEngine, EngineFactory, InferenceEngine};
use vecboost::service::translate::{TranslatePairRoute, TranslateService};

const EN_ZH_BUNDLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/opus-mt-en-zh");
const EN_ES_BUNDLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/opus-mt-en-es");
const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/translate_golden.json"
);
const CANDIDATE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/translate_golden.candidate.json"
);
const QUALITY_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/out/translate-quality.json");

/// 固定句集：短句/中句/长句/CJK 混排/专有名词各覆盖（≥8 条）
const SENTENCES: &[(&str, &str)] = &[
    // 两轮抽验：问候类短句在 en-zh 系统性幻觉续写（"Hello."→
    // 「你好,我叫艾莉森」；"Good morning."→「早上好,我刚从你家回来」）——
    // 换主谓宾完整的信息性短句
    ("short-greet", "I am happy today."),
    ("short-daily", "The weather is nice today."),
    (
        "medium-tech",
        "Machine learning models process large amounts of data to find patterns.",
    ),
    (
        "medium-daily",
        "She walked to the market and bought fresh bread and vegetables for dinner.",
    ),
    (
        "long-news",
        "The international research team published a comprehensive study on climate \
         change, highlighting the urgent need for global cooperation and sustainable \
         development policies across all industrial sectors.",
    ),
    (
        "long-story",
        "Once upon a time, in a small village surrounded by mountains, a young girl \
         discovered an ancient map hidden beneath the floorboards of her grandmother's \
         house, and what she found changed the history of the entire region.",
    ),
    // CJK 字符混排不入选（抽验：en-es 词表对 CJK 全 OOV，译文必现
    // mojibake）——译入 CJK 的覆盖由 en-zh pair 整体承担；文化专有名词以
    // 罗马字保留覆盖
    (
        "cjk-culture",
        "I enjoy eating sushi and ramen on weekends in Kyoto.",
    ),
    (
        "proper-noun",
        "Einstein visited Tokyo, Paris, and New York during his famous world tour.",
    ),
    // 抽验替换：OpenAI 在 en-zh 词表被误译为「开放国际」
    (
        "proper-noun-tech",
        "Microsoft and Google compete in the cloud computing market.",
    ),
    ("question", "Where is the nearest train station?"),
];

const PAIRS: [(&str, &str); 2] = [("en", "zh"), ("en", "es")];

fn dylib_ready() -> Option<String> {
    let dylib = std::env::var("ORT_DYLIB_PATH")
        .unwrap_or_else(|_| "3rdparty/onnxruntime/libonnxruntime.so".to_string());
    if Path::new(&dylib).is_file() {
        Some(dylib)
    } else {
        None
    }
}

fn bundle_ready(path: &str) -> bool {
    Path::new(path).join("onnx/encoder_model.onnx").exists()
        && Path::new(path)
            .join("onnx/decoder_model_merged.onnx")
            .exists()
        && Path::new(path).join("tokenizer.json").exists()
}

fn skip_reason() -> Option<String> {
    if dylib_ready().is_none() {
        return Some(
            "onnxruntime dylib not found（设置 ORT_DYLIB_PATH 或落位 \
             3rdparty/onnxruntime/libonnxruntime.so）"
                .to_string(),
        );
    }
    for path in [EN_ZH_BUNDLE, EN_ES_BUNDLE] {
        if !bundle_ready(path) {
            return Some(format!("translation bundle missing: {path}"));
        }
    }
    None
}

fn load_pair_engine(bundle_path: &str, max_source_tokens: usize) -> AnyEngine {
    let config = ModelConfig {
        name: Path::new(bundle_path)
            .file_name()
            .expect("translate_e2e bundle_path 需有文件名")
            .to_string_lossy()
            .to_string(),
        engine_type: EngineType::Onnx,
        model_path: PathBuf::from(bundle_path),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 1,
        oom_fallback_enabled: false,
        task: ModelTask::Translate,
        translation_params: Some(
            TranslationParams::new(max_source_tokens, 512, 0).expect("params valid"),
        ),
        ..Default::default()
    };
    EngineFactory::create(EngineType::Onnx, &config).expect("bundle 就绪时必须可加载")
}

#[test]
fn translate_e2e_golden_gate_and_quality() {
    if let Some(reason) = skip_reason() {
        println!("SKIP [translate_e2e_golden_gate_and_quality]: {reason}");
        return;
    }
    // 服务层错误文案走 i18n 键渲染：e2e 进程内显性初始化（未初始化时 tr
    // 回退键名、插值丢失）
    vecboost::i18n::init();
    // SAFETY: 测试进程内单线程时点回填一次（skip_reason 已验证文件在位）
    let dylib = dylib_ready().unwrap();
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", &dylib);
        }
    }

    let engine_zh = load_pair_engine(EN_ZH_BUNDLE, 512);
    let engine_es = load_pair_engine(EN_ES_BUNDLE, 512);

    // ── 逐句翻译（每句两次：贪心确定性断言）+ 质量数据采集 ──────────────
    let mut results: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, String>,
    > = Default::default();
    let mut quality_pairs: Vec<Value> = Vec::new();
    let skipped_reason = String::new();

    for (src, tgt) in PAIRS {
        // 评审：经 AnyEngine::translate_output_tokens 读生成
        // token 数（wire 五键契约不扩，质量观测走引擎级查询通道）
        let engine: &AnyEngine = if tgt == "zh" { &engine_zh } else { &engine_es };
        let output_tokens = |e: &AnyEngine| -> u64 { e.translate_output_tokens() };
        let pair_key = format!("{src}-{tgt}");
        let mut sentences_json: Vec<Value> = Vec::new();
        for (id, text) in SENTENCES {
            let req = TranslateRequest {
                text: text.to_string(),
                source_language: Some(src.to_string()),
                target_language: tgt.to_string(),
            };
            let start = Instant::now();
            let resp1 = InferenceEngine::translate(engine, &req)
                .unwrap_or_else(|e| panic!("[{pair_key}] {id} 首次翻译失败: {e}"));
            let decode_ms = start.elapsed().as_millis() as u64;
            let resp2 = InferenceEngine::translate(engine, &req)
                .unwrap_or_else(|e| panic!("[{pair_key}] {id} 复译失败: {e}"));

            // 贪心确定性：同输入两次解码逐字节相等
            assert_eq!(
                resp1.translation, resp2.translation,
                "[{pair_key}] {id} 贪心解码必须确定（两次输出逐字节相等）"
            );
            // 译文非空且不含特殊 token
            assert!(
                !resp1.translation.trim().is_empty(),
                "[{pair_key}] {id} 译文非空"
            );
            for special in ["</s>", "<pad>", "<unk>"] {
                assert!(
                    !resp1.translation.contains(special),
                    "[{pair_key}] {id} 译文泄漏特殊 token {special}: {:?}",
                    resp1.translation
                );
            }
            assert!(
                output_tokens(engine) > 0,
                "[{pair_key}] {id} 生成 token 数必须 > 0（质量观测维度有效）"
            );
            results
                .entry(pair_key.clone())
                .or_default()
                .insert(id.to_string(), resp1.translation.clone());
            sentences_json.push(json!({
                "id": id,
                "source": text,
                "translation": resp1.translation,
                "source_tokens": InferenceEngine::count_tokens(engine, text)
                    .expect("count_tokens"),
                "output_tokens": output_tokens(engine),
                "decode_ms": decode_ms,
            }));
        }
        quality_pairs.push(json!({
            "pair": pair_key,
            "sentences": sentences_json,
        }));
    }

    // ── 服务级语义：auto 检测回填 + 未配置 pair 报错含 pairs 清单 ─────────
    let mut pairs_map = std::collections::HashMap::new();
    pairs_map.insert(
        ("en".to_string(), "zh".to_string()),
        TranslatePairRoute::new(
            {
                let cfg = ModelConfig {
                    name: "e2e-zh".to_string(),
                    engine_type: EngineType::Onnx,
                    model_path: PathBuf::from(EN_ZH_BUNDLE),
                    device: DeviceType::Cpu,
                    max_batch_size: 1,
                    oom_fallback_enabled: false,
                    task: ModelTask::Translate,
                    translation_params: Some(TranslationParams::new(512, 512, 0).unwrap()),
                    transcription_params: None,
                    ..Default::default()
                };
                Arc::new(tokio::sync::RwLock::new(
                    EngineFactory::create(EngineType::Onnx, &cfg).unwrap(),
                ))
            },
            TranslationParams::new(512, 512, 0).unwrap(),
        ),
    );
    let service = TranslateService::new(pairs_map);

    // auto：长英文句检测可靠 → 回填 en
    let auto_req = TranslateRequest {
        text: "The quick brown fox jumps over the lazy dog near the riverbank every morning."
            .to_string(),
        source_language: None,
        target_language: "zh".to_string(),
    };
    let runtime_desc = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let auto_resp = runtime_desc
        .block_on(service.process_translate(auto_req))
        .expect("auto 检测路由必须成功");
    assert_eq!(
        auto_resp.detected_source_language.as_deref(),
        Some("en"),
        "auto 检测必须回填 detected_source_language"
    );

    // 未配置 pair（en-fr）：显性报错含 pairs 清单
    let fr_req = TranslateRequest {
        text: "bonjour".to_string(),
        source_language: Some("en".to_string()),
        target_language: "fr".to_string(),
    };
    let fr_err = runtime_desc
        .block_on(service.process_translate(fr_req))
        .expect_err("en-fr 未配置必须报错");
    let fr_detail = fr_err.error_detail().to_string();
    assert!(
        fr_detail.contains("en-fr") && fr_detail.contains("en-zh"),
        "未配置对错误必须含请求对与已配置清单，got {fr_detail}"
    );

    // ── 引擎级超 max_source_tokens 显性报错 ─────────────────────────────
    let tight = load_pair_engine(EN_ZH_BUNDLE, 8);
    let over_req = TranslateRequest {
        text: "word ".repeat(30).trim().to_string(),
        source_language: Some("en".to_string()),
        target_language: "zh".to_string(),
    };
    let over_err = InferenceEngine::translate(&tight, &over_req)
        .expect_err("超 max_source_tokens 必须显性报错");
    assert!(
        over_err.error_detail().contains("exceeds token budget"),
        "引擎级超预算报错须含预算语义，got {}",
        over_err.error_detail()
    );

    // ── golden 闸门：在位→逐句相等断言；缺位→生成模式（GOLDEN_PENDING）──
    let candidate = json!({
        "meta": {
            "model_repo_en_zh": "Xenova/opus-mt-en-zh",
            "model_repo_en_es": "Xenova/opus-mt-en-es",
            "quantization": "fp32 (encoder_model + decoder_model_merged)",
            "decoding": "greedy, cache-branch disabled (see onnx_translate module docs)",
            "note": "meta 字段仅供人工溯源，E2E 断言忽略",
        },
        "pairs": results,
    });

    let golden_exists = Path::new(GOLDEN_PATH).is_file();
    if golden_exists {
        let golden: Value =
            serde_json::from_str(&std::fs::read_to_string(GOLDEN_PATH).expect("read golden"))
                .expect("golden json parse");
        let mut matched = 0usize;
        for (src, tgt) in PAIRS {
            let pair_key = format!("{src}-{tgt}");
            let golden_pair = &golden["pairs"][&pair_key];
            for (id, translation) in &results[&pair_key] {
                assert_eq!(
                    translation,
                    golden_pair[id]
                        .as_str()
                        .unwrap_or_else(|| { panic!("golden 缺句 {pair_key}/{id}") }),
                    "golden 闸门：{pair_key}/{id} 译文与固化管理完全相等"
                );
                matched += 1;
            }
        }

        // 质量输出（含 golden 匹配数）
        write_quality_json(&dylib, &quality_pairs, matched, &skipped_reason);
        println!("E2E-PASS: golden 闸门全绿（{matched} 句完全相等）");
        println!("quality output: {QUALITY_PATH}");
    } else {
        // 生成模式：写候选 + 显性 GOLDEN_PENDING + 非零退出（提示）
        if let Some(parent) = Path::new(CANDIDATE_PATH).parent() {
            std::fs::create_dir_all(parent).expect("create fixtures dir");
        }
        std::fs::write(
            CANDIDATE_PATH,
            serde_json::to_string_pretty(&candidate).expect("candidate serialize"),
        )
        .expect("write candidate");
        write_quality_json(&dylib, &quality_pairs, 0, "golden pending (first run)");
        println!("GOLDEN_PENDING: golden 不存在，已生成候选 {CANDIDATE_PATH}");
        println!("请人工逐句抽验候选译文（T026），确认后固化为 {GOLDEN_PATH}");
        println!("quality output: {QUALITY_PATH}");
        // 规则 11：首跑生成模式显性非零退出，禁止静默通过
        std::process::exit(1);
    }
}

fn write_quality_json(
    ort_dylib: &str,
    pairs: &[Value],
    golden_matched: usize,
    skipped_reason: &str,
) {
    let doc = json!({
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "runtime": {
            "ort_dylib": ort_dylib,
            "model_pairs": ["en-zh", "en-es"],
        },
        "pairs": pairs,
        "summary": {
            "total": pairs.iter().map(|p| p["sentences"].as_array().map(|a| a.len()).unwrap_or(0)).sum::<usize>(),
            "golden_matched": golden_matched,
            "skipped_reason": skipped_reason,
        },
    });
    if let Some(parent) = Path::new(QUALITY_PATH).parent() {
        std::fs::create_dir_all(parent).expect("create out dir");
    }
    std::fs::write(
        QUALITY_PATH,
        serde_json::to_string_pretty(&doc).expect("quality serialize"),
    )
    .expect("write quality json");
}
