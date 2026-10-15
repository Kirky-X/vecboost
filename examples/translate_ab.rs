// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! A/B 对比：opus-mt-en-zh vs m2m100_418M 论文摘要翻译。
//!
//! 语料：`out/paper-translate-test/corpus_1..4.json`（liuxiang arXiv 检索），
//! 取评测轮的 4 篇成功 + 4 篇 504 长摘要。译文与耗时写
//! `out/paper-translate-test/ab/<model_name>.md`。
//!
//! 运行：`cargo run --features onnx --example translate_ab`

use std::path::PathBuf;
use std::time::Instant;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask};
use vecboost::domain::TranslateRequest;
use vecboost::engine::{EngineFactory, InferenceEngine};

const ROOT: &str = "out/paper-translate-test";
const DOMAINS: [&str; 4] = ["nlp-llm", "gnn-drug", "bft-consensus", "rl-manipulation"];
/// 评测轮成功的 4 篇（短）+ 504 的代表 4 篇（长）
const PICKS: [&str; 8] = [
    "nlp-llm-1",
    "nlp-llm-2",
    "gnn-drug-1",
    "gnn-drug-2",
    "nlp-llm-3",
    "gnn-drug-3",
    "bft-consensus-1",
    "rl-manipulation-1",
];

fn config_for(model_path: &str) -> ModelConfig {
    ModelConfig {
        name: format!("ab-{model_path}"),
        engine_type: EngineType::Onnx,
        model_path: PathBuf::from(model_path),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 1,
        pooling_mode: None,
        expected_dimension: None,
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
        task: ModelTask::Translate,
        quantized: false,
        decision_params: None,
        translation_params: None,
        transcription_params: None,
    }
}

/// 重建语料：与 run_translate_eval.py 同款 id 规则（域名-序号，摘要 >500 字符）。
fn load_corpus_all() -> Vec<(String, String)> {
    let mut all: Vec<(String, String)> = Vec::new();
    for (i, dom) in DOMAINS.iter().enumerate() {
        let path = PathBuf::from(ROOT).join(format!("corpus_{}.json", i + 1));
        let papers: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read corpus"))
                .expect("parse corpus");
        let mut n = 0;
        for p in papers {
            let ab = (p.get("abstract").and_then(|a| a.as_str()).unwrap_or("")).trim();
            if ab.len() > 500 {
                n += 1;
                all.push((format!("{dom}-{n}"), ab.to_string()));
            }
        }
    }
    all
}

fn load_corpus() -> Vec<(String, String)> {
    let all = load_corpus_all();
    PICKS
        .iter()
        .filter_map(|id| {
            all.iter()
                .find(|(key, _)| key == id)
                .map(|(key, ab)| (key.clone(), ab.clone()))
        })
        .collect()
}

fn main() {
    // 实验旋钮：AB_MODELS=逗号分隔模型路径；AB_IDS=逗号分隔语料 id；
    // TRANSLATE_INTRA=每 session intra 线程数（覆盖全局默认）
    if let Ok(v) = std::env::var("TRANSLATE_INTRA")
        && let Ok(n) = v.parse::<usize>()
    {
        vecboost::engine::set_session_intra_threads(n);
        eprintln!("translate_ab: intra threads override = {n}");
    }
    let all_items = load_corpus();
    let items: Vec<(String, String)> = match std::env::var("AB_IDS") {
        Ok(ids) => {
            let want: Vec<&str> = ids
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            // 显式 id 绕过 PICKS 白名单（实验任意语料用）：白名单外 id 从全量语料取
            let pool = if want.iter().all(|id| all_items.iter().any(|(k, _)| k == id)) {
                all_items
            } else {
                load_corpus_all()
            };
            pool.into_iter()
                .filter(|(id, _)| want.contains(&id.as_str()))
                .collect()
        }
        Err(_) => all_items,
    };
    println!(
        "语料 {} 篇：{:?}",
        items.len(),
        items.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    let out_dir = PathBuf::from(ROOT).join("ab");
    std::fs::create_dir_all(&out_dir).expect("mkdir ab");
    let models: Vec<String> = match std::env::var("AB_MODELS") {
        Ok(v) => v
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Err(_) => vec![
            "models/opus-mt-en-zh".to_string(),
            "models/m2m100_418M".to_string(),
        ],
    };
    for model in models {
        let t0 = Instant::now();
        let engine = match EngineFactory::create(EngineType::Onnx, &config_for(&model)) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("{model}: 构造失败 {e}");
                continue;
            }
        };
        println!("{model} 加载 {:?}", t0.elapsed());
        let mut md = format!("# A/B — {model}\n\n");
        for (id, text) in &items {
            let req = TranslateRequest {
                text: text.clone(),
                source_language: Some("en".into()),
                target_language: "zh".into(),
            };
            let s = Instant::now();
            let outcome = InferenceEngine::translate(&engine, &req);
            let dt = s.elapsed();
            match outcome {
                Ok(resp) => {
                    md.push_str(&format!(
                        "## {id}（{:.1}s）\n\n{}\n\n",
                        dt.as_secs_f64(),
                        resp.translation
                    ));
                    println!(
                        "{model} {id}: {:.1}s {} 字",
                        dt.as_secs_f64(),
                        resp.translation.chars().count()
                    );
                }
                Err(e) => {
                    md.push_str(&format!(
                        "## {id}（{:.1}s）\n\nERROR: {e}\n\n",
                        dt.as_secs_f64()
                    ));
                    println!("{model} {id}: ERROR {e}");
                }
            }
        }
        let name = model.rsplit('/').next().unwrap_or(model.as_str());
        std::fs::write(out_dir.join(format!("{name}.md")), md).expect("write ab md");
    }
}
