// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Grafana 仪表板资产回归：防止 JSON 语法损坏与关键面板静默丢失。
//!
//! 仪表板是运行时导入资产而非编译产物，Rust 测试只校验结构契约：
//! datasource 模板变量、必需面板标题与关键 PromQL 片段。

use serde_json::Value;

fn dashboard() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("deploy/grafana/vecboost-dashboard.json");
    let raw =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

#[test]
fn dashboard_has_datasource_template_variable() {
    let doc = dashboard();
    let list = doc["templating"]["list"]
        .as_array()
        .expect("templating.list must be an array");
    let ds = list
        .iter()
        .find(|item| item["name"] == "DS_PROMETHEUS")
        .expect("DS_PROMETHEUS variable must exist");
    assert_eq!(ds["type"], "datasource");
    assert_eq!(ds["query"], "prometheus");
}

#[test]
fn dashboard_covers_required_panels() {
    let doc = dashboard();
    let panels = doc["panels"].as_array().expect("panels must be an array");
    let titles: Vec<&str> = panels.iter().filter_map(|p| p["title"].as_str()).collect();
    for required in [
        "HTTP RPS",
        "批大小",
        "队列深度",
        "语义缓存模糊命中",
        "决策延迟",
    ] {
        assert!(
            titles.contains(&required),
            "missing panel {required:?}; titles: {titles:?}"
        );
    }
    assert!(
        doc["schemaVersion"].as_u64().unwrap_or(0) >= 39,
        "schemaVersion must be >= 39"
    );
}

#[test]
fn dashboard_queries_core_http_metric() {
    let doc = dashboard();
    let panels = doc["panels"].as_array().expect("panels must be an array");
    let exprs: Vec<&str> = panels
        .iter()
        .flat_map(|p| p["targets"].as_array().into_iter().flatten())
        .filter_map(|t| t["expr"].as_str())
        .collect();
    assert!(
        exprs.iter().any(|e| e.contains("http_requests_total")),
        "no panel queries http_requests_total"
    );
    assert!(
        exprs
            .iter()
            .any(|e| e.contains("vecboost_pipeline_queue_depth")),
        "no panel queries vecboost_pipeline_queue_depth"
    );
}
