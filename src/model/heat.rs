// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 模型热度持久化（port 自 colibri `.coli_usage` 原子写 + warmstart pin）。
//!
//! - 落盘 `data/model_heat.json`：原子写（tmp + rename）；
//! - 头部含模型指纹（name + schema 版本）；启动加载作 warmstart 初始 heat；
//! - 损坏/指纹不匹配即弃用并 warn（绝不污染新会话）。
use log::warn;
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// 热度表 schema 版本（指纹组成部分）。
pub const HEAT_SCHEMA_VERSION: u32 = 1;

/// 热度表默认落盘路径（server 接线：启动 warmstart + 卸载/切换后保存）。
pub const DEFAULT_HEAT_PATH: &str = "data/model_heat.json";

/// 单条目指纹：`{name}:v{schema}`。
pub fn heat_fingerprint(name: &str) -> String {
    format!("{}:v{}", name, HEAT_SCHEMA_VERSION)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct HeatEntry {
    heat: u32,
    fingerprint: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct HeatFile {
    schema_version: u32,
    entries: HashMap<String, HeatEntry>,
}

/// 原子保存热度表（tmp + rename）。
pub fn save_heat(path: &Path, heats: &HashMap<String, u32>) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let entries: HashMap<String, HeatEntry> = heats
        .iter()
        .map(|(name, heat)| {
            (
                name.clone(),
                HeatEntry {
                    heat: *heat,
                    fingerprint: heat_fingerprint(name),
                },
            )
        })
        .collect();
    let file = HeatFile {
        schema_version: HEAT_SCHEMA_VERSION,
        entries,
    };
    let text = serde_json::to_string_pretty(&file)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    // 同目录 tmp + rename：崩溃时要么旧文件完整、要么新文件完整。
    // tmp 文件名唯一（pid + 计数器 + 纳秒），并发写互不踩踏。
    static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let uniq = TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_extension(format!(
        "tmp.{}.{}.{}",
        std::process::id(),
        uniq,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&tmp, text)?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// 加载热度表作 warmstart。损坏/版本/指纹不匹配的条目即弃并 warn；
/// 文件整体不可读/不可解析时返回空表（服务正常启动）。
pub fn load_heat(path: &Path) -> HashMap<String, u32> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            warn!(
                "{}",
                crate::i18n::tr_with_args(
                    "heat-read-failed",
                    crate::i18n::tr_args(&[
                        ("path", &path.display().to_string()),
                        ("detail", &e.to_string()),
                    ]),
                )
            );
            return HashMap::new();
        }
    };
    let file: HeatFile = match serde_json::from_str(&text) {
        Ok(f) => f,
        Err(e) => {
            warn!(
                "{}",
                crate::i18n::tr_with_args(
                    "heat-parse-failed",
                    crate::i18n::tr_args(&[
                        ("path", &path.display().to_string()),
                        ("detail", &e.to_string()),
                    ]),
                )
            );
            return HashMap::new();
        }
    };
    if file.schema_version != HEAT_SCHEMA_VERSION {
        warn!(
            "{}",
            crate::i18n::tr_with_args(
                "heat-schema-mismatch",
                crate::i18n::tr_args(&[
                    ("got", &file.schema_version.to_string()),
                    ("expected", &HEAT_SCHEMA_VERSION.to_string()),
                ]),
            )
        );
        return HashMap::new();
    }
    let mut out = HashMap::with_capacity(file.entries.len());
    for (name, entry) in file.entries {
        if entry.fingerprint != heat_fingerprint(&name) {
            warn!(
                "{}",
                crate::i18n::tr_with_args(
                    "heat-entry-fingerprint-mismatch",
                    crate::i18n::tr_args(&[("name", &name)]),
                )
            );
            continue;
        }
        out.insert(name, entry.heat);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample() -> HashMap<String, u32> {
        [
            ("bge-small".to_string(), 42u32),
            ("bge-m3".to_string(), 7u32),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn test_heat_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model_heat.json");
        save_heat(&path, &sample()).unwrap();
        let loaded = load_heat(&path);
        assert_eq!(loaded.get("bge-small"), Some(&42));
        assert_eq!(loaded.get("bge-m3"), Some(&7));
    }

    #[test]
    fn test_heat_truncated_file_discarded() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model_heat.json");
        save_heat(&path, &sample()).unwrap();
        // 模拟崩溃截断：只保留前一半字节。
        let mut text = std::fs::read_to_string(&path).unwrap();
        text.truncate(text.len() / 2);
        std::fs::write(&path, text).unwrap();
        assert!(load_heat(&path).is_empty(), "截断文件必须弃用");
    }

    #[test]
    fn test_heat_fingerprint_mismatch_discarded() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model_heat.json");
        save_heat(&path, &sample()).unwrap();
        // 篡改指纹：bge-small 条目的 fingerprint 改掉。
        let mut text = std::fs::read_to_string(&path).unwrap();
        let fp = heat_fingerprint("bge-small");
        text = text.replacen(&fp, "tampered:v9", 1);
        std::fs::write(&path, text).unwrap();
        let loaded = load_heat(&path);
        assert!(!loaded.contains_key("bge-small"), "指纹不匹配条目必须弃用");
        assert_eq!(loaded.get("bge-m3"), Some(&7), "合法条目不受影响");
    }

    #[test]
    fn test_heat_missing_file_is_empty() {
        let dir = tempdir().unwrap();
        assert!(load_heat(&dir.path().join("nope.json")).is_empty());
    }

    #[test]
    fn test_heat_concurrent_writes_stay_readable() {
        use std::thread;
        let dir = tempdir().unwrap();
        let path = dir.path().join("model_heat.json");
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let path = path.clone();
                thread::spawn(move || {
                    let mut h = HashMap::new();
                    h.insert(format!("model-{}", i), i * 10);
                    save_heat(&path, &h).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // 最后一次完整写入必可读（tmp+rename 保证无半文件）。
        let loaded = load_heat(&path);
        assert_eq!(loaded.len(), 1, "并发写后仍可读出完整单写入: {:?}", loaded);
    }
}
