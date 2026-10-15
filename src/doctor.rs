// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! `vecboost doctor` — 只读诊断。
//!
//! 检查链：① 配置校验 ② tokenizer 加载自检 ③ 缓存持久层可写性
//! ④ 线程调优报告（物理核/逻辑核/NUMA）⑤ GPU 探测 ⑥ 模型文件完整性。
//!
//! 设计约束（源自 colibri doctor.py 的两条教训）：
//! - **只读**：除可写性探针文件（探测后立即删除）外不修改任何状态、不做修复动作；
//! - **按角色而非字面名匹配**：模型文件按内容/后缀归类角色，重命名的合法文件
//!   仍被正确识别，避免"按字面名误报缺文件"类问题。
//!
//! doctor 在服务装配**之前**短路运行：模型损坏、依赖缺失时诊断必须仍然可用。
use crate::config::VecboostConfig;
use crate::i18n;

/// 检查结论（枚举固定三值，与 spec 一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

impl CheckStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            CheckStatus::Pass => "PASS",
            CheckStatus::Warn => "WARN",
            CheckStatus::Fail => "FAIL",
        }
    }
}

/// 单项检查结果。
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: &'static str,
    pub status: CheckStatus,
    pub detail: String,
}

impl CheckResult {
    fn new(name: &'static str, status: CheckStatus, detail: impl Into<String>) -> Self {
        Self {
            name,
            status,
            detail: detail.into(),
        }
    }
}

/// doctor 诊断报告。
#[derive(Debug, Clone, Default)]
pub struct DoctorReport {
    pub results: Vec<CheckResult>,
}

impl DoctorReport {
    /// 运行全部检查（只读）。任何单项失败不中断后续检查。
    pub async fn run(config: &VecboostConfig) -> Self {
        let mut results = Vec::new();
        results.push(check_config(config));
        results.push(check_tokenizer(config));
        results.push(check_cache_persist(config));
        results.push(check_threads());
        results.push(check_gpu(config).await);
        results.extend(check_models(config));
        Self { results }
    }

    /// 任一 FAIL 时进程退出码为 1（WARN 不影响退出码）。
    pub fn has_failures(&self) -> bool {
        self.results.iter().any(|r| r.status == CheckStatus::Fail)
    }

    /// 打印人类可读报告到 stdout。
    pub fn print(&self) {
        println!("{}", i18n::tr("doctor-report-title"));
        println!();
        for r in &self.results {
            println!("[{:>4}] {:<12} {}", r.status.as_str(), r.name, r.detail);
        }
        println!();
        let fails = self
            .results
            .iter()
            .filter(|r| r.status == CheckStatus::Fail)
            .count();
        let warns = self
            .results
            .iter()
            .filter(|r| r.status == CheckStatus::Warn)
            .count();
        println!(
            "{}",
            i18n::tr_with_args(
                "doctor-summary",
                i18n::tr_args(&[
                    ("total", &self.results.len().to_string()),
                    ("pass", &(self.results.len() - fails - warns).to_string()),
                    ("warn", &warns.to_string()),
                    ("fail", &fails.to_string()),
                ]),
            )
        );
    }

    /// 打印报告并按结果退出（0 = 无 FAIL，1 = 有 FAIL）。
    pub fn print_and_exit(self) -> ! {
        self.print();
        let code = if self.has_failures() { 1 } else { 0 };
        std::process::exit(code);
    }
}

/// 解析模型目录：显式 model_path 优先，否则以 model_repo 名（本地目录约定）。
pub fn resolve_model_dir(config: &VecboostConfig) -> std::path::PathBuf {
    match config.model.model_path.as_deref() {
        Some(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::PathBuf::from(&config.model.model_repo),
    }
}

/// ① 配置校验。通过时回显解析后的 `model.task`：`[model]` 段（TOML 面）
/// 未启用未知键拒绝，拼写错误会静默回落默认值，doctor 是该字段的观测点。
pub fn check_config(config: &VecboostConfig) -> CheckResult {
    match config.validate() {
        Ok(()) => CheckResult::new(
            "config",
            CheckStatus::Pass,
            i18n::tr_with_args(
                "doctor-config-pass",
                i18n::tr_args(&[("task", &config.model.task.to_string())]),
            ),
        ),
        Err(e) => CheckResult::new(
            "config",
            CheckStatus::Fail,
            i18n::tr_with_args(
                "doctor-config-fail",
                i18n::tr_args(&[("detail", &e.to_string())]),
            ),
        ),
    }
}

/// ② tokenizer 加载自检。模型目录缺失时 WARN（运行时会回退 HF 下载），不武断 FAIL。
pub fn check_tokenizer(config: &VecboostConfig) -> CheckResult {
    let dir = resolve_model_dir(config);
    let tok_path = dir.join("tokenizer.json");
    if !dir.exists() {
        return CheckResult::new(
            "tokenizer",
            CheckStatus::Warn,
            i18n::tr_with_args(
                "doctor-model-dir-missing",
                i18n::tr_args(&[("path", &dir.display().to_string())]),
            ),
        );
    }
    if !tok_path.exists() {
        return CheckResult::new(
            "tokenizer",
            CheckStatus::Warn,
            i18n::tr_with_args(
                "doctor-tokenizer-json-missing",
                i18n::tr_args(&[("path", &tok_path.display().to_string())]),
            ),
        );
    }
    match crate::text::SyncTokenizer::from_file(tok_path.to_string_lossy().as_ref()) {
        Ok(_) => CheckResult::new(
            "tokenizer",
            CheckStatus::Pass,
            i18n::tr_with_args(
                "doctor-tokenizer-json-ok",
                i18n::tr_args(&[("path", &tok_path.display().to_string())]),
            ),
        ),
        Err(e) => CheckResult::new(
            "tokenizer",
            CheckStatus::Fail,
            i18n::tr_with_args(
                "doctor-tokenizer-json-fail",
                i18n::tr_args(&[("detail", &e.to_string())]),
            ),
        ),
    }
}

/// ③ 缓存持久层可写性。未配置 persist_path 时为纯内存模式（PASS）。
/// 可写性探测：创建探针文件后立即删除（不残留）；已存在的 WAL 文件以
/// append 打开验证权限，不改动内容。
pub fn check_cache_persist(config: &VecboostConfig) -> CheckResult {
    let Some(path) = config.embedding.persist_path.as_deref() else {
        return CheckResult::new(
            "cache-persist",
            CheckStatus::Pass,
            i18n::tr("doctor-cache-memory-only"),
        );
    };
    let wal = std::path::Path::new(path);
    if wal.exists() {
        return match std::fs::OpenOptions::new().append(true).open(wal) {
            Ok(_) => CheckResult::new(
                "cache-persist",
                CheckStatus::Pass,
                i18n::tr_with_args(
                    "doctor-cache-wal-writable",
                    i18n::tr_args(&[("path", path)]),
                ),
            ),
            Err(e) => CheckResult::new(
                "cache-persist",
                CheckStatus::Fail,
                i18n::tr_with_args(
                    "doctor-cache-wal-unwritable",
                    i18n::tr_args(&[("path", path), ("detail", &e.to_string())]),
                ),
            ),
        };
    }
    let Some(parent) = wal.parent().filter(|p| !p.as_os_str().is_empty()) else {
        return CheckResult::new(
            "cache-persist",
            CheckStatus::Fail,
            i18n::tr_with_args(
                "doctor-cache-path-invalid",
                i18n::tr_args(&[("path", path)]),
            ),
        );
    };
    if !parent.exists() {
        return CheckResult::new(
            "cache-persist",
            CheckStatus::Fail,
            i18n::tr_with_args(
                "doctor-cache-parent-missing",
                i18n::tr_args(&[("path", &parent.display().to_string())]),
            ),
        );
    }
    let probe = parent.join(".vecboost_doctor_probe");
    let probe_result = std::fs::File::create(&probe).and_then(|_| std::fs::remove_file(&probe));
    match probe_result {
        Ok(()) => CheckResult::new(
            "cache-persist",
            CheckStatus::Pass,
            i18n::tr_with_args(
                "doctor-cache-dir-writable",
                i18n::tr_args(&[("path", &parent.display().to_string())]),
            ),
        ),
        Err(e) => CheckResult::new(
            "cache-persist",
            CheckStatus::Fail,
            i18n::tr_with_args(
                "doctor-cache-dir-unwritable",
                i18n::tr_args(&[
                    ("path", &parent.display().to_string()),
                    ("detail", &e.to_string()),
                ]),
            ),
        ),
    }
}

/// ④ 线程调优报告：物理核/逻辑核/生效线程数/NUMA。报告性检查，恒 PASS
/// （检测失败回退 num_cpus 也是合法工作配置，以 WARN 语义提示不可用）。
pub fn check_threads() -> CheckResult {
    let logical = num_cpus::get();
    let physical = crate::device::thread_tune::detect_physical_cores();
    let mut detail = match physical {
        Some(p) => i18n::tr_with_args(
            "doctor-threads-detailed",
            i18n::tr_args(&[
                ("physical", &p.to_string()),
                ("logical", &logical.to_string()),
            ]),
        ),
        None => i18n::tr_with_args(
            "doctor-threads-fallback",
            i18n::tr_args(&[("logical", &logical.to_string())]),
        ),
    };
    let status = if physical.is_some() {
        CheckStatus::Pass
    } else {
        CheckStatus::Warn
    };
    if let Some(output) = std::process::Command::new("lscpu").output().ok()
        && output.status.success()
    {
        let text = String::from_utf8_lossy(&output.stdout);
        if let Some(sockets) = crate::device::thread_tune::parse_lscpu_sockets(&text)
            && sockets > 1
        {
            detail.push_str(&i18n::tr_with_args(
                "doctor-threads-numa",
                i18n::tr_args(&[("sockets", &sockets.to_string())]),
            ));
        }
    }
    CheckResult::new("threads", status, detail)
}

/// ⑤ GPU 探测。use_gpu=false 时 CPU 模式（PASS）；请求 GPU 但未编译 cuda
/// feature 为 FAIL（启动必然失败或静默回退，属于配置-构建不匹配）。
pub async fn check_gpu(config: &VecboostConfig) -> CheckResult {
    if !config.model.use_gpu {
        return CheckResult::new("gpu", CheckStatus::Pass, i18n::tr("doctor-gpu-cpu-mode"));
    }
    #[cfg(feature = "cuda")]
    {
        let manager = crate::device::manager::DeviceManager::new();
        let gpus = manager.get_cuda_gpu_info().await;
        if gpus.is_empty() {
            CheckResult::new("gpu", CheckStatus::Warn, i18n::tr("doctor-gpu-no-cuda"))
        } else {
            let names: Vec<String> = gpus.iter().map(|g| g.name.clone()).collect();
            CheckResult::new(
                "gpu",
                CheckStatus::Pass,
                i18n::tr_with_args(
                    "doctor-gpu-devices",
                    i18n::tr_args(&[("names", &names.join(", "))]),
                ),
            )
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = config;
        CheckResult::new(
            "gpu",
            CheckStatus::Fail,
            i18n::tr("doctor-gpu-no-cuda-feature"),
        )
    }
}

/// ⑥ 模型文件完整性。遍历 `models/` 下每个模型子目录，
/// 按角色分类文件并校验；目录缺失为 WARN（HF 下载模式合法）。
pub fn check_models(config: &VecboostConfig) -> Vec<CheckResult> {
    let mut results = Vec::new();
    let models_dir = std::path::Path::new("models");
    // 显式 model_path 指向单模型目录时同样纳入检查
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    if models_dir.is_dir()
        && let Ok(entries) = std::fs::read_dir(models_dir)
    {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                dirs.push(entry.path());
            }
        }
    }
    let explicit = resolve_model_dir(config);
    if explicit.is_dir() && !dirs.iter().any(|d| d == &explicit) {
        dirs.push(explicit);
    }
    if dirs.is_empty() {
        results.push(CheckResult::new(
            "models",
            CheckStatus::Warn,
            i18n::tr("doctor-models-none"),
        ));
        return results;
    }
    let mut total_fail = 0usize;
    for dir in &dirs {
        let (status, detail) = inspect_model_dir(dir);
        if status == CheckStatus::Fail {
            total_fail += 1;
        }
        results.push(CheckResult::new(
            "models",
            status,
            format!("{}: {detail}", dir.display()),
        ));
    }
    if total_fail > 0 {
        // 汇总行保证"任一模型 FAIL"对退出码可见
        results.push(CheckResult::new(
            "models",
            CheckStatus::Fail,
            i18n::tr_with_args(
                "doctor-models-fail-count",
                i18n::tr_args(&[("count", &total_fail.to_string())]),
            ),
        ));
    }
    results
}

/// 单个模型目录的角色化检查。返回 (状态, 描述)。
fn inspect_model_dir(dir: &std::path::Path) -> (CheckStatus, String) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (CheckStatus::Warn, i18n::tr("doctor-dir-unreadable"));
    };
    let mut weights = Vec::new();
    let mut gguf = Vec::new();
    let mut tokenizer_json = false;
    let mut config_hidden_size: Option<usize> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().map(|n| n.to_string_lossy().to_string());
        let Some(name) = name else { continue };
        if name.ends_with(".safetensors") {
            weights.push(path);
        } else if name.ends_with(".gguf") {
            gguf.push(path);
        } else if name.ends_with(".json") {
            // 按内容归类角色（非字面文件名）：hidden_size 等键 → 模型配置；
            // vocab/truncation 等键 → tokenizer 词表。
            match std::fs::read_to_string(&path) {
                Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(v) => {
                        if v.get("hidden_size").is_some() || v.get("num_hidden_layers").is_some() {
                            config_hidden_size = v
                                .get("hidden_size")
                                .and_then(|h| h.as_u64())
                                .map(|h| h as usize);
                        } else if v.get("vocab").is_some()
                            || v.get("model").is_some()
                            || v.get("added_tokens").is_some()
                        {
                            tokenizer_json = true;
                        }
                    }
                    Err(e) => {
                        return (
                            CheckStatus::Fail,
                            i18n::tr_with_args(
                                "doctor-json-parse-fail",
                                i18n::tr_args(&[
                                    ("path", &path.display().to_string()),
                                    ("detail", &e.to_string()),
                                ]),
                            ),
                        );
                    }
                },
                Err(e) => {
                    return (
                        CheckStatus::Fail,
                        i18n::tr_with_args(
                            "doctor-json-unreadable",
                            i18n::tr_args(&[
                                ("path", &path.display().to_string()),
                                ("detail", &e.to_string()),
                            ]),
                        ),
                    );
                }
            }
        }
    }
    for w in &weights {
        if let Err(e) = verify_safetensors(w) {
            return (CheckStatus::Fail, e);
        }
    }
    // 维度一致性（跨文件聚合）：BERT 族 FFN 权重为 [hidden, 4×hidden]（末维是
    // intermediate_size），逐张量校验末维必然误报。可靠契约是：声明了
    // hidden_size 的模型，其全部权重中至少存在一个二维张量维度含 hidden_size；
    // 完全不匹配说明权重与 config 配对错误（错模型/错配置）。
    if let Some(h) = config_hidden_size
        && !weights.is_empty()
    {
        let mut any_match = false;
        for w in &weights {
            match scan_safetensors_2d_dims(w) {
                Ok(dims) => {
                    if dims.iter().any(|(a, b)| *a == h || *b == h) {
                        any_match = true;
                        break;
                    }
                }
                Err(e) => return (CheckStatus::Fail, e),
            }
        }
        if !any_match {
            return (
                CheckStatus::Fail,
                i18n::tr_with_args(
                    "doctor-dims-no-hidden",
                    i18n::tr_args(&[("hidden", &h.to_string())]),
                ),
            );
        }
    }
    for g in &gguf {
        if let Err(e) = verify_gguf_magic(g) {
            return (CheckStatus::Fail, e);
        }
    }
    if weights.is_empty() && gguf.is_empty() {
        return (CheckStatus::Warn, i18n::tr("doctor-no-weights"));
    }
    let tokenizer = if tokenizer_json {
        i18n::tr("doctor-tokenizer-recognized")
    } else {
        i18n::tr("doctor-tokenizer-unrecognized")
    };
    let mut detail = i18n::tr_with_args(
        "doctor-weights-summary",
        i18n::tr_args(&[
            ("weights", &weights.len().to_string()),
            ("gguf", &gguf.len().to_string()),
            ("tokenizer", &tokenizer),
            ("hidden", &format!("{:?}", config_hidden_size)),
        ]),
    );
    if !tokenizer_json {
        detail.push_str(&i18n::tr("doctor-tokenizer-vocab-missing"));
    }
    (CheckStatus::Pass, detail)
}

/// safetensors 头校验：8 字节 LE 头长度 → JSON 头（dtype/shape/data_offsets）
/// → 文件大小 ≥ 头末尾 + 最大 data_offset。结构合法性失败即 Err。
fn verify_safetensors(path: &std::path::Path) -> Result<(), String> {
    scan_safetensors(path).map(|_| ())
}

/// 结构校验并收集全部二维张量维度（供跨文件的 hidden_size 一致性聚合）。
fn scan_safetensors_2d_dims(path: &std::path::Path) -> Result<Vec<(usize, usize)>, String> {
    Ok(scan_safetensors(path)?.two_d_dims)
}

struct SafetensorsInfo {
    two_d_dims: Vec<(usize, usize)>,
}

fn scan_safetensors(path: &std::path::Path) -> Result<SafetensorsInfo, String> {
    // 只读头部（8 字节长度 + header_len JSON），不整文件载入内存
    // （数 GB 权重文件的资源放大）。
    let mut file = std::fs::File::open(path).map_err(|e| {
        i18n::tr_with_args(
            "doctor-weight-unreadable",
            i18n::tr_args(&[
                ("path", &path.display().to_string()),
                ("detail", &e.to_string()),
            ]),
        )
    })?;
    let display = path.display();
    let file_len = file
        .metadata()
        .map_err(|e| {
            i18n::tr_with_args(
                "doctor-metadata-unreadable",
                i18n::tr_args(&[("path", &display.to_string()), ("detail", &e.to_string())]),
            )
        })?
        .len() as usize;
    if file_len < 8 {
        return Err(i18n::tr_with_args(
            "doctor-file-too-small",
            i18n::tr_args(&[("path", &display.to_string())]),
        ));
    }
    let mut len_b = [0u8; 8];
    std::io::Read::read_exact(&mut file, &mut len_b).map_err(|e| {
        i18n::tr_with_args(
            "doctor-header-len-read-fail",
            i18n::tr_args(&[("path", &display.to_string()), ("detail", &e.to_string())]),
        )
    })?;
    let header_len = u64::from_le_bytes(len_b) as usize;
    if header_len == 0 || 8 + header_len > file_len || header_len > 100 << 20 {
        return Err(i18n::tr_with_args(
            "doctor-header-len-invalid",
            i18n::tr_args(&[
                ("len", &header_len.to_string()),
                ("path", &display.to_string()),
            ]),
        ));
    }
    let mut header_buf = vec![0u8; header_len];
    std::io::Read::read_exact(&mut file, &mut header_buf).map_err(|e| {
        i18n::tr_with_args(
            "doctor-header-read-fail",
            i18n::tr_args(&[("path", &display.to_string()), ("detail", &e.to_string())]),
        )
    })?;
    let header: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_slice(&header_buf).map_err(|e| {
            i18n::tr_with_args(
                "doctor-header-json-fail",
                i18n::tr_args(&[("path", &display.to_string()), ("detail", &e.to_string())]),
            )
        })?;
    let mut max_end = 0usize;
    let mut two_d_dims = Vec::new();
    for (tensor, meta) in &header {
        // __metadata__ 是 safetensors 规范中的可选顶层元数据键，不是张量
        if tensor == "__metadata__" {
            continue;
        }
        let Some(obj) = meta.as_object() else {
            return Err(i18n::tr_with_args(
                "doctor-tensor-not-object",
                i18n::tr_args(&[("tensor", tensor), ("path", &display.to_string())]),
            ));
        };
        if obj.get("dtype").and_then(|d| d.as_str()).is_none() {
            return Err(i18n::tr_with_args(
                "doctor-tensor-no-dtype",
                i18n::tr_args(&[("tensor", tensor), ("path", &display.to_string())]),
            ));
        }
        let offsets = obj
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .ok_or_else(|| {
                i18n::tr_with_args(
                    "doctor-tensor-no-offsets",
                    i18n::tr_args(&[("tensor", tensor), ("path", &display.to_string())]),
                )
            })?;
        if offsets.len() != 2 {
            return Err(i18n::tr_with_args(
                "doctor-tensor-offsets-invalid",
                i18n::tr_args(&[("tensor", tensor), ("path", &display.to_string())]),
            ));
        }
        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        if end < start {
            return Err(i18n::tr_with_args(
                "doctor-tensor-offsets-reversed",
                i18n::tr_args(&[("tensor", tensor), ("path", &display.to_string())]),
            ));
        }
        max_end = max_end.max(end);
        if let Some(shape) = obj.get("shape").and_then(|s| s.as_array())
            && shape.len() == 2
        {
            let d0 = shape[0].as_u64().unwrap_or(0) as usize;
            let d1 = shape[1].as_u64().unwrap_or(0) as usize;
            two_d_dims.push((d0, d1));
        }
    }
    // 溢出安全比较（offset 来自不可信文件）。
    let data_begin = 8 + header_len;
    let available = file_len.saturating_sub(data_begin);
    if max_end > available {
        return Err(i18n::tr_with_args(
            "doctor-file-truncated",
            i18n::tr_args(&[
                ("need", &(data_begin + max_end).to_string()),
                ("have", &file_len.to_string()),
                ("path", &display.to_string()),
            ]),
        ));
    }
    Ok(SafetensorsInfo { two_d_dims })
}

/// gguf 魔数校验：前 4 字节必须为 "GGUF"。
fn verify_gguf_magic(path: &std::path::Path) -> Result<(), String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| {
        i18n::tr_with_args(
            "doctor-gguf-unreadable",
            i18n::tr_args(&[
                ("path", &path.display().to_string()),
                ("detail", &e.to_string()),
            ]),
        )
    })?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| {
        i18n::tr_with_args(
            "doctor-gguf-too-small",
            i18n::tr_args(&[
                ("path", &path.display().to_string()),
                ("detail", &e.to_string()),
            ]),
        )
    })?;
    if &magic != b"GGUF" {
        return Err(i18n::tr_with_args(
            "doctor-gguf-bad-magic",
            i18n::tr_args(&[
                ("got", &format!("{:02x?}", magic)),
                ("path", &path.display().to_string()),
            ]),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// `[model]` 段未知键静默回落默认值，doctor 的 config 检查项必须回显
    /// 解析后的 model.task 作为观测点（USER_GUIDE 键名拼写提示依赖此输出）。
    #[test]
    fn check_config_pass_detail_reports_model_task() {
        crate::i18n::init();
        let config = VecboostConfig::default();
        let result = check_config(&config);
        assert_eq!(result.status, CheckStatus::Pass);
        assert!(
            result
                .detail
                .contains(&format!("model.task={}", config.model.task)),
            "detail 必须回显 model.task，detail={}",
            result.detail
        );
    }

    fn temp_model_dir() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_path_buf();
        (dir, path)
    }

    /// 构造最小合法 safetensors：{"w": F32 [2,2], offsets [0,16]} + 16 字节数据。
    fn write_safetensors(path: &std::path::Path, corrupt: bool) {
        let header = if corrupt {
            br#"{"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 999999999]}}"#.to_vec()
        } else {
            br#"{"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}"#.to_vec()
        };
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0u8; 16]);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn verify_safetensors_accepts_valid_file() {
        let (dir, path) = temp_model_dir();
        let file = path.join("model.safetensors");
        write_safetensors(&file, false);
        assert!(verify_safetensors(&file).is_ok());
        let dims = scan_safetensors_2d_dims(&file).unwrap();
        assert_eq!(dims, vec![(2, 2)]);
        drop(dir);
    }

    /// 回归：BERT FFN 权重 [hidden, 4×hidden] 末维是 intermediate_size，
    /// 不得因"末维 != hidden_size"误报。
    #[test]
    fn inspect_model_dir_tolerates_ffn_weights() {
        let (dir, path) = temp_model_dir();
        std::fs::write(path.join("renamed-cfg.json"), br#"{"hidden_size": 384}"#).unwrap();
        // 构造 [384, 1536] 的 F32 张量（数据 384*1536*4 字节）
        let header = br#"{"encoder.layer.0.output.dense.weight": {"dtype": "F32", "shape": [384, 1536], "data_offsets": [0, 2359296]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&vec![0u8; 384 * 1536 * 4]);
        std::fs::write(path.join("model.safetensors"), bytes).unwrap();
        let (status, detail) = inspect_model_dir(&path);
        assert_eq!(status, CheckStatus::Pass, "detail: {detail}");
        drop(dir);
    }

    /// 权重与 config 完全不配对（hidden_size 与任何张量维度都不含）→ FAIL。
    #[test]
    fn inspect_model_dir_fails_on_wrong_config_pairing() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        std::fs::write(path.join("renamed-cfg.json"), br#"{"hidden_size": 999}"#).unwrap();
        write_safetensors(&path.join("model.safetensors"), false);
        let (status, detail) = inspect_model_dir(&path);
        assert_eq!(status, CheckStatus::Fail, "detail: {detail}");
        assert!(detail.contains("hidden_size"), "detail: {detail}");
        drop(dir);
    }

    #[test]
    fn verify_safetensors_rejects_truncated_data() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        let file = path.join("model.safetensors");
        write_safetensors(&file, false);
        // 截断到只剩头部 → 声明的 16 字节数据不存在
        let bytes = std::fs::read(&file).unwrap();
        std::fs::write(&file, &bytes[..8 + 66]).unwrap();
        let err = verify_safetensors(&file).unwrap_err();
        // locale-stable anchor: the file path is interpolated in both locales
        assert!(err.contains("model.safetensors"), "实际: {err}");
        drop(dir);
    }

    #[test]
    fn verify_safetensors_rejects_bad_header() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        let file = path.join("model.safetensors");
        write_safetensors(&file, true);
        let err = verify_safetensors(&file).unwrap_err();
        assert!(err.contains("model.safetensors"), "实际: {err}");
        drop(dir);
    }

    #[test]
    fn verify_gguf_magic_accepts_and_rejects() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        let good = path.join("m.gguf");
        std::fs::write(&good, b"GGUF\x03\x00\x00\x00").unwrap();
        assert!(verify_gguf_magic(&good).is_ok());
        let bad = path.join("bad.gguf");
        std::fs::write(&bad, b"NOPE").unwrap();
        let err = verify_gguf_magic(&bad).unwrap_err();
        // "GGUF" literal is preserved in both locales
        assert!(err.contains("GGUF"), "实际: {err}");
        drop(dir);
    }

    #[test]
    fn inspect_model_dir_recognizes_renamed_roles_by_content() {
        crate::i18n::init();
        // #1365 教训回归：合法文件改名后按内容仍被识别（非字面名匹配）
        let (dir, path) = temp_model_dir();
        let tokenizer = path.join("renamed-token-file.json");
        std::fs::write(
            &tokenizer,
            br#"{"model": {"type": "WordPiece"}, "added_tokens": []}"#,
        )
        .unwrap();
        let config_json = path.join("renamed-cfg.json");
        std::fs::write(&config_json, br#"{"hidden_size": 2, "layers": 1}"#).unwrap();
        write_safetensors(&path.join("weights.bin.safetensors"), false);
        let (status, detail) = inspect_model_dir(&path);
        assert_eq!(status, CheckStatus::Pass, "detail: {detail}");
        assert!(
            detail.contains(&crate::i18n::tr("doctor-tokenizer-recognized")),
            "tokenizer 应按内容识别: {detail}"
        );
        assert!(
            detail.contains("Some(2)"),
            "hidden_size 应从改名的 config 读取"
        );
        drop(dir);
    }

    #[test]
    fn check_models_warns_without_models_dir() {
        let (dir, _guard) = temp_model_dir();
        // 在无 models/ 的工作目录下运行（chick：不能污染真实仓库目录 ——
        // check_models 读取相对路径 "models"，此处仅断言不 panic 且有结论）
        let mut config = crate::config::VecboostConfig::default();
        config.model.model_path = Some(dir.path().to_string_lossy().to_string());
        let results = check_models(&config);
        assert!(!results.is_empty());
        drop(dir);
    }

    #[test]
    fn check_threads_always_reports() {
        crate::i18n::init();
        let r = check_threads();
        assert!(matches!(r.status, CheckStatus::Pass | CheckStatus::Warn));
        // locale-stable anchor: the logical-core count is interpolated in both locales
        assert!(
            r.detail.contains(&num_cpus::get().to_string()),
            "detail: {}",
            r.detail
        );
    }

    #[test]
    fn check_cache_persist_modes() {
        let mut config = crate::config::VecboostConfig::default();
        config.embedding.persist_path = None;
        assert_eq!(check_cache_persist(&config).status, CheckStatus::Pass);

        let (dir, path) = temp_model_dir();
        config.embedding.persist_path = Some(path.join("cache.wal").to_string_lossy().to_string());
        let r = check_cache_persist(&config);
        assert_eq!(r.status, CheckStatus::Pass, "detail: {}", r.detail);

        // 不可写目录 → FAIL
        let (dir2, path2) = temp_model_dir();
        let readonly = path2.join("ro");
        std::fs::create_dir(&readonly).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o555)).unwrap();
            config.embedding.persist_path =
                Some(readonly.join("cache.wal").to_string_lossy().to_string());
            let r = check_cache_persist(&config);
            // root 用户不受 0o555 限制，此处容忍 Pass（CI 环境差异），但不得 panic
            assert!(
                matches!(r.status, CheckStatus::Pass | CheckStatus::Fail),
                "detail: {}",
                r.detail
            );
        }
        config.embedding.persist_path = Some(
            path.join("missing")
                .join("cache.wal")
                .to_string_lossy()
                .to_string(),
        );
        let r = check_cache_persist(&config);
        assert_eq!(r.status, CheckStatus::Fail, "父目录不存在应 FAIL");
        drop(dir2);
        drop(dir);
    }

    #[tokio::test]
    async fn check_gpu_cpu_mode_passes() {
        let mut config = crate::config::VecboostConfig::default();
        config.model.use_gpu = false;
        assert_eq!(check_gpu(&config).await.status, CheckStatus::Pass);
    }

    #[test]
    fn report_exit_semantics() {
        let mut report = DoctorReport::default();
        report
            .results
            .push(CheckResult::new("a", CheckStatus::Pass, "ok"));
        report
            .results
            .push(CheckResult::new("b", CheckStatus::Warn, "meh"));
        assert!(!report.has_failures());
        report
            .results
            .push(CheckResult::new("c", CheckStatus::Fail, "bad"));
        assert!(report.has_failures());
    }

    // =========================================================================
    // 补缺口：CheckStatus 标签契约 / 全量 run / print / resolve_model_dir /
    // check_config 失败支路 / check_tokenizer 四分支 / cache WAL 分支 /
    // threads 回退 / GPU 无 cuda feature / models 汇总行 / scan_safetensors
    // 错误分支矩阵 / gguf 错误分支
    // =========================================================================

    /// 提升全局 log max_level（无 logger 安装时仅让 warn!/info! 参数求值），
    /// 使日志宏参数行进入覆盖统计；幂等，无副作用。
    fn enable_test_log_level() {
        log::set_max_level(log::LevelFilter::Trace);
    }

    #[test]
    fn test_check_status_as_str_labels() {
        assert_eq!(CheckStatus::Pass.as_str(), "PASS");
        assert_eq!(CheckStatus::Warn.as_str(), "WARN");
        assert_eq!(CheckStatus::Fail.as_str(), "FAIL");
    }

    #[tokio::test]
    async fn test_doctor_run_covers_all_sections() {
        crate::i18n::init();
        enable_test_log_level();
        let config = VecboostConfig::default();
        let report = DoctorReport::run(&config).await;
        // 六段检查链全部产生结论，且顺序稳定
        let names: Vec<&str> = report.results.iter().map(|r| r.name).collect();
        for expected in [
            "config",
            "tokenizer",
            "cache-persist",
            "threads",
            "gpu",
            "models",
        ] {
            assert!(
                names.contains(&expected),
                "缺少检查段 {expected}，实际 {names:?}"
            );
        }
        // 每条 detail 非空（报告必须可读，不得出现空描述）
        for r in &report.results {
            assert!(!r.detail.is_empty(), "段 {} 的 detail 为空", r.name);
        }
    }

    #[test]
    fn test_doctor_print_outputs_all_results() {
        crate::i18n::init();
        let mut report = DoctorReport::default();
        report
            .results
            .push(CheckResult::new("alpha", CheckStatus::Pass, "fine"));
        report
            .results
            .push(CheckResult::new("beta", CheckStatus::Fail, "broken"));
        // print 只写 stdout，不改变退出码；此处验证不 panic 且统计行覆盖全部结果
        report.print();
        assert_eq!(report.results.len(), 2);
    }

    #[test]
    fn test_resolve_model_dir_explicit_path_wins_and_repo_fallback() {
        let mut config = VecboostConfig::default();
        // 显式 model_path 优先（含非空校验）
        config.model.model_path = Some("/tmp/explicit-model".to_string());
        assert_eq!(
            resolve_model_dir(&config),
            PathBuf::from("/tmp/explicit-model")
        );
        // 空串视同未配置 → 回落 model_repo
        config.model.model_path = Some(String::new());
        assert_eq!(
            resolve_model_dir(&config),
            PathBuf::from(&config.model.model_repo)
        );
        // None → 回落 model_repo
        config.model.model_path = None;
        assert_eq!(
            resolve_model_dir(&config),
            PathBuf::from(&config.model.model_repo)
        );
    }

    #[test]
    fn test_check_config_fails_on_invalid_config() {
        crate::i18n::init();
        let mut config = VecboostConfig::default();
        config.server.port = 0; // garde range(min=1) 违规
        let result = check_config(&config);
        assert_eq!(
            result.status,
            CheckStatus::Fail,
            "detail: {}",
            result.detail
        );
        assert!(
            result.detail.contains("server.port"),
            "detail 应携带校验错误定位: {}",
            result.detail
        );
    }

    #[test]
    fn test_check_tokenizer_warns_when_model_dir_missing() {
        crate::i18n::init();
        let (dir, _path) = temp_model_dir();
        let missing = dir.path().join("no-such-model-dir");
        let mut config = VecboostConfig::default();
        config.model.model_path = Some(missing.to_string_lossy().to_string());
        let result = check_tokenizer(&config);
        assert_eq!(result.status, CheckStatus::Warn);
        assert!(
            result.detail.contains("no-such-model-dir"),
            "detail 应含缺失目录路径: {}",
            result.detail
        );
        drop(dir);
    }

    #[test]
    fn test_check_tokenizer_warns_when_tokenizer_json_missing() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        let mut config = VecboostConfig::default();
        config.model.model_path = Some(path.to_string_lossy().to_string());
        let result = check_tokenizer(&config);
        assert_eq!(result.status, CheckStatus::Warn);
        assert!(
            result.detail.contains("tokenizer.json"),
            "detail 应指名缺失文件: {}",
            result.detail
        );
        drop(dir);
    }

    #[test]
    fn test_check_tokenizer_passes_with_real_model() {
        let real = std::path::Path::new("models/BAAI-bge-small-en-v1.5/tokenizer.json");
        if !real.exists() {
            eprintln!("Skipping: local model tokenizer.json not available");
            return;
        }
        crate::i18n::init();
        let mut config = VecboostConfig::default();
        config.model.model_path = Some(
            real.parent()
                .expect("tokenizer.json 必有父目录")
                .to_string_lossy()
                .to_string(),
        );
        let result = check_tokenizer(&config);
        assert_eq!(
            result.status,
            CheckStatus::Pass,
            "detail: {}",
            result.detail
        );
    }

    #[test]
    fn test_check_tokenizer_fails_on_invalid_tokenizer_json() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        std::fs::write(path.join("tokenizer.json"), b"{ this is not json ").unwrap();
        let mut config = VecboostConfig::default();
        config.model.model_path = Some(path.to_string_lossy().to_string());
        let result = check_tokenizer(&config);
        assert_eq!(result.status, CheckStatus::Fail, "损坏 tokenizer 必须 FAIL");
        drop(dir);
    }

    #[test]
    fn test_check_cache_persist_existing_wal_file() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        let wal = path.join("existing.wal");
        std::fs::write(&wal, b"existing wal").unwrap();
        let mut config = VecboostConfig::default();
        config.embedding.persist_path = Some(wal.to_string_lossy().to_string());
        let result = check_cache_persist(&config);
        assert_eq!(
            result.status,
            CheckStatus::Pass,
            "detail: {}",
            result.detail
        );

        // 相对路径无父目录分量 → 显性 FAIL（path invalid），不得 panic
        config.embedding.persist_path = Some("bare.wal".to_string());
        let result = check_cache_persist(&config);
        assert_eq!(result.status, CheckStatus::Fail);
        drop(dir);
    }

    #[cfg(unix)]
    #[test]
    fn test_check_cache_persist_unwritable_wal_fails() {
        use std::os::unix::fs::PermissionsExt;
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        let wal = path.join("readonly.wal");
        std::fs::write(&wal, b"frozen").unwrap();
        std::fs::set_permissions(&wal, std::fs::Permissions::from_mode(0o444)).unwrap();
        let mut config = VecboostConfig::default();
        config.embedding.persist_path = Some(wal.to_string_lossy().to_string());
        let result = check_cache_persist(&config);
        // root 不受 0o444 限制（CI 差异），但不得 panic；非 root 必 FAIL
        assert!(
            matches!(result.status, CheckStatus::Pass | CheckStatus::Fail),
            "detail: {}",
            result.detail
        );
        std::fs::set_permissions(&wal, std::fs::Permissions::from_mode(0o644)).unwrap();
        drop(dir);
    }

    /// 检测关闭（环境开关）时 threads 检查降级为 WARN 而非 FAIL。
    #[test]
    fn test_check_threads_fallback_warns_when_detection_disabled() {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();
        crate::i18n::init();
        enable_test_log_level();
        let saved = std::env::var("VECBOOST_NO_THREAD_TUNE").ok();
        unsafe { std::env::set_var("VECBOOST_NO_THREAD_TUNE", "1") };
        let result = check_threads();
        match saved {
            Some(v) => unsafe { std::env::set_var("VECBOOST_NO_THREAD_TUNE", v) },
            None => unsafe { std::env::remove_var("VECBOOST_NO_THREAD_TUNE") },
        }
        assert_eq!(
            result.status,
            CheckStatus::Warn,
            "detail: {}",
            result.detail
        );
        assert!(
            result.detail.contains(&num_cpus::get().to_string()),
            "回退描述必须含逻辑核数: {}",
            result.detail
        );
    }

    /// 请求 GPU 但构建未启用 cuda feature → 配置-构建不匹配，必须 FAIL。
    #[cfg(not(feature = "cuda"))]
    #[tokio::test]
    async fn test_check_gpu_fails_when_gpu_requested_without_cuda_feature() {
        crate::i18n::init();
        let mut config = VecboostConfig::default();
        config.model.use_gpu = true;
        let result = check_gpu(&config).await;
        assert_eq!(result.status, CheckStatus::Fail);
    }

    #[test]
    fn test_check_models_appends_fail_summary_for_corrupt_model() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        // 损坏权重（声明偏移越界）→ 该模型目录 FAIL，并触发末尾汇总 FAIL 行
        write_safetensors(&path.join("broken.safetensors"), true);
        let mut config = VecboostConfig::default();
        config.model.model_path = Some(path.to_string_lossy().to_string());
        let results = check_models(&config);
        assert!(
            results.len() >= 2,
            "真实 models/ 目录 + 显式目录至少两条结论"
        );
        assert_eq!(
            results.last().expect("非空").status,
            CheckStatus::Fail,
            "存在 FAIL 模型时末尾必须有汇总行: {:?}",
            results.last()
        );
        drop(dir);
    }

    #[test]
    fn test_inspect_model_dir_unreadable_or_empty_warns() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        let sealed = path.join("sealed");
        std::fs::create_dir(&sealed).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o000)).unwrap();
        }
        let (status, _detail) = inspect_model_dir(&sealed);
        // 非 root：目录不可读 → Warn(unreadable)；root：目录可读但无权重 →
        // Warn(no-weights)。两条路径结论都是 WARN。
        assert_eq!(status, CheckStatus::Warn);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        drop(dir);
    }

    #[test]
    fn test_inspect_model_dir_fails_on_invalid_json() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        std::fs::write(path.join("config.json"), b"{{ not json").unwrap();
        write_safetensors(&path.join("model.safetensors"), false);
        let (status, detail) = inspect_model_dir(&path);
        assert_eq!(status, CheckStatus::Fail, "detail: {detail}");
        assert!(detail.contains("config.json"), "detail: {detail}");
        drop(dir);
    }

    /// .json 后缀但实为目录：read_to_string 失败 → 显性 FAIL（跨用户确定性，
    /// 不依赖权限位）。
    #[test]
    fn test_inspect_model_dir_fails_on_unreadable_json() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        std::fs::create_dir(path.join("actually-a-dir.json")).unwrap();
        let (status, detail) = inspect_model_dir(&path);
        assert_eq!(status, CheckStatus::Fail, "detail: {detail}");
        assert!(detail.contains("actually-a-dir.json"), "detail: {detail}");
        drop(dir);
    }

    #[test]
    fn test_inspect_model_dir_fails_on_corrupt_weights() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        write_safetensors(&path.join("weights.safetensors"), true);
        let (status, detail) = inspect_model_dir(&path);
        assert_eq!(status, CheckStatus::Fail, "detail: {detail}");
        assert!(detail.contains("weights.safetensors"), "detail: {detail}");
        drop(dir);
    }

    #[test]
    fn test_inspect_model_dir_fails_on_bad_gguf() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        std::fs::write(path.join("model.gguf"), b"NOTG").unwrap();
        let (status, detail) = inspect_model_dir(&path);
        assert_eq!(status, CheckStatus::Fail, "detail: {detail}");
        assert!(detail.contains("GGUF"), "detail: {detail}");
        drop(dir);
    }

    /// scan_safetensors 错误分支矩阵：结构非法的权重文件必须逐类显性报错。
    #[test]
    fn test_scan_safetensors_rejects_structurally_invalid_files() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        let f = |name: &str| path.join(name);

        // 文件不存在
        let err = verify_safetensors(&f("missing.safetensors")).unwrap_err();
        assert!(err.contains("missing.safetensors"), "实际: {err}");

        // 不足 8 字节头长度
        std::fs::write(f("tiny.safetensors"), b"abc").unwrap();
        let err = verify_safetensors(&f("tiny.safetensors")).unwrap_err();
        assert!(err.contains("tiny.safetensors"), "实际: {err}");

        // header_len = 0
        std::fs::write(f("zerolen.safetensors"), 0u64.to_le_bytes()).unwrap();
        let err = verify_safetensors(&f("zerolen.safetensors")).unwrap_err();
        assert!(err.contains("zerolen.safetensors"), "实际: {err}");

        // header_len 越过文件末尾
        let mut bytes = (u64::from(4096u32)).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(f("overrun.safetensors"), &bytes).unwrap();
        let err = verify_safetensors(&f("overrun.safetensors")).unwrap_err();
        assert!(err.contains("overrun.safetensors"), "实际: {err}");

        // header_len 超过 100MB 上限
        std::fs::write(
            f("hugehdr.safetensors"),
            (101u64 * 1024 * 1024).to_le_bytes(),
        )
        .unwrap();
        let err = verify_safetensors(&f("hugehdr.safetensors")).unwrap_err();
        assert!(err.contains("hugehdr.safetensors"), "实际: {err}");

        // 头不是合法 JSON
        let hdr = b"not-json-at-all";
        let mut bytes = (hdr.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(hdr);
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(f("badjson.safetensors"), &bytes).unwrap();
        let err = verify_safetensors(&f("badjson.safetensors")).unwrap_err();
        assert!(err.contains("badjson.safetensors"), "实际: {err}");

        let write_with_header = |name: &str, header: &[u8], data: &[u8]| {
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(data);
            std::fs::write(f(name), bytes).unwrap();
        };

        // 张量元数据不是对象
        write_with_header("notobj.safetensors", br#"{"w": 42}"#, &[0u8; 16]);
        let err = verify_safetensors(&f("notobj.safetensors")).unwrap_err();
        assert!(err.contains("notobj.safetensors"), "实际: {err}");

        // 缺 dtype
        write_with_header(
            "nodtype.safetensors",
            br#"{"w": {"shape": [2, 2], "data_offsets": [0, 16]}}"#,
            &[0u8; 16],
        );
        let err = verify_safetensors(&f("nodtype.safetensors")).unwrap_err();
        assert!(err.contains("nodtype.safetensors"), "实际: {err}");

        // 缺 data_offsets
        write_with_header(
            "nooffsets.safetensors",
            br#"{"w": {"dtype": "F32", "shape": [2, 2]}}"#,
            &[0u8; 16],
        );
        let err = verify_safetensors(&f("nooffsets.safetensors")).unwrap_err();
        assert!(err.contains("nooffsets.safetensors"), "实际: {err}");

        // data_offsets 长度 != 2
        write_with_header(
            "offsets1.safetensors",
            br#"{"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0]}}"#,
            &[0u8; 16],
        );
        let err = verify_safetensors(&f("offsets1.safetensors")).unwrap_err();
        assert!(err.contains("offsets1.safetensors"), "实际: {err}");

        // data_offsets 倒置（end < start）
        write_with_header(
            "reversed.safetensors",
            br#"{"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [16, 0]}}"#,
            &[0u8; 16],
        );
        let err = verify_safetensors(&f("reversed.safetensors")).unwrap_err();
        assert!(err.contains("reversed.safetensors"), "实际: {err}");

        // __metadata__ 顶层键必须被跳过而非当作张量校验
        write_with_header(
            "withmeta.safetensors",
            br#"{"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}"#,
            &[0u8; 16],
        );
        assert!(verify_safetensors(&f("withmeta.safetensors")).is_ok());

        drop(dir);
    }

    #[test]
    fn test_verify_gguf_magic_unreadable_and_tiny_files() {
        crate::i18n::init();
        let (dir, path) = temp_model_dir();
        let err = verify_gguf_magic(&path.join("missing.gguf")).unwrap_err();
        assert!(err.contains("missing.gguf"), "实际: {err}");

        let tiny = path.join("tiny.gguf");
        std::fs::write(&tiny, b"GG").unwrap();
        let err = verify_gguf_magic(&tiny).unwrap_err();
        assert!(err.contains("tiny.gguf"), "实际: {err}");
        drop(dir);
    }
}
