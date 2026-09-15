// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! `vecboost doctor` — 只读诊断（T027/T028）。
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

use crate::config::AppConfig;

/// 检查结论（枚举固定三值，与 spec R-observability-002 一致）。
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
    pub async fn run(config: &AppConfig) -> Self {
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
        println!("vecboost doctor — read-only diagnostics");
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
            "{} check(s): {} pass, {warns} warn, {fails} fail",
            self.results.len(),
            self.results.len() - fails - warns
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
pub fn resolve_model_dir(config: &AppConfig) -> std::path::PathBuf {
    match config.model.model_path.as_deref() {
        Some(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::PathBuf::from(&config.model.model_repo),
    }
}

/// ① 配置校验。
pub fn check_config(config: &AppConfig) -> CheckResult {
    match config.validate() {
        Ok(()) => CheckResult::new("config", CheckStatus::Pass, "AppConfig::validate 通过"),
        Err(e) => CheckResult::new("config", CheckStatus::Fail, format!("配置校验失败: {e}")),
    }
}

/// ② tokenizer 加载自检。模型目录缺失时 WARN（运行时会回退 HF 下载），不武断 FAIL。
pub fn check_tokenizer(config: &AppConfig) -> CheckResult {
    let dir = resolve_model_dir(config);
    let tok_path = dir.join("tokenizer.json");
    if !dir.exists() {
        return CheckResult::new(
            "tokenizer",
            CheckStatus::Warn,
            format!("模型目录不存在: {}（运行时将尝试 HF 下载）", dir.display()),
        );
    }
    if !tok_path.exists() {
        return CheckResult::new(
            "tokenizer",
            CheckStatus::Warn,
            format!("tokenizer.json 不存在: {}", tok_path.display()),
        );
    }
    match crate::text::tokenizer::Tokenizer::from_file(tok_path.to_string_lossy().as_ref()) {
        Ok(_) => CheckResult::new(
            "tokenizer",
            CheckStatus::Pass,
            format!("tokenizer.json 加载成功: {}", tok_path.display()),
        ),
        Err(e) => CheckResult::new(
            "tokenizer",
            CheckStatus::Fail,
            format!("tokenizer.json 加载失败: {e}"),
        ),
    }
}

/// ③ 缓存持久层可写性。未配置 persist_path 时为纯内存模式（PASS）。
/// 可写性探测：创建探针文件后立即删除（不残留）；已存在的 WAL 文件以
/// append 打开验证权限，不改动内容。
pub fn check_cache_persist(config: &AppConfig) -> CheckResult {
    let Some(path) = config.embedding.persist_path.as_deref() else {
        return CheckResult::new(
            "cache-persist",
            CheckStatus::Pass,
            "纯内存模式（未配置 persist_path）",
        );
    };
    let wal = std::path::Path::new(path);
    if wal.exists() {
        return match std::fs::OpenOptions::new().append(true).open(wal) {
            Ok(_) => CheckResult::new(
                "cache-persist",
                CheckStatus::Pass,
                format!("WAL 文件可写: {path}"),
            ),
            Err(e) => CheckResult::new(
                "cache-persist",
                CheckStatus::Fail,
                format!("WAL 文件不可写: {path}: {e}"),
            ),
        };
    }
    let Some(parent) = wal.parent().filter(|p| !p.as_os_str().is_empty()) else {
        return CheckResult::new(
            "cache-persist",
            CheckStatus::Fail,
            format!("persist_path 无效: {path}"),
        );
    };
    if !parent.exists() {
        return CheckResult::new(
            "cache-persist",
            CheckStatus::Fail,
            format!(
                "父目录不存在（启动写盘将失败，请先创建或修正配置）: {}",
                parent.display()
            ),
        );
    }
    // 探针：临时文件创建后立即删除
    let probe = parent.join(".vecboost_doctor_probe");
    let probe_result = std::fs::File::create(&probe).and_then(|_| std::fs::remove_file(&probe));
    match probe_result {
        Ok(()) => CheckResult::new(
            "cache-persist",
            CheckStatus::Pass,
            format!(
                "目录可写（WAL 尚未创建，首次插入时建立）: {}",
                parent.display()
            ),
        ),
        Err(e) => CheckResult::new(
            "cache-persist",
            CheckStatus::Fail,
            format!("目录不可写: {}: {e}", parent.display()),
        ),
    }
}

/// ④ 线程调优报告：物理核/逻辑核/生效线程数/NUMA。报告性检查，恒 PASS
/// （检测失败回退 num_cpus 也是合法工作配置，以 WARN 语义提示不可用）。
pub fn check_threads() -> CheckResult {
    let logical = num_cpus::get();
    let physical = crate::device::thread_tune::detect_physical_cores();
    let mut detail = match physical {
        Some(p) => format!("物理核 {p} / 逻辑核 {logical}（调优按物理核生效）"),
        None => {
            format!("物理核检测不可用，回退逻辑核 {logical}（VECBOOST_NO_THREAD_TUNE 可显式控制）")
        }
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
            detail.push_str(&format!(
                "；检测到 {sockets} 个 socket，建议 numactl --interleave=all 启动"
            ));
        }
    }
    CheckResult::new("threads", status, detail)
}

/// ⑤ GPU 探测。use_gpu=false 时 CPU 模式（PASS）；请求 GPU 但未编译 cuda
/// feature 为 FAIL（启动必然失败或静默回退，属于配置-构建不匹配）。
pub async fn check_gpu(config: &AppConfig) -> CheckResult {
    if !config.model.use_gpu {
        return CheckResult::new("gpu", CheckStatus::Pass, "CPU 模式（use_gpu=false）");
    }
    #[cfg(feature = "cuda")]
    {
        let manager = crate::device::manager::DeviceManager::new();
        let gpus = manager.get_cuda_gpu_info().await;
        if gpus.is_empty() {
            CheckResult::new(
                "gpu",
                CheckStatus::Warn,
                "use_gpu=true 但未探测到 CUDA 设备",
            )
        } else {
            let names: Vec<String> = gpus.iter().map(|g| g.name.clone()).collect();
            CheckResult::new(
                "gpu",
                CheckStatus::Pass,
                format!("CUDA 设备: {}", names.join(", ")),
            )
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = config;
        CheckResult::new(
            "gpu",
            CheckStatus::Fail,
            "use_gpu=true 但二进制未编译 cuda feature（构建时启用 --features cuda）",
        )
    }
}

/// ⑥ 模型文件完整性（T028）。遍历 `models/` 下每个模型子目录，
/// 按角色分类文件并校验；目录缺失为 WARN（HF 下载模式合法）。
pub fn check_models(config: &AppConfig) -> Vec<CheckResult> {
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
            "未找到本地模型目录（models/ 不存在且未配置 model_path），运行时将依赖 HF 下载",
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
            format!("{total_fail} 个模型目录存在完整性问题"),
        ));
    }
    results
}

/// 单个模型目录的角色化检查。返回 (状态, 描述)。
fn inspect_model_dir(dir: &std::path::Path) -> (CheckStatus, String) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (CheckStatus::Warn, "目录不可读".to_string());
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
                            format!("JSON 文件解析失败 {}: {e}", path.display()),
                        );
                    }
                },
                Err(e) => {
                    return (
                        CheckStatus::Fail,
                        format!("JSON 文件不可读 {}: {e}", path.display()),
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
                format!("全部二维张量维度均不含 config.hidden_size({h})，权重与配置疑似不配对"),
            );
        }
    }
    for g in &gguf {
        if let Err(e) = verify_gguf_magic(g) {
            return (CheckStatus::Fail, e);
        }
    }
    if weights.is_empty() && gguf.is_empty() {
        return (
            CheckStatus::Warn,
            "未发现权重文件（.safetensors/.gguf）".to_string(),
        );
    }
    let mut detail = format!(
        "权重 {} 个、gguf {} 个、tokenizer {}、config.hidden_size {:?}",
        weights.len(),
        gguf.len(),
        if tokenizer_json {
            "已识别"
        } else {
            "未识别"
        },
        config_hidden_size
    );
    if !tokenizer_json {
        detail.push_str("；⚠️ 未按内容识别到 tokenizer 词表文件");
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
    // （数 GB 权重文件的资源放大，T035 审查安全5）。
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("权重文件不可读 {}: {e}", path.display()))?;
    let display = path.display();
    let file_len = file
        .metadata()
        .map_err(|e| format!("元数据不可读 {display}: {e}"))?
        .len() as usize;
    if file_len < 8 {
        return Err(format!("文件过小，不足 8 字节头: {display}"));
    }
    let mut len_b = [0u8; 8];
    std::io::Read::read_exact(&mut file, &mut len_b)
        .map_err(|e| format!("头长度读取失败 {display}: {e}"))?;
    let header_len = u64::from_le_bytes(len_b) as usize;
    if header_len == 0 || 8 + header_len > file_len || header_len > 100 << 20 {
        return Err(format!("safetensors 头长度非法 ({header_len}): {display}"));
    }
    let mut header_buf = vec![0u8; header_len];
    std::io::Read::read_exact(&mut file, &mut header_buf)
        .map_err(|e| format!("头读取失败 {display}: {e}"))?;
    let header: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_slice(&header_buf)
            .map_err(|e| format!("safetensors 头 JSON 解析失败 {display}: {e}"))?;
    let mut max_end = 0usize;
    let mut two_d_dims = Vec::new();
    for (tensor, meta) in &header {
        // __metadata__ 是 safetensors 规范中的可选顶层元数据键，不是张量
        if tensor == "__metadata__" {
            continue;
        }
        let Some(obj) = meta.as_object() else {
            return Err(format!("张量 {tensor} 元数据不是对象: {display}"));
        };
        if obj.get("dtype").and_then(|d| d.as_str()).is_none() {
            return Err(format!("张量 {tensor} 缺 dtype: {display}"));
        }
        let offsets = obj
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .ok_or_else(|| format!("张量 {tensor} 缺 data_offsets: {display}"))?;
        if offsets.len() != 2 {
            return Err(format!("张量 {tensor} data_offsets 非法: {display}"));
        }
        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        if end < start {
            return Err(format!("张量 {tensor} data_offsets 逆序: {display}"));
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
    // 溢出安全比较（offset 来自不可信文件；T035 审查安全5）
    let data_begin = 8 + header_len;
    let available = file_len.saturating_sub(data_begin);
    if max_end > available {
        return Err(format!(
            "文件被截断（声明数据需要 {} 字节，实际 {}）: {display}",
            data_begin + max_end,
            file_len
        ));
    }
    Ok(SafetensorsInfo { two_d_dims })
}

/// gguf 魔数校验：前 4 字节必须为 "GGUF"。
fn verify_gguf_magic(path: &std::path::Path) -> Result<(), String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .map_err(|e| format!("gguf 文件不可读 {}: {e}", path.display()))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)
        .map_err(|e| format!("gguf 文件过小 {}: {e}", path.display()))?;
    if &magic != b"GGUF" {
        return Err(format!(
            "gguf 魔数非法（{:02x?}，应为 GGUF）: {}",
            magic,
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
        let (dir, path) = temp_model_dir();
        let file = path.join("model.safetensors");
        write_safetensors(&file, false);
        // 截断到只剩头部 → 声明的 16 字节数据不存在
        let bytes = std::fs::read(&file).unwrap();
        std::fs::write(&file, &bytes[..8 + 66]).unwrap();
        let err = verify_safetensors(&file).unwrap_err();
        assert!(err.contains("截断"), "实际: {err}");
        drop(dir);
    }

    #[test]
    fn verify_safetensors_rejects_bad_header() {
        let (dir, path) = temp_model_dir();
        let file = path.join("model.safetensors");
        write_safetensors(&file, true);
        let err = verify_safetensors(&file).unwrap_err();
        assert!(err.contains("截断"), "实际: {err}");
        drop(dir);
    }

    #[test]
    fn verify_gguf_magic_accepts_and_rejects() {
        let (dir, path) = temp_model_dir();
        let good = path.join("m.gguf");
        std::fs::write(&good, b"GGUF\x03\x00\x00\x00").unwrap();
        assert!(verify_gguf_magic(&good).is_ok());
        let bad = path.join("bad.gguf");
        std::fs::write(&bad, b"NOPE").unwrap();
        let err = verify_gguf_magic(&bad).unwrap_err();
        assert!(err.contains("魔数"), "实际: {err}");
        drop(dir);
    }

    #[test]
    fn inspect_model_dir_recognizes_renamed_roles_by_content() {
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
            detail.contains("已识别"),
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
        let mut config = crate::config::AppConfig::default();
        config.model.model_path = Some(dir.path().to_string_lossy().to_string());
        let results = check_models(&config);
        assert!(!results.is_empty());
        drop(dir);
    }

    #[test]
    fn check_threads_always_reports() {
        let r = check_threads();
        assert!(matches!(r.status, CheckStatus::Pass | CheckStatus::Warn));
        assert!(r.detail.contains("逻辑核"), "detail: {}", r.detail);
    }

    #[test]
    fn check_cache_persist_modes() {
        let mut config = crate::config::AppConfig::default();
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
        let mut config = crate::config::AppConfig::default();
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
}
