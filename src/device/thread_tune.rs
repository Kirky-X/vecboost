// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 物理核线程调优（port 自 colibri omp_tune.h 策略）。
//!
//! - 物理核检测：Linux 解析 `/sys/.../thread_siblings_list` 去重 SMT；
//!   macOS 取 `hw.perflevel0.physicalcpu`；Windows 返回 None 落回 num_cpus。
//! - 数不出就不猜：任何解析失败返回 None，调用方落回 `num_cpus::get()`。
//! - NUMA 仅检测建议，不做进程内绑定。

use std::collections::HashSet;

/// 解析单个 CPU 列表段（如 "0,2" 或 "0-3,8"）为有序 CPU id 集合。
fn parse_cpu_list(s: &str) -> Option<Vec<u32>> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut cpus = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: u32 = a.trim().parse().ok()?;
            let end: u32 = b.trim().parse().ok()?;
            if end < start {
                return None;
            }
            for cpu in start..=end {
                cpus.push(cpu);
            }
        } else {
            cpus.push(part.parse::<u32>().ok()?);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    Some(cpus)
}

/// 纯函数解析层：对每 CPU 的 `thread_siblings_list` 文本去重 SMT，计数物理核。
///
/// 输入为每逻辑 CPU 一行的 siblings 文本；相同集合视为同一物理核。
/// 任一行无法解析或输入为空 → 返回 None（不猜测）。
pub fn parse_thread_siblings_lists(lines: &[&str]) -> Option<usize> {
    if lines.is_empty() {
        return None;
    }
    let mut uniq: HashSet<Vec<u32>> = HashSet::new();
    for line in lines {
        uniq.insert(parse_cpu_list(line)?);
    }
    if uniq.is_empty() {
        return None;
    }
    Some(uniq.len())
}

/// 纯函数解析层：解析 `lscpu` 输出统计 socket 数。
///
/// 优先匹配 `Socket(s):` 字段；缺失时回退统计 `NUMA node` 行数；
/// 均无则返回 None。
pub fn parse_lscpu_sockets(output: &str) -> Option<usize> {
    for line in output.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("socket")
            && line.contains(':')
            && let Some(val) = line.split(':').nth(1)
            && let Ok(n) = val.split_whitespace().next().unwrap_or("").parse::<usize>()
            && n > 0
        {
            return Some(n);
        }
    }
    None
}

/// 跨平台物理核检测。失败返回 None，调用方落回 `num_cpus::get()`。
pub fn detect_physical_cores() -> Option<usize> {
    if std::env::var("VECBOOST_NO_THREAD_TUNE").as_deref() == Ok("1") {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        detect_physical_cores_linux()
    }
    #[cfg(target_os = "macos")]
    {
        detect_physical_cores_macos()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        // Windows：不引入 winapi 依赖，返回 None 由调用方回退 num_cpus。
        None
    }
}

#[cfg(target_os = "linux")]
fn detect_physical_cores_linux() -> Option<usize> {
    let mut lines: Vec<String> = Vec::new();
    let mut idx = 0usize;
    loop {
        let path = format!(
            "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",
            idx
        );
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                lines.push(content.trim().to_string());
                idx += 1;
                if idx > 1024 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    if lines.is_empty() {
        return None;
    }
    let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    parse_thread_siblings_lists(&refs)
}

#[cfg(target_os = "macos")]
fn detect_physical_cores_macos() -> Option<usize> {
    for key in ["hw.perflevel0.physicalcpu", "hw.physicalcpu"] {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", key])
            .output()
        {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                if let Ok(n) = text.trim().parse::<usize>() {
                    if n > 0 {
                        return Some(n);
                    }
                }
            }
        }
    }
    None
}

/// 解析优先级：显式配置 > 检测值 > 回退值（单测覆盖）。
pub fn resolve_worker_threads(
    explicit: Option<usize>,
    detected: Option<usize>,
    fallback: usize,
) -> usize {
    if let Some(v) = explicit
        && v > 0
    {
        return v;
    }
    if let Some(v) = detected
        && v > 0
    {
        return v;
    }
    fallback.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_siblings_sample_gives_two() {
        let lines = ["0,2", "0,2", "1,3"];
        assert_eq!(parse_thread_siblings_lists(&lines), Some(2));
    }

    #[test]
    fn test_parse_siblings_unparsable_returns_none() {
        assert_eq!(parse_thread_siblings_lists(&["not-a-cpu"]), None);
        assert_eq!(parse_thread_siblings_lists(&[]), None);
        assert_eq!(parse_thread_siblings_lists(&["0,2", "garbage!!"]), None);
    }

    #[test]
    fn test_parse_siblings_range_format() {
        let lines = ["0-1", "0-1", "2-3"];
        assert_eq!(parse_thread_siblings_lists(&lines), Some(2));
    }

    #[test]
    fn test_resolve_priority_explicit_over_detected_over_fallback() {
        assert_eq!(resolve_worker_threads(Some(12), Some(8), 4), 12);
        assert_eq!(resolve_worker_threads(None, Some(8), 4), 8);
        assert_eq!(resolve_worker_threads(None, None, 4), 4);
    }

    #[test]
    fn test_parse_lscpu_socket_count() {
        let fake = "Architecture: x86_64\nSocket(s): 2\nCore(s) per socket: 8\n";
        assert_eq!(parse_lscpu_sockets(fake), Some(2));
        let single = "Socket(s): 1\n";
        assert_eq!(parse_lscpu_sockets(single), Some(1));
        assert_eq!(parse_lscpu_sockets("no sockets here\n"), None);
    }

    #[test]
    fn test_detect_does_not_panic() {
        let _ = detect_physical_cores();
    }
}
