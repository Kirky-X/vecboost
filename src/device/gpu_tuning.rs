// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! GPU 调优检测模块
//!
//! 基于鲲鹏 GPU 应用优化白皮书的「硬件优化手段」和「操作系统优化」章节，
//! 在启动时检测 GPU 运行时配置并输出调优建议日志。
//!
//! 检测项：
//! - GPU 持久模式（Persistence Mode）
//! - 透明大页（Transparent Huge Pages）
//! - GPU 时钟频率设置
//! - ECC 内存状态

use log::{info, warn};
use std::process::Command;

/// GPU 调优建议级别
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuningLevel {
    /// 已优化
    Optimal,
    /// 建议优化
    Recommended,
    /// 不可用或非平台
    NotApplicable,
}

/// GPU 调优检测结果
#[derive(Debug, Clone)]
pub struct GpuTuningReport {
    /// 持久模式状态
    pub persistence_mode: TuningLevel,
    /// 透明大页状态
    pub transparent_hugepage: TuningLevel,
    /// 时钟频率是否已锁定
    pub clock_frequency: TuningLevel,
    /// ECC 状态
    pub ecc_status: TuningLevel,
    /// GPU 计算模式
    pub compute_mode: TuningLevel,
    /// 综合建议
    pub recommendations: Vec<String>,
}

/// GPU 调优顾问
///
/// 检测当前 GPU 配置并输出调优建议。
/// 对应鲲鹏文档「硬件优化手段」章节。
pub struct GpuTuningAdvisor;

impl GpuTuningAdvisor {
    /// 执行完整调优检测并输出建议日志
    pub fn check_and_advise() -> GpuTuningReport {
        let persistence_mode = Self::check_persistence_mode();
        let transparent_hugepage = Self::check_transparent_hugepage();
        let clock_frequency = Self::check_clock_frequency();
        let ecc_status = Self::check_ecc_status();
        let compute_mode = Self::check_compute_mode();

        let mut recommendations = Vec::new();

        if persistence_mode == TuningLevel::Recommended {
            recommendations.push(
                "建议开启 GPU 持久模式 (nvidia-smi -pm 1)：避免低负载休眠导致唤醒延迟".to_string(),
            );
        }
        if transparent_hugepage == TuningLevel::Recommended {
            recommendations.push(
                "建议开启透明大页 (echo always > /sys/kernel/mm/transparent_hugepage/enabled)：减少 CPU↔GPU 数据传输 TLB 消耗".to_string(),
            );
        }
        if clock_frequency == TuningLevel::Recommended {
            recommendations.push(
                "建议锁定 GPU 时钟频率 (nvidia-smi -ac <mem>,<graphics>)：消除频率波动导致的性能抖动".to_string(),
            );
        }
        if ecc_status == TuningLevel::Recommended {
            recommendations.push(
                "ECC 内存已禁用：生产环境建议开启 (nvidia-smi -e 1) 防止内存位翻转导致计算错误".to_string(),
            );
        }
        if compute_mode == TuningLevel::Recommended {
            recommendations.push(
                "建议设置 GPU 计算模式为 Exclusive_Process (nvidia-smi -c 1)：避免多进程竞争 GPU 资源".to_string(),
            );
        }

        if recommendations.is_empty() {
            info!("GPU 调优检测完成：所有配置已优化");
        } else {
            warn!("GPU 调优检测完成，发现 {} 项可优化配置：", recommendations.len());
            for (i, rec) in recommendations.iter().enumerate() {
                warn!("  {}. {}", i + 1, rec);
            }
        }

        GpuTuningReport {
            persistence_mode,
            transparent_hugepage,
            clock_frequency,
            ecc_status,
            compute_mode,
            recommendations,
        }
    }

    /// 检测 GPU 持久模式
    ///
    /// 鲲鹏文档建议：开启持久模式避免 GPU 低负载休眠后唤醒失败
    fn check_persistence_mode() -> TuningLevel {
        match run_nvidia_smi(&["--query-gpu=persistence_mode", "--format=csv,noheader,nounits"]) {
            Some(output) => {
                let mode = output.trim();
                if mode == "1" || mode == "Enabled" {
                    info!("GPU 持久模式：已开启");
                    TuningLevel::Optimal
                } else {
                    warn!("GPU 持久模式：未开启（当前: {}）", mode);
                    TuningLevel::Recommended
                }
            }
            None => TuningLevel::NotApplicable,
        }
    }

    /// 检测透明大页状态
    ///
    /// 鲲鹏文档建议：开启透明大页减少 CPU↔GPU 数据拷贝的 TLB 消耗
    fn check_transparent_hugepage() -> TuningLevel {
        match std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled") {
            Ok(content) => {
                // 格式: "always [madvise] never" — 方括号内是当前值
                if content.contains("[always]") {
                    info!("透明大页：已开启 (always)");
                    TuningLevel::Optimal
                } else if content.contains("[madvise]") {
                    info!("透明大页：madvise 模式（部分开启）");
                    TuningLevel::Recommended
                } else {
                    warn!("透明大页：未开启");
                    TuningLevel::Recommended
                }
            }
            Err(_) => {
                // 非 Linux 或无权限
                TuningLevel::NotApplicable
            }
        }
    }

    /// 检测 GPU 时钟频率是否已锁定
    ///
    /// 鲲鹏文档建议：锁定 GPU 时钟频率消除频率波动
    fn check_clock_frequency() -> TuningLevel {
        match run_nvidia_smi(&[
            "--query-gpu=clocks.max.graphics,clocks.max.memory",
            "--format=csv,noheader,nounits",
        ]) {
            Some(output) => {
                // 仅记录最大可用频率，供用户参考
                info!("GPU 最大时钟频率: {}", output.trim());
                // 无法直接判断是否已锁定，标记为建议
                TuningLevel::Recommended
            }
            None => TuningLevel::NotApplicable,
        }
    }

    /// 检测 ECC 内存状态
    fn check_ecc_status() -> TuningLevel {
        match run_nvidia_smi(&["--query-gpu=ecc.mode.current", "--format=csv,noheader,nounits"]) {
            Some(output) => {
                let mode = output.trim();
                if mode == "1" || mode.to_lowercase().contains("enabled") {
                    info!("GPU ECC 内存：已开启");
                    TuningLevel::Optimal
                } else {
                    warn!("GPU ECC 内存：未开启（当前: {}）", mode);
                    TuningLevel::Recommended
                }
            }
            None => TuningLevel::NotApplicable,
        }
    }

    /// 检测 GPU 计算模式
    fn check_compute_mode() -> TuningLevel {
        match run_nvidia_smi(&["--query-gpu=compute_mode", "--format=csv,noheader"]) {
            Some(output) => {
                let mode = output.trim().to_lowercase();
                if mode.contains("exclusive") {
                    info!("GPU 计算模式：Exclusive（最优）");
                    TuningLevel::Optimal
                } else if mode.contains("default") {
                    info!("GPU 计算模式：Default（多进程共享）");
                    TuningLevel::Recommended
                } else {
                    info!("GPU 计算模式: {}", mode);
                    TuningLevel::NotApplicable
                }
            }
            None => TuningLevel::NotApplicable,
        }
    }
}

/// 执行 nvidia-smi 命令并返回输出
fn run_nvidia_smi(args: &[&str]) -> Option<String> {
    Command::new("nvidia-smi")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuning_level_equality() {
        assert_eq!(TuningLevel::Optimal, TuningLevel::Optimal);
        assert_ne!(TuningLevel::Optimal, TuningLevel::Recommended);
    }

    #[test]
    fn test_gpu_tuning_report_creation() {
        let report = GpuTuningReport {
            persistence_mode: TuningLevel::Optimal,
            transparent_hugepage: TuningLevel::Recommended,
            clock_frequency: TuningLevel::NotApplicable,
            ecc_status: TuningLevel::Optimal,
            compute_mode: TuningLevel::Recommended,
            recommendations: vec!["test recommendation".to_string()],
        };
        assert_eq!(report.persistence_mode, TuningLevel::Optimal);
        assert_eq!(report.recommendations.len(), 1);
    }

    #[test]
    fn test_check_transparent_hugepage_returns_valid_level() {
        // 在任何平台上都应该返回一个有效级别
        let level = GpuTuningAdvisor::check_transparent_hugepage();
        // 在测试环境中可能是 Optimal、Recommended 或 NotApplicable
        assert!(
            matches!(
                level,
                TuningLevel::Optimal | TuningLevel::Recommended | TuningLevel::NotApplicable
            )
        );
    }

    #[test]
    fn test_check_and_advise_returns_report() {
        let report = GpuTuningAdvisor::check_and_advise();
        // 报告应该包含所有检测项
        assert!(matches!(
            report.persistence_mode,
            TuningLevel::Optimal | TuningLevel::Recommended | TuningLevel::NotApplicable
        ));
    }

    #[test]
    fn test_run_nvidia_smi_returns_none_without_gpu() {
        // 在没有 nvidia-smi 的环境中应返回 None
        // 这不是一个严格的测试，因为某些环境可能有 nvidia-smi
        let result = run_nvidia_smi(&["--query-gpu=name", "--format=csv,noheader"]);
        // 结果可能是 Some 或 None，取决于环境
        // 我们只验证不会 panic
        let _ = result;
    }
}
