// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 硬件感知启动规划（port 自 colibri `resource_plan.py`）。
//!
//! 纯函数设计：探测层结果以 [`Probes`] 注入，`plan()` 输出 [`HardwarePlan`]。
//! 规则保守默认；探测值为 None 的项不参与规则（不猜测）；全部探测失败时
//! 返回保守默认值，绝不阻塞启动。

/// 硬件探测结果（None = 探测失败，不参与规则）。
#[derive(Debug, Clone, Default)]
pub struct Probes {
    /// 可用内存（MB）。
    pub avail_ram_mb: Option<u64>,
    /// 物理核数（thread_tune 检测）。
    pub physical_cores: Option<usize>,
    /// 逻辑核数（num_cpus）。
    pub logical_cores: Option<usize>,
    /// GPU 是否存在。
    pub gpu_present: Option<bool>,
    /// 模型常驻大小（MB，模型目录扫描）。
    pub model_resident_mb: Option<u64>,
}

/// 瓶颈分类（仅日志化，不阻塞启动）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    Compute,
    Memory,
    Io,
    Mixed,
}

impl std::fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bottleneck::Compute => write!(f, "Compute"),
            Bottleneck::Memory => write!(f, "Memory"),
            Bottleneck::Io => write!(f, "Io"),
            Bottleneck::Mixed => write!(f, "Mixed"),
        }
    }
}

/// 启动规划输出。
#[derive(Debug, Clone)]
pub struct HardwarePlan {
    pub worker_threads: usize,
    pub max_batch_size: usize,
    pub batch_wait_ms: u64,
    pub quantized_recommended: bool,
    pub bottleneck: Bottleneck,
    pub rationale: String,
}

/// 默认批等待窗口（ms，与 WorkerConfig 默认一致）。
pub const DEFAULT_BATCH_WAIT_MS: u64 = 5;
/// 默认最大批大小。
pub const DEFAULT_MAX_BATCH_SIZE: usize = 8;

/// 由探测结果计算启动规划。
pub fn plan(p: &Probes) -> HardwarePlan {
    let mut rationale = Vec::new();

    // worker 线程：物理核 > 逻辑核 > num_cpus 回退（回退非"猜测探针"，仅默认值）。
    let cores = p.physical_cores.or(p.logical_cores);
    let worker_threads = match cores {
        Some(c) if c > 0 => {
            rationale.push(format!("worker_threads=物理/逻辑核数 {}", c));
            c
        }
        _ => {
            let fb = num_cpus::get().max(1);
            rationale.push(format!("worker_threads=回退 num_cpus {}", fb));
            fb
        }
    };

    // 内存规则：可用内存 < 2×模型常驻 → 推荐量化（边界：恰好 2× 不推荐）。
    let mem_pressure = match (p.avail_ram_mb, p.model_resident_mb) {
        (Some(avail), Some(model)) if model > 0 => {
            let pressured = avail < model.saturating_mul(2);
            rationale.push(format!(
                "内存规则：可用 {}MB vs 模型 {}MB（2×={}MB）→ {}",
                avail,
                model,
                model.saturating_mul(2),
                if pressured {
                    "quantized 推荐"
                } else {
                    "内存充足"
                }
            ));
            Some(pressured)
        }
        _ => {
            rationale.push("内存规则：探测缺失，不参与".to_string());
            None
        }
    };
    let quantized_recommended = mem_pressure.unwrap_or(false);

    // 核数规则：物理核 ≥16 → max_batch_size=32（边界含 16），否则维持默认。
    let max_batch_size = match p.physical_cores {
        Some(c) if c >= 16 => {
            rationale.push(format!("核数规则：物理核 {}≥16 → max_batch_size=32", c));
            32
        }
        Some(c) => {
            rationale.push(format!(
                "核数规则：物理核 {}<16 → max_batch_size 维持 {}",
                c, DEFAULT_MAX_BATCH_SIZE
            ));
            DEFAULT_MAX_BATCH_SIZE
        }
        None => {
            rationale.push("核数规则：物理核未知 → max_batch_size 维持默认".to_string());
            DEFAULT_MAX_BATCH_SIZE
        }
    };

    // 瓶颈分类（仅日志化）。
    let few_cores = cores.map(|c| c <= 4).unwrap_or(false);
    let big_model = p.model_resident_mb.map(|m| m >= 1024).unwrap_or(false);
    let bottleneck = match (mem_pressure, few_cores, big_model) {
        (Some(true), true, _) => Bottleneck::Mixed,
        (Some(true), _, _) => Bottleneck::Memory,
        (_, true, _) => Bottleneck::Compute,
        (_, _, true) => Bottleneck::Io,
        _ => Bottleneck::Compute,
    };
    rationale.push(format!("瓶颈分类：{}", bottleneck));

    HardwarePlan {
        worker_threads,
        max_batch_size,
        batch_wait_ms: DEFAULT_BATCH_WAIT_MS,
        quantized_recommended,
        bottleneck,
        rationale: rationale.join("; "),
    }
}

/// 显式配置掩码：标记用户显式设置过的字段（显式值优先于计划）。
#[derive(Debug, Clone, Default)]
pub struct PlanOverride {
    pub worker_threads: Option<usize>,
    pub max_batch_size: Option<usize>,
    pub batch_wait_ms: Option<u64>,
}

/// 应用计划：仅填充未显式配置的字段。
pub fn apply_plan(
    worker_threads: &mut usize,
    max_batch_size: &mut usize,
    batch_wait_ms: &mut u64,
    plan: &HardwarePlan,
    explicit: &PlanOverride,
) {
    if let Some(v) = explicit.worker_threads {
        *worker_threads = v;
    } else {
        *worker_threads = plan.worker_threads;
    }
    if let Some(v) = explicit.max_batch_size {
        *max_batch_size = v;
    } else {
        *max_batch_size = plan.max_batch_size;
    }
    if let Some(v) = explicit.batch_wait_ms {
        *batch_wait_ms = v;
    } else {
        *batch_wait_ms = plan.batch_wait_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_memory_boundary() {
        // 恰好 2× → 不推荐；低 1MB → 推荐。
        let exact = plan(&Probes {
            avail_ram_mb: Some(2000),
            model_resident_mb: Some(1000),
            ..Default::default()
        });
        assert!(!exact.quantized_recommended, "恰好 2× 不应推荐量化");
        let below = plan(&Probes {
            avail_ram_mb: Some(1999),
            model_resident_mb: Some(1000),
            ..Default::default()
        });
        assert!(below.quantized_recommended, "低于 2× 应推荐量化");
    }

    #[test]
    fn test_plan_core_boundary() {
        let at16 = plan(&Probes {
            physical_cores: Some(16),
            ..Default::default()
        });
        assert_eq!(at16.max_batch_size, 32, "核数恰好 16 应提至 32");
        let at15 = plan(&Probes {
            physical_cores: Some(15),
            ..Default::default()
        });
        assert_eq!(at15.max_batch_size, DEFAULT_MAX_BATCH_SIZE);
    }

    #[test]
    fn test_plan_all_none_conservative() {
        let p = plan(&Probes::default());
        assert!(!p.quantized_recommended);
        assert_eq!(p.max_batch_size, DEFAULT_MAX_BATCH_SIZE);
        assert_eq!(p.batch_wait_ms, DEFAULT_BATCH_WAIT_MS);
        assert!(p.worker_threads >= 1, "全 None 也必须给出可用计划");
        assert!(!p.rationale.is_empty());
    }

    #[test]
    fn test_plan_none_probes_ignored() {
        // 仅给核数：内存规则不参与 → 不推荐量化。
        let p = plan(&Probes {
            physical_cores: Some(32),
            ..Default::default()
        });
        assert!(!p.quantized_recommended);
        assert_eq!(p.max_batch_size, 32);
    }

    #[test]
    fn test_apply_plan_explicit_wins() {
        let plan_out = plan(&Probes {
            physical_cores: Some(16),
            ..Default::default()
        });
        let (mut wt, mut bs, mut bw) = (0usize, 0usize, 0u64);
        // 显式 batch_wait_ms 不被计划覆盖。
        apply_plan(
            &mut wt,
            &mut bs,
            &mut bw,
            &plan_out,
            &PlanOverride {
                batch_wait_ms: Some(0),
                ..Default::default()
            },
        );
        assert_eq!(bw, 0, "显式 batch_wait_ms=0（kill-switch）不得被覆盖");
        assert_eq!(bs, 32, "未显式字段取计划值");
        assert_eq!(wt, plan_out.worker_threads);
    }

    #[test]
    fn test_bottleneck_classification() {
        let mem = plan(&Probes {
            avail_ram_mb: Some(1000),
            model_resident_mb: Some(1000),
            physical_cores: Some(32),
            ..Default::default()
        });
        assert_eq!(mem.bottleneck, Bottleneck::Memory);
        let mixed = plan(&Probes {
            avail_ram_mb: Some(1000),
            model_resident_mb: Some(1000),
            physical_cores: Some(2),
            ..Default::default()
        });
        assert_eq!(mixed.bottleneck, Bottleneck::Mixed);
    }
}
