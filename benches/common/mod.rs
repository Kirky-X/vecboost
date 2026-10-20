// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! bench 目标共享的执行设备选择器。
//!
//! 目录约定：Cargo 只把 `benches/*.rs` 顶层文件当作 bench target（子目录不会被
//! 自动发现），故本模块以 `#[path = "common/mod.rs"] mod common;` 由各 bench 显式
//! 引入——与 `tests/common/mod.rs` 同一惯例。
//!
//! 设备经 `VECBOOST_BENCH_DEVICE` 选择（`cpu`（缺省）/ `cuda`），未知值显性退出：
//! 静默回退会把 CPU 数字记到 GPU 档位名下，与 `VECBOOST_BENCH_ENGINE` 的
//! 「拒绝静默回退」口径同源。

use vecboost::config::model::DeviceType;

/// `VECBOOST_BENCH_DEVICE` → 执行设备。缺省 `Cpu`（与历史 bench 行为逐字一致）。
///
/// `cuda` 的编译前提**按引擎分流**（两者的 CUDA 能力来自不同 feature，混用会把 CPU
/// 数字记成 GPU）：
/// - ort 引擎：需 `transcribe`（`ort/cuda` 仅经该特性引入），否则 `build_session`
///   只 warn 后落 CPU EP；
/// - candle 引擎：需 `candle-cuda`（`candle_core::utils::cuda_is_available` 在无该
///   feature 时恒假）。
///
/// `candle_engine` 由调用方从已选引擎派生（`matches!(engine_type, EngineType::Candle)`）——
/// 不直接传 `EngineType`，因为 `Onnx` 变体本身受 `onnx` feature 门控。
pub fn bench_device(bench_name: &str, candle_engine: bool) -> DeviceType {
    match std::env::var("VECBOOST_BENCH_DEVICE").as_deref() {
        Ok("cuda") => {
            if candle_engine {
                #[cfg(feature = "candle-cuda")]
                {
                    DeviceType::Cuda
                }
                #[cfg(not(feature = "candle-cuda"))]
                {
                    eprintln!(
                        "{bench_name}: 引擎=candle 的 device=cuda 需以 --features candle-cuda \
                         构建（否则 cuda_is_available() 恒假、引擎会落 CPU），拒绝执行"
                    );
                    std::process::exit(1);
                }
            } else {
                #[cfg(feature = "transcribe")]
                {
                    DeviceType::Cuda
                }
                #[cfg(not(feature = "transcribe"))]
                {
                    eprintln!(
                        "{bench_name}: 引擎=ort 的 device=cuda 需以 --features transcribe \
                         构建（该特性引入 ort/cuda，否则 CUDA 执行提供器不可用并静默回落 \
                         CPU），拒绝执行"
                    );
                    std::process::exit(1);
                }
            }
        }
        // 未设置与 "cpu" → Cpu（缺省臂与 main.rs 的未配置口径一致）
        Ok(other) if other != "cpu" => {
            eprintln!(
                "{bench_name}: 未知 VECBOOST_BENCH_DEVICE={other}（可选值: cpu/cuda），拒绝执行"
            );
            std::process::exit(1);
        }
        _ => DeviceType::Cpu,
    }
}
