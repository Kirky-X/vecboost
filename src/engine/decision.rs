// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Laya 决策管线（文档 temp/laya-vecboost-feasibility.md §2.1/§4.2/§9.6）：
//! 预处理（`super::decision_protocol` 共享协议层）→ 5 张量 ONNX 推理
//! → 按题 softmax + per-cardinality 温度校准 → choice/score/noul 后处理。
//!
//! 本 mod 为 `pub(crate)`：对外唯一入口是
//! [`crate::engine::InferenceEngine::decide`]（经 `AnyEngine::Decision`）。
//! 序列构造 / collate / 校准 / 后处理的协议实现与测试见
//! [`super::decision_protocol`]（onnx 与 candle 两路共享，无 feature 门）。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::config::model::{DecisionParams, DeviceType, ModelConfig, ModelTask, Precision};
use crate::domain::{DecisionRequest, DecisionResponse};
use crate::error::VecboostError;
use ndarray::Array2;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::InferenceEngine;
use super::decision_protocol::{
    FixedMarkers, TemperatureCalibration, answer_for_question, build_question_row, collate_batch,
    encode_ids, qtype_code, state_text,
};

/// bundle 内模型文件探测顺序（任务协议：model.onnx → model_quantized.onnx
/// → laya.onnx → laya_int8.onnx；决策侧要求 fp32 主模型优先 + 显式枚举
/// laya 变体名。与 onnx_engine::resolve_local_bundle 的 embedding 口径
/// ——quantized 优先 + 唯一 `*.onnx` 扫描——分歧为任务协议约定，两侧
/// 均有测试钉，改动须同步评估另一侧）
const MODEL_CANDIDATES: [&str; 4] = [
    "model.onnx",
    "model_quantized.onnx",
    "laya.onnx",
    "laya_int8.onnx",
];

/// Laya 决策管线：bundle 自持（模型 + tokenizer + 可选校准表）+ ort Session。
pub(crate) struct DecisionPipeline {
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
    calibration: TemperatureCalibration,
    /// score/noul 固定 marker 的预编码 id（加载期一次，热路径复用）
    fixed_markers: FixedMarkers,
    /// per-checkpoint 序列预算（加载期定值，热路径逐题消费）
    params: DecisionParams,
    precision: Precision,
    /// 已探明的本地 bundle 目录（try_fallback_to_cpu 重建 CPU Session 用，
    /// 无网络依赖）
    bundle_dir: PathBuf,
    fallback_triggered: bool,
    fallback_lock: Arc<Mutex<()>>,
}

/// bundle 内模型文件探测（顺序见 [`MODEL_CANDIDATES`]）；
/// 全 miss 报 `ModelLoadError` 含尝试路径清单（onnx_engine.rs 同款显性化）
fn probe_model_file(bundle_dir: &Path) -> Result<PathBuf, VecboostError> {
    let mut tried = Vec::with_capacity(MODEL_CANDIDATES.len());
    for name in MODEL_CANDIDATES {
        let candidate = bundle_dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
        tried.push(candidate.display().to_string());
    }
    Err(VecboostError::ModelLoadError(format!(
        "No Laya ONNX model found in {}; tried: {}",
        bundle_dir.display(),
        tried.join(", ")
    )))
}

/// 探测命中的模型文件名 → 对外精度标签：`*_quantized`/`*_int8` 候选命中时
/// 报 Int8，其余 Fp32（supports_mixed_precision 恒 false 已诚实，仅加载期
/// 标签与探测结果一致，避免 quantized/int8 bundle 对外失真报 Fp32）
fn precision_for_model_file(model_file: &Path) -> Precision {
    let name = model_file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if name.contains("quantized") || name.contains("int8") {
        Precision::Int8
    } else {
        Precision::Fp32
    }
}

/// Session 构建：Level3 图优化 + intra threads + CUDA EP 分支
/// （onnx_engine.rs:128-159 同模式）
fn build_session(model_file: &Path, device: &DeviceType) -> Result<Session, VecboostError> {
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    log::info!("Initializing ONNX Runtime session for decision pipeline...");
    let builder = Session::builder()
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        .with_intra_threads(num_threads)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

    let mut session = if *device == DeviceType::Cuda {
        log::info!("Attempting to configure CUDA execution provider for decision pipeline");
        #[cfg(feature = "cuda")]
        {
            builder
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                ])
                .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            log::warn!(
                "CUDA execution provider requires the cuda feature flag; \
                 using CPU execution provider for decision pipeline"
            );
            builder
        }
    } else {
        builder
    };

    session
        .commit_from_file(model_file)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))
}

impl DecisionPipeline {
    /// 从 bundle 目录加载：模型按 [`MODEL_CANDIDATES`] 探测，tokenizer 经
    /// `local_bundle::resolve_tokenizer_path` 共享契约解析——
    /// `config.tokenizer_path` 显式路径（已配置但不存在时显性报错，不回落）
    /// → bundle 根 `tokenizer.json` → `tokenizer/tokenizer.json` 子目录；
    /// 皆缺报 `ModelLoadError` 含检查路径清单。校准表缺失 warn + 空表兜底；
    /// 损坏报 `ModelFileCorrupted`。
    ///
    /// # 完整性校验覆盖边界（威胁模型声明）
    /// `config.model_sha256` 仅校验探测命中的主模型文件。bundle 内
    /// tokenizer.json 与 laya_config.json 温度校准表**不受 sha256 校验**——
    /// 它们与本管线同目录读取，威胁模型将 bundle 目录视为可信本地资产；
    /// 能写 bundle 目录的攻击者无需替换模型即可经篡改校准温度（分布尖锐化/
    /// 操纵置信呈现）或词表（分词漂移）改变下游语义。部署上以目录权限而非
    /// 文件哈希作为该资产的边界；bundle 清单化校验待 config 契约扩展任务组。
    pub(crate) fn load(config: &ModelConfig) -> Result<Self, VecboostError> {
        // decision_params 理论恒经合法路径构造（预设表解析期校验 + 启动
        // fail-fast + switch_override 复检），但字段全 pub 且 derive
        // Deserialize，库内直构非法值可绕过 new——加载期防御性复检（与
        // switch_override 同一原则的第二处落点），先于资产探测显性拒绝，
        // 而非延迟到每个决策请求期才以 InvalidInput 暴露
        let params = match config.decision_params {
            Some(p) => DecisionParams::new(p.head_max_len, p.state_max_tokens)?,
            None => DecisionParams::default(),
        };
        let bundle_dir = config.model_path.clone();
        if !bundle_dir.is_dir() {
            return Err(VecboostError::ModelLoadError(format!(
                "decision bundle path is not a directory: {}",
                bundle_dir.display()
            )));
        }
        let model_file = probe_model_file(&bundle_dir)?;

        if let Some(ref expected_hash) = config.model_sha256 {
            log::info!("Verifying decision model file SHA256 hash...");
            let is_valid =
                crate::utils::hash::verify_sha256(&model_file, expected_hash).map_err(|e| {
                    VecboostError::ModelLoadError(format!("Failed to verify SHA256: {e}"))
                })?;
            if !is_valid {
                return Err(VecboostError::ModelLoadError(format!(
                    "Model file SHA256 verification failed. Expected: {expected_hash}, File: {:?}",
                    model_file
                )));
            }
        }

        let tokenizer_file = super::local_bundle::resolve_tokenizer_path(
            &bundle_dir,
            config.tokenizer_path.as_deref(),
        )?;
        // bundle tokenizer.json 常自带 truncation/padding 配置（如 MiniLM 的
        // fixed-128 padding）——决策协议自管特殊 token 拼接与 collate 填充，
        // 两者必须清除，否则序列被静默 pad/截断（协议漂移）
        let mut tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(|e| {
            VecboostError::ModelLoadError(format!(
                "Failed to load tokenizer {}: {e}",
                tokenizer_file.display()
            ))
        })?;
        tokenizer.with_truncation(None).map_err(|e| {
            VecboostError::ModelLoadError(format!("Failed to clear truncation: {e}"))
        })?;
        tokenizer.with_padding(None);

        let fixed_markers = FixedMarkers::new(&tokenizer)?;
        let session = build_session(&model_file, &config.device)?;
        let calibration = TemperatureCalibration::from_bundle(&bundle_dir)?;
        let precision = precision_for_model_file(&model_file);

        log::info!(
            "Decision pipeline initialized: bundle={}, model={:?}, precision={:?}, calibration_buckets={}",
            bundle_dir.display(),
            model_file,
            precision,
            calibration.temperatures.len()
        );

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            calibration,
            fixed_markers,
            params,
            precision,
            bundle_dir,
            fallback_triggered: false,
            fallback_lock: Arc::new(Mutex::new(())),
        })
    }

    /// 决策管线本体：校验 → 逐题预处理 → collate → 5 张量推理 →
    /// 按行 softmax+温度校准 → 三类后处理。
    ///
    /// logits **绝不经过 `l2_normalize_in_place`**——embedding 出口的
    /// 归一化契约不适用于决策 logits（docDrift：对 logits 归一化会让
    /// softmax 前的概率分布整体塌缩到单位球面，校准与概率语义全毁）。
    pub(crate) fn decision(
        &self,
        req: &DecisionRequest,
    ) -> Result<DecisionResponse, VecboostError> {
        let started = std::time::Instant::now();
        let logits_rows = self.decision_logits(req)?;
        let mut answers = Vec::with_capacity(logits_rows.len());
        for (b, question) in req.questions.iter().enumerate() {
            // 行长度 = 该题有效 marker 数（decision_logits 已做前缀切片）
            let temperature = self.calibration.temperature_for(logits_rows[b].len());
            answers.push(answer_for_question(question, &logits_rows[b], temperature)?);
        }
        let elapsed = started.elapsed();
        // 埋点：决策链路调用时延直方图（histogram _count 即调用计数）与
        // 批大小。collector 未设置时零开销跳过。
        // Stage 三值豁免口径：Stage 枚举固定 tokenize/inference/pool（指标
        // 标签稳定性），语义属 embedding 管线分阶段；决策管线不强行映射，
        // take_stage_snapshot 恒 None，观测走本处的独立 decision 指标
        #[cfg(feature = "http")]
        if let Some(collector) = crate::metrics::prometheus_exporter::global_collector() {
            collector.observe_decision_seconds(elapsed.as_secs_f64());
            collector.record_batch_size("decision", req.questions.len() as f64);
        }
        Ok(DecisionResponse {
            answers,
            processing_time_ms: elapsed.as_millis(),
        })
    }

    /// 校验 → 逐题预处理 → collate → 5 张量推理 → 形状校验 → 逐行有效
    /// marker 前缀的**未校准** logits（温度施加前，后处理共享上游）。决策
    /// 主链路（[`Self::decision`]）与对齐闸门（`decide_logits` trait 出口）
    /// 的共享实现；决策埋点仅在 decision()，诊断直调不污染指标。
    pub(crate) fn decision_logits(
        &self,
        req: &DecisionRequest,
    ) -> Result<Vec<Vec<f32>>, VecboostError> {
        // trait 契约：实现方须假定请求已过 validate 或自行调用——
        // 管线自行调用（crate 内直调场景同样被输入防线覆盖）
        req.validate()?;

        // 同一请求内 state 全题共享：只 encode 一次（32 题上界下消除 31 次
        // 重复 tokenize）。取前 256 token 由 build_question_row 内防御性切片
        // 完成——tokenizers 0.23.2 的 encode 无编码期截断（normalize→
        // pre_tokenize→tokenize 全量完成后 post_process 才丢弃尾部），编码期
        // 截断与后置切片逐 token 等价；64KB 上界的全量 tokenize 为毫秒级、
        // 相对单次推理非瓶颈，如需消除须字符级前缀粗剪（预剪点须对齐
        // pre-token 边界才保证等价），待真实负载数据立项后再做
        let state = state_text(&req.state);
        let state_ids = encode_ids(&self.tokenizer, &state)?;
        let mut rows = Vec::with_capacity(req.questions.len());
        let mut qtype_codes = Vec::with_capacity(req.questions.len());
        for question in &req.questions {
            rows.push(build_question_row(
                &self.tokenizer,
                question,
                &state_ids,
                &self.fixed_markers,
                self.params,
            )?);
            qtype_codes.push(qtype_code(&question.qtype));
        }
        let batch = collate_batch(&rows, &qtype_codes)?;

        let input_ids = Array2::from_shape_vec((batch.batch_size, batch.seq_len), batch.input_ids)
            .map_err(|e| VecboostError::InferenceError(format!("input_ids shape: {e}")))?;
        let attention_mask =
            Array2::from_shape_vec((batch.batch_size, batch.seq_len), batch.attention_mask)
                .map_err(|e| VecboostError::InferenceError(format!("attention_mask shape: {e}")))?;
        let marker_pos =
            Array2::from_shape_vec((batch.batch_size, batch.max_markers), batch.marker_pos)
                .map_err(|e| VecboostError::InferenceError(format!("marker_pos shape: {e}")))?;
        let marker_mask =
            Array2::from_shape_vec((batch.batch_size, batch.max_markers), batch.marker_mask)
                .map_err(|e| VecboostError::InferenceError(format!("marker_mask shape: {e}")))?;
        let qtype = ndarray::Array1::from(batch.qtype);

        // marker_mask dtype 为 bool（§2.2）。若与真实导出图不符，
        // P0 数值对照时改 i64 并在此留注释
        let logits = {
            let mut session_guard = self
                .session
                .lock()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
            let outputs = session_guard
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array(input_ids.into_dyn())
                        .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                    "attention_mask" => Tensor::from_array(attention_mask.into_dyn())
                        .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                    "marker_pos" => Tensor::from_array(marker_pos.into_dyn())
                        .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                    "marker_mask" => Tensor::from_array(marker_mask.into_dyn())
                        .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                    "qtype" => Tensor::from_array(qtype.into_dyn())
                        .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                ])
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
            // get 而非 Index：Index 缺名时 panic（ort output.rs:189-192），
            // 与全文件显性失败口径不符——缺名属 bundle 模型资产与协议不符，
            // 显性报错并列出实际输出名清单
            let logits_value = outputs.get("logits").ok_or_else(|| {
                let names: Vec<&str> = outputs.iter().map(|(k, _)| k).collect();
                VecboostError::InferenceError(format!(
                    "decision model has no `logits` output; actual outputs: {names:?}"
                ))
            })?;
            logits_value
                .try_extract_array::<f32>()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?
                .to_owned()
        };

        let logits_shape = logits.shape().to_vec();
        // §2.2 dense gather 协议：logits 宽度恰为 [B, N]（N=batch 内最大
        // marker 数）。等值校验一次到位——宽度异常偏大（如误导出的
        // [B, seq_len] 图）在此显性报错，而非静默取前 N 个 logit 错答
        if logits.ndim() != 2
            || logits_shape[0] != batch.batch_size
            || logits_shape[1] != batch.max_markers
        {
            return Err(VecboostError::InferenceError(format!(
                "unexpected logits shape {logits_shape:?}, expected [{}, {}]",
                batch.batch_size, batch.max_markers
            )));
        }
        let logits_2d = logits
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| VecboostError::InferenceError(format!("logits dims: {e}")))?;

        // [B,N] 行主序 C-contiguous：行前缀切片必连续，零分配借用；
        // 非连续属数组构造异常，显性报错而非 panic。pad 位（onnx 图输出值
        // 任意且无语义）不进结果——逐 marker 对齐只看有效位
        let logits_rows = req
            .questions
            .iter()
            .enumerate()
            .map(|(b, _)| {
                let marker_count = rows[b].marker_pos.len();
                let row = logits_2d.row(b);
                let row_view = row.slice(ndarray::s![..marker_count]);
                let row_logits = row_view.to_slice().ok_or_else(|| {
                    VecboostError::InferenceError("logits row slice is not contiguous".to_string())
                })?;
                Ok(row_logits.to_vec())
            })
            .collect::<Result<Vec<_>, VecboostError>>()?;
        Ok(logits_rows)
    }
}

#[async_trait::async_trait]
impl InferenceEngine for DecisionPipeline {
    /// 决策主链路唯一 trait 入口：委托固有方法 [`DecisionPipeline::decision`]。
    /// 漏覆盖则继承 trait 默认 `Err(UnsupportedTask)`——`/api/1/decisions`
    /// 恒 400 且服务层经 trait 调用走不到管线（全部离线测试盲区）。
    /// 固有方法名为 `decision()`，与 trait 方法无同名遮蔽，方法解析自然进管线。
    fn decide(&self, req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
        self.decision(req)
    }

    /// 对齐/诊断出口：委托 [`DecisionPipeline::decision_logits`]
    fn decide_logits(&self, req: &DecisionRequest) -> Result<Vec<Vec<f32>>, VecboostError> {
        self.decision_logits(req)
    }

    /// 决策引擎不产向量（诚实语义）：task=decision 时 embedding/rerank
    /// 端点以 UnsupportedTask 400 显性拒绝
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        Err(VecboostError::unsupported_task(
            "decision pipeline does not produce embeddings".to_string(),
        ))
    }

    fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Err(VecboostError::unsupported_task(
            "decision pipeline does not produce embeddings".to_string(),
        ))
    }

    fn precision(&self) -> &Precision {
        &self.precision
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    /// 默认实现依赖 `embed`（bi-encoder rerank）——决策引擎不能谎报
    fn supports_rerank(&self) -> bool {
        false
    }

    fn is_fallback_triggered(&self) -> bool {
        self.fallback_triggered
    }

    fn supports_task(&self, task: ModelTask) -> bool {
        task == ModelTask::Decision
    }

    /// 用已探明的本地 bundle 路径重建 CPU Session（onnx_engine.rs:419-471
    /// 同模式，无 HF 网络依赖）
    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        let _lock = self.fallback_lock.lock().map_err(|e| {
            VecboostError::InferenceError(format!("Failed to acquire fallback lock: {e}"))
        })?;
        // 双重检查：获取锁后再次确认未降级
        if self.fallback_triggered {
            return Ok(());
        }
        log::info!("Attempting fallback to CPU for decision pipeline");
        let model_file = probe_model_file(&self.bundle_dir)?;
        let session = build_session(&model_file, &DeviceType::Cpu)?;
        let mut session_guard = self
            .session
            .lock()
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
        *session_guard = session;
        drop(session_guard);
        // 与加载路径单一事实源：降级复用同一 bundle（可能为 quantized/int8 图），
        // 标签按探测文件名映射，不得硬编码 Fp32 失真（onnx_engine 降级真下载
        // Fp32 model.onnx，其硬编码在那边语义成立，此处不同）
        self.precision = precision_for_model_file(&model_file);
        // 置位于 session 替换成功之后：probe/build 失败时状态位保持 false，
        // 后续 OOM 降级尝试不会被双重检查恒短路掩盖
        self.fallback_triggered = true;
        log::info!("Successfully fell back to CPU for decision pipeline");
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 测试：管线加载/分发面（协议层测试在 decision_protocol.rs）。
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 防回归钉①：decide 必须作为 trait 方法对 DecisionPipeline 可解析 ──
    //
    // 如实声明局限：fn 引用无法区分默认实现与覆盖——「漏覆盖 → 继承默认
    // Err(UnsupportedTask)」由 bundle 守卫测试（运行级）与 P0 example
    // （在线级）兜住，本钉只防 trait 接入整体缺失（编译级）。

    #[test]
    fn test_compile_pin_decide_is_trait_method_on_pipeline() {
        let _: fn(&DecisionPipeline, &DecisionRequest) -> Result<DecisionResponse, VecboostError> =
            <DecisionPipeline as InferenceEngine>::decide;
    }

    // ── 防回归钉②（运行级，bundle 就绪守卫，离线 SKIP）：
    // DecisionPipeline 经 &dyn InferenceEngine 调 decide 必须走真实管线
    // （钉死「漏覆盖 decide → 默认 UnsupportedTask」的分发回归）。──

    #[test]
    fn test_bundle_guard_decide_via_trait_object() {
        let bundle = Path::new("models/laya");
        if !bundle.is_dir() {
            eprintln!(
                "Skipping test: laya bundle not found at {:?}（真 bundle 环境下本测试钉死 DecisionPipeline 级分发链）",
                bundle
            );
            return;
        }
        // ort/load-dynamic 缺动态库时 Session::builder 触发 non-unwinding
        // panic（SIGABRT 拖垮整个测试进程）——守卫必须与 bundle 探测并列，
        // 口径同 tests/grpc_e2e 的 decision_assets_ready / tests/common 的
        // ensure_ort_env（ORT_DYLIB_PATH 显式优先，缺省回填仓库默认落位）
        let dylib = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| "3rdparty/onnxruntime/libonnxruntime.so".to_string());
        if !Path::new(&dylib).is_file() {
            eprintln!(
                "Skipping test: onnxruntime dylib not found at {dylib:?} \
                 （设置 ORT_DYLIB_PATH 或落位默认路径后本测试钉死 DecisionPipeline 级分发链）"
            );
            return;
        }
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            // SAFETY: 单线程测试环境回填一次，无并发读者
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &dylib);
            }
        }
        let mut config = test_config();
        config.model_path = bundle.to_path_buf();
        let pipeline = DecisionPipeline::load(&config).expect("bundle 就绪时必须可加载");
        let engine: &dyn InferenceEngine = &pipeline;
        // task=decision 时 embedding 端点必须显性 UnsupportedTask（诚实语义）
        match engine.embed("text") {
            Err(VecboostError::UnsupportedTask(_)) => {}
            other => panic!("决策引擎 embed 必须显性 UnsupportedTask，got {other:?}"),
        }
        assert!(
            engine.supports_task(crate::config::model::ModelTask::Decision),
            "决策引擎必须自报支持 Decision"
        );
        assert!(
            !engine.supports_rerank(),
            "决策引擎不得谎报 rerank 能力（默认实现依赖 embed）"
        );
        let req: DecisionRequest = serde_json::from_str(
            r#"{"state":"billed twice","questions":[
                {"name":"department","qtype":"choice","instructions":"which?","options":["billing","other"]},
                {"name":"urgency","qtype":"score","instructions":"how urgent?"},
                {"name":"churn_risk","qtype":"noul","instructions":"churning?"}
            ]}"#,
        )
        .expect("guard request");
        match engine.decide(&req) {
            Ok(resp) => assert_eq!(resp.answers.len(), 3, "三题型各一答案"),
            Err(e) => panic!("bundle 就绪时 decide 必须进真实管线而非报错：{e:?}"),
        }
    }

    // ── 加载类：bundle 探测失败显性 ModelLoadError（含尝试路径清单）──

    #[test]
    fn test_decision_pipeline_load_empty_dir_reports_tried_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = test_config();
        config.model_path = dir.path().to_path_buf();
        match DecisionPipeline::load(&config) {
            Err(VecboostError::ModelLoadError(msg)) => {
                assert!(
                    msg.contains("model.onnx") && msg.contains("laya.onnx"),
                    "错误必须列出尝试的模型路径清单，msg={msg}"
                );
                // 顺序钉 MODEL_CANDIDATES 探测优先级（fp32 主模型优先的任务
                // 协议约定）：tried 清单按候选顺序生成，长度钉挡不住乱序
                // 回归；与 onnx_engine::resolve_local_bundle 的 embedding 侧
                // 优先级钉配对（两侧注释宣称的「均有测试钉」以此为准）
                let positions = [
                    msg.find("model.onnx"),
                    msg.find("model_quantized.onnx"),
                    msg.find("laya.onnx"),
                    msg.find("laya_int8.onnx"),
                ];
                assert!(
                    positions.iter().all(|p| p.is_some())
                        && positions.windows(2).all(|w| w[0] < w[1]),
                    "tried 清单必须按 MODEL_CANDIDATES 顺序排列，msg={msg}"
                );
            }
            Err(other) => panic!("期望 ModelLoadError，got {other:?}"),
            Ok(_) => panic!("空 bundle 必须加载失败"),
        }
    }

    #[test]
    fn test_decision_pipeline_load_model_without_tokenizer_lists_tokenizer_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("model.onnx"), b"fake onnx").expect("write model");
        let mut config = test_config();
        config.model_path = dir.path().to_path_buf();
        match DecisionPipeline::load(&config) {
            Err(VecboostError::ModelLoadError(msg)) => {
                assert!(
                    msg.contains("tokenizer.json"),
                    "错误必须列出尝试的 tokenizer 路径清单，msg={msg}"
                );
                assert!(
                    msg.contains("tokenizer") && msg.contains("tokenizer.json"),
                    "清单必须含 tokenizer/ 子目录候选，msg={msg}"
                );
            }
            Err(other) => panic!("期望 ModelLoadError，got {other:?}"),
            // 正常情况在 fake onnx 的 session 构建前即失败（探测先于 Session::builder，
            // 不触发 ort 环境崩溃问题）
            Ok(_) => panic!("缺 tokenizer 必须加载失败"),
        }
    }

    // tokenizer_path 已配置但路径不存在时必须显性报错——即使 bundle 根目录
    // 有 tokenizer.json 也不得静默回落（静默换用其他 tokenizer 产生分词漂移，
    // 与 onnx_engine::resolve_local_bundle 同一契约，探测先于 Session 构建）
    #[test]
    fn test_decision_pipeline_load_bad_explicit_tokenizer_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("model.onnx"), b"fake onnx").expect("write model");
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").expect("write root tokenizer");
        let mut config = test_config();
        config.model_path = dir.path().to_path_buf();
        config.tokenizer_path = Some(dir.path().join("nowhere/tok.json"));
        match DecisionPipeline::load(&config) {
            Err(VecboostError::ModelLoadError(msg)) => {
                assert!(
                    msg.contains("nowhere"),
                    "错误必须点名配置的 override 路径，msg={msg}"
                );
                assert!(
                    msg.contains("tokenizer_path"),
                    "错误必须指向 [model].tokenizer_path 配置面，msg={msg}"
                );
            }
            Err(other) => panic!("期望 ModelLoadError，got {other:?}"),
            Ok(_) => panic!("坏 explicit tokenizer 路径不得静默回落加载成功"),
        }
    }

    // 直构非法 decision_params 必须在加载期被拒绝（先于资产探测）
    #[test]
    fn test_pipeline_load_rejects_invalid_decision_params_before_asset_probe() {
        // 字段全 pub + derive Deserialize：库内直构非法值可绕过
        // DecisionParams::new——load 处防御性复检（与 switch_override 同一
        // 原则），加载期显性拒绝且先于资产探测（model_path 不存在也先报
        // 参数错，本测试无需 bundle 资产）
        let config = ModelConfig {
            name: "laya-bad-params".to_string(),
            engine_type: crate::config::model::EngineType::Candle,
            model_path: PathBuf::from("__definitely_missing_bundle__"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            task: ModelTask::Decision,
            quantized: false,
            decision_params: Some(DecisionParams {
                head_max_len: 0,
                state_max_tokens: 256,
            }),
        };
        let err = match DecisionPipeline::load(&config) {
            Err(e) => e,
            Ok(_) => panic!("直构非法 decision_params 必须在加载期被拒绝"),
        };
        assert!(
            matches!(err, VecboostError::ConfigError(_)),
            "直构非法参数必须在加载期显性拒绝（ConfigError），got {err:?}"
        );
        assert!(
            err.error_detail().contains("head_max_len"),
            "错误必须点名字段，detail={}",
            err.error_detail()
        );
    }

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "test-decision".to_string(),
            engine_type: crate::config::model::EngineType::Onnx,
            model_path: PathBuf::from("/nonexistent/laya"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            task: crate::config::model::ModelTask::Decision,
            quantized: false,
            decision_params: None,
        }
    }

    // ── precision 标签与探测文件一致 ──

    #[test]
    fn test_precision_for_model_file_names() {
        assert_eq!(
            precision_for_model_file(Path::new("bundle/model.onnx")),
            Precision::Fp32
        );
        assert_eq!(
            precision_for_model_file(Path::new("bundle/laya.onnx")),
            Precision::Fp32
        );
        assert_eq!(
            precision_for_model_file(Path::new("bundle/model_quantized.onnx")),
            Precision::Int8
        );
        assert_eq!(
            precision_for_model_file(Path::new("bundle/laya_int8.onnx")),
            Precision::Int8
        );
    }

    #[test]
    fn test_marker_constants_consistent() {
        assert_eq!(MODEL_CANDIDATES.len(), 4);
    }
}
