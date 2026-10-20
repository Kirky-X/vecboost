// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Detect 服务实现：`/api/v1/detect` 端点与翻译 auto 检测共用的服务编排层。

use crate::domain::{DetectRequest, DetectResponse};
use crate::error::VecboostError;
use std::time::Instant;

/// 语言检测服务：无状态、无引擎字段。
///
/// 检测为 whatlang n-gram 统计（µs 级纯 CPU），无模型推理无 Session 锁：
/// 不进 ModelManager、不设 ModelTask 变体（rerank「能力而非任务」同款
/// 先例）；与 `/api/v1/translate` 的 auto 检测共用底层
/// [`crate::text::langdetect::detect_language`]（单一事实源，禁止第二份实现）。
#[derive(Debug, Clone, Copy, Default)]
pub struct DetectService;

impl DetectService {
    pub fn new() -> Self {
        Self
    }

    /// 语言检测：校验链（空文本/超长/allowlist 非法码，i18n 文案
    /// InvalidInput 语义）→ 域层 validate（控制字符防线与格式兜底，
    /// ValidationError 单一实现）→ 检测核心。
    ///
    /// # 不走 spawn_blocking 的差异依据（与 decide 对比）
    /// decide 是数十至数百毫秒的阻塞模型推理，须移出 tokio worker 调度面；
    /// 本方法为 whatlang n-gram 统计且无 await 点——10k 字符上限文本实测
    /// p50=93µs / p99=201µs（评审 PERF-6 基准，release 构建、200 样本），
    /// 亚毫秒级同步完成即返回，spawn_blocking 的任务切换与 blocking 池
    /// 占用反而放大延迟。
    ///
    /// 不可判定（空/纯数字/纯符号）返回 200 语义的
    /// `language: null + is_reliable: false`——显性表达，禁止静默成功或
    /// 虚构语言。
    ///
    /// 热路径计数挂点在校验受理之后、检测之前（400 拒绝不计；
    /// global_collector 为 None 时零开销跳过——library 模式/单测环境），
    /// 计数器本体随 i18n/metrics 任务接线。
    pub async fn process_detect(
        &self,
        req: DetectRequest,
    ) -> Result<DetectResponse, VecboostError> {
        Self::validate_request(&req)?;

        // 热路径埋点：请求计数（校验受理后计数，400 拒绝不计；全局
        // collector 未设置时零开销跳过——library 模式/单测环境）
        #[cfg(feature = "http")]
        if let Some(collector) = crate::metrics::prometheus_exporter::global_collector() {
            collector.inc_detect_requests();
        }

        let start = Instant::now();
        let detection = crate::text::detect_language(&req.text, req.allow.as_deref());
        // 不可判定（text 层 language 为空串信号）→ 200 语义的 null 字段
        let undetectable = detection.language.is_empty();
        Ok(DetectResponse {
            language: (!undetectable).then_some(detection.language),
            script: (!undetectable).then_some(detection.script),
            confidence: detection.confidence,
            is_reliable: detection.is_reliable,
            processing_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// 检测请求前置校验的单一实现（直连路径共用）：空文本/超长/allowlist
    /// 非法码以 i18n 文案 InvalidInput 显性拒绝，随后域层 validate 兜底
    /// （控制字符/格式，ValidationError）。
    pub fn validate_request(req: &DetectRequest) -> Result<(), VecboostError> {
        if req.text.is_empty() {
            return Err(VecboostError::InvalidInput(
                crate::i18n::tr("detect-empty-text").to_string(),
            ));
        }
        let text_chars = req.text.chars().count();
        if text_chars > crate::utils::constants::MAX_TEXT_LENGTH {
            return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
                "detect-too-long",
                crate::i18n::tr_args(&[
                    ("length", &text_chars.to_string()),
                    ("max", &crate::utils::constants::MAX_TEXT_LENGTH.to_string()),
                ]),
            )));
        }
        if let Some(allow) = &req.allow {
            for code in allow {
                // 双重校验（评审）：格式 + 发布集合成员——"sw"/"xx" 等
                // 格式合法码不在 whatlang 发布集，穿透会触发收窄失效
                if !crate::utils::validator::input::is_valid_language_code(code)
                    || !crate::text::is_published_language_code(code)
                {
                    return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
                        "detect-invalid-allowlist",
                        crate::i18n::tr_args(&[(
                            "code",
                            &crate::utils::validator::input::echo(code),
                        )]),
                    )));
                }
            }
        }
        req.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_process_detect_returns_language_script_confidence() {
        crate::i18n::init();
        let service = DetectService::new();
        let resp = service
            .process_detect(DetectRequest {
                text:
                    "The quick brown fox jumps over the lazy dog near the riverbank every morning."
                        .to_string(),
                allow: None,
            })
            .await
            .unwrap();
        assert_eq!(resp.language.as_deref(), Some("en"));
        assert_eq!(resp.script.as_deref(), Some("Latin"));
        assert!((0.0..=1.0).contains(&resp.confidence));
        assert!(resp.is_reliable);
    }

    #[tokio::test]
    async fn test_process_detect_rejects_empty_text_as_invalid_input() {
        crate::i18n::init();
        let service = DetectService::new();
        let err = service
            .process_detect(DetectRequest {
                text: String::new(),
                allow: None,
            })
            .await
            .unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "空文本必须 InvalidInput（i18n 文案语义在服务层），got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_process_detect_undetectable_returns_null_language_200_semantics() {
        crate::i18n::init();
        let service = DetectService::new();
        // 纯数字：不可判定必须以 Ok（200）+ language:null + is_reliable:false
        // 显性表达，断言非 Err（禁止把不可判定伪装成 4xx/5xx）
        let resp = service
            .process_detect(DetectRequest {
                text: "1234567890".to_string(),
                allow: None,
            })
            .await
            .unwrap();
        assert!(resp.language.is_none(), "纯数字不得虚构语言");
        assert!(resp.script.is_none());
        assert!(!resp.is_reliable);
        assert_eq!(resp.confidence, 0.0);
    }

    #[tokio::test]
    async fn test_process_detect_rejects_invalid_allowlist_as_invalid_input() {
        crate::i18n::init();
        let service = DetectService::new();
        let err = service
            .process_detect(DetectRequest {
                text: "hello world".to_string(),
                allow: Some(vec!["xx1".to_string()]),
            })
            .await
            .unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "allowlist 非法码走服务层 i18n 语义（先于域层 ValidationError），got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_process_detect_rejects_unpublished_allow_code_400_not_panic() {
        // 评审 回归钉："sw"（格式合法、未发布）必须 400
        // 显性拒绝，禁止穿透到检测核心 panic（panic 在 async handler 内即
        // 连接重置）
        crate::i18n::init();
        let service = DetectService::new();
        let err = service
            .process_detect(DetectRequest {
                text: "hello world".to_string(),
                allow: Some(vec!["sw".to_string(), "xx".to_string()]),
            })
            .await
            .unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "未发布码必须 400 InvalidInput，got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_process_detect_detects_with_allowlist() {
        crate::i18n::init();
        let service = DetectService::new();
        let resp = service
            .process_detect(DetectRequest {
                text: "今天天气非常好，我们一起去公园散步，欣赏美丽的风景。".to_string(),
                allow: Some(vec!["zh".to_string(), "en".to_string()]),
            })
            .await
            .unwrap();
        assert_eq!(resp.language.as_deref(), Some("zh"));
        assert!(resp.is_reliable);
    }

    #[test]
    fn test_validate_request_accepts_clean_request() {
        crate::i18n::init();
        DetectService::validate_request(&DetectRequest {
            text: "hello world".to_string(),
            allow: Some(vec!["zh".to_string()]),
        })
        .unwrap();
    }

    #[test]
    fn test_validate_request_rejects_too_long_text() {
        crate::i18n::init();
        let err = DetectService::validate_request(&DetectRequest {
            text: "水".repeat(crate::utils::constants::MAX_TEXT_LENGTH + 1),
            allow: None,
        })
        .unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "超长走服务层 i18n 语义（detect-too-long），got {err:?}"
        );
    }
}
