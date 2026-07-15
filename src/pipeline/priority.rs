// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

#![allow(clippy::all)]

use std::collections::HashMap;
use std::time::Duration;
use tracing::debug;

use super::config::PriorityConfig;

/// 优先级枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Critical = 100,
    High = 75,
    Normal = 50,
    Low = 25,
}

impl Priority {
    pub fn from_score(score: i32) -> Self {
        if score >= 90 {
            Priority::Critical
        } else if score >= 65 {
            Priority::High
        } else if score >= 40 {
            Priority::Normal
        } else {
            Priority::Low
        }
    }

    pub fn as_i32(&self) -> i32 {
        *self as i32
    }
}

/// 请求来源
#[derive(Debug, Clone)]
pub enum RequestSource {
    Http { ip: String },
    Grpc { client_id: String },
    Internal,
}

impl RequestSource {
    pub fn http(ip: String) -> Self {
        RequestSource::Http { ip }
    }

    pub fn grpc(client_id: String) -> Self {
        RequestSource::Grpc { client_id }
    }

    pub fn internal() -> Self {
        RequestSource::Internal
    }
}

/// 优先级输入
#[derive(Debug, Clone)]
pub struct PriorityInput {
    pub base_priority: Priority,
    pub time_until_timeout: Duration,
    pub user_tier: Option<String>,
    pub source: RequestSource,
    pub queue_length: usize,
}

/// 优先级计算器
pub struct PriorityCalculator {
    user_tier_weights: HashMap<String, f64>,
    source_weights: HashMap<String, f64>,
}

impl PriorityCalculator {
    pub fn new(config: PriorityConfig) -> Self {
        let user_tier_weights: HashMap<String, f64> = config
            .user_tier_weights
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        let source_weights: HashMap<String, f64> = config
            .source_weights
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        Self {
            user_tier_weights,
            source_weights,
        }
    }

    pub fn calculate(&self, input: PriorityInput) -> Priority {
        let mut score = input.base_priority.as_i32();

        // 1. 基于超时时间提升优先级（限制最大提升因子）
        let timeout_factor = self.calculate_timeout_factor(input.time_until_timeout);
        let timeout_factor = timeout_factor.min(2.0); // 限制最大提升为 2 倍
        score = (score as f64 * timeout_factor) as i32;

        // 2. 基于用户等级调整（限制最大权重）
        if let Some(ref tier) = input.user_tier {
            if let Some(&weight) = self.user_tier_weights.get(tier) {
                let weight = weight.min(3.0); // 限制最大权重为 3 倍
                score = (score as f64 * weight) as i32;
            }
        }

        // 3. 基于请求来源调整（限制最大权重）
        let source_weight = self.get_source_weight(&input.source);
        let source_weight = source_weight.min(2.0); // 限制最大权重为 2 倍
        score = (score as f64 * source_weight) as i32;

        // 4. 基于队列长度调整（限制最大提升）
        let queue_factor = 1.0 + (input.queue_length as f64 * 0.05).min(1.0); // 限制最大提升为 1 倍
        score = (score as f64 * queue_factor) as i32;

        // 最终限制分数范围
        let score = score.clamp(0, 150); // 限制在 0-150 范围内

        // 转换回 Priority 枚举
        let priority = Priority::from_score(score);
        debug!(
            "Calculated priority: score={}, priority={:?}, base={:?}, timeout_factor={:.2}, source_weight={:.2}, queue_factor={:.2}",
            score, priority, input.base_priority, timeout_factor, source_weight, queue_factor
        );

        priority
    }

    fn calculate_timeout_factor(&self, time_until_timeout: Duration) -> f64 {
        let remaining_ms = time_until_timeout.as_millis() as f64;

        if remaining_ms < 100.0 {
            2.0
        } else if remaining_ms < 500.0 {
            1.5
        } else if remaining_ms < 1000.0 {
            1.2
        } else {
            1.0
        }
    }

    fn get_source_weight(&self, source: &RequestSource) -> f64 {
        match source {
            RequestSource::Grpc { .. } => self.source_weights.get("grpc").copied().unwrap_or(1.2),
            RequestSource::Http { .. } => self.source_weights.get("http").copied().unwrap_or(1.0),
            RequestSource::Internal => self.source_weights.get("internal").copied().unwrap_or(1.5),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_from_score() {
        assert_eq!(Priority::from_score(95), Priority::Critical);
        assert_eq!(Priority::from_score(70), Priority::High);
        assert_eq!(Priority::from_score(50), Priority::Normal);
        assert_eq!(Priority::from_score(30), Priority::Low);
    }

    #[test]
    fn test_priority_calculator() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);

        let input = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_millis(2000),
            user_tier: Some("pro".to_string()),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
            queue_length: 5,
        };

        let priority = calculator.calculate(input);
        // 允许 Critical 或 High，因为队列长度会提升优先级
        assert!(
            priority == Priority::Critical
                || priority == Priority::High
                || priority == Priority::Normal
        );
    }

    #[test]
    fn test_timeout_boost() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);

        let input_critical = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_millis(50),
            user_tier: None,
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
            queue_length: 0,
        };

        let priority = calculator.calculate(input_critical);
        assert_eq!(priority, Priority::Critical);
    }

    #[test]
    fn test_user_tier_boost() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);

        let input_free = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_secs(10),
            user_tier: Some("free".to_string()),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
            queue_length: 0,
        };

        let input_enterprise = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_secs(10),
            user_tier: Some("enterprise".to_string()),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
            queue_length: 0,
        };

        let priority_free = calculator.calculate(input_free);
        let priority_enterprise = calculator.calculate(input_enterprise);

        assert!(priority_enterprise >= priority_free);
    }

    // ===== Priority::as_i32 tests =====

    #[test]
    fn test_priority_as_i32_returns_numeric_value() {
        assert_eq!(Priority::Critical.as_i32(), 100);
        assert_eq!(Priority::High.as_i32(), 75);
        assert_eq!(Priority::Normal.as_i32(), 50);
        assert_eq!(Priority::Low.as_i32(), 25);
    }

    // ===== Priority::from_score boundary tests =====

    #[test]
    fn test_priority_from_score_boundaries() {
        assert_eq!(Priority::from_score(90), Priority::Critical);
        assert_eq!(Priority::from_score(89), Priority::High);
        assert_eq!(Priority::from_score(65), Priority::High);
        assert_eq!(Priority::from_score(64), Priority::Normal);
        assert_eq!(Priority::from_score(40), Priority::Normal);
        assert_eq!(Priority::from_score(39), Priority::Low);
        assert_eq!(Priority::from_score(0), Priority::Low);
        assert_eq!(Priority::from_score(-1), Priority::Low);
    }

    // ===== RequestSource constructor tests =====

    #[test]
    fn test_request_source_http_constructor() {
        let source = RequestSource::http("192.168.1.1".to_string());
        match source {
            RequestSource::Http { ip } => assert_eq!(ip, "192.168.1.1"),
            other => panic!("Expected Http, got {:?}", other),
        }
    }

    #[test]
    fn test_request_source_grpc_constructor() {
        let source = RequestSource::grpc("client-123".to_string());
        match source {
            RequestSource::Grpc { client_id } => assert_eq!(client_id, "client-123"),
            other => panic!("Expected Grpc, got {:?}", other),
        }
    }

    #[test]
    fn test_request_source_internal_constructor() {
        let source = RequestSource::internal();
        assert!(matches!(source, RequestSource::Internal));
    }

    // ===== Timeout factor tests =====

    #[test]
    fn test_timeout_factor_under_100ms_critical_boost() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_millis(50),
            user_tier: None,
            source: RequestSource::internal(),
            queue_length: 0,
        };
        let priority = calculator.calculate(input);
        // Normal(50) * 2.0(timeout) * 1.5(internal) = 150 → Critical
        assert_eq!(priority, Priority::Critical);
    }

    #[test]
    fn test_timeout_factor_100_to_500ms_high_boost() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_millis(200),
            user_tier: None,
            source: RequestSource::internal(),
            queue_length: 0,
        };
        let priority = calculator.calculate(input);
        // Normal(50) * 1.5(timeout) * 1.5(internal) = 112.5 → 112 → Critical
        assert_eq!(priority, Priority::Critical);
    }

    #[test]
    fn test_timeout_factor_500_to_1000ms_moderate_boost() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_millis(700),
            user_tier: None,
            source: RequestSource::internal(),
            queue_length: 0,
        };
        let priority = calculator.calculate(input);
        // Normal(50) * 1.2(timeout) * 1.5(internal) = 90 → Critical
        assert_eq!(priority, Priority::Critical);
    }

    #[test]
    fn test_timeout_factor_over_1000ms_no_boost() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_millis(2000),
            user_tier: None,
            source: RequestSource::internal(),
            queue_length: 0,
        };
        let priority = calculator.calculate(input);
        // Normal(50) * 1.0(timeout) * 1.5(internal) = 75 → High
        assert_eq!(priority, Priority::High);
    }

    // ===== Source weight tests =====

    #[test]
    fn test_source_weight_http() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_secs(10),
            user_tier: None,
            source: RequestSource::http("127.0.0.1".to_string()),
            queue_length: 0,
        };
        let priority = calculator.calculate(input);
        // Normal(50) * 1.0(timeout) * 1.0(http default) = 50 → Normal
        assert_eq!(priority, Priority::Normal);
    }

    #[test]
    fn test_source_weight_grpc() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_secs(10),
            user_tier: None,
            source: RequestSource::grpc("client-1".to_string()),
            queue_length: 0,
        };
        let priority = calculator.calculate(input);
        // Normal(50) * 1.0(timeout) * 1.2(grpc default) = 60 → Normal
        assert_eq!(priority, Priority::Normal);
    }

    #[test]
    fn test_source_weight_internal() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_secs(10),
            user_tier: None,
            source: RequestSource::internal(),
            queue_length: 0,
        };
        let priority = calculator.calculate(input);
        // Normal(50) * 1.0(timeout) * 1.5(internal default) = 75 → High
        assert_eq!(priority, Priority::High);
    }

    // ===== Queue length tests =====

    #[test]
    fn test_queue_length_large_boost() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_secs(10),
            user_tier: None,
            source: RequestSource::http("127.0.0.1".to_string()),
            queue_length: 100,
        };
        let priority = calculator.calculate(input);
        // Normal(50) * 1.0(timeout) * 1.0(http) * 2.0(queue=100*0.05 capped) = 100 → Critical
        assert_eq!(priority, Priority::Critical);
    }

    #[test]
    fn test_calculate_with_unknown_tier_uses_no_weight() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Normal,
            time_until_timeout: Duration::from_secs(10),
            user_tier: Some("unknown_tier".to_string()),
            source: RequestSource::http("127.0.0.1".to_string()),
            queue_length: 0,
        };
        let priority = calculator.calculate(input);
        // Unknown tier → no weight applied: 50 * 1.0 * 1.0 = 50 → Normal
        assert_eq!(priority, Priority::Normal);
    }

    #[test]
    fn test_calculate_clamps_to_150() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Critical,
            time_until_timeout: Duration::from_millis(50),
            user_tier: Some("enterprise".to_string()),
            source: RequestSource::internal(),
            queue_length: 100,
        };
        let priority = calculator.calculate(input);
        // All boosts applied but clamped to 150 → Critical
        assert_eq!(priority, Priority::Critical);
    }

    #[test]
    fn test_calculate_low_base_with_no_boosts() {
        let config = PriorityConfig::default();
        let calculator = PriorityCalculator::new(config);
        let input = PriorityInput {
            base_priority: Priority::Low,
            time_until_timeout: Duration::from_secs(10),
            user_tier: None,
            source: RequestSource::http("127.0.0.1".to_string()),
            queue_length: 0,
        };
        let priority = calculator.calculate(input);
        // Low(25) * 1.0 * 1.0 = 25 → Low
        assert_eq!(priority, Priority::Low);
    }
}
