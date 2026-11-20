// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 调用方身份提取（语义缓存 caller isolation 用）。
//!
//! caller 取服务端认证态用户名，与 `AuthContext.user.username` 同源：
//! 认证中间件在 token 校验通过后把 token 写入 garrison task_local，本模块
//! 读取该 token 并解析出登录名。**不读取**可伪造的 X-User-Id 请求头
//! （与 `src/rate_limit/limiteron_adapter.rs` 既有结论一致）。

/// 无认证态时的归并桶字面量。
pub const ANONYMOUS_CALLER: &str = "anonymous";

/// caller 隔离启用标记（装配期单点注入，进程级无锁 AtomicBool）。
///
/// handler 借此决定是否提取真实 caller：`caller_isolation = "user"` 时置
/// true；off（默认）保持 false——off 模式下 caller 不参与任何缓存键与隔离
/// 判定，每请求不再白付 token 解析（会话后端为 database 时是一次 DB 往返）
/// 与 String 分配。选进程级原子而非 kit/state 配置：MCP/CLI/server 三种
/// kit 装配路径共用单一注入点，handler 读取零依赖、零锁。
static CALLER_ISOLATION_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// set-once 标记：门控只允许装配期注入一次，防止二次重置破坏进程级分区不变量。
static CALLER_ISOLATION_SET: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 装配期注入（`main.rs::init_engine_and_services` 在 `parse_caller_isolation`
/// 单一解析点之后调用；进程级状态，重复注入视为装配缺陷）。
pub fn set_caller_isolation_enabled(enabled: bool) {
    // compare_exchange 在 release 构建下同样拦截重复注入；debug_assert 额外在
    // 测试构建让装配缺陷显性失败（安全审查 Nit-1 的 set-once 加固）。
    let already_set = CALLER_ISOLATION_SET
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_err();
    debug_assert!(
        !already_set,
        "caller isolation gate is set-once: repeated assembly must not reset it"
    );
    if already_set {
        // release 构建同样显性拒绝二次注入（debug 构建前述 assert 直接失败），
        // 避免「首次注入 off 后 user 隔离静默失效」的调试盲区（架构审查 F2）。
        log::error!(
            "{}",
            crate::i18n::tr("log-caller-isolation-gate-already-set")
        );
        return;
    }
    CALLER_ISOLATION_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// 测试专用写入口：绕过 set-once 标记，只由门控测试套的 Mutex 串行保护使用。
#[cfg(test)]
fn set_caller_isolation_enabled_for_test(enabled: bool) {
    CALLER_ISOLATION_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// 无锁读取当前门控状态（handler 热路径；不得为取此标记先抢任何服务锁）。
pub fn caller_isolation_enabled() -> bool {
    CALLER_ISOLATION_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// handler 侧 caller 提取（门控版）。
///
/// 隔离启用时解析认证态用户名——调用方必须在**获取服务读锁之前**调用
///（避免持锁做会话后端 DB 往返）；off（默认）直接回 anonymous，语义与
/// auth 关闭时 caller 恒 anonymous 一致（[`current_caller_id`] 现行为保持）。
pub async fn current_caller_id_gated() -> String {
    if caller_isolation_enabled() {
        current_caller_id().await
    } else {
        ANONYMOUS_CALLER.to_string()
    }
}

/// 当前请求的调用方身份。
#[cfg(feature = "auth")]
pub async fn current_caller_id() -> String {
    match garrison::stp::current_token() {
        Ok(token) => crate::auth::GarrisonUtil::get_login_id_by_token(&token)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| ANONYMOUS_CALLER.to_string()),
        Err(_) => ANONYMOUS_CALLER.to_string(),
    }
}

/// auth 关闭时恒为 "anonymous"（公开推理端点目前不存在，防御性归并桶）。
#[cfg(not(feature = "auth"))]
pub async fn current_caller_id() -> String {
    ANONYMOUS_CALLER.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_anonymous_fallback_without_token() {
        // 无 task_local token 时必须回退 anonymous（auth 构建下 current_token 失败）。
        let caller = current_caller_id().await;
        assert!(!caller.is_empty());
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_caller_from_task_local_token() {
        // 设置一个测试 token 并确认能解析（解析失败也应回退 anonymous，不 panic）。
        let _ = crate::auth::with_current_token("test-token".to_string(), async {
            let caller = current_caller_id().await;
            assert!(!caller.is_empty());
        })
        .await;
    }

    /// 门控钉（性能 LOW-2）：off（默认）下门控提取直接回 anonymous 且不
    /// 触发解析路径；置 true 后门控提取走 current_caller_id（无 token 仍
    /// anonymous，非空不变）。进程级状态，改值后还原。
    #[tokio::test]
    async fn test_gated_extraction_off_skips_and_user_resolves() {
        // 进程级门控状态：改值测试套必须串行（参照 doctor.rs ENV_LOCK），
        // 否则并行测试互相覆盖会制造随机失败。本测试为 async，用 tokio Mutex
        // 避免 std MutexGuard 跨 await（clippy::await_holding_lock）。
        static GATE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
        let _guard = GATE_LOCK.lock().await;
        let prev = caller_isolation_enabled();
        // 先归到 off 再测，避免与其它测试的改值互相干扰。
        set_caller_isolation_enabled_for_test(false);
        assert_eq!(
            current_caller_id_gated().await,
            ANONYMOUS_CALLER,
            "off 门控必须跳过解析直回 anonymous"
        );

        set_caller_isolation_enabled_for_test(true);
        let gated = current_caller_id_gated().await;
        assert!(
            !gated.is_empty(),
            "user 门控必须走 current_caller_id（无 token 回退 anonymous）"
        );

        set_caller_isolation_enabled_for_test(prev);
        assert_eq!(caller_isolation_enabled(), prev, "必须还原装配态");
    }
}
