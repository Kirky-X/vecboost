// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! jemalloc decay 调优（wsl2-oom-hardening-r2 D3，实施期修正版）。
//!
//! 全局分配器为 tikv-jemallocator（main.rs，linux glibc）：默认 dirty/muzzy
//! decay 10s，大请求（长翻译/批量）释放的解码缓冲滞留 RSS，放大与并发构建
//! 的内存竞争。
//!
//! 配置注入机制（probe 实测三分支后的定案）：
//! - `je_malloc_conf` 符号覆盖不可行——jemalloc.c 自带 weak 定义且编译带
//!   `-fvisibility=hidden`（configure.ac:972），同 TU 引用绑定隐藏本地定义，
//!   应用侧强符号永不生效（二进制内实测两个定义共存，opt 仍回落默认）；
//! - 运行时 mallctl 不可行——无前缀 `dirty_decay_ms`/`muzzy_decay_ms` 在本
//!   build 不存在，`arena.N.*` 仅已创建 arena 可写；
//! - **定案**：`.cargo/config.toml [env] JEMALLOC_SYS_WITH_MALLOC_CONF` →
//!   jemalloc-sys build.rs 转 `--with-malloc-conf` 编译期烘焙，全域（含未来
//!   arena）生效。优先级最低：运行期 `MALLOC_CONF` 环境变量可覆盖（运维
//!   逃生口）。
//!
//! cfg 门控与 main.rs 全局分配器完全一致（linux + 非 musl），非门控平台零
//! 编译影响。

/// 读 boot 配置生效值（`opt.*` 为只读视图），供启动日志与测试验证。
pub fn effective_boot_conf() -> Result<(i64, i64), String> {
    let dirty = unsafe { tikv_jemalloc_ctl::raw::read::<i64>(b"opt.dirty_decay_ms\0") }
        .map_err(|e| format!("read opt.dirty_decay_ms failed: {e}"))?;
    let muzzy = unsafe { tikv_jemalloc_ctl::raw::read::<i64>(b"opt.muzzy_decay_ms\0") }
        .map_err(|e| format!("read opt.muzzy_decay_ms failed: {e}"))?;
    Ok((dirty, muzzy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boot_conf_effective_in_test_binary() {
        // .cargo/config.toml [env] JEMALLOC_SYS_WITH_MALLOC_CONF →
        // --with-malloc-conf 编译期烘焙；断言实际生效值
        assert_eq!(
            effective_boot_conf().expect("opt probe must succeed on linux gnu test build"),
            (1000, 1000)
        );
    }
}
