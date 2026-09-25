// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! per-key single-flight：并发同 key 只放一个 compute，后到者等前者写缓存后 double-check 命中。

use std::collections::HashMap;
use std::sync::Arc;

/// single-flight 锁表：key → 每 key 一把 async mutex。
pub(crate) type FlightMap = Arc<std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>;

/// single-flight 许可守卫：释放时若无等待者则移除映射条目，
/// 防 per-key 锁映射随历史 key 无界增长。
pub(crate) struct SingleFlightPermit {
    map: FlightMap,
    key: String,
    entry: Arc<tokio::sync::Mutex<()>>,
    _guard: tokio::sync::OwnedMutexGuard<()>,
}

impl SingleFlightPermit {
    /// 注册（或复用）key 的锁并等待持有。
    pub(crate) async fn acquire(map: &FlightMap, key: &str) -> Self {
        let entry = {
            let mut m = map.lock().unwrap_or_else(|e| e.into_inner());
            m.entry(key.to_string()).or_default().clone()
        };
        let guard = entry.clone().lock_owned().await;
        Self {
            map: Arc::clone(map),
            key: key.to_string(),
            entry,
            _guard: guard,
        }
    }
}

impl Drop for SingleFlightPermit {
    fn drop(&mut self) {
        // 先持 map 锁再读 strong_count：注册方必经同一把 map 锁，
        // 闭合「读计数 → 持锁」窗口内新等待者插入的 TOCTOU。
        if let Ok(mut map) = self.map.lock()
            // map(1) + self.entry(1) + OwnedMutexGuard 内部(1)；等待者每多一个 +1。
            // == 3 即无等待者，可回收条目。
            && Arc::strong_count(&self.entry) == 3
        {
            map.remove(&self.key);
        }
    }
}
