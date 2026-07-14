// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! CacheGetOrInsert trait 的默认实现

use super::{Cache, CacheGetOrInsert};
use std::hash::Hash;

/// 所有缓存实现默认实现 get_or_insert
#[async_trait::async_trait]
impl<K, V, C> CacheGetOrInsert<K, V> for C
where
    K: Hash + Eq + Send + Sync + std::fmt::Debug + Clone + 'static,
    V: Clone + Send + Sync + 'static,
    C: Cache<K, V>,
{
    async fn get_or_insert(
        &self,
        key: K,
        value: V,
        size: usize,
    ) -> Result<V, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(v) = self.get(&key).await {
            return Ok(v);
        }
        self.put(key.clone(), value, size).await?;
        Ok(self.get(&key).await.ok_or("Value not found after insert")?)
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::{Cache, CacheGetOrInsert, CacheStats};
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// 简单的内存缓存,用于测试 trait 实现
    struct MockCache {
        entries: Mutex<HashMap<String, (String, usize)>>,
    }

    impl MockCache {
        fn new() -> Self {
            Self {
                entries: Mutex::new(HashMap::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Cache<String, String> for MockCache {
        async fn get(&self, key: &String) -> Option<String> {
            self.entries
                .lock()
                .unwrap()
                .get(key)
                .map(|(v, _)| v.clone())
        }

        async fn put(&self, key: String, value: String, size: usize) -> Result<(), String> {
            self.entries.lock().unwrap().insert(key, (value, size));
            Ok(())
        }

        async fn remove(&self, key: &String) -> bool {
            self.entries.lock().unwrap().remove(key).is_some()
        }

        async fn clear(&self) {
            self.entries.lock().unwrap().clear();
        }

        async fn len(&self) -> usize {
            self.entries.lock().unwrap().len()
        }

        async fn is_empty(&self) -> bool {
            self.entries.lock().unwrap().is_empty()
        }

        async fn stats(&self) -> CacheStats {
            CacheStats::default()
        }

        async fn warm_up(&self, entries: HashMap<String, (String, usize)>) -> Result<(), String> {
            self.entries.lock().unwrap().extend(entries);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_get_or_insert_inserts_when_missing() {
        let cache = MockCache::new();
        // key 不存在时应插入并返回值
        let result = cache
            .get_or_insert("k1".to_string(), "v1".to_string(), 2)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "v1");
        // 验证确实写入了缓存
        assert_eq!(cache.get(&"k1".to_string()).await, Some("v1".to_string()));
    }

    #[tokio::test]
    async fn test_get_or_insert_returns_existing_when_present() {
        let cache = MockCache::new();
        cache
            .put("k1".to_string(), "existing".to_string(), 8)
            .await
            .unwrap();
        // key 已存在时应返回已缓存值,不覆盖
        let result = cache
            .get_or_insert("k1".to_string(), "new_value".to_string(), 9)
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "existing");
    }

    #[tokio::test]
    async fn test_get_or_insert_multiple_distinct_keys() {
        let cache = MockCache::new();
        for i in 0..3 {
            let key = format!("k{}", i);
            let val = format!("v{}", i);
            let r = cache.get_or_insert(key.clone(), val.clone(), i + 1).await;
            assert!(r.is_ok());
            assert_eq!(r.unwrap(), val);
        }
        assert_eq!(cache.len().await, 3);
    }

    #[tokio::test]
    async fn test_get_or_insert_returns_inserted_size() {
        let cache = MockCache::new();
        cache
            .get_or_insert("k".to_string(), "v".to_string(), 42)
            .await
            .unwrap();
        // 验证 size 被正确写入
        let snapshot = cache.entries.lock().unwrap().clone();
        assert_eq!(snapshot.get("k").unwrap().1, 42);
    }
}
