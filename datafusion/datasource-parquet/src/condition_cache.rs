// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Query Condition Cache for Parquet files
//!
//! This module implements a lightweight, memory-efficient cache for storing
//! predicate evaluation results at the row group and page level, similar to
//! ClickHouse's query condition cache.
//!
//! The cache stores a bitmap (1 bit per row group/page) indicating whether
//! the row group/page contains any rows that match a given predicate.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use bitvec::prelude::*;
use dashmap::DashMap;
use datafusion_physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder,
};

/// Configuration for the condition cache
#[derive(Debug, Clone)]
pub struct ConditionCacheConfig {
    /// Enable the condition cache
    pub enable: bool,
    /// Maximum cache size in MB
    pub max_size_mb: usize,
    /// Enable page-level caching (more memory, better pruning)
    pub enable_page_cache: bool,
    /// TTL for cache entries in seconds
    pub ttl_seconds: u64,
    /// Minimum selectivity to cache (don't cache if >X% rows match)
    pub min_selectivity_to_cache: f64,
}

impl Default for ConditionCacheConfig {
    fn default() -> Self {
        Self {
            enable: true,
            max_size_mb: 100, // 100MB default like ClickHouse
            enable_page_cache: false,
            ttl_seconds: 3600,             // 1 hour default
            min_selectivity_to_cache: 0.9, // Don't cache if >90% rows match
        }
    }
}

/// Key for row group level cache entries
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct RowGroupCacheKey {
    /// File identifier (path or object store URL hash)
    pub file_id: u64,
    /// Hash of the normalized predicate expression
    pub predicate_hash: u64,
    /// File version/mtime for invalidation
    pub file_version: u64,
}

/// Key for page level cache entries
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct PageCacheKey {
    /// File identifier
    pub file_id: u64,
    /// Row group index
    pub row_group_idx: usize,
    /// Hash of the normalized predicate expression
    pub predicate_hash: u64,
    /// File version/mtime for invalidation
    pub file_version: u64,
}

/// Cache entry with metadata
#[derive(Clone, Debug)]
struct CacheEntry<T> {
    /// The cached data
    data: Arc<T>,
    /// When this entry was created
    created_at: SystemTime,
    /// Size in bytes
    size_bytes: usize,
}

/// Thread-safe condition cache with minimal locking
#[derive(Debug)]
pub struct ParquetConditionCache {
    /// Row group level cache (coarse-grained)
    /// BitVec where each bit represents whether a row group contains matching rows
    row_group_cache: DashMap<RowGroupCacheKey, CacheEntry<BitVec>>,

    /// Page level cache (fine-grained)
    /// BitVec where each bit represents whether a page contains matching rows
    /// Using DashMap for lock-free concurrent access
    page_cache: DashMap<PageCacheKey, CacheEntry<BitVec>>,

    /// Configuration
    config: ConditionCacheConfig,

    /// Current memory usage in bytes
    memory_usage_bytes: AtomicUsize,

    /// Metrics
    metrics: ConditionCacheMetrics,
}

/// Metrics for monitoring cache effectiveness
#[derive(Debug, Clone)]
pub struct ConditionCacheMetrics {
    /// Number of row group cache hits
    pub row_group_cache_hits: Count,
    /// Number of row group cache misses
    pub row_group_cache_misses: Count,
    /// Number of page cache hits
    pub page_cache_hits: Count,
    /// Number of page cache misses
    pub page_cache_misses: Count,
    /// Current memory usage in bytes
    pub memory_usage_bytes: Gauge,
    /// Number of evictions
    pub evictions: Count,
}

/// Cache statistics snapshot
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Number of row group cache hits
    pub row_group_hits: usize,
    /// Number of row group cache misses
    pub row_group_misses: usize,
    /// Number of page cache hits
    pub page_hits: usize,
    /// Number of page cache misses
    pub page_misses: usize,
    /// Current memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of evictions
    pub evictions: usize,
    /// Number of entries in row group cache
    pub row_group_entries: usize,
    /// Number of entries in page cache
    pub page_entries: usize,
    /// Cache configuration
    pub config: ConditionCacheConfig,
}

/// Cache hit rate statistics
#[derive(Debug, Clone)]
pub struct CacheHitRate {
    /// Row group cache hit rate percentage
    pub row_group_hit_rate: f64,
    /// Page cache hit rate percentage
    pub page_hit_rate: f64,
    /// Overall cache hit rate percentage
    pub overall_hit_rate: f64,
}

impl ConditionCacheMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            row_group_cache_hits: MetricBuilder::new(metrics)
                .counter("row_group_cache_hits", 0),
            row_group_cache_misses: MetricBuilder::new(metrics)
                .counter("row_group_cache_misses", 0),
            page_cache_hits: MetricBuilder::new(metrics).counter("page_cache_hits", 0),
            page_cache_misses: MetricBuilder::new(metrics)
                .counter("page_cache_misses", 0),
            memory_usage_bytes: MetricBuilder::new(metrics)
                .gauge("cache_memory_bytes", 0),
            evictions: MetricBuilder::new(metrics).counter("cache_evictions", 0),
        }
    }
}

impl ParquetConditionCache {
    /// Create a new condition cache
    pub fn new(config: ConditionCacheConfig, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            row_group_cache: DashMap::new(),
            page_cache: DashMap::new(),
            config,
            memory_usage_bytes: AtomicUsize::new(0),
            metrics: ConditionCacheMetrics::new(metrics),
        }
    }

    /// Create a new condition cache with default configuration
    pub fn new_default(metrics: &ExecutionPlanMetricsSet) -> Self {
        Self::new(ConditionCacheConfig::default(), metrics)
    }

    /// Get a row group bitmap from the cache
    pub fn get_row_group_bitmap(&self, key: &RowGroupCacheKey) -> Option<Arc<BitVec>> {
        if !self.config.enable {
            return None;
        }

        if let Some(entry) = self.row_group_cache.get(key) {
            // Check TTL
            if let Ok(elapsed) = entry.created_at.elapsed() {
                if elapsed < Duration::from_secs(self.config.ttl_seconds) {
                    self.metrics.row_group_cache_hits.add(1);
                    return Some(entry.data.clone());
                } else {
                    // Entry expired, remove it
                    drop(entry);
                    self.row_group_cache.remove(key);
                }
            }
        }

        self.metrics.row_group_cache_misses.add(1);
        None
    }

    /// Store a row group bitmap in the cache
    pub fn put_row_group_bitmap(&self, key: RowGroupCacheKey, bitmap: BitVec) {
        if !self.config.enable {
            return;
        }

        // Check selectivity - don't cache if too many matches
        let matches = bitmap.count_ones();
        let total = bitmap.len();
        let selectivity = matches as f64 / total as f64;

        if selectivity > self.config.min_selectivity_to_cache {
            // Too many matches, not worth caching
            return;
        }

        let size_bytes = Self::estimate_bitmap_memory(&bitmap);

        // Check if we need to evict
        self.evict_if_needed(size_bytes);

        let entry = CacheEntry {
            data: Arc::new(bitmap),
            created_at: SystemTime::now(),
            size_bytes,
        };

        // Insert and update memory usage
        if let Some(old_entry) = self.row_group_cache.insert(key.clone(), entry.clone()) {
            // Subtract old entry size
            self.memory_usage_bytes
                .fetch_sub(old_entry.size_bytes, Ordering::Relaxed);
        }
        self.memory_usage_bytes
            .fetch_add(size_bytes, Ordering::Relaxed);
        self.metrics
            .memory_usage_bytes
            .set(self.memory_usage_bytes.load(Ordering::Relaxed));
    }

    /// Get a page bitmap from the cache
    pub fn get_page_bitmap(&self, key: &PageCacheKey) -> Option<Arc<BitVec>> {
        if !self.config.enable || !self.config.enable_page_cache {
            return None;
        }

        if let Some(entry) = self.page_cache.get(key) {
            // Check TTL
            if let Ok(elapsed) = entry.created_at.elapsed() {
                if elapsed < Duration::from_secs(self.config.ttl_seconds) {
                    self.metrics.page_cache_hits.add(1);
                    return Some(entry.data.clone());
                } else {
                    // Entry expired, remove it
                    drop(entry);
                    self.page_cache.remove(key);
                }
            }
        }

        self.metrics.page_cache_misses.add(1);
        None
    }

    /// Store a page bitmap in the cache
    pub fn put_page_bitmap(&self, key: PageCacheKey, bitmap: BitVec) {
        if !self.config.enable || !self.config.enable_page_cache {
            return;
        }

        let size_bytes = Self::estimate_bitmap_memory(&bitmap);

        // Check if we need to evict
        self.evict_if_needed(size_bytes);

        let entry = CacheEntry {
            data: Arc::new(bitmap),
            created_at: SystemTime::now(),
            size_bytes,
        };

        // Insert and update memory usage
        if let Some(old_entry) = self.page_cache.insert(key.clone(), entry.clone()) {
            self.memory_usage_bytes
                .fetch_sub(old_entry.size_bytes, Ordering::Relaxed);
        }
        self.memory_usage_bytes
            .fetch_add(size_bytes, Ordering::Relaxed);
        self.metrics
            .memory_usage_bytes
            .set(self.memory_usage_bytes.load(Ordering::Relaxed));
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        self.row_group_cache.clear();
        self.page_cache.clear();
        self.memory_usage_bytes.store(0, Ordering::Relaxed);
        self.metrics.memory_usage_bytes.set(0);
    }

    /// Get current memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        self.memory_usage_bytes.load(Ordering::Relaxed)
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStatistics {
        CacheStatistics {
            row_group_hits: self.metrics.row_group_cache_hits.value(),
            row_group_misses: self.metrics.row_group_cache_misses.value(),
            page_hits: self.metrics.page_cache_hits.value(),
            page_misses: self.metrics.page_cache_misses.value(),
            memory_usage_bytes: self.memory_usage_bytes.load(Ordering::Relaxed),
            evictions: self.metrics.evictions.value(),
            row_group_entries: self.row_group_cache.len(),
            page_entries: self.page_cache.len(),
            config: self.config.clone(),
        }
    }

    /// Get hit rate statistics
    pub fn hit_rate(&self) -> CacheHitRate {
        let rg_hits = self.metrics.row_group_cache_hits.value();
        let rg_misses = self.metrics.row_group_cache_misses.value();
        let rg_total = rg_hits + rg_misses;

        let page_hits = self.metrics.page_cache_hits.value();
        let page_misses = self.metrics.page_cache_misses.value();
        let page_total = page_hits + page_misses;

        CacheHitRate {
            row_group_hit_rate: if rg_total > 0 {
                (rg_hits as f64 / rg_total as f64) * 100.0
            } else {
                0.0
            },
            page_hit_rate: if page_total > 0 {
                (page_hits as f64 / page_total as f64) * 100.0
            } else {
                0.0
            },
            overall_hit_rate: {
                let total_hits = rg_hits + page_hits;
                let total_requests = rg_total + page_total;
                if total_requests > 0 {
                    (total_hits as f64 / total_requests as f64) * 100.0
                } else {
                    0.0
                }
            },
        }
    }

    /// Estimate memory usage of a bitmap
    fn estimate_bitmap_memory(bitmap: &BitVec) -> usize {
        use std::mem::size_of;
        // BitVec size + overhead for Arc and HashMap entry
        (bitmap.len() + 7) / 8 + size_of::<CacheEntry<BitVec>>() + 64
    }

    /// Evict entries if needed to stay within memory limit
    fn evict_if_needed(&self, required_bytes: usize) {
        let max_bytes = self.config.max_size_mb * 1024 * 1024;
        let current = self.memory_usage_bytes.load(Ordering::Relaxed);

        if current + required_bytes > max_bytes {
            let to_evict = (current + required_bytes) - max_bytes;
            self.evict_lru(to_evict);
        }
    }

    /// Evict least recently used entries
    fn evict_lru(&self, bytes_to_evict: usize) {
        let mut evicted = 0usize;

        // First evict from page cache (more granular, likely larger)
        if self.config.enable_page_cache {
            // Find oldest entries - DashMap iteration is lock-free
            let mut entries: Vec<_> = self
                .page_cache
                .iter()
                .map(|entry| {
                    let key = entry.key().clone();
                    let created = entry.value().created_at;
                    let size = entry.value().size_bytes;
                    (key, created, size)
                })
                .collect();
            entries.sort_by_key(|(_, created, _)| *created);

            for (key, _, size) in entries {
                if evicted >= bytes_to_evict {
                    break;
                }
                self.page_cache.remove(&key);
                evicted += size;
                self.metrics.evictions.add(1);
            }
        }

        // Then evict from row group cache if needed
        if evicted < bytes_to_evict {
            // DashMap doesn't provide easy iteration with ordering,
            // so we collect all entries first
            let entries: Vec<_> = self
                .row_group_cache
                .iter()
                .map(|entry| {
                    let key = entry.key().clone();
                    let created = entry.value().created_at;
                    let size = entry.value().size_bytes;
                    (key, created, size)
                })
                .collect();

            let mut sorted_entries = entries;
            sorted_entries.sort_by_key(|(_, created, _)| *created);

            for (key, _, size) in sorted_entries {
                if evicted >= bytes_to_evict {
                    break;
                }
                self.row_group_cache.remove(&key);
                evicted += size;
                self.metrics.evictions.add(1);
            }
        }

        self.memory_usage_bytes
            .fetch_sub(evicted, Ordering::Relaxed);
        self.metrics
            .memory_usage_bytes
            .set(self.memory_usage_bytes.load(Ordering::Relaxed));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;

    #[test]
    fn test_cache_basic_operations() {
        let metrics = ExecutionPlanMetricsSet::new();
        let cache = ParquetConditionCache::new_default(&metrics);

        let key = RowGroupCacheKey {
            file_id: 1,
            predicate_hash: 123,
            file_version: 456,
        };

        // Test cache miss
        assert!(cache.get_row_group_bitmap(&key).is_none());
        assert_eq!(cache.metrics.row_group_cache_misses.value(), 1);

        // Put and get
        let mut bitmap = BitVec::new();
        bitmap.resize(10, false);
        bitmap.set(2, true);
        bitmap.set(5, true);

        cache.put_row_group_bitmap(key.clone(), bitmap.clone());

        let retrieved = cache.get_row_group_bitmap(&key).unwrap();
        assert_eq!(retrieved.as_ref(), &bitmap);
        assert_eq!(cache.metrics.row_group_cache_hits.value(), 1);
    }

    #[test]
    fn test_cache_selectivity_filtering() {
        let metrics = ExecutionPlanMetricsSet::new();
        let mut config = ConditionCacheConfig::default();
        config.min_selectivity_to_cache = 0.5; // Don't cache if >50% match
        let cache = ParquetConditionCache::new(config, &metrics);

        let key = RowGroupCacheKey {
            file_id: 1,
            predicate_hash: 123,
            file_version: 456,
        };

        // Create bitmap with 70% matches (should not be cached)
        let mut bitmap = BitVec::new();
        bitmap.resize(10, false);
        for i in 0..7 {
            bitmap.set(i, true);
        }

        cache.put_row_group_bitmap(key.clone(), bitmap);

        // Should not be cached due to high selectivity
        assert!(cache.get_row_group_bitmap(&key).is_none());
    }

    #[test]
    fn test_cache_memory_limit() {
        let metrics = ExecutionPlanMetricsSet::new();
        let mut config = ConditionCacheConfig::default();
        config.max_size_mb = 1; // Very small cache for testing
        let cache = ParquetConditionCache::new(config, &metrics);

        // Add multiple entries
        for i in 0..1000 {
            let key = RowGroupCacheKey {
                file_id: i,
                predicate_hash: i,
                file_version: i,
            };

            let mut bitmap = BitVec::new();
            bitmap.resize(1000, false);
            bitmap.set((i % 1000) as usize, true);

            cache.put_row_group_bitmap(key, bitmap);
        }

        // Check that memory usage is within limit
        let usage = cache.memory_usage_bytes();
        assert!(usage <= 1 * 1024 * 1024);

        // Check that evictions occurred
        assert!(cache.metrics.evictions.value() > 0);
    }

    #[test]
    fn test_page_cache() {
        let metrics = ExecutionPlanMetricsSet::new();
        let mut config = ConditionCacheConfig::default();
        config.enable_page_cache = true;
        let cache = ParquetConditionCache::new(config, &metrics);

        let key = PageCacheKey {
            file_id: 1,
            row_group_idx: 0,
            predicate_hash: 123,
            file_version: 456,
        };

        // Test page cache operations
        assert!(cache.get_page_bitmap(&key).is_none());

        let mut bitmap = BitVec::new();
        bitmap.resize(100, false); // 100 pages
        bitmap.set(10, true);
        bitmap.set(50, true);

        cache.put_page_bitmap(key.clone(), bitmap.clone());

        let retrieved = cache.get_page_bitmap(&key).unwrap();
        assert_eq!(retrieved.as_ref(), &bitmap);
        assert_eq!(cache.metrics.page_cache_hits.value(), 1);
    }
}
