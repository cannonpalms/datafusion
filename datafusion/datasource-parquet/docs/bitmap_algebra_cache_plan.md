# Bitmap Algebra for Predicate Cache - Implementation Plan

## Overview

This document outlines a plan to enhance the Parquet condition cache by decomposing complex predicates into atomic components and combining cached bitmaps using boolean algebra. This would dramatically improve cache hit rates and memory efficiency.

## Concept

Instead of caching only complete predicates, we cache individual predicate components and combine them at query time:

```sql
-- Query 1: WHERE a = 1           → Cache: bitmap_a1
-- Query 2: WHERE b = 2           → Cache: bitmap_b2
-- Query 3: WHERE a = 1 AND b = 2 → Result: bitmap_a1 & bitmap_b2 (bitwise AND)
-- Query 4: WHERE a = 1 OR b = 2  → Result: bitmap_a1 | bitmap_b2 (bitwise OR)
```

## Implementation Phases

### Phase 1: Simple AND Decomposition (1-2 days)

**Goal**: Handle conjunction of simple predicates

#### 1.1 Predicate Decomposition

```rust
// In condition_cache.rs
impl ParquetConditionCache {
    /// Decompose AND predicates into atomic components
    fn decompose_conjunction(expr: &dyn PhysicalExpr) -> Vec<Arc<dyn PhysicalExpr>> {
        if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
            if binary.op() == &Operator::And {
                let mut components = Vec::new();
                components.extend(Self::decompose_conjunction(binary.left().as_ref()));
                components.extend(Self::decompose_conjunction(binary.right().as_ref()));
                return components;
            }
        }
        // Base case: return the expression itself
        vec![Arc::from(expr)]
    }
}
```

#### 1.2 Cache Lookup with Composition

```rust
impl ParquetConditionCache {
    /// Get bitmap, attempting to compose from cached components
    pub fn get_or_compose_row_group_bitmap(
        &self,
        key: &RowGroupCacheKey,
        predicate: &dyn PhysicalExpr,
    ) -> Option<Arc<BitVec>> {
        // Try direct cache hit first
        if let Some(cached) = self.row_group_cache.get(key) {
            if !self.is_expired(&cached) {
                self.metrics.row_group_cache_hits.add(1);
                return Some(cached.data.clone());
            }
        }

        // Try to compose from cached components
        if let Some(composed) = self.try_compose_and_cache(key, predicate) {
            self.metrics.row_group_cache_hits.add(1); // Partial hit
            return Some(composed);
        }

        self.metrics.row_group_cache_misses.add(1);
        None
    }

    fn try_compose_and_cache(
        &self,
        base_key: &RowGroupCacheKey,
        predicate: &dyn PhysicalExpr,
    ) -> Option<Arc<BitVec>> {
        let components = Self::decompose_conjunction(predicate);

        if components.len() <= 1 {
            return None; // No composition needed
        }

        // Get bitmap for each component
        let mut component_bitmaps = Vec::new();
        for component in &components {
            let component_hash = hash_predicate(component.as_ref());
            let component_key = RowGroupCacheKey {
                file_id: base_key.file_id,
                predicate_hash: component_hash,
                file_version: base_key.file_version,
            };

            // Check if component is cached
            if let Some(entry) = self.row_group_cache.get(&component_key) {
                if !self.is_expired(&entry) {
                    component_bitmaps.push(entry.data.clone());
                } else {
                    return None; // Missing component
                }
            } else {
                return None; // Missing component
            }
        }

        // Combine all bitmaps with AND
        let mut result = component_bitmaps[0].as_ref().clone();
        for bitmap in &component_bitmaps[1..] {
            result &= bitmap.as_ref();
        }

        let composed = Arc::new(result);

        // Cache the composed result for future use
        self.put_row_group_bitmap(base_key.clone(), composed.as_ref().clone());

        Some(composed)
    }
}
```

### Phase 2: OR Support (1 day)

**Goal**: Handle disjunction of predicates

#### 2.1 OR Decomposition

```rust
fn decompose_disjunction(expr: &dyn PhysicalExpr) -> Vec<Arc<dyn PhysicalExpr>> {
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if binary.op() == &Operator::Or {
            let mut components = Vec::new();
            components.extend(Self::decompose_disjunction(binary.left().as_ref()));
            components.extend(Self::decompose_disjunction(binary.right().as_ref()));
            return components;
        }
    }
    vec![Arc::from(expr)]
}
```

#### 2.2 Combine with OR

```rust
fn combine_with_or(&self, bitmaps: Vec<Arc<BitVec>>) -> Arc<BitVec> {
    let mut result = bitmaps[0].as_ref().clone();
    for bitmap in &bitmaps[1..] {
        result |= bitmap.as_ref();
    }
    Arc::new(result)
}
```

### Phase 3: Mixed AND/OR with DNF (2-3 days)

**Goal**: Handle arbitrary boolean combinations

#### 3.1 Convert to Disjunctive Normal Form

```rust
/// Convert predicate to DNF: (A ∧ B) ∨ (C ∧ D) ∨ ...
fn convert_to_dnf(expr: &dyn PhysicalExpr) -> DnfExpression {
    // Each inner Vec is a conjunction (AND)
    // Outer Vec represents disjunction (OR)
    struct DnfExpression {
        conjunctions: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    }

    // Implementation would use distribution laws:
    // A ∧ (B ∨ C) = (A ∧ B) ∨ (A ∧ C)
    // (A ∨ B) ∧ (C ∨ D) = (A ∧ C) ∨ (A ∧ D) ∨ (B ∧ C) ∨ (B ∧ D)
}
```

#### 3.2 Cache and Combine DNF

```rust
fn get_dnf_bitmap(&self, dnf: &DnfExpression, file_id: u64) -> Option<Arc<BitVec>> {
    let mut disjunction_bitmaps = Vec::new();

    for conjunction in &dnf.conjunctions {
        // Get bitmap for each term in the conjunction
        let mut conjunction_bitmaps = Vec::new();
        for term in conjunction {
            let term_bitmap = self.get_atomic_bitmap(term, file_id)?;
            conjunction_bitmaps.push(term_bitmap);
        }

        // AND all terms in this conjunction
        let conjunction_result = self.combine_with_and(conjunction_bitmaps);
        disjunction_bitmaps.push(conjunction_result);
    }

    // OR all conjunctions
    Some(self.combine_with_or(disjunction_bitmaps))
}
```

## Advanced Features

### NOT Support

Handle negation using De Morgan's laws:

```rust
fn handle_not(expr: &NotExpr) -> DnfExpression {
    // NOT (A AND B) = (NOT A) OR (NOT B)
    // NOT (A OR B) = (NOT A) AND (NOT B)
    // For bitmap: NOT bitmap = !bitmap (bitwise NOT)
}
```

### Cache Entry Metadata

Track dependencies for smart invalidation:

```rust
struct CacheEntryMetadata {
    /// Direct predicate hash
    predicate_hash: u64,
    /// Component hashes this entry depends on
    component_hashes: Vec<u64>,
    /// Whether this is a composed entry
    is_composed: bool,
    /// Statistics for debugging
    composition_depth: u32,
    hit_count: AtomicU64,
}
```

## Memory and Performance Analysis

### Memory Efficiency

Without composition:
- 8 different filter combinations = 8 cache entries
- Memory: 8 × (100 bits for row groups + 1000 bits for pages) = 8800 bits

With composition:
- 3 atomic predicates = 3 cache entries
- Memory: 3 × 1100 bits = 3300 bits
- **Savings: 62.5%**

### Performance Impact

#### Cache Hit Rates

Typical workload simulation:
```
Queries:
1. WHERE a = 1                    (new)
2. WHERE b = 2                    (new)
3. WHERE a = 1 AND b = 2          (composed hit!)
4. WHERE c = 3                    (new)
5. WHERE a = 1 AND c = 3          (composed hit!)
6. WHERE b = 2 AND c = 3          (composed hit!)
7. WHERE a = 1 AND b = 2 AND c = 3 (composed hit!)

Traditional cache: 7 misses (0% hit rate)
Composition cache: 3 misses, 4 hits (57% hit rate)
```

#### Composition Overhead

- Bitwise AND/OR operations: ~1ns per 64 bits
- For 100 row groups: ~2ns overhead
- For 1000 pages: ~16ns overhead
- Total composition overhead: < 20ns (negligible)

## Integration Points

### 1. Modify `row_group_filter.rs`

```rust
impl RowGroupAccessPlanFilter {
    pub fn prune_by_statistics_with_cache(
        &mut self,
        cache: Option<Arc<ParquetConditionCache>>,
        // ...
    ) {
        if let Some(cache) = cache {
            // Use new composition-aware method
            if let Some(bitmap) = cache.get_or_compose_row_group_bitmap(&key, predicate) {
                self.apply_cached_bitmap(bitmap);
                return;
            }
        }
        // ... existing pruning logic
    }
}
```

### 2. Modify `page_filter.rs`

Similar changes for page-level cache with composition support.

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_and_composition() {
    let cache = ParquetConditionCache::new(config, &metrics);

    // Cache individual components
    cache.put_row_group_bitmap(key_a1, bitmap_a1);
    cache.put_row_group_bitmap(key_b2, bitmap_b2);

    // Query for AND combination
    let composed = cache.get_or_compose_row_group_bitmap(&key_composite, &expr_a1_and_b2);
    assert!(composed.is_some());
    assert_eq!(composed.unwrap(), expected_and_result);
}
```

### Benchmarks

```rust
#[bench]
fn bench_composition_overhead(b: &mut Bencher) {
    // Measure overhead of composition vs direct cache hit
}

#[bench]
fn bench_memory_savings(b: &mut Bencher) {
    // Compare memory usage with/without composition
}
```

## Rollout Plan

1. **Phase 1 (Week 1)**: Implement AND composition, test in development
2. **Phase 2 (Week 2)**: Add OR support, extensive testing
3. **Phase 3 (Week 3-4)**: Full DNF support if needed
4. **Monitoring**: Track composition rates, memory usage, performance impact

## Success Metrics

- Cache hit rate improvement: Target 3x increase (20% → 60%)
- Memory efficiency: Target 50% reduction in cache memory usage
- Query latency: No regression (composition overhead < 1µs)
- Code complexity: Keep under 500 additional lines

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Composition overhead exceeds benefits | Add heuristics to avoid composition for simple queries |
| Memory tracking becomes complex | Use simple reference counting for shared components |
| DNF conversion causes exponential blowup | Limit DNF conversion depth, fall back to direct caching |
| Cache invalidation complexity | Version components independently, use generation counters |

## Future Enhancements

1. **Adaptive Composition**: Learn which predicates are worth decomposing based on usage patterns
2. **Predicate Subsumption**: Recognize when one predicate implies another (e.g., `a > 10` subsumes `a > 5`)
3. **Cross-File Sharing**: Share atomic predicate results across files with same schema
4. **Statistics-Based Estimation**: Estimate bitmap results without evaluation using statistics
5. **Compression**: Use run-length encoding for sparse bitmaps

## Conclusion

Implementing bitmap algebra for the predicate cache would provide significant benefits:
- **3x higher cache hit rates** through component reuse
- **50% memory savings** by eliminating redundant storage
- **Negligible overhead** (<1µs per query)
- **Future-proof architecture** for advanced optimizations

The phased approach allows incremental delivery of value while managing complexity. Phase 1 alone (AND decomposition) would provide immediate benefits for most workloads.