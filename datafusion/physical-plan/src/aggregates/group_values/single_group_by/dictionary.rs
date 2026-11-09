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

use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::{Arc, Weak};

use crate::aggregates::AGGREGATION_HASH_SEED;
use crate::aggregates::group_values::GroupValues;

use ahash::RandomState;
use arrow::array::{
    Array, ArrayRef, AsArray, DictionaryArray, PrimitiveArray, StringArray,
};
use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{ArrowDictionaryKeyType, ArrowNativeType, DataType};
use datafusion_common::{Result, exec_err};
use datafusion_execution::memory_pool::proxy::HashTableAllocExt;
use datafusion_expr::EmitTo;
use hashbrown::hash_table::HashTable;

const UNRESOLVED: usize = usize::MAX;

/// A [`GroupValues`] for a single `Dictionary(K, Utf8)` group column.
///
/// Rows of one batch usually reference far fewer distinct values than there
/// are rows, and batches downstream of `take`, `filter` or repartitioning
/// usually share one values array. So instead of hashing every row, each
/// dictionary value is resolved to a group once (on first use) and rows map
/// to groups through their key. The mapping is kept while later batches carry
/// the same values array.
///
/// Emits the same array as `GroupValuesColumn` does for this column: a
/// `DictionaryArray<K>` whose values are the non-null groups in group order,
/// with a null key for the null group.
pub struct GroupValuesDictionary<K: ArrowDictionaryKeyType> {
    store: GroupStore,
    /// `value_to_group[i]` is the group of `cached_values[i]`, or
    /// [`UNRESOLVED`] if no row has referenced it yet.
    value_to_group: Vec<usize>,
    /// Weak reference to the values array that `value_to_group` was built
    /// for. A matching live array reuses the mapping without retaining an
    /// input dictionary after its batch is released.
    cached_values: Option<Weak<dyn Array>>,
    _phantom: PhantomData<fn() -> K>,
}

/// Distinct group values, deduplicated by their bytes.
struct GroupStore {
    /// `(hash, group)` for every non-null group.
    map: HashTable<(u64, usize)>,
    map_size: usize,
    /// Bytes of every group in group order. The null group's range is empty,
    /// so this is also the concatenation of the non-null groups' values.
    buffer: Vec<u8>,
    /// Group `g` is `buffer[offsets[g]..offsets[g + 1]]`.
    offsets: Vec<usize>,
    null_group: Option<usize>,
    random_state: RandomState,
}

impl GroupStore {
    fn new() -> Self {
        Self {
            map: HashTable::new(),
            map_size: 0,
            buffer: Vec::new(),
            offsets: vec![0],
            null_group: None,
            random_state: AGGREGATION_HASH_SEED,
        }
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn null_group(&mut self) -> usize {
        if let Some(g) = self.null_group {
            return g;
        }
        let g = self.len();
        self.offsets.push(self.buffer.len());
        self.null_group = Some(g);
        g
    }

    /// Kept out of line so the per-row loop in `intern` stays small: it runs
    /// once per distinct value per values array, not once per row.
    #[inline(never)]
    fn resolve(&mut self, values: &StringArray, val_idx: usize) -> usize {
        if values.is_null(val_idx) {
            return self.null_group();
        }
        let bytes = values.value(val_idx).as_bytes();
        let hash = self.random_state.hash_one(bytes);
        let buffer = &self.buffer;
        let offsets = &self.offsets;
        if let Some(&(_, g)) = self.map.find(hash, |&(h, g)| {
            h == hash && &buffer[offsets[g]..offsets[g + 1]] == bytes
        }) {
            return g;
        }
        let g = self.len();
        self.buffer.extend_from_slice(bytes);
        self.offsets.push(self.buffer.len());
        self.map
            .insert_accounted((hash, g), |&(h, _)| h, &mut self.map_size);
        g
    }

    fn non_null_len(&self) -> usize {
        self.len() - self.null_group.is_some() as usize
    }
}

impl<K: ArrowDictionaryKeyType> GroupValuesDictionary<K> {
    pub fn new() -> Self {
        Self {
            store: GroupStore::new(),
            value_to_group: Vec::new(),
            cached_values: None,
            _phantom: PhantomData,
        }
    }

    fn is_cached(&self, values: &ArrayRef) -> bool {
        self.cached_values
            .as_ref()
            .is_some_and(|cached| std::ptr::addr_eq(cached.as_ptr(), Arc::as_ptr(values)))
    }

    fn sync_value_cache(&mut self, values: &ArrayRef) {
        if self.is_cached(values) {
            return;
        }
        self.value_to_group.clear();
        self.value_to_group.resize(values.len(), UNRESOLVED);
        self.cached_values = Some(Arc::downgrade(values));
    }

    fn drop_value_cache(&mut self) {
        self.cached_values = None;
        self.value_to_group.clear();
    }

    /// Same limit and message as `DictionaryGroupValuesColumn`, so both paths
    /// fail on the same input.
    fn check_key_overflow(&self) -> Result<()> {
        let num_values = self.store.non_null_len();
        let max: usize = match K::DATA_TYPE {
            DataType::Int8 => i8::MAX as usize,
            DataType::Int16 => i16::MAX as usize,
            DataType::Int32 => i32::MAX as usize,
            DataType::Int64 => i64::MAX as usize,
            DataType::UInt8 => u8::MAX as usize,
            DataType::UInt16 => u16::MAX as usize,
            DataType::UInt32 => u32::MAX as usize,
            DataType::UInt64 => usize::MAX,
            _ => 0,
        };
        if num_values == 0 || num_values - 1 <= max {
            return Ok(());
        }
        exec_err!(
            "Dictionary key type {:?} cannot represent {} distinct values",
            K::DATA_TYPE,
            num_values
        )
    }

    /// Builds the dictionary for groups `0..n`.
    fn build(&self, n: usize) -> Result<ArrayRef> {
        let store = &self.store;
        let end = store.offsets[n];
        if i32::try_from(end).is_err() {
            return exec_err!("Utf8 group values exceed {} bytes", i32::MAX);
        }
        let null_group = store.null_group.filter(|&g| g < n);

        let keys: PrimitiveArray<K> = match null_group {
            None => PrimitiveArray::from_iter_values((0..n).map(K::Native::usize_as)),
            Some(p) => (0..n)
                .map(|g| match g.cmp(&p) {
                    std::cmp::Ordering::Less => Some(K::Native::usize_as(g)),
                    std::cmp::Ordering::Equal => None,
                    std::cmp::Ordering::Greater => Some(K::Native::usize_as(g - 1)),
                })
                .collect(),
        };

        // The null group's range is empty, so dropping its end offset leaves
        // valid offsets for the non-null groups over the same bytes.
        let offsets: Vec<i32> = store.offsets[..=n]
            .iter()
            .enumerate()
            .filter(|&(i, _)| null_group.is_none_or(|p| i != p + 1))
            .map(|(_, &o)| o as i32)
            .collect();
        let bytes = Buffer::from(&store.buffer[..end]);
        // SAFETY: every group's bytes were copied whole from a valid Utf8
        // value, and the offsets are monotonic and end at `bytes.len()`.
        let values = unsafe {
            StringArray::new_unchecked(
                OffsetBuffer::new_unchecked(ScalarBuffer::from(offsets)),
                bytes,
                None,
            )
        };
        Ok(Arc::new(DictionaryArray::<K>::new(keys, Arc::new(values))))
    }

    /// Drops groups `0..n` and renumbers the rest from 0.
    fn remove_first(&mut self, n: usize) {
        let store = &mut self.store;
        let base = store.offsets[n];
        store.buffer.drain(..base);
        store.offsets.drain(..n);
        store.offsets.iter_mut().for_each(|o| *o -= base);
        store.null_group = store.null_group.and_then(|g| g.checked_sub(n));
        store.map.retain(|(_, g)| match g.checked_sub(n) {
            Some(shifted) => {
                *g = shifted;
                true
            }
            None => false,
        });
        for g in self.value_to_group.iter_mut() {
            if *g != UNRESOLVED {
                *g = g.checked_sub(n).unwrap_or(UNRESOLVED);
            }
        }
    }
}

#[inline(always)]
fn lookup(
    store: &mut GroupStore,
    value_to_group: &mut [usize],
    values: &StringArray,
    val_idx: usize,
) -> usize {
    let g = value_to_group[val_idx];
    if g != UNRESOLVED {
        return g;
    }
    let g = store.resolve(values, val_idx);
    value_to_group[val_idx] = g;
    g
}

impl<K: ArrowDictionaryKeyType> Default for GroupValuesDictionary<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: ArrowDictionaryKeyType> GroupValues for GroupValuesDictionary<K> {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);
        let dict = cols[0].as_dictionary::<K>();
        let values = dict.values();
        self.sync_value_cache(values);
        let strings = values.as_string::<i32>();
        let keys = dict.keys();
        let raw_keys = keys.values();

        groups.clear();
        groups.reserve(raw_keys.len());
        let store = &mut self.store;
        let value_to_group = self.value_to_group.as_mut_slice();
        match keys.nulls().filter(|n| n.null_count() > 0) {
            None => groups.extend(
                raw_keys
                    .iter()
                    .map(|k| lookup(store, value_to_group, strings, k.as_usize())),
            ),
            Some(nulls) => {
                // A null key's slot need not be a valid values index, so it
                // must not reach `lookup`.
                for (row, k) in raw_keys.iter().enumerate() {
                    groups.push(if nulls.is_null(row) {
                        store.null_group()
                    } else {
                        lookup(store, value_to_group, strings, k.as_usize())
                    });
                }
            }
        }

        self.check_key_overflow()
    }

    fn size(&self) -> usize {
        self.store.map_size
            + self.store.buffer.capacity()
            + self.store.offsets.capacity() * size_of::<usize>()
            + self.value_to_group.capacity() * size_of::<usize>()
            + size_of::<Self>()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let n = match emit_to {
            EmitTo::All => self.len(),
            EmitTo::First(n) => n,
        };
        let output = self.build(n)?;
        if n == self.len() {
            self.store = GroupStore::new();
            self.drop_value_cache();
        } else {
            self.remove_first(n);
        }
        Ok(vec![output])
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        let store = &mut self.store;
        store.map.clear();
        store.map.shrink_to(num_rows, |_| 0);
        store.map_size = store.map.capacity() * size_of::<(u64, usize)>();
        store.buffer.clear();
        store.buffer.shrink_to(num_rows);
        store.offsets.clear();
        store.offsets.shrink_to(num_rows);
        store.offsets.push(0);
        store.null_group = None;
        self.drop_value_cache();
        self.value_to_group.shrink_to(num_rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregates::group_values::multi_group_by::GroupValuesColumn;
    use arrow::array::{Int8Array, Int32Array};
    use arrow::datatypes::{Field, Int8Type, Int32Type, Schema, UInt16Type};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    enum Step {
        Intern(ArrayRef),
        /// `EmitTo::First`, clamped to the current group count; skipped when
        /// there are no groups.
        EmitFirst(usize),
        EmitAll,
    }

    fn strings(values: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(values.to_vec()))
    }

    fn dict_i32(keys: &[Option<i32>], values: &ArrayRef) -> ArrayRef {
        Arc::new(DictionaryArray::<Int32Type>::new(
            Int32Array::from(keys.to_vec()),
            Arc::clone(values),
        ))
    }

    fn generic_impls<K: ArrowDictionaryKeyType>() -> Vec<Box<dyn GroupValues>> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "k",
            DataType::Dictionary(Box::new(K::DATA_TYPE), Box::new(DataType::Utf8)),
            true,
        )]));
        vec![
            Box::new(GroupValuesColumn::<false>::try_new(Arc::clone(&schema)).unwrap()),
            Box::new(GroupValuesColumn::<true>::try_new(schema).unwrap()),
        ]
    }

    fn assert_same_dictionary<K: ArrowDictionaryKeyType>(
        ours: &ArrayRef,
        generic: &ArrayRef,
    ) {
        assert_eq!(ours.data_type(), generic.data_type());
        let (ours, generic) = (ours.as_dictionary::<K>(), generic.as_dictionary::<K>());
        assert_eq!(ours.keys(), generic.keys());
        assert_eq!(ours.values().as_ref(), generic.values().as_ref());
    }

    fn emit_both<K: ArrowDictionaryKeyType>(
        ours: &mut GroupValuesDictionary<K>,
        generic: &mut Box<dyn GroupValues>,
        emit_to: EmitTo,
    ) {
        let ours_out = ours.emit(emit_to).unwrap();
        let generic_out = generic.emit(emit_to).unwrap();
        // `GroupValuesColumn` emits no columns at all when nothing was
        // interned since it was last cleared.
        match generic_out.first() {
            Some(generic_out) => assert_same_dictionary::<K>(&ours_out[0], generic_out),
            None => assert!(ours_out[0].is_empty()),
        }
    }

    /// Runs `steps` through this implementation and through both
    /// `GroupValuesColumn` variants, requiring identical group ids, errors and
    /// emitted arrays, then emits whatever is left.
    fn check_against_generic<K: ArrowDictionaryKeyType>(steps: &[Step]) {
        for mut generic in generic_impls::<K>() {
            let mut ours = GroupValuesDictionary::<K>::new();
            for (i, step) in steps.iter().enumerate() {
                match step {
                    Step::Intern(array) => {
                        let (mut ours_groups, mut generic_groups) = (vec![], vec![]);
                        let ours_result =
                            ours.intern(std::slice::from_ref(array), &mut ours_groups);
                        let generic_result = generic
                            .intern(std::slice::from_ref(array), &mut generic_groups);
                        match (ours_result, generic_result) {
                            (Ok(()), Ok(())) => {
                                assert_eq!(ours_groups, generic_groups, "step {i}")
                            }
                            (Err(a), Err(b)) => {
                                assert_eq!(a.strip_backtrace(), b.strip_backtrace());
                                return;
                            }
                            (a, b) => panic!("step {i}: ours {a:?}, generic {b:?}"),
                        }
                    }
                    Step::EmitFirst(n) if !ours.is_empty() => {
                        let n = (*n).min(ours.len());
                        emit_both(&mut ours, &mut generic, EmitTo::First(n));
                    }
                    Step::EmitFirst(_) => {}
                    Step::EmitAll => {
                        // `GroupValuesColumn` keeps its hash table after
                        // `EmitTo::All`; the aggregate stream only interns
                        // again after `clear_shrink`, as it does on spill.
                        emit_both(&mut ours, &mut generic, EmitTo::All);
                        ours.clear_shrink(8);
                        generic.clear_shrink(8);
                    }
                }
                assert_eq!(ours.len(), generic.len(), "step {i}");
                assert_eq!(ours.is_empty(), generic.is_empty(), "step {i}");
            }
            emit_both(&mut ours, &mut generic, EmitTo::All);
            assert!(ours.is_empty());
        }
    }

    #[test]
    fn value_cache_does_not_retain_dictionary_values() {
        let mut gv = GroupValuesDictionary::<Int32Type>::new();
        let values = strings(&[Some("retained"), Some("other")]);
        let weak_values = Arc::downgrade(&values);
        let input = dict_i32(&[Some(0), Some(0)], &values);
        let mut groups = Vec::new();
        gv.intern(&[Arc::clone(&input)], &mut groups).unwrap();
        assert!(gv.is_cached(&values));

        // The cache is only an identity hint: it must not keep an input
        // dictionary allocation alive after its batch is released.
        drop(values);
        drop(input);
        assert!(weak_values.upgrade().is_none());

        let next = dict_i32(
            &[Some(1), Some(0)],
            &strings(&[Some("new"), Some("retained")]),
        );
        gv.intern(&[next], &mut groups).unwrap();
        assert_eq!(groups, vec![0, 1]);
    }

    #[test]
    fn shared_dictionary_with_unused_values_matches_generic() {
        let values = strings(&[Some("a"), Some("b"), Some("c"), Some("d"), Some("e")]);
        check_against_generic::<Int32Type>(&[
            Step::Intern(dict_i32(&[Some(1), Some(3), Some(1), Some(1)], &values)),
            Step::Intern(dict_i32(&[Some(0), Some(3), Some(3), Some(1)], &values)),
        ]);
    }

    #[test]
    fn repeated_values_across_dictionaries_match_generic() {
        let first = strings(&[Some("x"), Some("y")]);
        let second = strings(&[Some("y"), Some("z"), Some("x"), Some("")]);
        let duplicated = strings(&[Some("z"), Some("z"), Some("x")]);
        check_against_generic::<Int32Type>(&[
            Step::Intern(dict_i32(&[Some(1), Some(0), Some(1)], &first)),
            Step::Intern(dict_i32(&[Some(3), Some(2), Some(1), Some(0)], &second)),
            Step::Intern(dict_i32(&[Some(1), Some(0), Some(2)], &duplicated)),
            Step::Intern(dict_i32(&[Some(0), Some(1)], &first)),
        ]);
    }

    #[test]
    fn null_keys_and_null_values_share_null_group() {
        let first = strings(&[Some("a"), None, Some("b")]);
        let second = strings(&[None, Some("c")]);
        check_against_generic::<Int32Type>(&[
            Step::Intern(dict_i32(&[Some(0), None, Some(1), Some(2)], &first)),
            Step::Intern(dict_i32(&[Some(1), Some(0), None], &second)),
            Step::Intern(dict_i32(&[None, None], &first)),
        ]);
    }

    #[test]
    fn null_key_first_matches_generic() {
        let values = strings(&[Some("a"), Some("b")]);
        check_against_generic::<Int32Type>(&[
            Step::Intern(dict_i32(&[None, Some(1), None, Some(0)], &values)),
            Step::Intern(dict_i32(&[Some(1), None], &values)),
        ]);
    }

    #[test]
    fn emit_first_n_renumbers_groups_and_value_cache() {
        let values = strings(&[Some("a"), None, Some("b"), Some("c"), Some("d")]);
        check_against_generic::<Int32Type>(&[
            Step::Intern(dict_i32(&[Some(2), Some(1), Some(0), Some(3)], &values)),
            Step::EmitFirst(1),
            Step::Intern(dict_i32(&[Some(3), Some(2), Some(4), Some(1)], &values)),
            Step::EmitFirst(2),
            Step::Intern(dict_i32(&[Some(1), Some(0), Some(2), None], &values)),
            Step::EmitFirst(3),
            Step::Intern(dict_i32(&[Some(4), Some(0)], &values)),
        ]);
    }

    #[test]
    fn clear_after_emit_all_then_reuse_shared_dictionary() {
        let values = strings(&[Some("a"), Some("b"), None]);
        check_against_generic::<Int32Type>(&[
            Step::Intern(dict_i32(&[Some(1), Some(2), Some(0)], &values)),
            Step::EmitAll,
            Step::Intern(dict_i32(&[Some(0), Some(2), Some(1)], &values)),
        ]);
    }

    #[test]
    fn emit_all_restarts_group_ids_for_shared_dictionary() {
        let values = strings(&[Some("a"), Some("b"), None]);
        let batch = dict_i32(&[Some(1), Some(2), Some(0), Some(1)], &values);
        let mut groups = GroupValuesDictionary::<Int32Type>::new();
        let mut ids = vec![];
        groups
            .intern(std::slice::from_ref(&batch), &mut ids)
            .unwrap();
        groups
            .intern(std::slice::from_ref(&batch), &mut ids)
            .unwrap();
        assert_eq!(ids, vec![0, 1, 2, 0]);
        groups.emit(EmitTo::All).unwrap();
        let batch = dict_i32(&[Some(2), Some(0)], &values);
        groups
            .intern(std::slice::from_ref(&batch), &mut ids)
            .unwrap();
        assert_eq!(ids, vec![0, 1]);
        let out = groups.emit(EmitTo::All).unwrap();
        let expected = dict_i32(&[None, Some(0)], &strings(&[Some("a")]));
        assert_same_dictionary::<Int32Type>(&out[0], &expected);
    }

    #[test]
    fn empty_batches_match_generic() {
        let values = strings(&[Some("a")]);
        let empty = strings(&[]);
        check_against_generic::<Int32Type>(&[
            Step::Intern(dict_i32(&[], &empty)),
            Step::Intern(dict_i32(&[], &values)),
            Step::Intern(dict_i32(&[Some(0)], &values)),
            Step::Intern(dict_i32(&[], &empty)),
        ]);
    }

    #[test]
    fn int8_key_overflow_fails_like_generic() {
        let batch = |range: std::ops::Range<usize>| -> ArrayRef {
            let values: ArrayRef = Arc::new(StringArray::from_iter_values(
                range.map(|i| format!("v{i}")),
            ));
            let keys = Int8Array::from_iter_values(0..values.len() as i8);
            Arc::new(DictionaryArray::<Int8Type>::new(keys, values))
        };
        check_against_generic::<Int8Type>(&[
            Step::Intern(batch(0..100)),
            Step::Intern(batch(100..127)),
            Step::Intern(batch(127..129)),
        ]);
    }

    /// Batches that switch between reusing the previous values array and
    /// building a new one, with null keys, null and empty values, duplicate
    /// and unused values, and emits in between.
    fn random_steps<K: ArrowDictionaryKeyType>(
        rng: &mut StdRng,
        vocab_len: usize,
    ) -> Vec<Step> {
        let vocab: Vec<String> = (0..vocab_len).map(|i| format!("v{i}")).collect();
        let mut steps = vec![];
        let mut values: Option<ArrayRef> = None;
        for _ in 0..40 {
            if values.is_none() || rng.random_bool(0.4) {
                let n = rng.random_range(0..30);
                values = Some(Arc::new(StringArray::from_iter((0..n).map(|_| {
                    match rng.random_range(0..10) {
                        0 => None,
                        1 => Some(String::new()),
                        _ => Some(vocab[rng.random_range(0..vocab_len)].clone()),
                    }
                }))));
            }
            let values = values.as_ref().unwrap();
            let rows = rng.random_range(0..40);
            let keys = PrimitiveArray::<K>::from_iter((0..rows).map(|_| {
                if values.is_empty() || rng.random_bool(0.1) {
                    None
                } else {
                    Some(K::Native::usize_as(rng.random_range(0..values.len())))
                }
            }));
            steps.push(Step::Intern(Arc::new(DictionaryArray::<K>::new(
                keys,
                Arc::clone(values),
            ))));
            match rng.random_range(0..10) {
                0 => steps.push(Step::EmitAll),
                1 | 2 => steps.push(Step::EmitFirst(rng.random_range(1..20))),
                _ => {}
            }
        }
        steps
    }

    #[test]
    fn random_batches_match_generic() {
        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            check_against_generic::<Int32Type>(&random_steps::<Int32Type>(&mut rng, 60));
            check_against_generic::<UInt16Type>(&random_steps::<UInt16Type>(&mut rng, 8));
            check_against_generic::<Int8Type>(&random_steps::<Int8Type>(&mut rng, 20));
        }
    }
}
