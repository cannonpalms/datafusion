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

use std::any::Any;
use std::borrow::Cow;
use std::fmt::Debug;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use arrow::array::{Array, AsArray, BooleanArray, Float16Array, ListArray};
use arrow::buffer::OffsetBuffer;
use arrow::compute::{filter, is_not_null};
use arrow::datatypes::FieldRef;
use arrow::datatypes::{Float64Type, UInt64Type};
use arrow::{
    array::{
        ArrayRef, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
        Int64Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, Field},
};
use datafusion_common::{
    DataFusionError, Result, ScalarValue, downcast_value, internal_err, not_impl_err,
    plan_err,
};
use datafusion_expr::expr::{AggregateFunction, Sort};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::type_coercion::aggregates::{INTEGERS, NUMERICS};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, EmitTo, Expr, GroupsAccumulator,
    Signature, TypeSignature, Volatility,
};
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::nulls::filtered_null_mask;
use datafusion_functions_aggregate_common::tdigest::{DEFAULT_MAX_SIZE, TDigest};
use datafusion_macros::user_doc;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

use crate::utils::{get_scalar_value, validate_percentile_expr};

create_func!(ApproxPercentileCont, approx_percentile_cont_udaf);

/// Computes the approximate percentile continuous of a set of numbers
pub fn approx_percentile_cont(
    order_by: Sort,
    percentile: Expr,
    centroids: Option<Expr>,
) -> Expr {
    let expr = order_by.expr.clone();

    let args = if let Some(centroids) = centroids {
        vec![expr, percentile, centroids]
    } else {
        vec![expr, percentile]
    };

    Expr::AggregateFunction(AggregateFunction::new_udf(
        approx_percentile_cont_udaf(),
        args,
        false,
        None,
        vec![order_by],
        None,
    ))
}

#[user_doc(
    doc_section(label = "Approximate Functions"),
    description = "Returns the approximate percentile of input values using the t-digest algorithm.",
    syntax_example = "approx_percentile_cont(percentile [, centroids]) WITHIN GROUP (ORDER BY expression)",
    sql_example = r#"```sql
> SELECT approx_percentile_cont(0.75) WITHIN GROUP (ORDER BY column_name) FROM table_name;
+------------------------------------------------------------------+
| approx_percentile_cont(0.75) WITHIN GROUP (ORDER BY column_name) |
+------------------------------------------------------------------+
| 65.0                                                             |
+------------------------------------------------------------------+
> SELECT approx_percentile_cont(0.75, 100) WITHIN GROUP (ORDER BY column_name) FROM table_name;
+-----------------------------------------------------------------------+
| approx_percentile_cont(0.75, 100) WITHIN GROUP (ORDER BY column_name) |
+-----------------------------------------------------------------------+
| 65.0                                                                  |
+-----------------------------------------------------------------------+
```
An alternate syntax is also supported:
```sql
> SELECT approx_percentile_cont(column_name, 0.75) FROM table_name;
+-----------------------------------------------+
| approx_percentile_cont(column_name, 0.75)     |
+-----------------------------------------------+
| 65.0                                          |
+-----------------------------------------------+

> SELECT approx_percentile_cont(column_name, 0.75, 100) FROM table_name;
+----------------------------------------------------------+
| approx_percentile_cont(column_name, 0.75, 100)           |
+----------------------------------------------------------+
| 65.0                                                     |
+----------------------------------------------------------+
```
"#,
    standard_argument(name = "expression",),
    argument(
        name = "percentile",
        description = "Percentile to compute. Must be a float value between 0 and 1 (inclusive)."
    ),
    argument(
        name = "centroids",
        description = "Number of centroids to use in the t-digest algorithm. _Default is 100_. A higher number results in more accurate approximation but requires more memory."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ApproxPercentileCont {
    signature: Signature,
}

impl Default for ApproxPercentileCont {
    fn default() -> Self {
        Self::new()
    }
}

impl ApproxPercentileCont {
    /// Create a new [`ApproxPercentileCont`] aggregate function.
    pub fn new() -> Self {
        let mut variants = Vec::with_capacity(NUMERICS.len() * (INTEGERS.len() + 1));
        // Accept any numeric value paired with a float64 percentile
        for num in NUMERICS {
            variants.push(TypeSignature::Exact(vec![num.clone(), DataType::Float64]));
            // Additionally accept an integer number of centroids for T-Digest
            for int in INTEGERS {
                variants.push(TypeSignature::Exact(vec![
                    num.clone(),
                    DataType::Float64,
                    int.clone(),
                ]))
            }
        }
        Self {
            signature: Signature::one_of(variants, Volatility::Immutable),
        }
    }

    pub(crate) fn create_accumulator(
        &self,
        args: &AccumulatorArgs,
    ) -> Result<ApproxPercentileAccumulator> {
        let (percentile, max_size, data_type) = Self::accumulator_params(args)?;
        Ok(ApproxPercentileAccumulator::new_with_max_size(
            percentile, data_type, max_size,
        ))
    }

    /// The percentile (already flipped for a descending `WITHIN GROUP`
    /// order), the digest size and the input type, shared by the
    /// [`Accumulator`] and the [`GroupsAccumulator`] so both see the same
    /// arguments.
    fn accumulator_params(args: &AccumulatorArgs) -> Result<(f64, usize, DataType)> {
        let percentile =
            validate_percentile_expr(&args.exprs[1], "APPROX_PERCENTILE_CONT")?;

        let is_descending = args
            .order_bys
            .first()
            .map(|sort_expr| sort_expr.options.descending)
            .unwrap_or(false);

        let percentile = if is_descending {
            1.0 - percentile
        } else {
            percentile
        };

        let tdigest_max_size = if args.exprs.len() == 3 {
            validate_input_max_size_expr(&args.exprs[2])?
        } else {
            DEFAULT_MAX_SIZE
        };

        let data_type = args.expr_fields[0].data_type();
        if !is_supported_input_type(data_type) {
            return not_impl_err!(
                "Support for 'APPROX_PERCENTILE_CONT' for data type {data_type} is not implemented"
            );
        }

        Ok((percentile, tdigest_max_size, data_type.clone()))
    }
}

pub(crate) fn is_supported_input_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
    )
}

fn validate_input_max_size_expr(expr: &Arc<dyn PhysicalExpr>) -> Result<usize> {
    let scalar_value = get_scalar_value(expr).map_err(|_e| {
        DataFusionError::Plan(
            "Tdigest max_size value for 'APPROX_PERCENTILE_CONT' must be a literal"
                .to_string(),
        )
    })?;

    let max_size = match scalar_value {
        ScalarValue::UInt8(Some(q)) => q as usize,
        ScalarValue::UInt16(Some(q)) => q as usize,
        ScalarValue::UInt32(Some(q)) => q as usize,
        ScalarValue::UInt64(Some(q)) => q as usize,
        ScalarValue::Int32(Some(q)) if q > 0 => q as usize,
        ScalarValue::Int64(Some(q)) if q > 0 => q as usize,
        ScalarValue::Int16(Some(q)) if q > 0 => q as usize,
        ScalarValue::Int8(Some(q)) if q > 0 => q as usize,
        sv => {
            return plan_err!(
                "Tdigest max_size value for 'APPROX_PERCENTILE_CONT' must be UInt > 0 literal (got data type {}).",
                sv.data_type()
            );
        }
    };

    Ok(max_size)
}

impl AggregateUDFImpl for ApproxPercentileCont {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// See [`TDigest::to_scalar_state()`] for a description of the serialized
    /// state.
    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new(
                format_state_name(args.name, "max_size"),
                DataType::UInt64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "sum"),
                DataType::Float64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "count"),
                DataType::Float64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "max"),
                DataType::Float64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "min"),
                DataType::Float64,
                false,
            ),
            Field::new_list(
                format_state_name(args.name, "centroids"),
                Field::new_list_field(DataType::Float64, true),
                false,
            ),
        ]
        .into_iter()
        .map(Arc::new)
        .collect())
    }

    fn name(&self) -> &str {
        "approx_percentile_cont"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    #[inline]
    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(self.create_accumulator(&acc_args)?))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        !args.is_distinct && is_supported_input_type(args.expr_fields[0].data_type())
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        let (percentile, max_size, data_type) = Self::accumulator_params(&args)?;
        Ok(Box::new(ApproxPercentileGroupsAccumulator::new(
            percentile, data_type, max_size,
        )))
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if !arg_types[0].is_numeric() {
            return plan_err!("approx_percentile_cont requires numeric input types");
        }
        if arg_types.len() == 3 && !arg_types[2].is_integer() {
            return plan_err!(
                "approx_percentile_cont requires integer centroids input types"
            );
        }
        Ok(arg_types[0].clone())
    }

    fn supports_within_group_clause(&self) -> bool {
        true
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[derive(Debug)]
pub struct ApproxPercentileAccumulator {
    digest: TDigest,
    percentile: f64,
    return_type: DataType,
}

impl ApproxPercentileAccumulator {
    pub fn new(percentile: f64, return_type: DataType) -> Self {
        Self {
            digest: TDigest::new(DEFAULT_MAX_SIZE),
            percentile,
            return_type,
        }
    }

    pub fn new_with_max_size(
        percentile: f64,
        return_type: DataType,
        max_size: usize,
    ) -> Self {
        Self {
            digest: TDigest::new(max_size),
            percentile,
            return_type,
        }
    }

    // pub(crate) for approx_percentile_cont_with_weight
    pub(crate) fn max_size(&self) -> usize {
        self.digest.max_size()
    }

    // pub(crate) for approx_percentile_cont_with_weight
    pub(crate) fn merge_digests(&mut self, digests: &[TDigest]) {
        let digests = digests.iter().chain(std::iter::once(&self.digest));
        self.digest = TDigest::merge_digests(digests)
    }

    // pub(crate) for approx_percentile_cont_with_weight
    pub(crate) fn convert_to_float(values: &ArrayRef) -> Result<Vec<f64>> {
        debug_assert!(
            values.null_count() == 0,
            "convert_to_float assumes nulls have already been filtered out"
        );
        match values.data_type() {
            DataType::Float64 => {
                let array = downcast_value!(values, Float64Array);
                Ok(array.values().iter().copied().collect::<Vec<_>>())
            }
            DataType::Float32 => {
                let array = downcast_value!(values, Float32Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            DataType::Float16 => {
                let array = downcast_value!(values, Float16Array);
                Ok(array
                    .values()
                    .iter()
                    .map(|v| v.to_f64())
                    .collect::<Vec<_>>())
            }
            DataType::Int64 => {
                let array = downcast_value!(values, Int64Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            DataType::Int32 => {
                let array = downcast_value!(values, Int32Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            DataType::Int16 => {
                let array = downcast_value!(values, Int16Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            DataType::Int8 => {
                let array = downcast_value!(values, Int8Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            DataType::UInt64 => {
                let array = downcast_value!(values, UInt64Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            DataType::UInt32 => {
                let array = downcast_value!(values, UInt32Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            DataType::UInt16 => {
                let array = downcast_value!(values, UInt16Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            DataType::UInt8 => {
                let array = downcast_value!(values, UInt8Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            e => internal_err!(
                "APPROX_PERCENTILE_CONT is not expected to receive the type {e:?}"
            ),
        }
    }
}

impl Accumulator for ApproxPercentileAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(self.digest.to_scalar_state().into_iter().collect())
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        // Remove any nulls before computing the percentile
        let mut values = Arc::clone(&values[0]);
        if values.null_count() > 0 {
            values = filter(&values, &is_not_null(&values)?)?;
        }
        let sorted_values = &arrow::compute::sort(&values, None)?;
        let sorted_values = ApproxPercentileAccumulator::convert_to_float(sorted_values)?;
        self.digest = self.digest.merge_sorted_f64(&sorted_values);
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.digest.count() == 0.0 {
            return ScalarValue::try_from(self.return_type.clone());
        }
        let q = self.digest.estimate_quantile(self.percentile);

        // These acceptable return types MUST match the validation in
        // ApproxPercentile::create_accumulator.
        Ok(match &self.return_type {
            DataType::Int8 => ScalarValue::Int8(Some(q as i8)),
            DataType::Int16 => ScalarValue::Int16(Some(q as i16)),
            DataType::Int32 => ScalarValue::Int32(Some(q as i32)),
            DataType::Int64 => ScalarValue::Int64(Some(q as i64)),
            DataType::UInt8 => ScalarValue::UInt8(Some(q as u8)),
            DataType::UInt16 => ScalarValue::UInt16(Some(q as u16)),
            DataType::UInt32 => ScalarValue::UInt32(Some(q as u32)),
            DataType::UInt64 => ScalarValue::UInt64(Some(q as u64)),
            DataType::Float16 => ScalarValue::Float16(Some(half::f16::from_f64(q))),
            DataType::Float32 => ScalarValue::Float32(Some(q as f32)),
            DataType::Float64 => ScalarValue::Float64(Some(q)),
            v => unreachable!("unexpected return type {}", v),
        })
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }

        let states = (0..states[0].len())
            .map(|index| {
                states
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, index))
                    .collect::<Result<Vec<_>>>()
                    .map(|state| TDigest::from_scalar_state(&state))
            })
            .collect::<Result<Vec<_>>>()?;

        self.merge_digests(&states);

        Ok(())
    }

    fn size(&self) -> usize {
        size_of_val(self) + self.digest.size() - size_of_val(&self.digest)
            + self.return_type.size()
            - size_of_val(&self.return_type)
    }
}

/// [`GroupsAccumulator`] for `approx_percentile_cont` and `approx_median`,
/// with one [`TDigest`] per group.
///
/// It produces the same digests as [`GroupsAccumulatorAdapter`] wrapping
/// [`ApproxPercentileAccumulator`]: each batch's values for a group are
/// sorted and merged into that group's digest with one
/// [`TDigest::merge_sorted_f64`], and `merge_batch` merges a group's incoming
/// states, in row order, ahead of its own digest with one
/// [`TDigest::merge_digests`]. Where the digests merge decides the result, so
/// both stay per batch even though buffering values across batches would
/// call `merge_sorted_f64` less often.
///
/// What it saves over the adapter: the per-group `take`, slice, sort kernel
/// and `Vec<f64>` copies, the walk over every group on every batch, and the
/// [`ScalarValue`] round trip of every state row. Values are bucketed by
/// group with a counting sort and each bucket is sorted as `i64` keys that
/// order exactly as [`f64::total_cmp`], which the sort kernel uses.
///
/// [`GroupsAccumulatorAdapter`]: datafusion_functions_aggregate_common::aggregate::groups_accumulator::GroupsAccumulatorAdapter
#[derive(Debug)]
pub(crate) struct ApproxPercentileGroupsAccumulator {
    percentile: f64,
    return_type: DataType,
    max_size: usize,
    digests: Vec<TDigest>,
    /// Sum of the heap bytes of `digests`, kept up to date so that `size()`
    /// does not walk every group.
    digests_heap_bytes: usize,
    /// Per-group row counts for the batch in hand, then bucket write
    /// positions. Zero for every group between batches.
    counts: Vec<u32>,
    /// Groups the batch in hand has rows for, in order of first appearance.
    touched: Vec<usize>,
    /// Start of each touched group's bucket in `keys` or `rows`.
    starts: Vec<u32>,
    keys: Vec<i64>,
    rows: Vec<u32>,
    sorted: Vec<f64>,
    /// One group's incoming digests in `merge_batch`. Empty between calls.
    incoming: Vec<TDigest>,
}

impl ApproxPercentileGroupsAccumulator {
    pub(crate) fn new(percentile: f64, return_type: DataType, max_size: usize) -> Self {
        Self {
            percentile,
            return_type,
            max_size,
            digests: vec![],
            digests_heap_bytes: 0,
            counts: vec![],
            touched: vec![],
            starts: vec![],
            keys: vec![],
            rows: vec![],
            sorted: vec![],
            incoming: vec![],
        }
    }

    fn ensure_groups(&mut self, total_num_groups: usize) {
        if self.digests.len() < total_num_groups {
            self.digests
                .resize(total_num_groups, TDigest::new(self.max_size));
        }
        if self.counts.len() < total_num_groups {
            self.counts.resize(total_num_groups, 0);
        }
    }

    /// Counts a row for `group`, recording the group on its first row.
    #[inline]
    fn count_row(counts: &mut [u32], touched: &mut Vec<usize>, group: usize) {
        let count = &mut counts[group];
        if *count == 0 {
            touched.push(group);
        }
        *count += 1;
    }

    /// Turns the counts of the touched groups into bucket write positions
    /// and returns the total number of rows.
    fn bucket_starts(&mut self) -> usize {
        self.starts.clear();
        let mut offset = 0u32;
        for &group in &self.touched {
            let n = self.counts[group];
            self.starts.push(offset);
            self.counts[group] = offset;
            offset += n;
        }
        offset as usize
    }

    fn replace_digest(&mut self, group: usize, digest: TDigest) {
        let added = heap_bytes(&digest);
        let old = std::mem::replace(&mut self.digests[group], digest);
        self.digests_heap_bytes =
            (self.digests_heap_bytes + added).saturating_sub(heap_bytes(&old));
    }

    fn take_digests(&mut self, emit_to: EmitTo) -> Vec<TDigest> {
        let digests = emit_to.take_needed(&mut self.digests);
        let emitted: usize = digests.iter().map(heap_bytes).sum();
        self.digests_heap_bytes = self.digests_heap_bytes.saturating_sub(emitted);
        digests
    }

    fn state_arrays(digests: &[TDigest]) -> Vec<ArrayRef> {
        let max_size: UInt64Array =
            digests.iter().map(|d| Some(d.max_size() as u64)).collect();
        let sum: Float64Array = digests.iter().map(|d| Some(d.sum())).collect();
        let count: Float64Array = digests.iter().map(|d| Some(d.count())).collect();
        let max: Float64Array = digests.iter().map(|d| Some(d.max())).collect();
        let min: Float64Array = digests.iter().map(|d| Some(d.min())).collect();

        let n_values = digests.iter().map(|d| d.centroids().len() * 2).sum();
        let mut values = Vec::with_capacity(n_values);
        let mut offsets = Vec::with_capacity(digests.len() + 1);
        offsets.push(0i32);
        for d in digests {
            for c in d.centroids() {
                values.push(c.mean());
                values.push(c.weight());
            }
            offsets.push(values.len() as i32);
        }
        let centroids = ListArray::new(
            Arc::new(Field::new_list_field(DataType::Float64, true)),
            OffsetBuffer::new(offsets.into()),
            Arc::new(Float64Array::from(values)),
            None,
        );

        vec![
            Arc::new(max_size),
            Arc::new(sum),
            Arc::new(count),
            Arc::new(max),
            Arc::new(min),
            Arc::new(centroids),
        ]
    }
}

fn heap_bytes(digest: &TDigest) -> usize {
    digest.size() - size_of::<TDigest>()
}

/// Maps `v` to an `i64` whose order is [`f64::total_cmp`]'s order of `v`.
/// The same expression maps the key back.
#[inline]
fn f64_sort_key(v: f64) -> i64 {
    let bits = v.to_bits() as i64;
    bits ^ ((((bits >> 63) as u64) >> 1) as i64)
}

#[inline]
fn f64_from_sort_key(key: i64) -> f64 {
    f64::from_bits((key ^ ((((key >> 63) as u64) >> 1) as i64)) as u64)
}

/// The input values as `f64`, converted as
/// [`ApproxPercentileAccumulator::convert_to_float`] does. Null slots hold
/// arbitrary values.
fn values_as_f64(values: &ArrayRef) -> Result<Cow<'_, [f64]>> {
    macro_rules! convert {
        ($ARRAY:ident) => {{
            let array = downcast_value!(values, $ARRAY);
            Cow::Owned(array.values().iter().map(|v| *v as f64).collect())
        }};
    }
    Ok(match values.data_type() {
        DataType::Float64 => {
            Cow::Borrowed(downcast_value!(values, Float64Array).values().as_ref())
        }
        DataType::Float32 => convert!(Float32Array),
        DataType::Float16 => Cow::Owned(
            downcast_value!(values, Float16Array)
                .values()
                .iter()
                .map(|v| v.to_f64())
                .collect(),
        ),
        DataType::Int64 => convert!(Int64Array),
        DataType::Int32 => convert!(Int32Array),
        DataType::Int16 => convert!(Int16Array),
        DataType::Int8 => convert!(Int8Array),
        DataType::UInt64 => convert!(UInt64Array),
        DataType::UInt32 => convert!(UInt32Array),
        DataType::UInt16 => convert!(UInt16Array),
        DataType::UInt8 => convert!(UInt8Array),
        e => {
            return internal_err!(
                "APPROX_PERCENTILE_CONT is not expected to receive the type {e:?}"
            );
        }
    })
}

impl GroupsAccumulator for ApproxPercentileGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values[0].len(), group_indices.len());
        self.ensure_groups(total_num_groups);

        let nulls = filtered_null_mask(opt_filter, values[0].as_ref())
            .filter(|n| n.null_count() > 0);
        let floats = values_as_f64(&values[0])?;

        // A group whose rows are all null or filtered out is left alone: the
        // adapter's accumulator merges an empty slice, which is a no-op.
        self.touched.clear();
        match &nulls {
            None => {
                for &group in group_indices {
                    Self::count_row(&mut self.counts, &mut self.touched, group);
                }
            }
            Some(nulls) => {
                for row in nulls.valid_indices() {
                    Self::count_row(
                        &mut self.counts,
                        &mut self.touched,
                        group_indices[row],
                    );
                }
            }
        }

        let total = self.bucket_starts();
        self.keys.clear();
        self.keys.resize(total, 0);
        {
            let mut place = |row: usize| {
                let group = group_indices[row];
                let slot = self.counts[group];
                self.keys[slot as usize] = f64_sort_key(floats[row]);
                self.counts[group] = slot + 1;
            };
            match &nulls {
                None => (0..group_indices.len()).for_each(&mut place),
                Some(nulls) => nulls.valid_indices().for_each(&mut place),
            }
        }

        for i in 0..self.touched.len() {
            let group = self.touched[i];
            let start = self.starts[i] as usize;
            let end = self.counts[group] as usize;
            self.counts[group] = 0;

            let bucket = &mut self.keys[start..end];
            bucket.sort_unstable();
            self.sorted.clear();
            self.sorted
                .extend(bucket.iter().map(|&key| f64_from_sort_key(key)));
            let digest = self.digests[group].merge_sorted_f64(&self.sorted);
            self.replace_digest(group, digest);
        }
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let digests = self.take_digests(emit_to);
        let percentile = self.percentile;
        let quantiles = digests
            .iter()
            .map(|d| (d.count() != 0.0).then(|| d.estimate_quantile(percentile)));

        // These casts MUST match ApproxPercentileAccumulator::evaluate.
        let array: ArrayRef = match &self.return_type {
            DataType::Int8 => {
                Arc::new(quantiles.map(|q| q.map(|q| q as i8)).collect::<Int8Array>())
            }
            DataType::Int16 => Arc::new(
                quantiles
                    .map(|q| q.map(|q| q as i16))
                    .collect::<Int16Array>(),
            ),
            DataType::Int32 => Arc::new(
                quantiles
                    .map(|q| q.map(|q| q as i32))
                    .collect::<Int32Array>(),
            ),
            DataType::Int64 => Arc::new(
                quantiles
                    .map(|q| q.map(|q| q as i64))
                    .collect::<Int64Array>(),
            ),
            DataType::UInt8 => Arc::new(
                quantiles
                    .map(|q| q.map(|q| q as u8))
                    .collect::<UInt8Array>(),
            ),
            DataType::UInt16 => Arc::new(
                quantiles
                    .map(|q| q.map(|q| q as u16))
                    .collect::<UInt16Array>(),
            ),
            DataType::UInt32 => Arc::new(
                quantiles
                    .map(|q| q.map(|q| q as u32))
                    .collect::<UInt32Array>(),
            ),
            DataType::UInt64 => Arc::new(
                quantiles
                    .map(|q| q.map(|q| q as u64))
                    .collect::<UInt64Array>(),
            ),
            DataType::Float16 => Arc::new(
                quantiles
                    .map(|q| q.map(half::f16::from_f64))
                    .collect::<Float16Array>(),
            ),
            DataType::Float32 => Arc::new(
                quantiles
                    .map(|q| q.map(|q| q as f32))
                    .collect::<Float32Array>(),
            ),
            DataType::Float64 => Arc::new(quantiles.collect::<Float64Array>()),
            v => return internal_err!("unexpected return type {v}"),
        };
        Ok(array)
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let digests = self.take_digests(emit_to);
        Ok(Self::state_arrays(&digests))
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        assert_eq!(values.len(), 6, "invalid TDigest state");
        assert_eq!(values[0].len(), group_indices.len());
        self.ensure_groups(total_num_groups);

        let max_size = values[0].as_primitive::<UInt64Type>().values();
        let sum = values[1].as_primitive::<Float64Type>().values();
        let count = values[2].as_primitive::<Float64Type>().values();
        let max = values[3].as_primitive::<Float64Type>().values();
        let min = values[4].as_primitive::<Float64Type>().values();
        let centroids = values[5].as_list::<i32>();
        let offsets = centroids.value_offsets();
        let centroid_values = centroids.values().as_primitive::<Float64Type>().values();

        // Unlike update_batch, a group whose rows are all filtered out still
        // gets merged, with no incoming states: the adapter's accumulator
        // re-merges its own digest when handed an empty batch of states.
        self.touched.clear();
        for &group in group_indices {
            Self::count_row(&mut self.counts, &mut self.touched, group);
        }
        let total = self.bucket_starts();
        self.rows.clear();
        self.rows.resize(total, 0);
        for (row, &group) in group_indices.iter().enumerate() {
            let slot = self.counts[group];
            self.rows[slot as usize] = row as u32;
            self.counts[group] = slot + 1;
        }

        for i in 0..self.touched.len() {
            let group = self.touched[i];
            let start = self.starts[i] as usize;
            let end = self.counts[group] as usize;
            self.counts[group] = 0;

            self.incoming.clear();
            for &row in &self.rows[start..end] {
                let row = row as usize;
                if let Some(filter) = opt_filter
                    && !(filter.is_valid(row) && filter.value(row))
                {
                    continue;
                }
                let (lo, hi) = (offsets[row] as usize, offsets[row + 1] as usize);
                self.incoming.push(TDigest::from_state_parts(
                    max_size[row] as usize,
                    sum[row],
                    count[row],
                    max[row],
                    min[row],
                    &centroid_values[lo..hi],
                ));
            }
            let digest = TDigest::merge_digests(
                self.incoming
                    .iter()
                    .chain(std::iter::once(&self.digests[group])),
            );
            self.replace_digest(group, digest);
        }
        self.incoming.clear();
        Ok(())
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let nulls = filtered_null_mask(opt_filter, values[0].as_ref());
        let floats = values_as_f64(&values[0])?;
        let empty = TDigest::new(self.max_size);
        let digests: Vec<TDigest> = floats
            .iter()
            .enumerate()
            .map(|(row, &v)| {
                if nulls.as_ref().is_none_or(|n| n.is_valid(row)) {
                    empty.merge_sorted_f64(&[v])
                } else {
                    empty.clone()
                }
            })
            .collect();
        Ok(Self::state_arrays(&digests))
    }

    fn supports_convert_to_state(&self) -> bool {
        true
    }

    fn size(&self) -> usize {
        size_of_val(self)
            + self.digests_heap_bytes
            + self.digests.capacity() * size_of::<TDigest>()
            + self.counts.capacity() * size_of::<u32>()
            + self.touched.capacity() * size_of::<usize>()
            + self.starts.capacity() * size_of::<u32>()
            + self.keys.capacity() * size_of::<i64>()
            + self.rows.capacity() * size_of::<u32>()
            + self.sorted.capacity() * size_of::<f64>()
            + self.incoming.capacity() * size_of::<TDigest>()
            + self.return_type.size()
            - size_of_val(&self.return_type)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        Array, ArrayRef, BooleanArray, Float16Array, Float32Array, Float64Array,
        Int64Array, UInt8Array,
    };
    use arrow::datatypes::DataType;
    use datafusion_common::Result;
    use datafusion_expr::{Accumulator, EmitTo, GroupsAccumulator};
    use datafusion_functions_aggregate_common::aggregate::groups_accumulator::GroupsAccumulatorAdapter;
    use datafusion_functions_aggregate_common::tdigest::TDigest;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::approx_percentile_cont::{
        ApproxPercentileAccumulator, ApproxPercentileGroupsAccumulator,
        f64_from_sort_key, f64_sort_key,
    };

    /// How the rows of a batch spread over its groups.
    #[derive(Clone, Copy, Debug)]
    enum Spread {
        Uniform,
        /// Most rows land in the first few groups.
        Skewed,
        /// Runs of equal group indices, as rows sorted by series give.
        Runs,
    }

    #[derive(Clone, Copy, Debug)]
    struct Shape {
        groups: usize,
        rows: usize,
        batches: usize,
        spread: Spread,
        null_fraction: f64,
        filter: bool,
        /// Values rounded to two decimals, so buckets hold many duplicates.
        rounded: bool,
        /// Mixes in NaN, infinities and signed zeros.
        specials: bool,
    }

    struct Batch {
        values: ArrayRef,
        group_indices: Vec<usize>,
        filter: Option<BooleanArray>,
        total_num_groups: usize,
    }

    fn random_value(rng: &mut StdRng, shape: &Shape) -> f64 {
        if shape.specials && rng.random_bool(0.05) {
            return [
                f64::NAN,
                -f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0.0,
                -0.0,
            ][rng.random_range(0..6)];
        }
        let v = rng.random::<f64>().powi(3) * 1000.0 - 100.0;
        if shape.rounded {
            (v * 100.0).round() / 100.0
        } else {
            v
        }
    }

    fn random_group(rng: &mut StdRng, shape: &Shape, row: usize, run: usize) -> usize {
        match shape.spread {
            Spread::Uniform => rng.random_range(0..shape.groups),
            Spread::Skewed => ((rng.random::<f64>().powi(4) * shape.groups as f64)
                as usize)
                .min(shape.groups - 1),
            Spread::Runs => (row / run) % shape.groups,
        }
    }

    fn random_batches(rng: &mut StdRng, shape: &Shape) -> Vec<Batch> {
        (0..shape.batches)
            .map(|b| {
                let rows = rng.random_range(0..=shape.rows);
                let run = rng.random_range(1..=64);
                let offset = b * 7 * run;
                let group_indices: Vec<usize> = (0..rows)
                    .map(|r| random_group(rng, shape, r + offset, run))
                    .collect();
                let values: Float64Array = (0..rows)
                    .map(|_| {
                        (!rng.random_bool(shape.null_fraction))
                            .then(|| random_value(rng, shape))
                    })
                    .collect();
                let filter = shape.filter.then(|| {
                    (0..rows)
                        .map(|_| (!rng.random_bool(0.1)).then(|| rng.random_bool(0.7)))
                        .collect::<BooleanArray>()
                });
                Batch {
                    values: Arc::new(values),
                    group_indices,
                    filter,
                    total_num_groups: shape.groups,
                }
            })
            .collect()
    }

    fn adapter(
        percentile: f64,
        data_type: DataType,
        max_size: usize,
    ) -> GroupsAccumulatorAdapter {
        GroupsAccumulatorAdapter::new(move || {
            Ok(Box::new(ApproxPercentileAccumulator::new_with_max_size(
                percentile,
                data_type.clone(),
                max_size,
            )) as Box<dyn Accumulator>)
        })
    }

    fn pair(
        percentile: f64,
        data_type: &DataType,
        max_size: usize,
    ) -> [Box<dyn GroupsAccumulator>; 2] {
        [
            Box::new(adapter(percentile, data_type.clone(), max_size)),
            Box::new(ApproxPercentileGroupsAccumulator::new(
                percentile,
                data_type.clone(),
                max_size,
            )),
        ]
    }

    fn update(acc: &mut dyn GroupsAccumulator, batches: &[Batch]) -> Result<()> {
        for b in batches {
            acc.update_batch(
                &[Arc::clone(&b.values)],
                &b.group_indices,
                b.filter.as_ref(),
                b.total_num_groups,
            )?;
        }
        Ok(())
    }

    /// Arrow's array equality compares primitive values byte for byte, so
    /// this also tells NaN payloads and signed zeros apart.
    fn assert_same(adapter: &[ArrayRef], grouped: &[ArrayRef], what: &str) {
        assert_eq!(adapter.len(), grouped.len(), "{what}: column count");
        for (i, (a, g)) in adapter.iter().zip(grouped).enumerate() {
            assert_eq!(a.to_data(), g.to_data(), "{what}: column {i}");
        }
    }

    fn shapes() -> Vec<Shape> {
        let base = Shape {
            groups: 50,
            rows: 1000,
            batches: 6,
            spread: Spread::Uniform,
            null_fraction: 0.0,
            filter: false,
            rounded: false,
            specials: false,
        };
        vec![
            base,
            Shape {
                rounded: true,
                ..base
            },
            Shape {
                spread: Spread::Skewed,
                null_fraction: 0.2,
                ..base
            },
            Shape {
                spread: Spread::Runs,
                rounded: true,
                ..base
            },
            Shape {
                groups: 1,
                rows: 3000,
                ..base
            },
            Shape {
                groups: 5000,
                rows: 2000,
                ..base
            },
            Shape {
                groups: 2000,
                rows: 2000,
                spread: Spread::Skewed,
                ..base
            },
            Shape {
                filter: true,
                null_fraction: 0.1,
                ..base
            },
            Shape {
                specials: true,
                null_fraction: 0.05,
                ..base
            },
            Shape {
                null_fraction: 1.0,
                ..base
            },
            Shape {
                rows: 3,
                batches: 20,
                ..base
            },
            Shape {
                groups: 3,
                rows: 4000,
                rounded: true,
                specials: true,
                null_fraction: 0.05,
                ..base
            },
        ]
    }

    #[test]
    fn sort_keys_order_as_total_cmp() {
        let mut rng = StdRng::seed_from_u64(4000);
        let specials = [
            f64::NAN,
            -f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
        ];
        for kind in 0..3 {
            let values: Vec<f64> = (0..5000)
                .map(|_| match kind {
                    0 => f64::from_bits(rng.random()),
                    1 => (rng.random::<f64>() * 10_000.0).round() / 100.0,
                    _ if rng.random_bool(0.2) => specials[rng.random_range(0..10)],
                    _ => rng.random::<f64>() * 2.0 - 1.0,
                })
                .collect();
            let mut expected = values.clone();
            expected.sort_by(|a, b| a.total_cmp(b));

            let mut keys: Vec<i64> = values.iter().map(|&v| f64_sort_key(v)).collect();
            keys.sort_unstable();
            let sorted: Vec<u64> = keys
                .iter()
                .map(|&k| f64_from_sort_key(k).to_bits())
                .collect();
            let expected: Vec<u64> = expected.iter().map(|v| v.to_bits()).collect();
            assert_eq!(sorted, expected, "kind {kind}");
        }
    }

    #[test]
    fn groups_accumulator_matches_adapter_on_update() -> Result<()> {
        for (seed, shape) in shapes().into_iter().enumerate() {
            for (percentile, max_size) in
                [(0.99, 100), (0.5, 100), (0.1, 10), (0.0, 100), (1.0, 7)]
            {
                let mut rng = StdRng::seed_from_u64(seed as u64);
                let batches = random_batches(&mut rng, &shape);
                let what = format!("{shape:?} p={percentile} max_size={max_size}");

                let [mut a, mut g] = pair(percentile, &DataType::Float64, max_size);
                update(a.as_mut(), &batches)?;
                update(g.as_mut(), &batches)?;
                assert_same(&a.state(EmitTo::All)?, &g.state(EmitTo::All)?, &what);

                let [mut a, mut g] = pair(percentile, &DataType::Float64, max_size);
                update(a.as_mut(), &batches)?;
                update(g.as_mut(), &batches)?;
                // The adapter cannot emit zero groups, so k starts at one.
                let k = rng.random_range(1..=shape.groups);
                assert_same(
                    &[a.evaluate(EmitTo::First(k))?],
                    &[g.evaluate(EmitTo::First(k))?],
                    &what,
                );
                // The groups left after the emit are renumbered from zero.
                let left = shape.groups - k;
                if left > 0 {
                    let more = random_batches(
                        &mut rng,
                        &Shape {
                            groups: left,
                            ..shape
                        },
                    );
                    update(a.as_mut(), &more)?;
                    update(g.as_mut(), &more)?;
                    assert_same(
                        &[a.evaluate(EmitTo::All)?],
                        &[g.evaluate(EmitTo::All)?],
                        &what,
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn groups_accumulator_matches_adapter_on_merge() -> Result<()> {
        for (seed, shape) in shapes().into_iter().enumerate() {
            for (percentile, max_size) in
                [(0.99, 100), (0.5, 100), (0.1, 10), (0.0, 100), (1.0, 7)]
            {
                let mut rng = StdRng::seed_from_u64(1000 + seed as u64);
                let final_groups = rng.random_range(1..=shape.groups);
                let what = format!(
                    "{shape:?} p={percentile} max_size={max_size} final_groups={final_groups}"
                );

                // Partial states from several partitions, alternating between
                // the two kinds of partial accumulator.
                let mut partials = vec![];
                for p in 0..4 {
                    let batches = random_batches(&mut rng, &shape);
                    let [mut a, mut g] = pair(percentile, &DataType::Float64, max_size);
                    update(a.as_mut(), &batches)?;
                    update(g.as_mut(), &batches)?;
                    let (sa, sg) = (a.state(EmitTo::All)?, g.state(EmitTo::All)?);
                    assert_same(&sa, &sg, &what);
                    partials.push(if p % 2 == 0 { sa } else { sg });
                }

                let [mut a, mut g] = pair(percentile, &DataType::Float64, max_size);
                for (p, state) in partials.iter().enumerate() {
                    // Several rows of one state batch can land in the same
                    // final group.
                    let rows = state[0].len();
                    let group_indices: Vec<usize> = (0..rows)
                        .map(|_| rng.random_range(0..final_groups))
                        .collect();
                    let filter = (shape.filter && p == 1).then(|| {
                        (0..rows)
                            .map(|_| {
                                (!rng.random_bool(0.1)).then(|| rng.random_bool(0.5))
                            })
                            .collect::<BooleanArray>()
                    });
                    // Merge each state batch as two slices, so the state
                    // arrays reach merge_batch with a non-zero offset.
                    let mid = rows / 2;
                    for (offset, len) in [(0, mid), (mid, rows - mid)] {
                        let sliced: Vec<ArrayRef> =
                            state.iter().map(|c| c.slice(offset, len)).collect();
                        let filter = filter.as_ref().map(|f| f.slice(offset, len));
                        for acc in [&mut a, &mut g] {
                            acc.merge_batch(
                                &sliced,
                                &group_indices[offset..offset + len],
                                filter.as_ref(),
                                final_groups,
                            )?;
                        }
                    }
                }
                if final_groups == 1 {
                    assert_same(&a.state(EmitTo::All)?, &g.state(EmitTo::All)?, &what);
                } else {
                    let first = EmitTo::First(final_groups / 2);
                    assert_same(&a.state(first)?, &g.state(first)?, &what);
                    assert_same(
                        &[a.evaluate(EmitTo::All)?],
                        &[g.evaluate(EmitTo::All)?],
                        &what,
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn groups_accumulator_matches_adapter_on_convert_to_state() -> Result<()> {
        for (seed, shape) in shapes().into_iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(2000 + seed as u64);
            let what = format!("{shape:?}");
            let [a, g] = pair(0.5, &DataType::Float64, 100);
            assert!(a.supports_convert_to_state() && g.supports_convert_to_state());
            for b in random_batches(&mut rng, &shape) {
                let values = [Arc::clone(&b.values)];
                assert_same(
                    &a.convert_to_state(&values, b.filter.as_ref())?,
                    &g.convert_to_state(&values, b.filter.as_ref())?,
                    &what,
                );
            }
        }
        Ok(())
    }

    #[test]
    fn groups_accumulator_matches_adapter_for_other_input_types() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(3000);
        let rows = 4000;
        let group_indices: Vec<usize> =
            (0..rows).map(|_| rng.random_range(0..40)).collect();
        let raw: Vec<f64> = (0..rows).map(|_| rng.random::<f64>() * 200.0).collect();
        let valid: Vec<bool> = (0..rows).map(|_| rng.random_bool(0.9)).collect();
        let pick = |i: usize| valid[i].then_some(raw[i]);
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(
                (0..rows)
                    .map(|i| pick(i).map(|v| v as i64 - 100))
                    .collect::<Int64Array>(),
            ),
            Arc::new(
                (0..rows)
                    .map(|i| pick(i).map(|v| v as u8))
                    .collect::<UInt8Array>(),
            ),
            Arc::new(
                (0..rows)
                    .map(|i| pick(i).map(|v| v as f32))
                    .collect::<Float32Array>(),
            ),
            Arc::new(
                (0..rows)
                    .map(|i| pick(i).map(half::f16::from_f64))
                    .collect::<Float16Array>(),
            ),
        ];
        for values in arrays {
            let data_type = values.data_type().clone();
            let [mut a, mut g] = pair(0.75, &data_type, 100);
            for chunk in 0..4 {
                let (lo, hi) = (chunk * rows / 4, (chunk + 1) * rows / 4);
                for acc in [&mut a, &mut g] {
                    acc.update_batch(
                        &[values.slice(lo, hi - lo)],
                        &group_indices[lo..hi],
                        None,
                        40,
                    )?;
                }
            }
            let what = format!("{data_type}");
            assert_same(
                &[a.evaluate(EmitTo::First(3))?],
                &[g.evaluate(EmitTo::First(3))?],
                &what,
            );
            assert_same(&a.state(EmitTo::All)?, &g.state(EmitTo::All)?, &what);
        }
        Ok(())
    }

    #[test]
    fn test_combine_approx_percentile_accumulator() {
        let mut digests: Vec<TDigest> = Vec::new();

        // one TDigest with 50_000 values from 1 to 1_000
        for _ in 1..=50 {
            let t = TDigest::new(100);
            let values: Vec<_> = (1..=1_000).map(f64::from).collect();
            let t = t.merge_unsorted_f64(values);
            digests.push(t)
        }

        let t1 = TDigest::merge_digests(&digests);
        let t2 = TDigest::merge_digests(&digests);

        let mut accumulator =
            ApproxPercentileAccumulator::new_with_max_size(0.5, DataType::Float64, 100);

        accumulator.merge_digests(&[t1]);
        assert_eq!(accumulator.digest.count(), 50_000.0);
        accumulator.merge_digests(&[t2]);
        assert_eq!(accumulator.digest.count(), 100_000.0);
    }
}
