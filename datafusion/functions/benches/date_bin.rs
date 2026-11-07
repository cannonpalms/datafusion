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

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, TimestampNanosecondArray, TimestampSecondArray};
use arrow::datatypes::Field;
use criterion::{Criterion, criterion_group};
use datafusion_common::ScalarValue;
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDF};
use datafusion_functions::datetime::date_bin;
use rand::rngs::{StdRng, ThreadRng};
use rand::{Rng, SeedableRng};

fn timestamps(rng: &mut ThreadRng) -> TimestampSecondArray {
    let mut seconds = vec![];
    for _ in 0..1000 {
        seconds.push(rng.random_range(0..1_000_000));
    }

    TimestampSecondArray::from(seconds)
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("date_bin_1000", |b| {
        let mut rng = rand::rng();
        let timestamps_array = Arc::new(timestamps(&mut rng)) as ArrayRef;
        let batch_len = timestamps_array.len();
        let interval = ColumnarValue::Scalar(ScalarValue::new_interval_dt(0, 1_000_000));
        let timestamps = ColumnarValue::Array(timestamps_array);
        let udf = date_bin();
        let return_type = udf
            .return_type(&[interval.data_type(), timestamps.data_type()])
            .unwrap();
        let return_field = Arc::new(Field::new("f", return_type, true));

        let arg_fields = vec![
            Field::new("a", interval.data_type(), true).into(),
            Field::new("b", timestamps.data_type(), true).into(),
        ];
        let config_options = Arc::new(ConfigOptions::default());

        b.iter(|| {
            black_box(
                udf.invoke_with_args(ScalarFunctionArgs {
                    args: vec![interval.clone(), timestamps.clone()],
                    arg_fields: arg_fields.clone(),
                    number_rows: batch_len,
                    return_field: Arc::clone(&return_field),
                    config_options: Arc::clone(&config_options),
                })
                .expect("date_bin should work on valid values"),
            )
        })
    });
}

const NANOS_PER_SEC: i64 = 1_000_000_000;
const BATCH_ROWS: usize = 8192;
const BATCHES: usize = 128;
/// 2026-01-01T00:00:00Z
const START_NANOS: i64 = 1_767_225_600 * NANOS_PER_SEC;

/// About one million nanosecond timestamps over 30 days, in query-sized
/// batches, the shape of an InfluxDB `time` column.
fn nanosecond_batches(null_rate: f64) -> Vec<ColumnarValue> {
    let mut rng = StdRng::seed_from_u64(42);
    let span = 30 * 86_400 * NANOS_PER_SEC;
    (0..BATCHES)
        .map(|_| {
            let array: TimestampNanosecondArray = (0..BATCH_ROWS)
                .map(|_| {
                    (!rng.random_bool(null_rate))
                        .then(|| START_NANOS + rng.random_range(0..span))
                })
                .collect();
            ColumnarValue::Array(Arc::new(array))
        })
        .collect()
}

struct NanosCase {
    name: &'static str,
    interval: ColumnarValue,
    batches: Vec<ColumnarValue>,
}

fn nanos_cases() -> Vec<NanosCase> {
    let dense = nanosecond_batches(0.0);
    let minute = ScalarValue::new_interval_dt(0, 60_000);
    let hour = ScalarValue::new_interval_dt(0, 3_600_000);
    let day = ScalarValue::new_interval_dt(1, 0);
    vec![
        NanosCase {
            name: "date_bin_nanos_1m_1min",
            interval: ColumnarValue::Scalar(minute),
            batches: dense.clone(),
        },
        NanosCase {
            name: "date_bin_nanos_1m_1h",
            interval: ColumnarValue::Scalar(hour.clone()),
            batches: dense.clone(),
        },
        NanosCase {
            name: "date_bin_nanos_1m_1d",
            interval: ColumnarValue::Scalar(day),
            batches: dense,
        },
        NanosCase {
            name: "date_bin_nanos_1m_1h_nulls",
            interval: ColumnarValue::Scalar(hour),
            batches: nanosecond_batches(0.1),
        },
    ]
}

fn run_nanos_case(udf: &ScalarUDF, case: &NanosCase) -> usize {
    let return_type = udf
        .return_type(&[case.interval.data_type(), case.batches[0].data_type()])
        .unwrap();
    let return_field = Arc::new(Field::new("f", return_type, true));
    let arg_fields = vec![
        Field::new("a", case.interval.data_type(), true).into(),
        Field::new("b", case.batches[0].data_type(), true).into(),
    ];
    let config_options = Arc::new(ConfigOptions::default());
    let mut nulls = 0;
    for batch in &case.batches {
        let result = udf
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![case.interval.clone(), batch.clone()],
                arg_fields: arg_fields.clone(),
                number_rows: BATCH_ROWS,
                return_field: Arc::clone(&return_field),
                config_options: Arc::clone(&config_options),
            })
            .expect("date_bin should work on valid values");
        if let ColumnarValue::Array(a) = black_box(result) {
            nulls += a.null_count();
        }
    }
    nulls
}

fn nanos_benchmark(c: &mut Criterion) {
    let udf = date_bin();
    for case in nanos_cases() {
        c.bench_function(case.name, |b| b.iter(|| run_nanos_case(&udf, &case)));
    }
}

/// `DATE_BIN_PERF=<case>:<iterations>` runs one case without criterion, for
/// counting instructions with `perf stat`.
fn perf_mode() -> bool {
    let Ok(spec) = std::env::var("DATE_BIN_PERF") else {
        return false;
    };
    let (name, iterations) = spec.split_once(':').expect("<case>:<iterations>");
    let iterations: usize = iterations.parse().unwrap();
    let udf = date_bin();
    let case = nanos_cases()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("unknown case {name}"));
    for _ in 0..iterations {
        black_box(run_nanos_case(&udf, &case));
    }
    true
}

criterion_group!(benches, criterion_benchmark, nanos_benchmark);

fn main() {
    if perf_mode() {
        return;
    }
    benches();
    Criterion::default().configure_from_args().final_summary();
}
