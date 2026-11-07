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

//! `GROUP BY date_bin(..)` over a time-series table held in a `MemTable`.

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Float64Array, RecordBatch, StringArray, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use criterion::{Criterion, criterion_group};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::prelude::SessionConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tokio::runtime::Runtime;

const NANOS_PER_SEC: i64 = 1_000_000_000;
const BATCH_ROWS: usize = 8192;
const BATCHES: usize = 128;
const HOSTS: usize = 100;
/// 2026-01-01T00:00:00Z
const START_NANOS: i64 = 1_767_225_600 * NANOS_PER_SEC;

const QUERIES: [(&str, &str); 2] = [
    (
        "date_bin_group_by_1h",
        "SELECT date_bin(INTERVAL '1 hour', time) AS bin, avg(usage) \
         FROM cpu GROUP BY bin",
    ),
    (
        "date_bin_group_by_host_1d",
        "SELECT host, date_bin(INTERVAL '1 day', time) AS bin, max(usage) \
         FROM cpu GROUP BY host, bin",
    ),
];

fn context() -> SessionContext {
    let schema = Arc::new(Schema::new(vec![
        Field::new("host", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("usage", DataType::Float64, true),
    ]));
    let mut rng = StdRng::seed_from_u64(42);
    let span = 30 * 86_400 * NANOS_PER_SEC;
    let hosts: Vec<String> = (0..HOSTS).map(|i| format!("host-{i:03}")).collect();
    let batches = (0..BATCHES)
        .map(|_| {
            let host: StringArray = (0..BATCH_ROWS)
                .map(|_| Some(hosts[rng.random_range(0..HOSTS)].as_str()))
                .collect();
            let time: TimestampNanosecondArray = (0..BATCH_ROWS)
                .map(|_| Some(START_NANOS + rng.random_range(0..span)))
                .collect();
            let usage: Float64Array = (0..BATCH_ROWS)
                .map(|_| Some(rng.random_range(0.0..100.0)))
                .collect();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(host), Arc::new(time), Arc::new(usage)],
            )
            .unwrap()
        })
        .collect();
    let table = MemTable::try_new(schema, vec![batches]).unwrap();
    let config = SessionConfig::new()
        .with_target_partitions(1)
        .with_batch_size(BATCH_ROWS);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("cpu", Arc::new(table)).unwrap();
    ctx
}

fn query(ctx: &SessionContext, rt: &Runtime, sql: &str) -> usize {
    let df = rt.block_on(ctx.sql(sql)).unwrap();
    let batches = rt.block_on(df.collect()).unwrap();
    black_box(batches.iter().map(|b| b.num_rows()).sum())
}

fn criterion_benchmark(c: &mut Criterion) {
    let ctx = context();
    let rt = Runtime::new().unwrap();
    for (name, sql) in QUERIES {
        c.bench_function(name, |b| b.iter(|| query(&ctx, &rt, sql)));
    }
}

/// `DATE_BIN_PERF=<query>:<iterations>` runs one query without criterion, for
/// counting instructions with `perf stat`.
fn perf_mode() -> bool {
    let Ok(spec) = std::env::var("DATE_BIN_PERF") else {
        return false;
    };
    let (name, iterations) = spec.split_once(':').expect("<query>:<iterations>");
    let iterations: usize = iterations.parse().unwrap();
    let (_, sql) = QUERIES
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("unknown query {name}"));
    let ctx = context();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for _ in 0..iterations {
        query(&ctx, &rt, sql);
    }
    true
}

criterion_group!(benches, criterion_benchmark);

fn main() {
    if perf_mode() {
        return;
    }
    benches();
    Criterion::default().configure_from_args().final_summary();
}
