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

//! This example demonstrates configuring the Parquet query condition cache
//! and shows how to set it up via SessionConfig settings.

use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::error::Result;
use datafusion::execution::config::SessionConfig;
use datafusion::prelude::*;
use tempfile::tempdir;

#[tokio::main]
async fn main() -> Result<()> {
    // Demonstrate various ways to configure the condition cache
    println!("Parquet Condition Cache Configuration Examples");
    println!("{}", "=".repeat(60));
    println!();

    // Method 1: Using builder methods
    let config_builder = SessionConfig::new()
        .with_parquet_condition_cache(true)
        .with_parquet_condition_cache_size_mb(100)
        .with_parquet_condition_cache_ttl(3600)
        .with_parquet_condition_cache_page_level(false)
        .with_parquet_condition_cache_min_selectivity(0.9);

    println!("Method 1 - Using builder methods:");
    println!(
        "  Cache enabled: {}",
        config_builder.parquet_condition_cache_enabled()
    );
    println!(
        "  Cache size: {} MB",
        config_builder
            .options()
            .execution
            .parquet
            .condition_cache_size_mb
    );
    println!(
        "  TTL: {} seconds",
        config_builder
            .options()
            .execution
            .parquet
            .condition_cache_ttl_seconds
    );
    println!(
        "  Page-level: {}",
        config_builder
            .options()
            .execution
            .parquet
            .condition_cache_page_level
    );
    println!(
        "  Min selectivity: {}",
        config_builder
            .options()
            .execution
            .parquet
            .condition_cache_min_selectivity
    );
    println!();

    // Method 2: Using string-based configuration
    let config_strings = SessionConfig::new()
        .set_bool("datafusion.execution.parquet.enable_condition_cache", true)
        .set_usize("datafusion.execution.parquet.condition_cache_size_mb", 50)
        .set_u64(
            "datafusion.execution.parquet.condition_cache_ttl_seconds",
            1800,
        )
        .set_bool(
            "datafusion.execution.parquet.condition_cache_page_level",
            true,
        )
        .set_str(
            "datafusion.execution.parquet.condition_cache_min_selectivity",
            "0.8",
        );

    println!("Method 2 - Using string-based configuration:");
    println!(
        "  Cache enabled: {}",
        config_strings.parquet_condition_cache_enabled()
    );
    println!(
        "  Cache size: {} MB",
        config_strings
            .options()
            .execution
            .parquet
            .condition_cache_size_mb
    );
    println!(
        "  TTL: {} seconds",
        config_strings
            .options()
            .execution
            .parquet
            .condition_cache_ttl_seconds
    );
    println!(
        "  Page-level: {}",
        config_strings
            .options()
            .execution
            .parquet
            .condition_cache_page_level
    );
    println!(
        "  Min selectivity: {}",
        config_strings
            .options()
            .execution
            .parquet
            .condition_cache_min_selectivity
    );
    println!();

    // Create a sample Parquet file and demonstrate usage
    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("demo.parquet");

    // Create context with cache enabled
    let ctx = SessionContext::new_with_config(config_builder);
    create_sample_parquet_file(&ctx, &file_path).await?;

    println!("Created sample Parquet file: {}", file_path.display());
    println!();

    // Register and query the table
    ctx.register_parquet(
        "demo",
        file_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    println!("Running sample queries with cache configuration enabled:");
    println!("{}", "-".repeat(60));

    let queries = vec![
        "SELECT COUNT(*) FROM demo WHERE value > 500",
        "SELECT category, AVG(value) FROM demo WHERE value > 500 GROUP BY category",
        "SELECT MIN(id), MAX(id) FROM demo WHERE value > 500",
    ];

    for (i, query) in queries.iter().enumerate() {
        let df = ctx.sql(query).await?;
        let results = df.collect().await?;

        println!("Query {}: {}", i + 1, query);
        if let Some(batch) = results.first() {
            if batch.num_rows() > 0 {
                println!("  Result rows: {}", batch.num_rows());
            }
        }
    }

    println!();
    println!("Cache Integration Architecture:");
    println!("{}", "=".repeat(60));
    println!("1. Configuration is stored in SessionConfig");
    println!("2. When ParquetExec is created, it checks if cache is enabled");
    println!("3. Cache instance would be created and shared across queries");
    println!("4. ParquetOpener uses the cache during row group pruning:");
    println!("   - Generates file ID from path for cache key");
    println!("   - Checks cache for existing predicate evaluation results");
    println!("   - If hit: Uses cached bitmap to skip row groups");
    println!("   - If miss: Evaluates predicate and stores result");
    println!("5. Cache entries are shared across queries with same predicates");
    println!();
    println!("Memory Efficiency:");
    println!("  - Each cache entry uses 1 bit per row group");
    println!("  - 100MB cache can store info for ~6.8 trillion rows");
    println!("  - Automatic LRU eviction when memory limit reached");
    println!();
    println!("Note: Full integration requires wiring the cache through");
    println!("SessionState → ParquetExec → ParquetSource → ParquetOpener");

    Ok(())
}

/// Create a small sample Parquet file
async fn create_sample_parquet_file(
    ctx: &SessionContext,
    path: &std::path::Path,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let mut batches = vec![];
    for batch_idx in 0..5 {
        let base = batch_idx * 100;
        let ids: Vec<i32> = (base..base + 100).collect();
        let categories: Vec<&str> = ids
            .iter()
            .map(|i| match i % 3 {
                0 => "A",
                1 => "B",
                _ => "C",
            })
            .collect();
        let values: Vec<i32> = ids.iter().map(|i| (i * 17) % 1000).collect();

        batches.push(RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(categories)),
                Arc::new(Int32Array::from(values)),
            ],
        )?);
    }

    let df = ctx.read_batches(batches)?;
    df.write_parquet(
        path.to_str().unwrap(),
        DataFrameWriteOptions::new().with_single_file_output(true),
        None,
    )
    .await?;

    Ok(())
}
