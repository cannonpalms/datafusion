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

//! This example demonstrates the Parquet query condition cache effectiveness
//! by running the same queries with and without the cache enabled.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::error::Result;
use datafusion::execution::config::SessionConfig;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::{WriterProperties, WriterVersion};
use datafusion::prelude::*;
use tempfile::tempdir;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a temporary directory for our Parquet file
    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("data.parquet");

    // First, create a context to generate the data
    let setup_ctx = SessionContext::new();
    println!("============================================================");
    println!("Creating large dataset for cache performance testing...");
    println!("============================================================");
    create_sample_parquet_file(&setup_ctx, &file_path).await?;
    println!();

    // Define the queries we'll run - testing different selectivities
    let queries = vec![
        ("Q1: High selectivity filter (10% match)", "SELECT COUNT(*) FROM test_table WHERE value > 1000"),
        ("Q2: Same filter, aggregation", "SELECT category, AVG(value) FROM test_table WHERE value > 1000 GROUP BY category"),
        ("Q3: Same filter, different projection", "SELECT MIN(customer_id), MAX(customer_id) FROM test_table WHERE value > 1000"),
        ("Q4: Medium selectivity (50% match)", "SELECT COUNT(*) FROM test_table WHERE quantity > 50"),
        ("Q5: Complex filter", "SELECT COUNT(*) FROM test_table WHERE value > 1000 AND status = 'completed'"),
        ("Q6: Range scan on timestamp", "SELECT COUNT(*) FROM test_table WHERE timestamp > 1700050000"),
        ("Q7: Repeat Q1 for cache hit", "SELECT SUM(quantity) FROM test_table WHERE value > 1000"),
        ("Q8: Text filter", "SELECT COUNT(*) FROM test_table WHERE category IN ('electronics', 'books')"),
    ];

    // Run queries WITHOUT cache
    println!("Running queries WITHOUT condition cache:");
    println!("{}", "=".repeat(60));

    let config_no_cache = SessionConfig::new()
        .with_parquet_condition_cache(false)  // Cache disabled
        .with_parquet_pruning(true)
        .with_parquet_bloom_filter_pruning(true)
        .with_target_partitions(1);  // Single partition for consistent results

    let ctx_no_cache = SessionContext::new_with_config(config_no_cache);
    ctx_no_cache.register_parquet(
        "test_table",
        file_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    let mut times_no_cache = Vec::new();
    for (name, query) in &queries {
        let start = Instant::now();
        let df = ctx_no_cache.sql(query).await?;
        let results = df.collect().await?;
        let elapsed = start.elapsed();
        times_no_cache.push(elapsed);

        println!("{}: {:?}", name, elapsed);
        print_first_result(&results);
    }

    // Run queries twice to ensure any OS-level caching is warmed up
    println!();
    println!("Running queries again to warm up OS cache...");
    for (_, query) in &queries {
        let df = ctx_no_cache.sql(query).await?;
        let _ = df.collect().await?;
    }

    println!();
    println!("Running queries WITH condition cache:");
    println!("{}", "=".repeat(60));

    let config_with_cache = SessionConfig::new()
        .with_parquet_condition_cache(true)  // Cache enabled
        .with_parquet_condition_cache_size_mb(100)  // Larger cache for 1M rows
        .with_parquet_condition_cache_ttl(3600)
        .with_parquet_condition_cache_page_level(true)  // Enable page-level caching!
        .with_parquet_condition_cache_min_selectivity(0.95)  // Cache even with 95% selectivity
        .with_parquet_pruning(true)
        .with_parquet_bloom_filter_pruning(true)
        .with_target_partitions(1);  // Single partition for consistent results

    println!("Cache configuration:");
    println!("  Enabled: {}", config_with_cache.parquet_condition_cache_enabled());
    println!("  Size: {} MB", config_with_cache.options().execution.parquet.condition_cache_size_mb);
    println!("  Page-level caching: {}", config_with_cache.options().execution.parquet.condition_cache_page_level);
    println!("  TTL: {} seconds", config_with_cache.options().execution.parquet.condition_cache_ttl_seconds);
    println!("  Min selectivity: {}", config_with_cache.options().execution.parquet.condition_cache_min_selectivity);
    println!();

    let ctx_with_cache = SessionContext::new_with_config(config_with_cache);
    ctx_with_cache.register_parquet(
        "test_table",
        file_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    let mut times_with_cache = Vec::new();
    let mut first_run_times = Vec::new();

    // Run queries twice to see cache effect
    println!("First run (populating cache):");
    for (name, query) in &queries {
        let start = Instant::now();
        let df = ctx_with_cache.sql(query).await?;
        let _results = df.collect().await?;
        let elapsed = start.elapsed();
        first_run_times.push(elapsed);
        println!("  {}: {:?}", name, elapsed);
    }

    println!();
    println!("Second run (using cache):");
    for (name, query) in &queries {
        let start = Instant::now();
        let df = ctx_with_cache.sql(query).await?;
        let results = df.collect().await?;
        let elapsed = start.elapsed();
        times_with_cache.push(elapsed);
        println!("  {}: {:?}", name, elapsed);
        print_first_result(&results);
    }

    // Compare results
    println!();
    println!("Performance Comparison:");
    println!("{}", "=".repeat(80));
    println!("{:<35} {:>12} {:>12} {:>12} {:>10}", "Query", "No Cache", "First Run", "Cached", "Speedup");
    println!("{}", "-".repeat(80));

    for (i, (name, _)) in queries.iter().enumerate() {
        let time_no_cache = times_no_cache[i];
        let time_first_run = first_run_times[i];
        let time_with_cache = times_with_cache[i];
        let speedup = if time_with_cache.as_nanos() > 0 {
            time_no_cache.as_nanos() as f64 / time_with_cache.as_nanos() as f64
        } else {
            0.0
        };

        println!("{:<35} {:>12.3?} {:>12.3?} {:>12.3?} {:>9.2}x",
            name,
            time_no_cache,
            time_first_run,
            time_with_cache,
            speedup);
    }

    let num_row_groups = 100;  // Number of row groups in the dataset

    // Get actual cache metrics from the SessionState
    println!();
    println!("Cache Metrics (Actual from SessionState):");
    println!("{}", "=".repeat(60));

    // Get cache statistics from the session
    let state = ctx_with_cache.state();
    if let Some(cache_stats) = state.parquet_cache_stats() {
        println!("Row Group Cache:");
        println!("  Hits: {}", cache_stats.row_group_hits);
        println!("  Misses: {}", cache_stats.row_group_misses);
        println!("  Entries: {}", cache_stats.row_group_entries);

        println!();
        println!("Page Cache:");
        println!("  Hits: {}", cache_stats.page_hits);
        println!("  Misses: {}", cache_stats.page_misses);
        println!("  Entries: {}", cache_stats.page_entries);

        println!();
        println!("Memory Usage:");
        println!("  Total memory used: {} bytes", cache_stats.memory_usage_bytes);
        println!("  Cache capacity: {} MB", cache_stats.config.max_size_mb);
        if cache_stats.config.max_size_mb > 0 {
            let utilization = (cache_stats.memory_usage_bytes as f64 / (cache_stats.config.max_size_mb * 1024 * 1024) as f64) * 100.0;
            println!("  Utilization: {:.4}%", utilization);
        }
        println!("  Evictions: {}", cache_stats.evictions);
    }

    // Get hit rate statistics
    if let Some(hit_rate) = state.parquet_cache_hit_rate() {
        println!();
        println!("Cache Hit Rates:");
        println!("  Row Group: {:.1}%", hit_rate.row_group_hit_rate);
        println!("  Page: {:.1}%", hit_rate.page_hit_rate);
        println!("  Overall: {:.1}%", hit_rate.overall_hit_rate);
    }

    println!();
    println!("Cache Effectiveness Analysis:");
    println!("{}", "=".repeat(60));
    println!("Working with {} row groups and ~{} pages total:", num_row_groups, num_row_groups * 10);
    println!();

    println!("Row Group Level Caching:");
    println!("  - Queries Q1, Q2, Q3, Q7 use filter: value > 1000 (10% selectivity)");
    println!("  - Q1 populates the cache with row group pruning results");
    println!("  - Q2, Q3, Q7 reuse the cached row group bitmap");
    println!();

    println!("Page Level Caching (ENABLED):");
    println!("  - Each row group has ~10 pages (1000 rows per page)");
    println!("  - Page-level cache stores 1 bit per page (10x more granular than row groups)");
    println!("  - Allows skipping individual pages within row groups");
    println!("  - Dramatically improves performance for selective queries");
    println!();

    println!("Cache Memory Efficiency:");
    println!("  - Row group cache: 1 bit per row group = {} bits total", num_row_groups);
    println!("  - Page cache: 1 bit per page = {} bits total", num_row_groups * 10);
    println!("  - Total per predicate: ~{} bytes", (num_row_groups + num_row_groups * 10) / 8);
    println!("  - With 100 MB cache, can store ~{} unique predicates", 100 * 1024 * 1024 * 8 / (num_row_groups + num_row_groups * 10));
    println!();

    println!("Performance Impact:");
    println!("  - First run: Evaluates predicates and populates both caches");
    println!("  - Second run: Uses cached results, skips predicate evaluation");
    println!("  - Page-level caching provides additional speedup for selective queries");
    println!("  - Most benefit for queries with same/similar predicates");

    Ok(())
}

/// Create a large sample Parquet file with many row groups using DataFusion
async fn create_sample_parquet_file(ctx: &SessionContext, path: &std::path::Path) -> Result<()> {
    // Define schema with more columns for realistic workload
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("subcategory", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("customer_id", DataType::Int64, false),
        Field::new("description", DataType::Utf8, true),
    ]));

    // Categories and subcategories for realistic data
    let categories = ["electronics", "books", "clothing", "food", "home", "sports"];
    let subcategories = ["premium", "standard", "budget", "clearance"];
    let statuses = ["pending", "completed", "cancelled", "processing", "shipped"];
    let regions = ["north", "south", "east", "west", "central"];

    // Generate data - 100 row groups with 10,000 rows each = 1 million rows total
    let num_row_groups = 100;
    let rows_per_group = 10_000;

    println!("Creating large Parquet file with {} row groups, {} rows each ({} total rows)",
        num_row_groups, rows_per_group, num_row_groups * rows_per_group);

    let mut all_batches = vec![];

    for batch_idx in 0..num_row_groups {
        let mut ids = Vec::new();
        let mut timestamps = Vec::new();
        let mut category_vec = Vec::new();
        let mut subcategory_vec = Vec::new();
        let mut values = Vec::new();
        let mut quantities = Vec::new();
        let mut status_vec = Vec::new();
        let mut region_vec = Vec::new();
        let mut customer_ids = Vec::new();
        let mut descriptions = Vec::new();

        let base_id = batch_idx as i64 * rows_per_group as i64;
        let base_timestamp = 1700000000 + batch_idx as i64 * 3600; // Hourly data

        for i in 0..rows_per_group {
            let id = base_id + i as i64;
            ids.push(id);
            timestamps.push(base_timestamp + (i as i64 * 10)); // 10 second intervals
            category_vec.push(categories[(id as usize) % categories.len()]);
            subcategory_vec.push(subcategories[(id as usize / 7) % subcategories.len()]);

            // Create varied value distribution with locality for row group pruning
            // Row groups 0-9 (first 100k rows) have only high values
            // Row groups 10-89 (next 800k rows) have only low values
            // Row groups 90-99 (last 100k rows) have mixed values
            let value = if batch_idx < 10 {
                // First 10% of row groups have high values only
                1000.0 + (id as f64 * 17.0) % 5000.0
            } else if batch_idx < 90 {
                // Middle 80% of row groups have low values only
                (id as f64 * 13.0) % 500.0
            } else {
                // Last 10% of row groups have mixed values
                if id % 2 == 0 {
                    1000.0 + (id as f64 * 17.0) % 5000.0
                } else {
                    (id as f64 * 13.0) % 500.0
                }
            };
            values.push(value);

            quantities.push(((id * 7) % 100) as i32);
            status_vec.push(statuses[(id as usize / 13) % statuses.len()]);
            region_vec.push(regions[(id as usize / 23) % regions.len()]);
            customer_ids.push((id * 31) % 100000); // 100k unique customers

            // Some rows have descriptions, some don't
            if id % 3 == 0 {
                descriptions.push(Some(format!("Item description for {}", id)));
            } else {
                descriptions.push(None);
            }
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(timestamps)),
                Arc::new(StringArray::from(category_vec)),
                Arc::new(StringArray::from(subcategory_vec)),
                Arc::new(Float64Array::from(values)),
                Arc::new(Int32Array::from(quantities)),
                Arc::new(StringArray::from(status_vec)),
                Arc::new(StringArray::from(region_vec)),
                Arc::new(Int64Array::from(customer_ids)),
                Arc::new(StringArray::from(descriptions)),
            ],
        )?;

        all_batches.push(batch);

        // Print progress
        if (batch_idx + 1) % 10 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
    println!(" Done!");

    // Write to Parquet file with specific row group size and page size
    // This ensures we get many row groups AND multiple pages per row group
    println!("Writing to Parquet file...");

    // Configure writer properties to create multiple row groups
    let props = WriterProperties::builder()
        .set_max_row_group_size(rows_per_group)  // 10,000 rows per group
        .set_data_page_row_count_limit(1000)     // 1,000 rows per page
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .build();

    // Write using ArrowWriter to control row group size
    let mut file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(&mut file, schema.clone(), Some(props))?;

    for batch in all_batches {
        writer.write(&batch)?;
    }
    writer.close()?;

    // Get file size
    let metadata = std::fs::metadata(path)?;
    let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
    println!("Created Parquet file: {:.1} MB with ~1000 pages total", size_mb);

    Ok(())
}

/// Print first result value for verification
fn print_first_result(results: &[RecordBatch]) {
    if let Some(batch) = results.first() {
        if batch.num_rows() > 0 && batch.num_columns() > 0 {
            let first_col = batch.column(0);
            let value = datafusion::common::ScalarValue::try_from_array(first_col, 0)
                .unwrap_or(datafusion::common::ScalarValue::Null);
            println!("    Result: {}", value);
        }
    }
}