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

//! Integration tests for the Parquet query condition cache

use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl};
use datafusion::datasource::physical_plan::ParquetExec;
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;
use datafusion_common::FileType;
use datafusion_datasource_parquet::{
    ConditionCacheConfig, ParquetAccessPlan, ParquetConditionCache, RowGroupAccessPlanFilter,
};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_expr::expressions as phys_expr;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_pruning::PruningPredicate;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use tempfile::TempDir;

#[tokio::test]
async fn test_condition_cache_basic() -> Result<()> {
    // Create temporary directory and file
    let tmp_dir = TempDir::new()?;
    let file_path = tmp_dir.path().join("test.parquet");
    create_test_parquet_file(&file_path)?;

    // Create cache
    let metrics = Arc::new(ExecutionPlanMetricsSet::new());
    let cache_config = ConditionCacheConfig::default();
    let cache = Arc::new(ParquetConditionCache::new(cache_config, &metrics));

    // Test that cache properly stores and retrieves row group pruning results
    let file_id = 12345u64;

    // Create a simple predicate: col1 > 50
    let schema = Arc::new(Schema::new(vec![
        Field::new("col1", DataType::Int32, false),
        Field::new("col2", DataType::Utf8, false),
    ]));

    let col_expr = phys_expr::col("col1", &schema)?;
    let lit_expr = phys_expr::lit(50i32);
    let predicate_expr = phys_expr::gt(col_expr, lit_expr, &schema)?;

    let pruning_predicate = PruningPredicate::try_new(predicate_expr, schema.clone())?;

    // Read parquet metadata
    let file = std::fs::File::open(&file_path)?;
    let reader = parquet::file::reader::SerializedFileReader::new(file)?;
    let metadata = reader.metadata();

    // Create access plan
    let num_row_groups = metadata.num_row_groups();
    let mut access_plan = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(num_row_groups));

    // First call - should miss cache and compute
    access_plan.prune_by_statistics_with_cache(
        Some(cache.clone()),
        file_id,
        Some(metadata.file_metadata()),
        &schema,
        metadata.file_metadata().schema_descr(),
        metadata.row_groups(),
        &pruning_predicate,
        &datafusion_datasource_parquet::ParquetFileMetrics::new(
            0,
            "test.parquet",
            &metrics,
        ),
    );

    let first_result = access_plan.build();

    // Second call with same predicate - should hit cache
    let mut access_plan2 = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(num_row_groups));

    access_plan2.prune_by_statistics_with_cache(
        Some(cache.clone()),
        file_id,
        Some(metadata.file_metadata()),
        &schema,
        metadata.file_metadata().schema_descr(),
        metadata.row_groups(),
        &pruning_predicate,
        &datafusion_datasource_parquet::ParquetFileMetrics::new(
            0,
            "test.parquet",
            &metrics,
        ),
    );

    let second_result = access_plan2.build();

    // Results should be identical
    assert_eq!(first_result, second_result);

    // Check cache metrics
    assert!(cache.memory_usage_bytes() > 0);

    Ok(())
}

#[tokio::test]
async fn test_condition_cache_selectivity() -> Result<()> {
    // Test that highly unselective predicates are not cached
    let tmp_dir = TempDir::new()?;
    let file_path = tmp_dir.path().join("test.parquet");
    create_test_parquet_file(&file_path)?;

    let metrics = Arc::new(ExecutionPlanMetricsSet::new());
    let mut cache_config = ConditionCacheConfig::default();
    cache_config.min_selectivity_to_cache = 0.5; // Don't cache if >50% rows match

    let cache = Arc::new(ParquetConditionCache::new(cache_config, &metrics));

    let schema = Arc::new(Schema::new(vec![
        Field::new("col1", DataType::Int32, false),
        Field::new("col2", DataType::Utf8, false),
    ]));

    // Create an unselective predicate: col1 > 10 (most rows will match)
    let col_expr = phys_expr::col("col1", &schema)?;
    let lit_expr = phys_expr::lit(10i32);
    let predicate_expr = phys_expr::gt(col_expr, lit_expr, &schema)?;
    let pruning_predicate = PruningPredicate::try_new(predicate_expr, schema.clone())?;

    let file = std::fs::File::open(&file_path)?;
    let reader = parquet::file::reader::SerializedFileReader::new(file)?;
    let metadata = reader.metadata();

    let mut access_plan = RowGroupAccessPlanFilter::new(
        ParquetAccessPlan::new_all(metadata.num_row_groups())
    );

    access_plan.prune_by_statistics_with_cache(
        Some(cache.clone()),
        123u64,
        Some(metadata.file_metadata()),
        &schema,
        metadata.file_metadata().schema_descr(),
        metadata.row_groups(),
        &pruning_predicate,
        &datafusion_datasource_parquet::ParquetFileMetrics::new(
            0,
            "test.parquet",
            &metrics,
        ),
    );

    // Cache should be empty or very small since predicate was unselective
    assert!(cache.memory_usage_bytes() < 1000);

    Ok(())
}

#[tokio::test]
async fn test_condition_cache_with_datafusion_query() -> Result<()> {
    // Create test data
    let tmp_dir = TempDir::new()?;
    let file_path = tmp_dir.path().join("test.parquet");
    create_test_parquet_file(&file_path)?;

    // Create session context
    let ctx = SessionContext::new();

    // Register the Parquet file
    ctx.register_parquet(
        "test_table",
        file_path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;

    // Run multiple queries with the same predicate
    let query1 = "SELECT COUNT(*) FROM test_table WHERE col1 > 75";
    let df1 = ctx.sql(query1).await?;
    let results1 = df1.collect().await?;

    // Second query with same WHERE clause but different projection
    let query2 = "SELECT col2 FROM test_table WHERE col1 > 75 LIMIT 10";
    let df2 = ctx.sql(query2).await?;
    let results2 = df2.collect().await?;

    // Both queries should succeed
    assert!(!results1.is_empty());
    assert!(!results2.is_empty());

    Ok(())
}

/// Create a test Parquet file with multiple row groups
fn create_test_parquet_file(path: &std::path::Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("col1", DataType::Int32, false),
        Field::new("col2", DataType::Utf8, false),
    ]));

    // Create small row groups for testing
    let props = WriterProperties::builder()
        .set_max_row_group_size(100)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .build();

    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    // Create 10 row groups with different value ranges
    for group in 0..10 {
        let base = group * 10;
        let col1_data: Vec<i32> = (0..100).map(|i| base + (i % 10)).collect();
        let col2_data: Vec<&str> = (0..100)
            .map(|i| match i % 3 {
                0 => "foo",
                1 => "bar",
                _ => "baz",
            })
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(col1_data)),
                Arc::new(StringArray::from(col2_data)),
            ],
        )?;

        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

#[tokio::test]
async fn test_condition_cache_memory_limit() -> Result<()> {
    let metrics = Arc::new(ExecutionPlanMetricsSet::new());
    let mut cache_config = ConditionCacheConfig::default();
    cache_config.max_size_mb = 1; // Very small limit for testing

    let cache = Arc::new(ParquetConditionCache::new(cache_config, &metrics));

    // Add many entries to trigger eviction
    for i in 0..1000 {
        let mut bitmap = bitvec::bitvec![0; 1000];
        bitmap.set(i % 1000, true);

        cache.put_row_group_bitmap(
            datafusion_datasource_parquet::condition_cache::RowGroupCacheKey {
                file_id: i,
                predicate_hash: i,
                file_version: i,
            },
            bitmap,
        );
    }

    // Check that memory usage is within limit
    assert!(cache.memory_usage_bytes() <= 1024 * 1024);

    Ok(())
}