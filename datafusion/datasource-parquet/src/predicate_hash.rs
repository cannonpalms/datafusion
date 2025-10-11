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

//! Predicate normalization and hashing for the condition cache
//!
//! This module provides functionality to normalize and hash predicates
//! to maximize cache hits across similar queries.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use datafusion_common::{Column, ScalarValue};
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column as PhysicalColumn, InListExpr, IsNotNullExpr,
    IsNullExpr, Literal, NotExpr,
};
use datafusion_physical_plan::PhysicalExpr;
use rustc_hash::FxHasher;

/// Hash a predicate expression for use as a cache key
///
/// This function normalizes the expression first to ensure that
/// semantically equivalent expressions produce the same hash.
pub fn hash_predicate(expr: &dyn PhysicalExpr) -> u64 {
    let mut hasher = FxHasher::default();

    // Hash the expression tree directly
    // Full normalization would require implementing TreeNode for PhysicalExpr
    hash_expr_tree(expr, &mut hasher);

    hasher.finish()
}

/// Hash an expression tree recursively
fn hash_expr_tree<H: Hasher>(expr: &dyn PhysicalExpr, hasher: &mut H) {
    // Hash the expression type
    let expr_type = expr_type_id(expr);
    expr_type.hash(hasher);

    // Handle specific expression types
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        // Hash operator
        format!("{:?}", binary.op()).hash(hasher);

        // For commutative operations, sort the operands before hashing
        if is_commutative(binary.op()) {
            let left_hash = hash_single_expr(binary.left().as_ref());
            let right_hash = hash_single_expr(binary.right().as_ref());

            if left_hash <= right_hash {
                hash_expr_tree(binary.left().as_ref(), hasher);
                hash_expr_tree(binary.right().as_ref(), hasher);
            } else {
                hash_expr_tree(binary.right().as_ref(), hasher);
                hash_expr_tree(binary.left().as_ref(), hasher);
            }
        } else {
            hash_expr_tree(binary.left().as_ref(), hasher);
            hash_expr_tree(binary.right().as_ref(), hasher);
        }
    } else if let Some(column) = expr.as_any().downcast_ref::<PhysicalColumn>() {
        column.name().hash(hasher);
        column.index().hash(hasher);
    } else if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        hash_scalar_value(literal.value(), hasher);
    } else if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        // Hash the expression
        hash_expr_tree(in_list.expr().as_ref(), hasher);

        // Sort and hash the list values
        let mut list_hashes: Vec<u64> = in_list.list()
            .iter()
            .map(|e| hash_single_expr(e.as_ref()))
            .collect();
        list_hashes.sort_unstable();

        for hash_val in list_hashes {
            hash_val.hash(hasher);
        }

        in_list.negated().hash(hasher);
    } else if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        hash_expr_tree(is_null.arg().as_ref(), hasher);
    } else if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        hash_expr_tree(is_not_null.arg().as_ref(), hasher);
    } else if let Some(not_expr) = expr.as_any().downcast_ref::<NotExpr>() {
        hash_expr_tree(not_expr.arg().as_ref(), hasher);
    } else if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        hash_expr_tree(cast.expr().as_ref(), hasher);
        format!("{:?}", cast.cast_type()).hash(hasher);
    } else if let Some(case) = expr.as_any().downcast_ref::<CaseExpr>() {
        // Hash the base expression if present
        if let Some(base) = case.expr() {
            hash_expr_tree(base.as_ref(), hasher);
        }

        // Hash when-then pairs
        for (when_expr, then_expr) in case.when_then_expr() {
            hash_expr_tree(when_expr.as_ref(), hasher);
            hash_expr_tree(then_expr.as_ref(), hasher);
        }

        // Hash else expression if present
        if let Some(else_expr) = case.else_expr() {
            hash_expr_tree(else_expr.as_ref(), hasher);
        }
    } else {
        // For other expression types, hash the debug representation
        let debug_str = format!("{:?}", expr);
        debug_str.hash(hasher);
    }
}

/// Get a single hash value for an expression (used for sorting)
fn hash_single_expr(expr: &dyn PhysicalExpr) -> u64 {
    let mut hasher = FxHasher::default();
    hash_expr_tree(expr, &mut hasher);
    hasher.finish()
}

/// Hash a scalar value consistently
fn hash_scalar_value<H: Hasher>(value: &ScalarValue, hasher: &mut H) {
    // Hash the variant type
    std::mem::discriminant(value).hash(hasher);

    // Hash the actual value
    match value {
        ScalarValue::Boolean(v) => v.hash(hasher),
        ScalarValue::Float32(v) => {
            if let Some(f) = v {
                f.to_bits().hash(hasher);
            }
        }
        ScalarValue::Float64(v) => {
            if let Some(f) = v {
                f.to_bits().hash(hasher);
            }
        }
        ScalarValue::Int8(v) => v.hash(hasher),
        ScalarValue::Int16(v) => v.hash(hasher),
        ScalarValue::Int32(v) => v.hash(hasher),
        ScalarValue::Int64(v) => v.hash(hasher),
        ScalarValue::UInt8(v) => v.hash(hasher),
        ScalarValue::UInt16(v) => v.hash(hasher),
        ScalarValue::UInt32(v) => v.hash(hasher),
        ScalarValue::UInt64(v) => v.hash(hasher),
        ScalarValue::Utf8(v) | ScalarValue::LargeUtf8(v) => v.hash(hasher),
        ScalarValue::Binary(v) | ScalarValue::LargeBinary(v) => v.hash(hasher),
        ScalarValue::Date32(v) => v.hash(hasher),
        ScalarValue::Date64(v) => v.hash(hasher),
        ScalarValue::TimestampSecond(v, tz) => {
            v.hash(hasher);
            tz.hash(hasher);
        }
        ScalarValue::TimestampMillisecond(v, tz) => {
            v.hash(hasher);
            tz.hash(hasher);
        }
        ScalarValue::TimestampMicrosecond(v, tz) => {
            v.hash(hasher);
            tz.hash(hasher);
        }
        ScalarValue::TimestampNanosecond(v, tz) => {
            v.hash(hasher);
            tz.hash(hasher);
        }
        _ => {
            // For other types, hash the string representation
            format!("{:?}", value).hash(hasher);
        }
    }
}

/// Get a type identifier for an expression (for hashing)
fn expr_type_id(expr: &dyn PhysicalExpr) -> &'static str {
    if expr.as_any().downcast_ref::<BinaryExpr>().is_some() {
        "BinaryExpr"
    } else if expr.as_any().downcast_ref::<PhysicalColumn>().is_some() {
        "Column"
    } else if expr.as_any().downcast_ref::<Literal>().is_some() {
        "Literal"
    } else if expr.as_any().downcast_ref::<InListExpr>().is_some() {
        "InList"
    } else if expr.as_any().downcast_ref::<IsNullExpr>().is_some() {
        "IsNull"
    } else if expr.as_any().downcast_ref::<IsNotNullExpr>().is_some() {
        "IsNotNull"
    } else if expr.as_any().downcast_ref::<NotExpr>().is_some() {
        "Not"
    } else if expr.as_any().downcast_ref::<CastExpr>().is_some() {
        "Cast"
    } else if expr.as_any().downcast_ref::<CaseExpr>().is_some() {
        "Case"
    } else {
        "Other"
    }
}

/// Check if an operator is commutative
fn is_commutative(op: &Operator) -> bool {
    matches!(
        op,
        Operator::Eq | Operator::NotEq | Operator::And | Operator::Or | Operator::Plus | Operator::Multiply
    )
}

/// Extract all columns referenced in a predicate
pub fn extract_columns(expr: &dyn PhysicalExpr) -> Vec<Column> {
    let mut columns = BTreeMap::new();
    collect_columns_recursive(expr, &mut columns);
    columns.into_values().collect()
}

/// Recursively collect columns from an expression tree
fn collect_columns_recursive(
    expr: &dyn PhysicalExpr,
    columns: &mut BTreeMap<String, Column>,
) {
    // Check if this node is a column
    if let Some(column) = expr.as_any().downcast_ref::<PhysicalColumn>() {
        columns.insert(
            column.name().to_string(),
            Column::new_unqualified(column.name()),
        );
    }

    // Recursively process children
    for child in expr.children() {
        collect_columns_recursive(child.as_ref(), columns);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_physical_expr::expressions;
    use std::sync::Arc;

    #[test]
    fn test_hash_predicate_deterministic() {
        // Create a test schema
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
        ]);

        // Create a simple predicate: column = 5
        let col = expressions::col("a", &schema).unwrap();
        let lit = expressions::lit(ScalarValue::Int32(Some(5)));
        let expr = expressions::binary(
            col.clone(),
            Operator::Eq,
            lit.clone(),
            &schema,
        ).unwrap();

        // Hash should be deterministic
        let hash1 = hash_predicate(expr.as_ref());
        let hash2 = hash_predicate(expr.as_ref());
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_commutative_operations() {
        // Create a test schema
        let schema = Schema::new(vec![
            Field::new("a", DataType::Boolean, true),
            Field::new("b", DataType::Boolean, true),
        ]);

        // Create two equivalent predicates with different operand order
        let col_a = expressions::col("a", &schema).unwrap();
        let col_b = expressions::col("b", &schema).unwrap();

        // a AND b
        let expr1 = expressions::binary(
            col_a.clone(),
            Operator::And,
            col_b.clone(),
            &schema,
        ).unwrap();

        // b AND a
        let expr2 = expressions::binary(
            col_b.clone(),
            Operator::And,
            col_a.clone(),
            &schema,
        ).unwrap();

        // Hashes should be the same for commutative operations
        let hash1 = hash_predicate(expr1.as_ref());
        let hash2 = hash_predicate(expr2.as_ref());
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_different_predicates() {
        // Create a test schema
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
        ]);

        // Create different predicates
        let col = expressions::col("a", &schema).unwrap();
        let lit5 = expressions::lit(ScalarValue::Int32(Some(5)));
        let lit10 = expressions::lit(ScalarValue::Int32(Some(10)));

        let expr1 = expressions::binary(
            col.clone(),
            Operator::Eq,
            lit5,
            &schema,
        ).unwrap();

        let expr2 = expressions::binary(
            col,
            Operator::Eq,
            lit10,
            &schema,
        ).unwrap();

        // Different predicates should have different hashes
        let hash1 = hash_predicate(expr1.as_ref());
        let hash2 = hash_predicate(expr2.as_ref());
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_extract_columns() {
        // Create a test schema
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]);

        // Create a predicate with multiple columns: (a > 5) AND (b = 'test')
        let col_a = expressions::col("a", &schema).unwrap();
        let col_b = expressions::col("b", &schema).unwrap();
        let lit5 = expressions::lit(ScalarValue::Int32(Some(5)));
        let lit_test = expressions::lit(ScalarValue::Utf8(Some("test".to_string())));

        let left = expressions::binary(
            col_a,
            Operator::Gt,
            lit5,
            &schema,
        ).unwrap();

        let right = expressions::binary(
            col_b,
            Operator::Eq,
            lit_test,
            &schema,
        ).unwrap();

        let expr = expressions::binary(
            left,
            Operator::And,
            right,
            &schema,
        ).unwrap();

        let columns = extract_columns(expr.as_ref());
        assert_eq!(columns.len(), 2);
        assert_eq!(columns[0].name(), "a");
        assert_eq!(columns[1].name(), "b");
    }
}