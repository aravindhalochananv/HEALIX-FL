"""
Phase 1: Data Audit Module

This module performs a comprehensive audit of the raw diabetic dataset.
It does NOT modify the raw CSV in any way.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


# Fixed random seed for reproducibility where randomness is used
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data(file_path):
    """Load raw dataset from CSV.
    
    Args:
        file_path: Path to the raw CSV file.
        
    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, na_values=["?"], keep_default_na=True)
    print(f"Dataset loaded successfully.")
    return df


def get_shape(df):
    """Return dataset shape."""
    return df.shape


def get_column_info(df):
    """Return column names and data types."""
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null_count": df.count().values,
        "null_count": df.isnull().sum().values
    })


def get_missing_summary(df):
    """Return missing value summary (count and percentage).
    
    Treats '?' as missing (already handled by na_values in load_data).
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    return pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values.round(2)
    })


def get_duplicate_rows(df):
    """Return duplicate row information."""
    duplicate_mask = df.duplicated()
    duplicate_count = duplicate_mask.sum()
    duplicate_pct = (duplicate_count / len(df)) * 100
    return duplicate_count, duplicate_pct, duplicate_mask


def analyze_target(df, target_column):
    """Analyze the target column distribution."""
    if target_column not in df.columns:
        return None
    
    value_counts = df[target_column].value_counts(dropna=False)
    target_df = pd.DataFrame({
        "target_value": value_counts.index.astype(str),
        "count": value_counts.values,
        "percentage": ((value_counts.values / len(df)) * 100).round(2)
    })
    return target_df


def classify_columns(df, target_column, id_columns=None):
    """Classify columns as numerical or categorical."""
    if id_columns is None:
        id_columns = []
    
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col == target_column or col in id_columns:
            continue
        
        # Consider column numerical if dtype is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numerical_cols, categorical_cols


def get_unique_value_counts(df):
    """Return unique-value counts for each column."""
    unique_counts = {}
    for col in df.columns:
        unique_counts[col] = df[col].nunique(dropna=False)
    return pd.DataFrame({
        "column": list(unique_counts.keys()),
        "unique_values": list(unique_counts.values())
    })


def get_descriptive_stats(df, numerical_cols):
    """Return descriptive statistics for numerical columns."""
    if not numerical_cols:
        return pd.DataFrame()
    
    stats = df[numerical_cols].describe().T
    stats = stats.round(2)
    return stats


def perform_quality_checks(df, target_column, id_columns=None, high_missing_threshold=0.5):
    """Perform basic data quality checks."""
    if id_columns is None:
        id_columns = []
    
    issues = []
    
    # 1. Completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        issues.append(f"Completely empty columns: {empty_cols}")
    
    # 2. Columns with extremely high missing percentage
    missing_pct = (df.isnull().sum() / len(df)) * 100
    high_missing_cols = missing_pct[missing_pct > (high_missing_threshold * 100)].index.tolist()
    if high_missing_cols:
        issues.append(f"Columns with >{high_missing_threshold*100}% missing: {high_missing_cols}")
    
    # 3. Possible identifier columns (very high unique count relative to rows)
    for col in df.columns:
        if col in id_columns:
            continue
        unique_ratio = df[col].nunique(dropna=False) / len(df)
        if unique_ratio > 0.95 and len(df) > 100:
            issues.append(f"Possible identifier column (high cardinality): {col} ({unique_ratio:.2%} unique)")
    
    # 4. Constant columns (only one unique non-null value)
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            issues.append(f"Constant column (single unique value): {col}")
    
    # 5. Suspicious values (negative values in columns that should be non-negative)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_column or col in id_columns:
            continue
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            issues.append(f"Suspicious negative values in '{col}': {negative_count} occurrences")
    
    return issues


def print_audit_summary(df, shape, col_info, missing_summary, duplicate_info,
                        target_analysis, numerical_cols, categorical_cols,
                        unique_counts, quality_issues, config):
    """Print a clean, human-readable audit summary to the terminal."""
    separator = "=" * 60
    thin_separator = "-" * 60
    
    print("\n" + separator)
    print("DATA AUDIT SUMMARY")
    print(separator)
    
    # 1. Dataset shape
    print(f"\n1. DATASET SHAPE")
    print(f"   Rows:    {shape[0]:,}")
    print(f"   Columns: {shape[1]:,}")
    
    # 2. Column names
    print(f"\n2. COLUMN NAMES ({shape[1]} columns)")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # 3. Data types
    print(f"\n3. DATA TYPES")
    print(f"   {'Column':<35} {'Type':<15} {'Non-Null':<12} {'Null':<10}")
    print(f"   {thin_separator}")
    for _, row in col_info.iterrows():
        print(f"   {row['column']:<35} {row['dtype']:<15} {row['non_null_count']:<12} {row['null_count']:<10}")
    
    # 4. Missing values
    print(f"\n4. MISSING VALUES")
    missing_df = missing_summary[missing_summary["missing_count"] > 0]
    if missing_df.empty:
        print("   No missing values found.")
    else:
        print(f"   {'Column':<35} {'Missing Count':<15} {'Missing %':<12}")
        print(f"   {thin_separator}")
        for _, row in missing_df.iterrows():
            print(f"   {row['column']:<35} {row['missing_count']:<15} {row['missing_pct']:<12}")
    
    # 5. Duplicate rows
    dup_count, dup_pct, _ = duplicate_info
    print(f"\n5. DUPLICATE ROWS")
    print(f"   Count: {dup_count:,}")
    print(f"   Percentage: {dup_pct:.2f}%")
    
    # 6. Target analysis
    print(f"\n6. TARGET ANALYSIS")
    print(f"   Target column: {config['dataset']['target_column']}")
    if target_analysis is not None:
        print(f"   Unique values: {target_analysis['target_value'].tolist()}")
        print(f"   {'Value':<15} {'Count':<10} {'Percentage':<12}")
        print(f"   {thin_separator}")
        for _, row in target_analysis.iterrows():
            print(f"   {str(row['target_value']):<15} {row['count']:<10} {row['percentage']:<12}")
    else:
        print(f"   Target column '{config['dataset']['target_column']}' not found in dataset.")
    
    # 7. Numerical columns
    print(f"\n7. NUMERICAL COLUMNS ({len(numerical_cols)} columns)")
    for col in numerical_cols:
        print(f"   - {col}")
    
    # 8. Categorical columns
    print(f"\n8. CATEGORICAL COLUMNS ({len(categorical_cols)} columns)")
    for col in categorical_cols:
        print(f"   - {col}")
    
    # 9. Unique-value counts
    print(f"\n9. UNIQUE-VALUE COUNTS")
    print(f"   {'Column':<35} {'Unique Values':<15}")
    print(f"   {thin_separator}")
    for _, row in unique_counts.iterrows():
        print(f"   {row['column']:<35} {row['unique_values']:<15}")
    
    # 10. Descriptive statistics
    print(f"\n10. DESCRIPTIVE STATISTICS (Numerical Columns)")
    if numerical_cols:
        stats = df[numerical_cols].describe().T
        print(stats.round(2).to_string())
    else:
        print("   No numerical columns to describe.")
    
    # 11. Basic data-quality checks
    print(f"\n11. DATA QUALITY CHECKS")
    if quality_issues:
        for issue in quality_issues:
            print(f"   [WARNING] {issue}")
    else:
        print("   No major data quality issues detected.")
    
    print(f"\n{separator}")


def save_audit_summary(df, shape, col_info, missing_summary, duplicate_info,
                       target_analysis, numerical_cols, categorical_cols,
                       unique_counts, quality_issues, output_path):
    """Save the complete audit summary to a CSV file.
    
    Creates a multi-sheet-like CSV with section headers for clarity.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    lines = []
    lines.append(["=" * 80])
    lines.append([f"HEALIX-FL DATA AUDIT - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
    lines.append(["=" * 80])
    lines.append([])
    
    # Section 1: Shape
    lines.append(["1. DATASET SHAPE"])
    lines.append(["Rows", shape[0]])
    lines.append(["Columns", shape[1]])
    lines.append([])
    
    # Section 2: Column info
    lines.append(["2. COLUMN INFORMATION"])
    lines.append(["Column", "Dtype", "Non-Null Count", "Null Count"])
    for _, row in col_info.iterrows():
        lines.append([row["column"], row["dtype"], row["non_null_count"], row["null_count"]])
    lines.append([])
    
    # Section 3: Missing values
    lines.append(["3. MISSING VALUES"])
    lines.append(["Column", "Missing Count", "Missing Percentage"])
    for _, row in missing_summary.iterrows():
        lines.append([row["column"], row["missing_count"], row["missing_pct"]])
    lines.append([])
    
    # Section 4: Duplicates
    dup_count, dup_pct, _ = duplicate_info
    lines.append(["4. DUPLICATE ROWS"])
    lines.append(["Duplicate Count", dup_count])
    lines.append(["Duplicate Percentage", f"{dup_pct:.2f}%"])
    lines.append([])
    
    # Section 5: Target analysis
    lines.append(["5. TARGET ANALYSIS"])
    lines.append(["Target Column", "readmitted"])
    if target_analysis is not None:
        lines.append(["Target Value", "Count", "Percentage"])
        for _, row in target_analysis.iterrows():
            lines.append([row["target_value"], row["count"], row["percentage"]])
    lines.append([])
    
    # Section 6: Column classification
    lines.append(["6. COLUMN CLASSIFICATION"])
    lines.append(["Numerical Columns", ", ".join(numerical_cols)])
    lines.append(["Categorical Columns", ", ".join(categorical_cols)])
    lines.append([])
    
    # Section 7: Unique counts
    lines.append(["7. UNIQUE VALUE COUNTS"])
    lines.append(["Column", "Unique Values"])
    for _, row in unique_counts.iterrows():
        lines.append([row["column"], row["unique_values"]])
    lines.append([])
    
    # Section 8: Quality issues
    lines.append(["8. DATA QUALITY ISSUES"])
    if quality_issues:
        for issue in quality_issues:
            lines.append(["WARNING", issue])
    else:
        lines.append(["OK", "No major data quality issues detected."])
    lines.append([])
    
    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        for line in lines:
            writer.writerow(line)
    
    print(f"Audit summary saved to: {output_path}")


def run_audit(config):
    """Main function to run the complete Phase 1 data audit."""
    
    # Load data
    file_path = config["paths"]["raw_data"]
    target_column = config["dataset"]["target_column"]
    id_columns = config["dataset"].get("id_columns", [])
    high_missing_threshold = config["phase1"]["audit"].get("high_missing_threshold", 0.5)
    output_csv = config["phase1"]["audit"]["output_csv"]
    
    df = load_data(file_path)
    
    # 1. Dataset shape
    shape = get_shape(df)
    
    # 2-3. Column names and data types
    col_info = get_column_info(df)
    
    # 4. Missing values
    missing_summary = get_missing_summary(df)
    
    # 5. Duplicate rows
    duplicate_info = get_duplicate_rows(df)
    
    # 6. Target analysis
    target_analysis = analyze_target(df, target_column)
    
    # 7-8. Column classification
    numerical_cols, categorical_cols = classify_columns(df, target_column, id_columns)
    
    # 9. Unique-value counts
    unique_counts = get_unique_value_counts(df)
    
    # 11. Data quality checks
    quality_issues = perform_quality_checks(
        df, target_column, id_columns, high_missing_threshold
    )
    
    # Print summary
    print_audit_summary(
        df, shape, col_info, missing_summary, duplicate_info,
        target_analysis, numerical_cols, categorical_cols,
        unique_counts, quality_issues, config
    )
    
    # Save summary
    save_audit_summary(
        df, shape, col_info, missing_summary, duplicate_info,
        target_analysis, numerical_cols, categorical_cols,
        unique_counts, quality_issues, output_csv
    )
