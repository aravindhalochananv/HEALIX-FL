"""
Phase 2: Data Preprocessing Module

This module performs reproducible preprocessing for the HEALIX-FL project.
It creates a binary readmission target, splits the data, and builds a
reusable sklearn preprocessing pipeline.

IMPORTANT:
- The raw CSV is never modified.
- Preprocessing is fitted ONLY on training data.
- The fitted preprocessor is saved for later reuse.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Fixed random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_raw_data(raw_path):
    """Load raw dataset and treat '?' as missing values.
    
    Args:
        raw_path: Path to the raw CSV file.
        
    Returns:
        pandas.DataFrame: Loaded dataset with '?' replaced by NaN.
    """
    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path, na_values=["?"], keep_default_na=True)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


def create_binary_target(df, target_column="readmitted"):
    """Create a binary target for 30-day readmission.
    
    Mapping:
        readmitted == "<30" -> target = 1
        readmitted == "NO"   -> target = 0
        readmitted == ">30"  -> target = 0
        
    Args:
        df: Input DataFrame.
        target_column: Name of the original target column.
        
    Returns:
        pandas.DataFrame: DataFrame with new 'target' column.
    """
    print("Creating binary target for 30-day readmission...")
    print(f"Original target values: {sorted(df[target_column].dropna().unique())}")
    
    # Create binary target
    df["target"] = df[target_column].apply(
        lambda x: 1 if x == "<30" else 0 if x in ["NO", ">30"] else np.nan
    )
    
    # Drop original target column
    df = df.drop(columns=[target_column])
    
    print(f"Binary target created:")
    print(f"  Class 0 (NO or >30): {(df['target'] == 0).sum():,}")
    print(f"  Class 1 (<30):       {(df['target'] == 1).sum():,}")
    
    return df


def remove_identifiers(df, id_columns):
    """Remove identifier columns that should not be used as features.
    
    Args:
        df: Input DataFrame.
        id_columns: List of column names to remove.
        
    Returns:
        pandas.DataFrame: DataFrame without identifier columns.
    """
    print(f"Removing identifier columns: {id_columns}")
    df = df.drop(columns=id_columns, errors="ignore")
    print(f"Remaining columns after ID removal: {len(df.columns)}")
    return df


def split_data(df, target_column="target", test_size=0.2, random_state=42):
    """Split data into stratified train and test sets.
    
    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"Splitting data: {1-test_size:.0%} train / {test_size:.0%} test")
    print(f"Stratified by: {target_column}")
    print(f"Random state: {random_state}")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples:  {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test


def identify_feature_types(X_train):
    """Automatically identify numerical and categorical features.
    
    Args:
        X_train: Training features DataFrame.
        
    Returns:
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
    """
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    
    print(f"\nDetected {len(numerical_cols)} numerical features:")
    for col in numerical_cols:
        print(f"  - {col}")
    
    print(f"\nDetected {len(categorical_cols)} categorical features:")
    for col in categorical_cols:
        print(f"  - {col}")
    
    return numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    """Build a sklearn ColumnTransformer preprocessing pipeline.
    
    Numerical pipeline:
        1. Median imputation
        2. StandardScaler
        
    Categorical pipeline:
        1. Most-frequent imputation
        2. OneHotEncoder (handle_unknown='ignore')
        
    Args:
        numerical_cols: List of numerical feature names.
        categorical_cols: List of categorical feature names.
        
    Returns:
        sklearn ColumnTransformer: Fitted preprocessor.
    """
    print("\nBuilding preprocessing pipeline...")
    
    # Numerical pipeline: median imputation + standard scaling
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline: most-frequent imputation + one-hot encoding
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )
    
    print("Preprocessing pipeline built successfully.")
    return preprocessor


def fit_preprocessor(preprocessor, X_train):
    """Fit the preprocessor on training data only.
    
    Args:
        preprocessor: sklearn ColumnTransformer.
        X_train: Training features.
        
    Returns:
        Fitted preprocessor.
    """
    print("\nFitting preprocessor on training data...")
    preprocessor.fit(X_train)
    print("Preprocessor fitted successfully.")
    return preprocessor


def transform_data(preprocessor, X_train, X_test):
    """Transform train and test data using the fitted preprocessor.
    
    Args:
        preprocessor: Fitted sklearn ColumnTransformer.
        X_train: Training features.
        X_test: Testing features.
        
    Returns:
        X_train_processed, X_test_processed, feature_names
    """
    print("\nTransforming train and test data...")
    
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Extract feature names from the fitted transformer
    feature_names = preprocessor.get_feature_names_out()
    
    print(f"Processed training shape:   {X_train_processed.shape}")
    print(f"Processed testing shape:    {X_test_processed.shape}")
    print(f"Total processed features:   {len(feature_names)}")
    
    return X_train_processed, X_test_processed, feature_names


def save_processed_data(X_train_processed, X_test_processed, y_train, y_test,
                        feature_names, processed_dir):
    """Save processed datasets and feature names to disk.
    
    Args:
        X_train_processed: Processed training features.
        X_test_processed: Processed testing features.
        y_train: Training target.
        y_test: Testing target.
        feature_names: Array of feature names.
        processed_dir: Directory to save outputs.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"\nSaving processed data to: {processed_dir}")
    
    # Save features as numpy arrays
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train_processed)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test_processed)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.values)
    
    # Save feature names as CSV
    feature_names_df = pd.DataFrame({"feature_name": feature_names})
    feature_names_df.to_csv(os.path.join(processed_dir, "feature_names.csv"), index=False)
    
    print("Saved:")
    print("  - X_train.npy")
    print("  - X_test.npy")
    print("  - y_train.npy")
    print("  - y_test.npy")
    print("  - feature_names.csv")


def save_preprocessor(preprocessor, processed_dir):
    """Save the fitted preprocessor using joblib.
    
    Args:
        preprocessor: Fitted sklearn ColumnTransformer.
        processed_dir: Directory to save the preprocessor.
    """
    preprocessor_path = os.path.join(processed_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Fitted preprocessor saved to: {preprocessor_path}")


def validate_preprocessing(X_train_processed, X_test_processed, y_train, y_test,
                          feature_names, id_columns, target_column_original="readmitted"):
    """Run validation checks after preprocessing.
    
    Args:
        X_train_processed: Processed training features.
        X_test_processed: Processed testing features.
        y_train: Training target.
        y_test: Testing target.
        feature_names: Array of feature names.
        id_columns: List of identifier columns that should not be present.
        target_column_original: Original target column name.
    """
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    
    all_passed = True
    
    # 1. No missing values in X_train
    missing_train = np.isnan(X_train_processed).sum()
    if missing_train == 0:
        print("[PASS] No missing values in X_train")
    else:
        print(f"[FAIL] Missing values in X_train: {missing_train}")
        all_passed = False
    
    # 2. No missing values in X_test
    missing_test = np.isnan(X_test_processed).sum()
    if missing_test == 0:
        print("[PASS] No missing values in X_test")
    else:
        print(f"[FAIL] Missing values in X_test: {missing_test}")
        all_passed = False
    
    # 3. Same number of features
    if X_train_processed.shape[1] == X_test_processed.shape[1]:
        print(f"[PASS] X_train and X_test have same feature count: {X_train_processed.shape[1]}")
    else:
        print(f"[FAIL] Feature mismatch: train={X_train_processed.shape[1]}, test={X_test_processed.shape[1]}")
        all_passed = False
    
    # 4. y_train and y_test contain only 0 and 1
    unique_train = set(y_train.unique())
    unique_test = set(y_test.unique())
    if unique_train.issubset({0, 1}) and unique_test.issubset({0, 1}):
        print(f"[PASS] y_train and y_test contain only binary values: {sorted(unique_train | unique_test)}")
    else:
        print(f"[FAIL] Non-binary values found: train={unique_train}, test={unique_test}")
        all_passed = False
    
    # 5. Feature names preserved
    if len(feature_names) == X_train_processed.shape[1]:
        print(f"[PASS] Feature names preserved: {len(feature_names)} features")
    else:
        print(f"[FAIL] Feature name count mismatch: {len(feature_names)} names vs {X_train_processed.shape[1]} features")
        all_passed = False
    
    # 6. No encounter_id
    if "encounter_id" not in feature_names:
        print("[PASS] No 'encounter_id' in features")
    else:
        print("[FAIL] 'encounter_id' found in features")
        all_passed = False
    
    # 7. No patient_nbr
    if "patient_nbr" not in feature_names:
        print("[PASS] No 'patient_nbr' in features")
    else:
        print("[FAIL] 'patient_nbr' found in features")
        all_passed = False
    
    # 8. No readmitted column
    if "readmitted" not in feature_names:
        print("[PASS] No 'readmitted' in features")
    else:
        print("[FAIL] 'readmitted' found in features")
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All validation checks PASSED.")
    else:
        print("Some validation checks FAILED. Review the output above.")
    print("=" * 60)


def print_target_distribution(y_train, y_test):
    """Print training and testing target distributions.
    
    Args:
        y_train: Training target.
        y_test: Testing target.
    """
    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)
    
    for name, y in [("TRAINING", y_train), ("TESTING", y_test)]:
        counts = y.value_counts().sort_index()
        total = len(y)
        print(f"\n{name} TARGET:")
        print(f"  Class 0 (NO or >30): {counts.get(0, 0):,} ({counts.get(0, 0)/total*100:.2f}%)")
        print(f"  Class 1 (<30):       {counts.get(1, 0):,} ({counts.get(1, 0)/total*100:.2f}%)")
        print(f"  Total:               {total:,}")


def print_preprocessing_summary(df_original, X_train_processed, X_test_processed,
                               y_train, y_test, missing_before):
    """Print a clean summary of preprocessing results.
    
    Args:
        df_original: Original DataFrame before preprocessing.
        X_train_processed: Processed training features.
        X_test_processed: Processed testing features.
        y_train: Training target.
        y_test: Testing target.
        missing_before: Number of missing values before preprocessing.
    """
    print("\n" + "=" * 60)
    print("HEALIX-FL PHASE 2")
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 60)
    
    print(f"\nOriginal dataset:")
    print(f"  Rows:    {len(df_original):,}")
    print(f"  Columns: {len(df_original.columns)}")
    
    print(f"\nTraining samples: {len(X_train_processed):,}")
    print(f"Testing samples:  {len(X_test_processed):,}")
    
    print(f"\nOriginal features: {len(df_original.columns) - 1}")  # -1 for target
    print(f"Final processed features: {X_train_processed.shape[1]}")
    
    # Target distribution
    pos_train = (y_train == 1).sum()
    neg_train = (y_train == 0).sum()
    pos_test = (y_test == 1).sum()
    neg_test = (y_test == 0).sum()
    
    print(f"\nPositive class (readmitted <30):")
    print(f"  Training: {pos_train:,} ({pos_train/len(y_train)*100:.2f}%)")
    print(f"  Testing:  {pos_test:,} ({pos_test/len(y_test)*100:.2f}%)")
    
    print(f"\nNegative class (readmitted NO or >30):")
    print(f"  Training: {neg_train:,} ({neg_train/len(y_train)*100:.2f}%)")
    print(f"  Testing:  {neg_test:,} ({neg_test/len(y_test)*100:.2f}%)")
    
    # Missing values
    missing_after = np.isnan(X_train_processed).sum() + np.isnan(X_test_processed).sum()
    print(f"\nMissing values before preprocessing: {missing_before:,}")
    print(f"Missing values after preprocessing:  {missing_after:,}")
    
    print("\n" + "=" * 60)


def run_preprocessing(config):
    """Execute the complete Phase 2 preprocessing pipeline.
    
    Args:
        config: Dictionary loaded from config.yaml.
    """
    print("=" * 60)
    print("HEALIX-FL PHASE 2: DATA PREPROCESSING")
    print("=" * 60)
    print()
    
    # Paths from config
    raw_path = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed_data"]
    id_columns = config["dataset"].get("id_columns", ["encounter_id", "patient_nbr"])
    target_column_original = config["dataset"]["target_column"]
    
    # 1. Load raw data
    df = load_raw_data(raw_path)
    missing_before = df.isnull().sum().sum()
    
    # 2. Create binary target
    df = create_binary_target(df, target_column=target_column_original)
    
    # 3. Remove identifiers
    df = remove_identifiers(df, id_columns=id_columns)
    
    # 4. Train/test split
    X_train, X_test, y_train, y_test = split_data(
        df, target_column="target", test_size=0.2, random_state=RANDOM_STATE
    )
    
    # 5. Identify feature types
    numerical_cols, categorical_cols = identify_feature_types(X_train)
    
    # 6-8. Build and fit preprocessor
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    preprocessor = fit_preprocessor(preprocessor, X_train)
    
    # 9. Transform data
    X_train_processed, X_test_processed, feature_names = transform_data(
        preprocessor, X_train, X_test
    )
    
    # 10. Save preprocessor
    save_preprocessor(preprocessor, processed_dir)
    
    # 11. Save processed data
    save_processed_data(
        X_train_processed, X_test_processed, y_train, y_test,
        feature_names, processed_dir
    )
    
    # 12. Target distribution
    print_target_distribution(y_train, y_test)
    
    # 13. Validation
    validate_preprocessing(
        X_train_processed, X_test_processed, y_train, y_test,
        feature_names, id_columns, target_column_original
    )
    
    # 14. Summary
    print_preprocessing_summary(
        df, X_train_processed, X_test_processed, y_train, y_test,
        missing_before
    )
    
    print("Phase 2 preprocessing completed successfully.")
    print(f"All outputs saved to: {processed_dir}")
    
    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols
    }
