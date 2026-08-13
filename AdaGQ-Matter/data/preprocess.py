"""
Unified data preprocessing for IoTID20 and CICIoT2023 datasets.

Supports:
1. IoTID20: from chunk files (processed_chunk_*.csv.gz) or single CSV
2. CICIoT2023: from HuggingFace download or local CSV

Output format: .npz files (train/test split) + partitions.json (Dirichlet)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Fix Python import path for package-level imports ---
# When running `python data/preprocess.py`, Python adds `data/` to sys.path,
# but `from data.xxx import ...` needs the project root in sys.path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.dirichlet_partition import dirichlet_partition, save_partitions


# ============================================================
# IoTID20 Preprocessing
# ============================================================

def preprocess_iotid20_from_chunks(
    chunk_dir: str,
    output_dir: str,
    alpha: float = 0.5,
    N: int = 10,
    test_ratio: float = 0.2,
    seed: int = 1,
    skip_scaler: bool = True,
) -> None:
    """
    Read IoTID20 from preprocessed .csv.gz chunk files (processed_chunk_1~7).
    These are already Z-score standardized, so skip_scaler=True by default.

    Args:
        chunk_dir: Directory containing processed_chunk_*.csv.gz files
        output_dir: Directory for output .npz and .json files
        alpha: Dirichlet concentration parameter
        N: Number of FL clients
        test_ratio: Train/test split ratio
        seed: Random seed
        skip_scaler: If True, skip Z-score (data already standardized)
    """
    print(f"\n{'='*60}")
    print(f"[IoTID20] Loading chunks from {chunk_dir} ...")
    print(f"{'='*60}")

    chunk_dir_path = Path(chunk_dir)
    chunk_files = sorted(chunk_dir_path.glob("processed_chunk_*.csv.gz"))
    if not chunk_files:
        raise FileNotFoundError(f"No 'processed_chunk_*.csv.gz' found in {chunk_dir}.")

    print(f"[IoTID20] Found {len(chunk_files)} chunk files")

    # Read & concatenate all chunks
    dfs = []
    for f in chunk_files:
        df_chunk = pd.read_csv(f, compression="gzip")
        print(f"  {f.name}: {len(df_chunk)} rows, {df_chunk.shape[1]} cols")
        dfs.append(df_chunk)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[IoTID20] Total: {len(df)} rows, {df.shape[1]} columns")

    # Process
    _process_iotid20(df, output_dir, alpha, N, test_ratio, seed, skip_scaler)


def preprocess_iotid20_single(
    raw_path: str,
    output_dir: str,
    alpha: float = 0.5,
    N: int = 10,
    test_ratio: float = 0.2,
    seed: int = 1,
    skip_scaler: bool = False,
) -> None:
    """
    Read IoTID20 from a single CSV file (public dataset format).
    Applies Z-score standardization by default.
    """
    print(f"\n{'='*60}")
    print(f"[IoTID20] Loading single CSV from {raw_path} ...")
    print(f"{'='*60}")

    df = pd.read_csv(raw_path)
    print(f"[IoTID20] Total: {len(df)} rows, {df.shape[1]} columns")

    _process_iotid20(df, output_dir, alpha, N, test_ratio, seed, skip_scaler)


def _process_iotid20(
    df: pd.DataFrame,
    output_dir: str,
    alpha: float,
    N: int,
    test_ratio: float,
    seed: int,
    skip_scaler: bool,
) -> None:
    """Common processing pipeline for IoTID20."""
    # Identify label columns
    label_cols = [c for c in df.columns if c.lower() in
                  ["label", "attack_type", "class", "cat", "subcategory", "sub_cat"]]
    feature_cols = [c for c in df.columns if c not in label_cols]

    # Extract features and ENCODE categorical columns (not just drop)
    X_df = df[feature_cols].copy()
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric) > 0:
        print(f"[IoTID20] Found {len(non_numeric)} non-numeric columns: {non_numeric}")
        for col in non_numeric:
            n_unique = X_df[col].nunique()
            # 高基数列（IP/MAC地址等 >500 唯一值）：丢弃（信息熵低且维度爆炸）
            if n_unique > 500:
                print(f"  [DROP] {col}: {n_unique} unique values (high-cardinality, likely IP/MAC)")
                X_df = X_df.drop(columns=[col])
            # 中低基数列（3~500 唯一值）：label encoding → 整数编码
            elif n_unique > 2:
                print(f"  [LABEL-ENC] {col}: {n_unique} unique values → label encoding")
                X_df[col] = X_df[col].astype('category').cat.codes.astype(np.float32)
            # 二元列（2 唯一值）：直接 label encoding
            else:
                print(f"  [LABEL-ENC] {col}: {n_unique} unique values → label encoding")
                X_df[col] = X_df[col].astype('category').cat.codes.astype(np.float32)

    # Clean infinities and NaNs
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X_df = X_df.dropna()

    actual_dim = X_df.shape[1]
    print(f"[IoTID20] Feature dimension: {actual_dim} (expected: 79)")

    # Z-score standardization
    if skip_scaler:
        X_scaled = X_df.values.astype(np.float32)
        print("[IoTID20] Skipping standardization (data already Z-scored)")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df.values).astype(np.float32)
        print("[IoTID20] Applied Z-score standardization")

    # Binary label extraction
    label_col_candidates = [c for c in label_cols if c.lower() == "label"]
    if label_col_candidates:
        label_col = label_col_candidates[0]
        y_raw = df.loc[X_df.index, label_col]
    else:
        # Use first label column found
        label_col = label_cols[0] if label_cols else None
        y_raw = df.loc[X_df.index, label_col]

    if y_raw.dtype in [np.int64, np.int32, np.float64, np.float32]:
        y = y_raw.astype(int).values
    else:
        y = (y_raw != "Normal").astype(int).values

    print(f"[IoTID20] Label distribution: 0(normal)={int((y == 0).sum())}, "
          f"1(attack)={int((y == 1).sum())}")

    # Save processed data
    _save_processed_data(X_scaled, y, output_dir, "iotid20", alpha, N, test_ratio, seed)


# ============================================================
# CICIoT2023 Preprocessing
# ============================================================

def preprocess_ciciot2023(
    raw_path: str = None,
    output_dir: str = "data/processed",
    alpha: float = 0.5,
    N: int = 10,
    test_ratio: float = 0.2,
    seed: int = 1,
    subsample_ratio: float = 0.03,  # ~1.3M from 46M
    attack_types: list = None,
) -> None:
    """
    Read CICIoT2023 dataset, optionally subsample, filter attack types,
    and prepare for FL training.

    Args:
        raw_path: Path to CICIoT2023 CSV file(s) or directory
        output_dir: Output directory
        alpha: Dirichlet concentration
        N: Number of FL clients
        test_ratio: Train/test split ratio
        seed: Random seed
        subsample_ratio: Fraction of data to keep (for CPU-friendliness)
        attack_types: List of attack types to include (None = all)
    """
    print(f"\n{'='*60}")
    print(f"[CICIoT2023] Loading data from {raw_path} ...")
    print(f"{'='*60}")

    # Try loading from HuggingFace if no local path
    if raw_path is None:
        try:
            from datasets import load_dataset
            print("[CICIoT2023] Downloading from HuggingFace...")
            ds = load_dataset("lacg030175/CIC-IoT-2023-raw", split="train")
            df = ds.to_pandas()
            print(f"[CICIoT2023] Loaded {len(df)} rows from HuggingFace")
        except Exception as e:
            print(f"[CICIoT2023] HuggingFace download failed: {e}")
            print("[CICIoT2023] Please provide local path via --raw_dir")
            return
    else:
        raw_path_obj = Path(raw_path)
        # Try raw_path first; if not found, try raw_path + .csv
        if not raw_path_obj.exists() and not raw_path_obj.is_dir():
            csv_candidate = Path(str(raw_path_obj) + ".csv")
            if csv_candidate.exists():
                raw_path_obj = csv_candidate
                raw_path = str(csv_candidate)
                print(f"[CICIoT2023] Auto-detected file: {raw_path}")
        if raw_path_obj.is_dir():
            # Load all CSV files in directory
            csv_files = sorted(raw_path_obj.glob("*.csv"))
            if not csv_files:
                csv_files = sorted(raw_path_obj.glob("*.csv.gz"))
            dfs = []
            for f in csv_files:
                if f.suffix == ".gz":
                    dfs.append(pd.read_csv(f, compression="gzip", nrows=None))
                else:
                    dfs.append(pd.read_csv(f, nrows=None))
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(raw_path)

    print(f"[CICIoT2023] Raw data: {len(df)} rows, {df.shape[1]} columns")

    # Filter attack types if specified
    if attack_types:
        label_col = [c for c in df.columns if c.lower() in
                     ["label", "attack_type", "class", "category"]]
        if label_col:
            df = df[df[label_col[0]].isin(attack_types + ["Normal"])]
            print(f"[CICIoT2023] After attack type filter: {len(df)} rows")

    # Subsample for CPU-friendly experiments
    if subsample_ratio < 1.0 and len(df) > 500000:
        rng = np.random.default_rng(seed)
        n_keep = int(len(df) * subsample_ratio)
        keep_idx = rng.choice(len(df), size=n_keep, replace=False)
        df = df.iloc[keep_idx].reset_index(drop=True)
        print(f"[CICIoT2023] Subsampled: {len(df)} rows (ratio={subsample_ratio})")

    # Identify columns
    label_cols = [c for c in df.columns if c.lower() in
                  ["label", "attack_type", "class", "category"]]
    feature_cols = [c for c in df.columns if c not in label_cols]

    # Extract numeric features
    X_df = df[feature_cols].copy()
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"[CICIoT2023] Dropping non-numeric: {list(non_numeric)}")
        X_df = X_df.drop(columns=non_numeric)

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X_df = X_df.dropna()

    actual_dim = X_df.shape[1]
    print(f"[CICIoT2023] Feature dimension: {actual_dim}")

    # Z-score standardization (always needed for CICIoT2023)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values).astype(np.float32)
    print("[CICIoT2023] Applied Z-score standardization")

    # Binary label: any non-"Normal" = attack
    label_col = label_cols[0] if label_cols else None
    if label_col:
        y_raw = df.loc[X_df.index, label_col]
        y = (y_raw != "Normal").astype(int).values
    else:
        raise ValueError("No label column found in CICIoT2023 data")

    print(f"[CICIoT2023] Label distribution: 0(normal)={int((y == 0).sum())}, "
          f"1(attack)={int((y == 1).sum())}")

    # Save
    _save_processed_data(X_scaled, y, output_dir, "ciciot2023", alpha, N, test_ratio, seed)


# ============================================================
# Common Save Function
# ============================================================

def _save_processed_data(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    dataset_name: str,
    alpha: float,
    N: int,
    test_ratio: float,
    seed: int,
) -> None:
    """Save processed data as .npz files and create Dirichlet partitions."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )

    print(f"[{dataset_name}] Train: {len(X_train)}, Test: {len(X_test)}")

    # Dirichlet partition on training data
    partitions = dirichlet_partition(y_train, N=N, alpha=alpha, seed=seed)

    # Save .npz files
    train_path = os.path.join(output_dir, f"{dataset_name}_train.npz")
    test_path = os.path.join(output_dir, f"{dataset_name}_test.npz")
    partition_path = os.path.join(output_dir, f"{dataset_name}_partitions.json")

    np.savez(train_path, X=X_train, y=y_train)
    np.savez(test_path, X=X_test, y=y_test)
    save_partitions(partitions, partition_path)

    print(f"[{dataset_name}] Saved train → {train_path}")
    print(f"[{dataset_name}] Saved test → {test_path}")
    print(f"[{dataset_name}] Saved partitions → {partition_path}")
    print(f"[{dataset_name}] ✅ Preprocessing complete!")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess IoT datasets for FL")
    parser.add_argument("--dataset", default="iotid20",
                        choices=["iotid20", "ciciot2023", "both"],
                        help="Dataset to process")
    parser.add_argument("--raw_dir", default="/root/datasets/raw",
                        help="Directory with raw data files (default: /root/datasets/raw)")
    parser.add_argument("--chunk_dir", default=None,
                        help="Directory with IoTID20 processed_chunk_*.csv.gz files")
    parser.add_argument("--output_dir", default="/root/datasets/processed",
                        help="Output directory for .npz files (default: /root/datasets/processed)")
    parser.add_argument("--N", type=int, default=10,
                        help="Number of FL clients")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration parameter")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Test set ratio")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--skip_scaler", action="store_true",
                        help="Skip Z-score (data already standardized)")
    parser.add_argument("--subsample_ratio", type=float, default=0.03,
                        help="CICIoT2023 subsample ratio")
    parser.add_argument("--ciciot_raw_dir", default=None,
                        help="Path to CICIoT2023 raw data (directory or file)")
    args = parser.parse_args()

    if args.dataset in ["iotid20", "both"]:
        if args.chunk_dir:
            preprocess_iotid20_from_chunks(
                args.chunk_dir, args.output_dir,
                alpha=args.alpha, N=args.N,
                test_ratio=args.test_ratio, seed=args.seed,
                skip_scaler=args.skip_scaler,
            )
        else:
            iotid20_path = os.path.join(args.raw_dir, "IoTID20.csv")
            preprocess_iotid20_single(
                iotid20_path, args.output_dir,
                alpha=args.alpha, N=args.N,
                test_ratio=args.test_ratio, seed=args.seed,
                skip_scaler=args.skip_scaler,
            )

    if args.dataset in ["ciciot2023", "both"]:
        # Auto-detect CICIoT2023 file path (with or without .csv extension)
        ciciot_path = args.ciciot_raw_dir
        if ciciot_path is None:
            ciciot_candidates = [
                os.path.join(args.raw_dir, "CICIoT2023.csv"),
                os.path.join(args.raw_dir, "CICIoT2023"),
            ]
            for candidate in ciciot_candidates:
                if os.path.exists(candidate):
                    ciciot_path = candidate
                    break
            if ciciot_path is None:
                ciciot_path = os.path.join(args.raw_dir, "CICIoT2023.csv")
        preprocess_ciciot2023(
            raw_path=ciciot_path,
            output_dir=args.output_dir,
            alpha=args.alpha, N=args.N,
            test_ratio=args.test_ratio, seed=args.seed,
            subsample_ratio=args.subsample_ratio,
        )


if __name__ == "__main__":
    main()
