"""
Download scripts for IoT datasets.

IoTID20:
- Option 1: From local chunk files (processed_chunk_*.csv.gz)
- Option 2: From Kaggle/HuggingFace public repository
- Option 3: From Mendeley Data (official source)

CICIoT2023:
- Option 1: From HuggingFace (lacg030175/CIC-IoT-2023-raw, 1.3M subsampled)
- Option 2: From the official CIC website (full 46M rows)
"""

import argparse
import os
import sys
from pathlib import Path


# --- Fix Python import path ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def download_iotid20_huggingface(output_dir: str = "/root/datasets/raw") -> None:
    """Download IoTID20 from HuggingFace dataset repository."""
    # AutoDL: use HF mirror if direct access fails
    hf_mirror = os.environ.get("HF_ENDPOINT", "")
    try:
        from datasets import load_dataset
        repo_name = "maruuf/iotid20_dataset"
        print(f"[IoTID20] Downloading from HuggingFace ({repo_name})...")
        if hf_mirror:
            print(f"[IoTID20] Using mirror: {hf_mirror}")
        ds = load_dataset(repo_name, split="train")
        df = ds.to_pandas()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        csv_path = os.path.join(output_dir, "IoTID20.csv")
        df.to_csv(csv_path, index=False)
        print(f"[IoTID20] ✅ Saved to {csv_path} ({len(df)} rows)")
    except Exception as e:
        print(f"[IoTID20] ❌ HuggingFace download failed: {e}")
        print("[IoTID20] Trying Mendeley Data direct download...")
        try:
            import subprocess
            mendeley_url = "https://data.mendeley.com/public-files/datasets/nzc7grj6jm/files/da3c4c83-5a3b-4d5b-8b7b-d5a73d3f5c7a/file_downloaded"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            csv_path = os.path.join(output_dir, "IoTID20.csv")
            result = subprocess.run(
                ["wget", "-q", "-O", csv_path, mendeley_url],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0 and os.path.exists(csv_path):
                print(f"[IoTID20] ✅ Downloaded from Mendeley Data: {csv_path}")
            else:
                print(f"[IoTID20] ❌ Mendeley Data download also failed")
                print("[IoTID20] Please upload manually (see manual §4)")
        except Exception as e2:
            print(f"[IoTID20] ❌ Mendeley fallback failed: {e2}")
            print("[IoTID20] Please upload manually (see manual §4)")


def download_ciciot2023_huggingface(
    output_dir: str = "/root/datasets/raw",
    subsample: bool = True,
) -> None:
    """Download CICIoT2023 from HuggingFace (1.3M subsampled version)."""
    hf_mirror = os.environ.get("HF_ENDPOINT", "")
    try:
        from datasets import load_dataset
        if subsample:
            print("[CICIoT2023] Downloading subsampled version (1.3M rows)...")
            repo_name = "lacg030175/CIC-IoT-2023-raw"
            if hf_mirror:
                print(f"[CICIoT2023] Using mirror: {hf_mirror}")
            ds = load_dataset(repo_name, split="train")
        else:
            print("[CICIoT2023] Downloading full dataset (46M rows)...")
            print("[CICIoT2023] ⚠️ Full dataset is ~4GB, consider subsampled version")
            ds = load_dataset("lacg030175/CIC-IoT-2023-raw", split="train")
        df = ds.to_pandas()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        csv_path = os.path.join(output_dir, "CICIoT2023.csv")
        df.to_csv(csv_path, index=False)
        print(f"[CICIoT2023] ✅ Saved to {csv_path} ({len(df)} rows)")
    except Exception as e:
        print(f"[CICIoT2023] ❌ HuggingFace download failed: {e}")
        print("[CICIoT2023] Please download manually from:")
        print("  - HuggingFace: https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-raw")
        print("  - CIC Website: https://www.unb.ca/cic/datasets/iotdataset-2023.html")


def main():
    parser = argparse.ArgumentParser(description="Download IoT datasets")
    parser.add_argument("--dataset", default="both",
                        choices=["iotid20", "ciciot2023", "both"])
    parser.add_argument("--output_dir", default="/root/datasets/raw")
    parser.add_argument("--ciciot_full", action="store_true",
                        help="Download full CICIoT2023 (46M rows)")
    args = parser.parse_args()

    if args.dataset in ["iotid20", "both"]:
        download_iotid20_huggingface(args.output_dir)

    if args.dataset in ["ciciot2023", "both"]:
        download_ciciot2023_huggingface(
            args.output_dir,
            subsample=not args.ciciot_full,
        )


if __name__ == "__main__":
    main()
