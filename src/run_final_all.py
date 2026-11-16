"""Optional master runner for final experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODULES = {
    "cdc": ("src.run_final_cdc", "cdc"),
    "cdc_dp": ("src.run_final_cdc_dp_noise", "cdc_dp"),
    "20news": ("src.run_final_20news", "20news"),
    "20news_topk": ("src.run_final_20news_topk", "20news_topk"),
    "20news_dp": ("src.run_final_20news_dp_noise", "20news_dp"),
    "20news_api": ("src.run_final_20news_api_exposure", "20news_api"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run final experiment suites in sequence.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    parser.add_argument("--output-dir", type=str, default="results_final")
    parser.add_argument("--only", choices=list(MODULES), default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    selected = [args.only] if args.only else list(MODULES)
    for key in selected:
        module, subdir = MODULES[key]
        cmd = [
            sys.executable,
            "-m",
            module,
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--output-dir",
            str(Path(args.output_dir) / subdir),
            "--device",
            args.device,
        ]
        if args.smoke:
            cmd.append("--smoke")
        print(f"[run_final_all] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
