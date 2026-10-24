from __future__ import annotations
import argparse
import json
from pathlib import Path
from .experiment import run


def main():
    parser = argparse.ArgumentParser(description="Run the ShiftBench distribution-shift research benchmark.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.n_seeds is not None:
        config["n_seeds"] = args.n_seeds
    if args.n_samples is not None:
        config["n_samples"] = args.n_samples
        config["shift_point"] = args.n_samples // 2
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if config["shift_point"] >= config["n_samples"] - 100 or config["shift_point"] < 100:
        parser.error("shift_point must be at least 100 from either stream boundary.")
    print(json.dumps(run(config), indent=2))


if __name__ == "__main__":
    main()
