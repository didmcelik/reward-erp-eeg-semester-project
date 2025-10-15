#!/usr/bin/env python
"""
EEG Reward ERP — end-to-end runner.

Runs the project in this order:
  1) step02_03_preproc_batch.py  (filter + reref + resample + ICA, all subjects)
  2) step04_epoching_batch.py    (events → epochs + evokeds, all subjects)
  3) step05_metrics.py           (subject-level FRN/P3, each subject)
  4) step06_group_stats.py       (group CSV + stats + figures)

Notes
- Code assumes your repo layout:
    project/
      data/ds004147/...
      notebooks/*.py
      derivatives/...
- All scripts already write outputs under `derivatives/`.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path, args: list[str] | None = None) -> None:
    """Run a Python script in a subprocess and stream its output; raise on error."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    print(f"\n=== RUN: {script_path.name} {' '.join(args or [])} ===")
    res = subprocess.run(cmd, cwd=str(script_path.parent))
    if res.returncode != 0:
        raise SystemExit(f"[FATAL] {script_path.name} failed with exit code {res.returncode}")


def parse_subjects(spec: str) -> list[str]:
    """
    Accept formats like:
      '27-38'           -> ['27', '28', ..., '38']
      '27,28,31'        -> ['27','28','31']
    """
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return [f"{i}" for i in range(int(a), int(b) + 1)]
    return [s.strip() for s in spec.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reward ERP pipeline end-to-end.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "ds004147",
        help="Path to ds004147 root (contains sub-XX folders).",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="27-38",
        help="Subjects to include. Examples: '27-38' or '27,29,31'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow scripts to overwrite existing outputs.",
    )
    parser.add_argument(
        "--no-ica",
        action="store_true",
        help="Skip ICA (still runs filtering/rereferencing/resampling).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="2,4,5,6",
        help="Which steps to run (comma-separated). Choices: 2,4,5,6",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    nb = repo_root / "notebooks"

    # Validate inputs
    if not args.dataset_root.exists():
        raise SystemExit(f"[FATAL] dataset-root not found: {args.dataset_root}")
    subjects = parse_subjects(args.subjects)
    if not subjects:
        raise SystemExit("[FATAL] empty subject list")

    # Common flags
    ow_flag = ["--overwrite"] if args.overwrite else []
    ica_flag = [] if not args.no-ica else ["--no-ica"]

    # Step selectors
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}

    # 1) Preprocess batch (filter + reref + resample + optional ICA)
    if "2" in steps:
        # step02_03_preproc_batch.py handles all subjects internally
        script = nb / "step02_03_preproc_batch.py"
        run_script(script, [*ica_flag, *ow_flag])

    # 2) Epoching batch (events → epochs + evokeds)
    if "4" in steps:
        script = nb / "step04_epoching_batch.py"
        run_script(script, [*ow_flag])

    # 3) Subject metrics (FRN / P3 per subject)
    if "5" in steps:
        script = nb / "step05_metrics.py"
        for sub in subjects:
            # step05_metrics reads from derivatives/sub-XX; no CLI needed
            # We pass subject via env var to keep the script unchanged if it uses the repo default.
            # If your step05 accepts --subject, switch to: run_script(script, ["--subject", sub])
            print(f"\n--- Subject metrics: sub-{sub} ---")
            # Run in a subprocess and let the file look up sub-XX automatically (as you already do)

            res = subprocess.run([sys.executable, str(script)], cwd=str(script.parent))
            if res.returncode != 0:
                raise SystemExit(f"[FATAL] step05_metrics failed for sub-{sub}")

    # 4) Group stats (CSV + plots)
    if "6" in steps:
        script = nb / "step06_group_stats.py"
        run_script(script, [])

    print("\n✅ Pipeline finished successfully.")


if __name__ == "__main__":
    main()
