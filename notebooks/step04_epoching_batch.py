from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne
from collections import Counter
import argparse

# ----------------------------
# Defaults (edit if needed)
# ----------------------------
TASK = "casinos"
REWARD_CODES   = {31, 32, 33}       # edit if your dataset uses different labels
NOREWARD_CODES = {34, 35, 36, 37}   # edit if your dataset uses different labels
TMIN, TMAX = -0.2, 0.8              # epoch window (s)
BASELINE = (None, 0.0)
REJECT = dict(eeg=200e-6)           # µV threshold; adjust for your data quality

# ----------------------------
# Helpers
# ----------------------------
def project_root_from_here(this_file: Path) -> Path:
    # Works whether this file is inside notebooks/ or deeper
    here = this_file.resolve()
    for p in [here, *here.parents]:
        root = p if p.is_dir() else p.parent
        if (root / "data" / "ds004147").exists():
            return root
    raise RuntimeError("Could not locate project root. Ensure 'data/ds004147' exists.")

def list_subjects(data_root: Path, pattern=r"sub-(\d+)$", include=None):
    subs = []
    for p in sorted(data_root.glob("sub-*")):
        m = re.match(pattern, p.name)
        if m:
            subs.append(m.group(1))
    if include:
        subs = [s for s in subs if s in set(include)]
    return subs

def load_raw_for_subject(deriv: Path, sub: str, task: str) -> mne.io.BaseRaw | None:
    fif_ica  = deriv / f"sub-{sub}_task-{task}_ica_raw.fif"
    fif_pre  = deriv / f"sub-{sub}_task-{task}_preproc_raw.fif"
    pick = fif_ica if fif_ica.exists() else (fif_pre if fif_pre.exists() else None)
    if pick is None:
        print(f"[WARN] No preprocessed raw for sub-{sub} (expected {fif_ica.name} or {fif_pre.name}). Skipping.")
        return None
    raw = mne.io.read_raw_fif(pick.as_posix(), preload=True, verbose=False)
    return raw

def parse_events(data_root: Path, sub: str, task: str, sfreq: float):
    ev_tsv = data_root / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-{task}_events.tsv"
    if not ev_tsv.exists():
        print(f"[WARN] Missing events.tsv for sub-{sub}. Skipping.")
        return None, None

    ev = pd.read_csv(ev_tsv, sep="\t")
    # Keep only BrainVision Stimulus 'S xx' markers
    is_stim = ev["trial_type"].astype(str).str.lower().eq("stimulus")
    is_Sxx  = ev["value"].astype(str).str.match(r"^\s*S\s*\d+\s*$", na=False)
    ev = ev.loc[is_stim & is_Sxx].copy()

    # 'S 31' -> 'S31'; extract numeric code
    ev["value"] = ev["value"].astype(str).str.replace(r"\s+", "", regex=True)
    ev["code_num"] = ev["value"].str.extract(r"S(\d+)").astype(int)

    # Deduplicate collisions by onset; keep last
    ev.sort_values(["onset", "code_num"], inplace=True)
    ev = ev.groupby("onset", as_index=False).last()

    # Quick counts to sanity-check mapping
    print("[INFO] S-code counts:", Counter(ev["value"]))

    def code_to_cond(n: int) -> str:
        if n in REWARD_CODES: return "reward"
        if n in NOREWARD_CODES: return "no_reward"
        return ""

    ev["cond"] = ev["code_num"].map(code_to_cond)
    ev = ev.loc[ev["cond"].isin(["reward", "no_reward"])].copy()

    if ev.empty:
        print("[WARN] No reward/no-reward events after mapping. Check REWARD_CODES/NOREWARD_CODES.")
        return None, None

    onset_samples = np.round(ev["onset"].to_numpy().astype(float) * sfreq).astype(int)
    event_id = {"reward": 1, "no_reward": 2}
    codes = ev["cond"].map(event_id).to_numpy()
    events = np.c_[onset_samples, np.zeros_like(codes), codes]

    # Ensure increasing unique sample indices
    order = np.argsort(events[:, 0], kind="mergesort")
    events = events[order]
    events = events[np.unique(events[:, 0], return_index=True)[1]]
    print(f"[INFO] Final events: {len(events)} "
          f"(reward={np.sum(events[:,2]==1)}, no_reward={np.sum(events[:,2]==2)})")
    return events, event_id

def process_subject(proj_root: Path, sub: str, overwrite: bool = True):
    data_root = proj_root / "data" / "ds004147"
    deriv_root = proj_root / "derivatives"
    out_dir = deriv_root / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_for_subject(deriv_root, sub, TASK)
    if raw is None:
        return

    # Build events
    events, event_id = parse_events(data_root, sub, TASK, raw.info["sfreq"])
    if events is None:
        return

    # Epochs
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                        preload=True, detrend=1, event_repeated="drop")
    epochs = epochs.drop_bad(reject=REJECT)
    print(epochs)

    # Evokeds
    evk_reward = epochs["reward"].average()
    evk_norew  = epochs["no_reward"].average()

    # Save
    epo_fif = out_dir / f"sub-{sub}_epo.fif"
    rew_fif = out_dir / f"sub-{sub}_reward-ave.fif"
    nor_fif = out_dir / f"sub-{sub}_noreward-ave.fif"
    epochs.save(epo_fif.as_posix(), overwrite=overwrite)
    evk_reward.save(rew_fif.as_posix(), overwrite=overwrite)
    evk_norew.save(nor_fif.as_posix(), overwrite=overwrite)
    print(f"[INFO] Saved: {epo_fif.name}, {rew_fif.name}, {nor_fif.name} → {out_dir}")
# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch epoching for ds004147 (reward vs no-reward)")
    ap.add_argument("--subs", nargs="*", help="Optional list of subject IDs to process (e.g., 27 28 29). Default: all found.")
    ap.add_argument("--reject_uv", type=float, default=200.0, help="EEG rejection threshold in µV (default 200).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing FIF files.")
    args = ap.parse_args()

    PROJECT = project_root_from_here(Path(__file__))
    DATA = PROJECT / "data" / "ds004147"
    DERIV = PROJECT / "derivatives"

    # update reject from CLI
    REJECT = dict(eeg=args.reject_uv * 1e-6)

    subs = list_subjects(DATA, include=args.subs)
    if not subs:
        raise SystemExit("No subjects found under data/ds004147.")

    print(f"[INFO] Subjects: {', '.join(subs)}")
    for s in subs:
        print(f"\n=== Processing sub-{s} ===")
        process_subject(PROJECT, s, overwrite=args.overwrite)

    print("\n[INFO] Batch epoching finished.")
