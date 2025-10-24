"""
02_events_epoch_erp.py

EEG Semester Project — ds004147
--------------------------------
From cleaned EEG (after ICA) → events → drop first 10 trials per condition →
create epochs and compute ERP + RewP.

Usage example:
    cd project\pipeline
    python .\02_events_epoch_erp.py --sub 27 --root ../data/ds004147 --deriv ../data/ds004147/derivatives

"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import mne


def parse_args():
    p = argparse.ArgumentParser(description="Epoch and ERP extraction for ds004147")
    p.add_argument("--sub", type=str, required=True, help="Subject ID (e.g., 27)")
    p.add_argument("--root", type=str, default="./data/ds004147", help="Dataset root path")
    p.add_argument("--deriv", type=str, default=None, help="Derivatives path (default: <root>/derivatives)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    sub = args.sub
    deriv = Path(args.deriv) if args.deriv else (root / "derivatives" / f"sub-{sub}")
    deriv.mkdir(parents=True, exist_ok=True)

    # ---------- File paths ----------
    fif_clean = deriv / f"sub-{sub}_task-casinos_clean_eeg_raw.fif"
    ev_tsv = root / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-casinos_events.tsv"
    ev_tsv_clean = deriv / "05_events_clean.tsv"
    epochs_fif = deriv / "06_epochs-feedback-0.2-0.6.fif"
    evokeds_fif = deriv / "07_evoked-feedback-avg.fif"
    rewp_fif = deriv / "08_evoked-RewP.fif"

    # ---------- Load cleaned EEG ----------
    raw = mne.io.read_raw_fif(fif_clean, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    print(f"[EEG] Loaded {fif_clean.name} ({len(raw.ch_names)} ch, {sfreq} Hz)")

    # ---------- Load and clean events ----------
    ev = pd.read_csv(ev_tsv, sep="\t")
    codes = pd.to_numeric(ev["value"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    mask_valid = codes.notna()
    ev = ev.loc[mask_valid].copy()
    ev["code"] = codes[mask_valid].astype(int)
    ev.reset_index(drop=True, inplace=True)

    # Onset is in milliseconds (check events.json) → convert to seconds
    ev["onset_sec"] = ev["onset"] / 1000.0

    # ---------- Trial structure per condition ----------
    TRIAL_STARTS = {1: "LL", 11: "ML", 21: "MH", 31: "HH"}
    cond_counters = {k: 0 for k in TRIAL_STARTS.values()}
    current_cond = None
    cond_for_row, trial_no_for_row = [], []

    for c in ev["code"].values:
        if c in TRIAL_STARTS:
            current_cond = TRIAL_STARTS[c]
            cond_counters[current_cond] += 1
        cond_for_row.append(current_cond if current_cond else "NA")
        trial_no_for_row.append(cond_counters.get(current_cond, 0) if current_cond else 0)

    ev["cond"] = cond_for_row
    ev["trial_no"] = trial_no_for_row

    # ---------- Drop first 10 trials per condition ----------
    keep_mask = (ev["trial_no"] == 0) | (
        ev["cond"].isin(TRIAL_STARTS.values()) & (ev["trial_no"] > 10)
    )
    ev_clean = ev.loc[keep_mask].reset_index(drop=True)
    ev_clean.to_csv(ev_tsv_clean, sep="\t", index=False)
    print(f"[EVENTS] Before={len(ev)} | After={len(ev_clean)} | Saved {ev_tsv_clean.name}")

    # ---------- Convert to MNE events array ----------
    on_samp = np.round(ev_clean["onset_sec"].values * sfreq).astype(int)
    codes_clean = ev_clean["code"].values.astype(int)
    events = np.c_[on_samp, np.zeros_like(on_samp), codes_clean]

    # ---------- Feedback codes ----------
    all_win = [6, 16, 26, 36]
    all_loss = [7, 17, 27, 37]
    avail = set(codes_clean.tolist())
    win_codes = [c for c in all_win if c in avail]
    loss_codes = [c for c in all_loss if c in avail]

    if not win_codes and not loss_codes:
        raise RuntimeError("No feedback events (win/loss) found after cleaning.")

    event_id = {f"S{c}": c for c in (win_codes + loss_codes)}
    groups = {"win": [f"S{c}" for c in win_codes], "loss": [f"S{c}" for c in loss_codes]}
    print(f"[EVENTS] Win: {win_codes} | Loss: {loss_codes}")

    # ---------- Create epochs ----------
    tmin, tmax = -0.2, 0.6
    baseline = (-0.2, 0)
    reject = dict(eeg=150e-6)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        detrend=None,
        verbose=False,
    )
    epochs.save(epochs_fif, overwrite=True)
    print(f"[EPOCHS] Saved {epochs_fif.name} (n={len(epochs)})")

    # ---------- ERP averages ----------
    evokeds = {}
    if groups["win"]:
        evokeds["win"] = epochs[groups["win"]].average()
    if groups["loss"]:
        evokeds["loss"] = epochs[groups["loss"]].average()
    mne.evoked.write_evokeds(evokeds_fif, list(evokeds.values()), overwrite=True)
    print(f"[ERP] Saved {evokeds_fif.name} | Keys={list(evokeds.keys())}")

    # ---------- RewP (Win - Loss) ----------
    if ("win" in evokeds) and ("loss" in evokeds):
        rewp = mne.combine_evoked([evokeds["win"], evokeds["loss"]], weights=[1, -1])
        rewp.comment = "RewP (Win-Loss)"
        rewp.save(rewp_fif, overwrite=True)
        print(f"[RewP] Saved {rewp_fif.name}")
    else:
        print("[RewP] Skipped (need both win and loss)")

    print("[DONE]")


if __name__ == "__main__":
    main()
