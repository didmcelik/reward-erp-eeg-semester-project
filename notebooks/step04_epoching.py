from pathlib import Path
import numpy as np
import pandas as pd
import mne
from collections import Counter
import re

# --- Paths ---
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "ds004147"
DERIV = PROJECT / "derivatives"
SUBJECT = "27"
TASK = "casinos"

# Load blink-cleaned raw (from Step 3). If you skipped ICA, you can load the preproc file instead.
raw_fif = DERIV / f"sub-{SUBJECT}_task-{TASK}_ica_raw.fif"
if not raw_fif.exists():
    raw_fif = DERIV / f"sub-{SUBJECT}_task-{TASK}_preproc_raw.fif"
raw = mne.io.read_raw_fif(raw_fif.as_posix(), preload=True)
print("[INFO] Loaded:", raw_fif)

# -------------------------
# Read BIDS events.tsv and reduce to BrainVision 'Sxx' Stimulus markers
# -------------------------
ev_tsv = DATA / f"sub-{SUBJECT}" / "eeg" / f"sub-{SUBJECT}_task-{TASK}_events.tsv"
ev = pd.read_csv(ev_tsv, sep="\t")

# Keep only proper Stimulus rows with S-codes like 'S 31' or 'S31'
is_stim = ev["trial_type"].astype(str).str.lower().eq("stimulus")
is_Sxx  = ev["value"].astype(str).str.match(r"^\s*S\s*\d+\s*$", na=False)
ev = ev.loc[is_stim & is_Sxx].copy()

# Normalize: 'S 31' -> 'S31', and extract the integer code
ev["value"] = ev["value"].astype(str).str.replace(r"\s+", "", regex=True)
ev["code_num"] = ev["value"].str.extract(r"S(\d+)").astype(int)

# Deduplicate by onset (keep the last of any collisions)
ev.sort_values(["onset", "code_num"], inplace=True)
ev = ev.groupby("onset", as_index=False).last()

print("[INFO] S-code counts:", Counter(ev["value"]))

# ---- Define mapping (based on your subject-27 counts) ----
REWARD_CODES   = {31, 32, 33}
NOREWARD_CODES = {34, 35, 36, 37}

def code_to_cond(n: int) -> str:
    if n in REWARD_CODES:
        return "reward"
    if n in NOREWARD_CODES:
        return "no_reward"
    return ""

ev["cond"] = ev["code_num"].map(code_to_cond)
ev = ev.loc[ev["cond"].isin(["reward", "no_reward"])].copy()

# Build MNE events array
sfreq = raw.info["sfreq"]
onset_samples = np.round(ev["onset"].to_numpy().astype(float) * sfreq).astype(int)
event_id = {"reward": 1, "no_reward": 2}
codes = ev["cond"].map(event_id).to_numpy()
events = np.c_[onset_samples, np.zeros_like(codes), codes]

# Ensure strictly increasing unique sample indices
order = np.argsort(events[:, 0], kind="mergesort")
events = events[order]
events = events[np.unique(events[:, 0], return_index=True)[1]]
print(f"[INFO] Final events: {len(events)}  "
      f"(reward={np.sum(events[:,2]==1)}, no_reward={np.sum(events[:,2]==2)})")

# -------------------------
# Epoching
# -------------------------
tmin, tmax = -0.2, 0.8
epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=tmin, tmax=tmax, baseline=(None, 0.0),
                    preload=True, detrend=1, event_repeated="drop")

# Light rejection (adjust if too many drops)
reject = dict(eeg=200e-6)  # was 150e-6; a bit more lenient
epochs = epochs.drop_bad(reject=reject)
print(epochs)

# -------------------------
# Condition ERPs + quick plots
# -------------------------
evk_reward = epochs["reward"].average()
evk_norew  = epochs["no_reward"].average()
frn_diff   = mne.combine_evoked([evk_reward, evk_norew], weights=[-1, 1])  # No-Reward − Reward

# If montage was set earlier, topomaps will work; otherwise plot traces.
try:
    evk_reward.plot_joint(title=f"sub-{SUBJECT} — Reward")
    evk_norew.plot_joint(title=f"sub-{SUBJECT} — No-Reward")
except Exception:
    evk_reward.plot(picks="FCz", titles=f"sub-{SUBJECT} — Reward (FCz)")
    evk_norew.plot(picks="FCz", titles=f"sub-{SUBJECT} — No-Reward (FCz)")

mne.viz.plot_compare_evokeds(
    {"Reward": evk_reward, "No-Reward": evk_norew},
    picks="FCz", combine="mean", title=f"sub-{SUBJECT} — FCz comparison"
)
frn_diff.plot(picks="FCz", titles=f"sub-{SUBJECT} — FRN (No-Reward − Reward)")

# -------------------------
# Save outputs
# -------------------------
out_dir = DERIV / f"sub-{SUBJECT}"
out_dir.mkdir(parents=True, exist_ok=True)
epochs.save((out_dir / f"sub-{SUBJECT}_epo.fif").as_posix(), overwrite=True)
evk_reward.save((out_dir / f"sub-{SUBJECT}_reward-ave.fif").as_posix(), overwrite=True)
evk_norew.save((out_dir / f"sub-{SUBJECT}_noreward-ave.fif").as_posix(), overwrite=True)
print("[INFO] Saved epochs and evokeds to:", out_dir)
