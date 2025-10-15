# Group-level ERP stats for ds004147
# - Scans subjects under data/ds004147/sub-xx
# - Loads evokeds saved in derivatives/sub-xx from Step 4
# - Computes FRN (No-Reward − Reward @ FCz, 200–350 ms) and P300 (Reward @ Pz, 300–500 ms)
# - Saves group CSV + figures + prints t-tests

from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import pingouin as pg

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "ds004147"
DERIV = PROJECT / "derivatives"
OUT   = DERIV / "group"
OUT.mkdir(parents=True, exist_ok=True)

# ---- analysis parameters ----
FRN_CH = "FCz"
FRN_WIN = (0.200, 0.350)   # s
P3_CH  = "Pz"
P3_WIN = (0.300, 0.500)    # s

def list_subjects():
    subs = []
    for p in sorted((DATA).glob("sub-*")):
        m = re.match(r"sub-(\d+)$", p.name)
        if m:
            subs.append(m.group(1))
    return subs

def mean_amp(evk: mne.Evoked, ch: str, tmin: float, tmax: float) -> float:
    """Return mean amplitude (µV) in [tmin, tmax] at channel."""
    evk = evk.copy().pick(ch)
    idx = (evk.times >= tmin) & (evk.times <= tmax)
    return float(evk.data[0, idx].mean() * 1e6)

def load_subject_metrics(sub: str):
    sub_dir = DERIV / f"sub-{sub}"
    f_rew   = sub_dir / f"sub-{sub}_reward-ave.fif"
    f_nor   = sub_dir / f"sub-{sub}_noreward-ave.fif"
    if not (f_rew.exists() and f_nor.exists()):
        print(f"[WARN] Missing evokeds for sub-{sub}; skipping.")
        return None

    evk_rew   = mne.read_evokeds(f_rew.as_posix())[0]
    evk_nor   = mne.read_evokeds(f_nor.as_posix())[0]
    diff      = mne.combine_evoked([evk_rew, evk_nor], weights=[-1, 1])

    frn_uv = mean_amp(diff, FRN_CH, *FRN_WIN)         # No-Reward − Reward
    p3_uv  = mean_amp(evk_rew, P3_CH, *P3_WIN)        # Reward condition

    return {
        "subject": sub,
        "n_reward": evk_rew.nave,
        "n_noreward": evk_nor.nave,
        "frn_uv": frn_uv,
        "p3_uv": p3_uv,
    }

# -------- run over all subjects --------
rows = []
for sub in list_subjects():
    rec = load_subject_metrics(sub)
    if rec is not None:
        rows.append(rec)

if not rows:
    raise SystemExit("No subjects with saved ERPs found. Run step04_epoching.py first.")

df = pd.DataFrame(rows).sort_values("subject")
csv_path = OUT / "group_erp_metrics.csv"
df.to_csv(csv_path, index=False)
print(df)
print("[INFO] Saved:", csv_path)

# -------- inferential stats --------
# One-sample t-tests against 0 µV (typical for difference waves and component amplitudes)
frn_test = pg.ttest(df["frn_uv"], 0.0, alternative="two-sided")
p3_test  = pg.ttest(df["p3_uv"],  0.0, alternative="two-sided")

# Cohen's d (uses unbiased Hedge's g by default in pingouin; we can report 'cohen-d' column)
frn_d = pg.compute_effsize(df["frn_uv"], np.zeros(len(df)), eftype="cohen")
p3_d  = pg.compute_effsize(df["p3_uv"],  np.zeros(len(df)), eftype="cohen")

print("\n=== FRN (No-Reward − Reward @ FCz, 200–350 ms) ===")
print(frn_test)
print(f"Cohen's d: {frn_d:.3f}")

print("\n=== P300 (Reward @ Pz, 300–500 ms) ===")
print(p3_test)
print(f"Cohen's d: {p3_d:.3f}")

# -------- simple plots (Matplotlib) --------
plt.figure(figsize=(6,4))
plt.boxplot(df["frn_uv"], vert=True, labels=["FRN (µV)"])
plt.axhline(0, linestyle="--")
plt.title("FRN across subjects")
plt.ylabel("µV")
plt.tight_layout()
plt.savefig(OUT / "frn_boxplot.png", dpi=150)

plt.figure(figsize=(6,4))
plt.boxplot(df["p3_uv"], vert=True, labels=["P300 (µV)"])
plt.axhline(0, linestyle="--")
plt.title("P300 across subjects (Reward @ Pz)")
plt.ylabel("µV")
plt.tight_layout()
plt.savefig(OUT / "p3_boxplot.png", dpi=150)

print("[INFO] Figures saved to:", OUT)
