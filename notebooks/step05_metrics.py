from pathlib import Path
import numpy as np
import pandas as pd
import mne

PROJECT = Path(__file__).resolve().parents[1]
DERIV = PROJECT / "derivatives"

# ---- choose subjects you have ----
SUBJECTS = ["27"]  # extend later: ["27","28","29",...]

# ---- analysis settings ----
frn_chan = "FCz"
frn_win  = (0.200, 0.350)  # s, mean amplitude of (No-Reward − Reward)

p3_chan  = "Pz"
p3_win   = (0.300, 0.500)  # s, mean amplitude of Reward (or use No-Reward if you prefer)

rows = []

for sub in SUBJECTS:
    sub_dir = DERIV / f"sub-{sub}"
    # load evokeds saved in step04
    evk_rew   = mne.read_evokeds((sub_dir / f"sub-{sub}_reward-ave.fif").as_posix())[0]
    evk_norew = mne.read_evokeds((sub_dir / f"sub-{sub}_noreward-ave.fif").as_posix())[0]

    # difference wave for FRN
    diff = mne.combine_evoked([evk_rew, evk_norew], weights=[-1, 1])

    # helper to compute mean amplitude in a window at a channel
    def mean_amp(evk, ch_name, tmin, tmax):
        pick = evk.copy().pick(ch_name)
        idx = (pick.times >= tmin) & (pick.times <= tmax)
        return float(pick.data[0, idx].mean() * 1e6)  # µV

    frn_uv = mean_amp(diff, frn_chan, *frn_win)
    p3_uv  = mean_amp(evk_rew, p3_chan, *p3_win)

    # trial counts (stored in comment or use epochs export if you prefer)
    n_rew = evk_rew.nave
    n_nor = evk_norew.nave

    rows.append(dict(
        subject=sub,
        n_reward=n_rew, n_noreward=n_nor,
        frn_chan=frn_chan, frn_tmin=frn_win[0], frn_tmax=frn_win[1], frn_uv=frn_uv,
        p3_chan=p3_chan, p3_tmin=p3_win[0], p3_tmax=p3_win[1], p3_uv=p3_uv
    ))

df = pd.DataFrame(rows)
out_csv = (DERIV / "erp_metrics.csv")
df.to_csv(out_csv, index=False)
print(df)
print("[INFO] Saved:", out_csv)
