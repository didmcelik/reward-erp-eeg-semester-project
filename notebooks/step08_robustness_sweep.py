from pathlib import Path
import itertools
import mne
import numpy as np
import pandas as pd
import re


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "ds004147"
DERIV = PROJECT / "derivatives"
OUT_CSV = DERIV / "group" / "robustness.csv"
DERIV.mkdir(exist_ok=True)
(DERIV / "group").mkdir(exist_ok=True)

SUBS = ["27", "31", "33"]   # fast subset; extend as you wish
TASK = "casinos"
HPFS = [0.5, 1.0]           # high-pass candidates
NCOMPS = [10, 15]           # ICA components
REJECT_UV = [150, 200]      # epoch reject thresholds (µV)

FRN_CH, FRN_TMIN, FRN_TMAX = "FCz", 0.2, 0.35
P3_CH,  P3_TMIN,  P3_TMAX  = "Pz",  0.3, 0.5

def find_vhdr(sub):
    eeg_dir = DATA / f"sub-{sub}" / "eeg"
    cand = sorted(eeg_dir.glob(f"sub-{sub}_task-*_eeg.vhdr"))
    if cand: return cand[0]
    raise FileNotFoundError(f"vhdr missing for sub-{sub}")

def parse_task(name):  # fallback if needed
    m = re.search(r"_task-([^-_]+)_eeg\.vhdr$", name)
    return m.group(1) if m else TASK

def amplitude_in_window(evk, ch, tmin, tmax, mode="mean"):
    evk = evk.copy().pick([ch])
    tmask = (evk.times >= tmin) & (evk.times <= tmax)
    data = evk.data.mean(axis=0)[tmask]
    return float(data.mean() if mode=="mean" else data.max())

def run_one(sub, hpf, ncomp, reject_uv):
    vhdr = find_vhdr(sub)
    task = parse_task(vhdr.name)

    # Load raw, minimal preproc variant per config
    raw = mne.io.read_raw_brainvision(vhdr.as_posix(), preload=True, verbose=False)
    raw.filter(hpf, 30.0, fir_design="firwin")
    raw.set_eeg_reference("average")
    raw.resample(250, npad="auto")

    # ICA
    ica = mne.preprocessing.ICA(n_components=ncomp, random_state=97)
    ica.fit(raw)
    try:
        inds, _ = ica.find_bads_eog(raw, ch_name="Fp2" if "Fp2" in raw.ch_names else None)
        ica.exclude = inds
    except Exception:
        pass
    raw = ica.apply(raw)

    # Events from TSV
    tsv = DATA / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-{task}_events.tsv"
    ev = pd.read_csv(tsv, sep="\t")
    mask = (ev["trial_type"].astype(str).str.lower() == "stimulus") & ev["value"].astype(str).str.match(r"^\s*S\s*(\d+)\s*$", na=False)
    ev = ev.loc[mask].copy()
    ev["value"] = ev["value"].astype(str).str.replace(r"\s+","",regex=True)
    ev["code"] = ev["value"].str.extract(r"S(\d+)").astype(int)

    reward = {31, 32, 33}
    norew  = {34, 35, 36, 37}
    ev["cond"] = ev["code"].map(lambda n: "reward" if n in reward else ("no_reward" if n in norew else ""))
    ev = ev.loc[ev["cond"].isin(["reward","no_reward"])]

    onsets = np.round(ev["onset"].to_numpy().astype(float) * raw.info["sfreq"]).astype(int)
    event_id = {"reward":1, "no_reward":2}
    events = np.c_[onsets, np.zeros_like(onsets), ev["cond"].map(event_id).to_numpy()]
    order = np.argsort(events[:,0], kind="mergesort")
    events = events[order]
    events = events[np.unique(events[:,0], return_index=True)[1]]

    # Epochs
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=0.8, baseline=(None, 0),
                        preload=True, detrend=1, event_repeated="drop")
    epochs.drop_bad(reject=dict(eeg=reject_uv*1e-6))

    # Evokeds
    evk_rew = epochs["reward"].average()
    evk_nor = epochs["no_reward"].average()

    # Metrics
    # FRN (No-Reward − Reward) at FCz, mean amplitude
    frn = mne.combine_evoked([evk_nor, evk_rew], weights=[1, -1])
    frn_uv = amplitude_in_window(frn, FRN_CH, FRN_TMIN, FRN_TMAX, mode="mean")
    # P3 (Reward) at Pz
    p3_uv  = amplitude_in_window(evk_rew, P3_CH, P3_TMIN, P3_TMAX, mode="mean")

    return dict(subject=sub, hpf=hpf, ncomp=ncomp, reject_uv=reject_uv,
                n_reward=len(epochs["reward"]), n_noreward=len(epochs["no_reward"]),
                frn_uv=frn_uv, p3_uv=p3_uv)

def main():
    rows = []
    for hpf, ncomp, rej in itertools.product(HPFS, NCOMPS, REJECT_UV):
        for sub in SUBS:
            try:
                rows.append(run_one(sub, hpf, ncomp, rej))
                print(f"[OK] sub-{sub} | HPF={hpf}Hz | ncomp={ncomp} | rej={rej}µV")
            except Exception as e:
                print(f"[FAIL] sub-{sub} {hpf}/{ncomp}/{rej}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
