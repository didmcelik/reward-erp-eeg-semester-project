from pathlib import Path
import re
import mne
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "ds004147"
DERIV = PROJECT / "derivatives"
QC_DIR = DERIV / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

def list_subjects():
    subs = []
    for p in sorted(DATA.glob("sub-*")):
        m = re.match(r"sub-(\d+)$", p.name)
        if m: subs.append(m.group(1))
    return subs

def load_evokeds(sub):
    sub_dir = DERIV / f"sub-{sub}"
    evk_rew = mne.read_evokeds(sub_dir / f"sub-{sub}_reward-ave.fif", condition=0, verbose=False)
    evk_nor = mne.read_evokeds(sub_dir / f"sub-{sub}_noreward-ave.fif", condition=0, verbose=False)
    return evk_rew, evk_nor

def code_counts(sub):
    tsv = DATA / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-casinos_events.tsv"
    if not tsv.exists(): return {}
    df = pd.read_csv(tsv, sep="\t")
    mask = (df["trial_type"].astype(str).str.lower() == "stimulus") & df["value"].astype(str).str.match(r"^\s*S\s*\d+\s*$", na=False)
    cnt = df.loc[mask, "value"].astype(str).str.replace(r"\s+","",regex=True).value_counts().to_dict()
    return cnt

def add_psd(report, raw, title):
    psd = raw.compute_psd(fmax=60)
    fig = psd.plot(picks="eeg", average=False, dB=True, show=False)
    report.add_figure(fig, title=title, section="PSD")

def main():
    subjects = list_subjects()
    cards = []

    for sub in subjects:
        rep = mne.Report(title=f"QC — sub-{sub}")
        # Paths
        fif_pre = DERIV / f"sub-{sub}_task-casinos_preproc_raw.fif"
        fif_ica = DERIV / f"sub-{sub}_task-casinos_ica_raw.fif"
        epo_fif = DERIV / f"sub-{sub}" / f"sub-{sub}_epo.fif"

        # Raw (preproc)
        if fif_pre.exists():
            raw_pre = mne.io.read_raw_fif(fif_pre.as_posix(), preload=False, verbose=False)
            try:
                add_psd(rep, raw_pre, "Preprocessed Raw — PSD")
            except Exception as e:
                rep.add_html(f"<p>PSD failed: {e}</p>", title="Preproc PSD")

        # Raw (ICA)
        if fif_ica.exists():
            raw_ica = mne.io.read_raw_fif(fif_ica.as_posix(), preload=False, verbose=False)
            try:
                add_psd(rep, raw_ica, "ICA-cleaned Raw — PSD")
            except Exception as e:
                rep.add_html(f"<p>PSD failed: {e}</p>", title="ICA PSD")

        # ICA diagnostics (if we can recover ICA from file name; not stored separately, so show evoked-based)
        # Tip: you could persist ICA .fif in derivatives/ica/ to plot components. Skipping here if absent.

        # Events counts
        cnt = code_counts(sub)
        if cnt:
            table_html = "<table><tr><th>Code</th><th>Count</th></tr>" + "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in sorted(cnt.items())
            ) + "</table>"
            rep.add_html(table_html, title="Stimulus S-codes", section="Events")

        # Epochs / drops
        if epo_fif.exists():
            epochs = mne.read_epochs(epo_fif.as_posix(), preload=False, verbose=False)
            try:
                fig = epochs.plot_drop_log(show=False)
                rep.add_figure(fig, title="Epoch Drop Log", section="Epochs")
            except Exception:
                pass

        # Evokeds
        try:
            evk_rew, evk_nor = load_evokeds(sub)
            # Overlay at FCz/Pz
            for picks, label in [(["FCz"], "FCz"), (["Pz"], "Pz")]:
                fig = mne.viz.plot_compare_evokeds(
                    {"Reward": evk_rew.copy().pick(picks),
                     "No-Reward": evk_nor.copy().pick(picks)},
                    combine="mean", show=False, title=f"Evoked @ {label}")
                rep.add_figure(fig, title=f"Evoked @ {label}", section="Evokeds")

            # Difference wave (FRN style: No-Reward − Reward)
            evk_diff = mne.combine_evoked([evk_nor, evk_rew], weights=[1, -1])
            fig = evk_diff.plot_joint(times=np.linspace(0.25, 0.35, 3), title="Difference (No-Reward − Reward)", show=False)
            rep.add_figure(fig, title="FRN diff (No-Reward − Reward)", section="Evokeds")

        except Exception as e:
            rep.add_html(f"<p>Evoked plotting failed: {e}</p>", title="Evokeds")

        out_html = QC_DIR / f"sub-{sub}_report.html"
        rep.save(out_html.as_posix(), overwrite=True, open_browser=False)
        cards.append((sub, out_html.name))

    # Simple index
    idx = QC_DIR / "index.html"
    with idx.open("w", encoding="utf-8") as f:
        f.write("<h1>QC Reports</h1><ul>")
        for sub, name in cards:
            f.write(f'<li><a href="{name}">sub-{sub}</a></li>')
        f.write("</ul>")
    print(f"[INFO] QC reports written to: {QC_DIR}")

if __name__ == "__main__":
    main()
