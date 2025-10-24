"""
02b_qc_raw_vs_clean.py

QA / QC for ds004147:
- Load preprocessed (filt+ref) raw and ICA-cleaned raw
- Compare PSDs (median across channels)
- Optional: also show the BrainVision (unfiltered) PSD with the 50 Hz spike
- Show 10-second example traces for a few channels
- Plot 50 Hz power reduction (raw -> clean) as a topomap

Usage examples (from project/pipeline):

  python 02b_qc_raw_vs_clean.py --sub 27 --root ..\data\ds004147 --deriv ..\data\ds004147\derivatives

  # If you also want the "unfiltered BrainVision PSD" panel (needs .vhdr present):
  python 02b_qc_raw_vs_clean.py --sub 27 --root ..\data\ds004147 --deriv ..\data\ds004147\derivatives --show_vhdr_psd 1
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import mne

from mne.viz import plot_topomap
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="QC: Raw (filt+ref) vs Clean (ICA)")
    p.add_argument("--sub", required=True, type=str, help="Subject ID, e.g., 27")
    p.add_argument("--root", default="./data/ds004147", type=str, help="Dataset root")
    p.add_argument("--deriv", default=None, type=str,
                   help="Derivatives dir (default: <root>/derivatives/sub-XX)")
    p.add_argument("--show_vhdr_psd", type=int, default=0,
                   help="If 1, also plot PSD of the original BrainVision file (unfiltered).")
    return p.parse_args()


def _safe_read_raw(path: Path, preload: bool = False):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return mne.io.read_raw_fif(path, preload=preload, verbose=False)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    sub = args.sub
    deriv = Path(args.deriv) if args.deriv else (root / "derivatives" / f"sub-{sub}")
    deriv.mkdir(parents=True, exist_ok=True)

    # ---- expected inputs from step 01 ----
    preproc_fif = deriv / f"sub-{sub}_task-casinos_preproc_eeg_raw.fif"
    clean_fif   = deriv / f"sub-{sub}_task-casinos_clean_eeg_raw.fif"

    # ---- optional raw BrainVision (unfiltered) for visualizing the 50 Hz spike ----
    vhdr = root / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-casinos_eeg.vhdr"

    print(f"[INFO] Loading preproc: {preproc_fif.name}")
    raw0 = _safe_read_raw(preproc_fif, preload=False)
    print(f"[INFO] Loading clean  : {clean_fif.name}")
    rawC = _safe_read_raw(clean_fif, preload=False)

    # ----------- PSD: median across channels (0.5–60 Hz) -----------
    def median_psd(raw, fmin=0.5, fmax=60.0):
        psd = raw.compute_psd(fmin=fmin, fmax=fmax, picks="eeg", verbose=False)
        f, p = psd.freqs, psd.get_data()  # shape: (n_channels, n_freqs)
        return f, np.median(p, axis=0)

    f0, P0 = median_psd(raw0)
    fC, PC = median_psd(rawC)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(f0, 10 * np.log10(P0 * 1e12), label="Raw (filt+ref)")   # dB/Hz re 1 µV^2
    ax.plot(fC, 10 * np.log10(PC * 1e12), label="Clean (ICA)")
    ax.axvline(50, ls="--", color="0.5")
    ax.set_title(f"Median PSD across channels — sub-{sub}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB/Hz re 1 µV²)")
    ax.legend(loc="best")
    psd_out = deriv / "12_psd_median_raw_vs_clean.png"
    fig.tight_layout()
    fig.savefig(psd_out, dpi=160)
    plt.close(fig)
    print(f"[OUT] {psd_out.name}")

    # ----------- Optional: Unfiltered BrainVision PSD to show the 50 Hz spike -----------
    if args.show_vhdr_psd and vhdr.exists():
        raw_vhdr = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)
        psd = raw_vhdr.compute_psd(fmin=0.5, fmax=60, picks="eeg", verbose=False)
        f, P = psd.freqs, psd.get_data()
        fig, ax = plt.subplots(figsize=(8.8, 4.5))
        for ch in P:
            ax.plot(f, 10 * np.log10(ch * 1e12), lw=1, alpha=0.7)
        ax.axvline(50, ls="--", color="0.4")
        ax.set_title(f"Unfiltered BrainVision PSD — sub-{sub}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB/Hz re 1 µV²)")
        fig.tight_layout()
        out = deriv / "11_psd_unfiltered_brainvision.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"[OUT] {out.name}")
    elif args.show_vhdr_psd:
        print(f"[WARN] --show_vhdr_psd=1 but missing {vhdr.name}; skipped.")

    # ----------- Example 10 s traces: Raw vs Clean -----------
    # pick a central window ~ middle of the recording
    dur = 10.0
    tmax = min(raw0.times[-1], rawC.times[-1])
    t0 = max(0.0, tmax / 2 - dur / 2)

    chs = ["FCz", "Cz", "Fp1", "Fp2"]
    picks = [ch for ch in chs if ch in raw0.ch_names and ch in rawC.ch_names]
    if not picks:
        picks = mne.pick_types(raw0.info, eeg=True)[:4]
        picks = [raw0.ch_names[p] for p in picks]

    def _data_slice(raw, ch_name, t0, dur):
        idx = raw.ch_names.index(ch_name)
        s0 = int(np.round(t0 * raw.info["sfreq"]))
        s1 = int(np.round((t0 + dur) * raw.info["sfreq"]))
        data, _ = raw[idx, s0:s1]
        return data[0], np.arange(s0, s1) / raw.info["sfreq"]

    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 7.5), sharex=True)
    for ax, ch in zip(np.atleast_1d(axes), picks):
        y0, t = _data_slice(raw0, ch, t0, dur)
        yC, _ = _data_slice(rawC, ch, t0, dur)
        ax.plot(t, y0 * 1e6, label="Raw", alpha=0.85)
        ax.plot(t, yC * 1e6, label="Clean", alpha=0.85)
        ax.set_ylabel(f"{ch} (µV)")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.grid(False)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"10 s example traces — sub-{sub}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    traces_out = deriv / "10_example_traces_raw_vs_clean.png"
    fig.savefig(traces_out, dpi=160)
    plt.close(fig)
    print(f"[OUT] {traces_out.name}")
    # ----------- 50 Hz power reduction topomap (Raw - Clean, in dB) -----------
    # ----------- 50 Hz power reduction topomap (Raw - Clean, in dB) -----------


    def band_power(raw, f_center=50.0, bw=1.0):
        fmin, fmax = f_center - bw, f_center + bw
        psd = raw.compute_psd(fmin=fmin, fmax=fmax, picks="eeg", verbose=False)
        P = psd.get_data()
        df = np.diff(psd.freqs).mean() if len(psd.freqs) > 1 else (fmax - fmin)
        return (P.sum(axis=1) * df).astype(float)

    P50_raw = band_power(raw0, f_center=50.0, bw=1.0)
    P50_cln = band_power(rawC, f_center=50.0, bw=1.0)

    # dB reduction (positive values mean Raw > Clean, i.e., reduction after cleaning)
    eps = 1e-30
    diff_db = 10 * np.log10((P50_raw + eps) / (P50_cln + eps))

    # Use EEG-only info for plotting
    info = mne.pick_info(raw0.info, mne.pick_types(raw0.info, eeg=True))

    # Robust color limits
    vmin = float(np.percentile(diff_db, 10))
    vmax = float(np.percentile(diff_db, 90))

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    try:
        # Newer MNE: accepts vmin/vmax
        plot_topomap(diff_db, info, axes=ax, cmap="Reds",
                     vmin=vmin, vmax=vmax, contours=4)
    except TypeError:
        # Older MNE: uses vlim instead of vmin/vmax
        plot_topomap(diff_db, info, axes=ax, cmap="Reds",
                     vlim=(vmin, vmax), contours=4)

    ax.set_title("50 Hz power reduction (dB)   Raw → Clean", pad=14)
    topo_out = deriv / "13_topomap_50Hz_reduction.png"
    fig.tight_layout()
    fig.savefig(topo_out, dpi=160)
    plt.close(fig)
    print(f"[OUT] {topo_out.name}")

if __name__ == "__main__":
    main()
