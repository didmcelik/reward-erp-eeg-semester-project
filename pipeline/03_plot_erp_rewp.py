"""
03_plot_erp_rewp.py

From subject evokeds → plot Win/Loss at FCz, RewP topomap, and extract metrics.

Usage:
  python 03_plot_erp_rewp.py --sub 27 --root ../data/ds004147 --deriv ../data/ds004147/derivatives
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot ERP & RewP and extract metrics")
    p.add_argument("--sub", type=str, required=True, help="Subject ID (e.g., 27)")
    p.add_argument("--root", type=str, default="./data/ds004147", help="Dataset root")
    p.add_argument("--deriv", type=str, default=None, help="Derivatives dir (default: <root>/derivatives/sub-XX)")
    p.add_argument("--fc", type=str, default="FCz", help="Target channel for waveforms (default: FCz)")
    p.add_argument("--win_start", type=float, default=0.24, help="RewP window start (s)")
    p.add_argument("--win_end", type=float, default=0.34, help="RewP window end (s)")
    p.add_argument("--peak_window", type=str, default="0.20,0.40", help="Peak search window in sec, e.g. '0.20,0.40'")
    return p.parse_args()


def _closest_channel(info, target):
    if target in info["ch_names"]:
        return target
    # fallback: nearest by position if montage present
    picks = mne.pick_types(info, eeg=True)
    pos = mne.channels.layout._find_topomap_coords(info, picks)  # (n,2)
    names = [info["ch_names"][p] for p in picks]
    # if target not found, prefer FCz-like (FC1/FC2/Cz etc.) by name
    prefer = [n for n in names if n.upper().startswith("FC") or n.upper()=="CZ"]
    return prefer[0] if prefer else names[0]


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    sub = args.sub
    deriv = Path(args.deriv) if args.deriv else (root / "derivatives" / f"sub-{sub}")
    deriv.mkdir(parents=True, exist_ok=True)

    evokeds_fif = deriv / "07_evoked-feedback-avg.fif"
    rewp_fif    = deriv / "08_evoked-RewP.fif"

    if not evokeds_fif.exists():
        raise FileNotFoundError(f"Missing evoked file: {evokeds_fif}")
    ev_list = mne.read_evokeds(evokeds_fif, condition=None, verbose=False)

    ev_by_comment = {e.comment or f"cond{idx}": e for idx, e in enumerate(ev_list)}

    # --- Find win/loss automatically if explicit labels not found ---
    ev_win = ev_by_comment.get("win")
    ev_loss = ev_by_comment.get("loss")

    if ev_win is None or ev_loss is None:
        # Try to detect win/loss from event IDs like "S6", "S16", etc.
        for key, evk in ev_by_comment.items():
            if any(k in key for k in ["S6", "S16", "S26", "S36"]):
                ev_win = evk
            elif any(k in key for k in ["S7", "S17", "S27", "S37"]):
                ev_loss = evk

    if ev_win is None and ev_loss is None:
        raise RuntimeError(f"No recognizable Win/Loss evokeds found. Found: {list(ev_by_comment.keys())}")

    # Load RewP if present or compute from win/loss
    if rewp_fif.exists():
        rewp = mne.Evoked(rewp_fif)
    else:
        if ev_win is None or ev_loss is None:
            raise RuntimeError("Cannot compute RewP without both win and loss.")
        rewp = mne.combine_evoked([ev_win, ev_loss], weights=[1, -1])
        rewp.comment = "RewP (Win-Loss)"

    # pick channel
    target_ch = _closest_channel(rewp.info, args.fc)
    print(f"[INFO] Using channel: {target_ch}")

    # -------- Plot Win/Loss at FCz (or closest) --------
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if ev_win is not None:
        ax.plot(ev_win.times, ev_win.data[ev_win.ch_names.index(target_ch)] * 1e6, label="Win", lw=2)
    if ev_loss is not None:
        ax.plot(ev_loss.times, ev_loss.data[ev_loss.ch_names.index(target_ch)] * 1e6, label="Loss", lw=2)
    ax.axvline(0.0, color="k", lw=1)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlim(-0.2, 0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{target_ch} (µV)")
    ax.set_title(f"Win / Loss at {target_ch} — sub-{sub}")
    ax.legend()
    fig.tight_layout()
    out1 = deriv / "21_fc-WinLoss.png"
    fig.savefig(out1, dpi=160)
    plt.close(fig)
    print(f"[OUT] {out1.name}")

    # -------- Plot RewP (Win-Loss) at FCz --------
    y = rewp.data[rewp.ch_names.index(target_ch)] * 1e6
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(rewp.times, y, color="C3", lw=2)
    ax.axvline(0.0, color="k", lw=1)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlim(-0.2, 0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{target_ch} (µV)")
    ax.set_title(f"RewP (Win-Loss) at {target_ch} — sub-{sub}")
    # highlight analysis window
    ax.axvspan(args.win_start, args.win_end, color="C3", alpha=0.12, label="RewP window")
    ax.legend(loc="best")
    fig.tight_layout()
    out2 = deriv / "22_fc-RewP.png"
    fig.savefig(out2, dpi=160)
    plt.close(fig)
    print(f"[OUT] {out2.name}")

    # -------- Topomap of RewP in analysis window --------
    # average across 240–340 ms to get a single map
    idx = np.where((rewp.times >= args.win_start) & (rewp.times <= args.win_end))[0]
    if len(idx) == 0:
        raise RuntimeError("RewP window does not overlap evoked time axis.")
    dat = rewp.copy().crop(args.win_start, args.win_end).data.mean(axis=1)[:, None]
    ev_for_map = mne.EvokedArray(dat, rewp.info, tmin=0.0, nave=rewp.nave)
    # robust limits
    vmin = float(np.percentile(dat, 10))
    vmax = float(np.percentile(dat, 90))
    fig = ev_for_map.plot_topomap(times=[0], scalings=1,
                                  time_unit="s", time_format="",
                                  outlines="head", size=3.0, show=False)
    # Matplotlib figure returned by MNE for EvokedArray topomap:
    fig.axes[0].set_title(f"RewP topography ({args.win_start:.0f}-{args.win_end:.0f} ms)", pad=16)
    out3 = deriv / "23_topomap-RewP-avgwindow.png"
    fig.savefig(out3, dpi=160)
    plt.close(fig)
    print(f"[OUT] {out3.name}")

    # -------- Metrics: mean amplitude & peak latency --------
    # mean amplitude over window
    mean_amp = float(y[(rewp.times >= args.win_start) & (rewp.times <= args.win_end)].mean())

    # peak (most negative by default for RewP; modify if you prefer positive peak)
    pmin, pmax = [float(x) for x in args.peak_window.split(",")]
    tmask = (rewp.times >= pmin) & (rewp.times <= pmax)
    tvec = rewp.times[tmask]
    yseg = y[tmask]
    # RewP often described as negativity; choose min amplitude (most negative)
    pk_idx = int(np.argmin(yseg))
    peak_lat = float(tvec[pk_idx])
    peak_amp = float(yseg[pk_idx])

    # save CSV
    metrics = pd.DataFrame([{
        "subject": sub,
        "channel": target_ch,
        "rewp_mean_uV": mean_amp,
        "rewp_peak_uV": peak_amp,
        "rewp_peak_latency_s": peak_lat,
        "window_start_s": args.win_start,
        "window_end_s": args.win_end,
        "peak_search_start_s": pmin,
        "peak_search_end_s": pmax
    }])
    csv_out = deriv / "24_rewp_metrics.csv"
    if csv_out.exists():
        old = pd.read_csv(csv_out)
        # replace if same subject exists
        old = old[old["subject"].astype(str) != str(sub)]
        metrics = pd.concat([old, metrics], ignore_index=True)
    metrics.to_csv(csv_out, index=False)
    print(f"[OUT] {csv_out.name}  (mean={mean_amp:.3f} µV, peak={peak_amp:.3f} µV @ {peak_lat*1000:.0f} ms)")

    print("[DONE]")


if __name__ == "__main__":
    main()
