"""
01_add_montage_and_clean_ICA.py

Load BrainVision EEG (ds004147), set montage from site2channellocations.locs,
apply preprocessing (0.1–30 Hz band-pass + 50 Hz notch + TP9/TP10 reference),
train ICA on a 1 Hz high-passed copy, remove blink components, and save:

- sub-XX_task-casinos_preproc_eeg_raw.fif
- sub-XX_task-casinos_ica-train_eeg_raw.fif
- sub-XX_task-casinos_ica.fif
- sub-XX_task-casinos_clean_eeg_raw.fif

Usage
-----
cd project\pipeline
python .\01_add_montage_and_clean_ICA.py --sub 27 --root ..\data\ds004147 --deriv ..\data\ds004147\derivatives --plot 0


"""

from __future__ import annotations
from pathlib import Path
import argparse
import gc

import numpy as np
import mne


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sub", type=str, required=True, help="Subject ID (e.g., 27)")
    p.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "ds004147"),
        help="Path to ds004147 dataset root",
    )
    p.add_argument(
        "--deriv",
        type=str,
        default=None,
        help="Derivatives output folder (default: <root>/derivatives/sub-XX/)",
    )
    p.add_argument("--plot", type=int, default=0, help="Show quick sensor/PSD plots (0/1)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    sub = args.sub

    vhdr = root / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-casinos_eeg.vhdr"
    if not vhdr.exists():
        raise FileNotFoundError(f"BrainVision header not found: {vhdr}")

    # Derivatives folder
    deriv = Path(args.deriv) if args.deriv else (root / "derivatives" / f"sub-{sub}")
    deriv.mkdir(parents=True, exist_ok=True)

    # Output names (MNE-compliant)
    fif_pre = deriv / f"sub-{sub}_task-casinos_preproc_eeg_raw.fif"
    fif_tmp = deriv / f"sub-{sub}_task-casinos_ica-train_eeg_raw.fif"
    fif_ica = deriv / f"sub-{sub}_task-casinos_ica.fif"
    fif_clean = deriv / f"sub-{sub}_task-casinos_clean_eeg_raw.fif"

    # ---------- 0) Load raw + set montage ----------
    print(f"Reading: {vhdr}")
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=True)

    # Montage from dataset root (site-2)
    locs = root / "site2channellocations.locs"
    if not locs.exists():
        raise FileNotFoundError(
            f"site2channellocations.locs not found at {locs} "
            "(place it in the ds004147 root)."
        )
    montage = mne.channels.read_custom_montage(locs)
    raw.set_montage(montage, on_missing="warn")

    if args.plot:
        raw.plot_sensors(kind="topomap", show_names=True, block=True)
        raw.compute_psd(fmax=60).plot(average=False, picks="eeg", exclude="bads", show=True)

    # ---------- 1) Preprocess: band-pass + notch + reference ----------
    print("Filtering 0.1–30 Hz (FIR, firwin) and applying 50 Hz notch...")
    raw.filter(0.1, 30.0, fir_design="firwin", verbose=False)
    raw.notch_filter(50.0, verbose=False)

    # Prefer mastoid reference if present, otherwise average
    if all(ch in raw.ch_names for ch in ("TP9", "TP10")):
        raw.set_eeg_reference(ref_channels=["TP9", "TP10"])
    else:
        raw.set_eeg_reference("average")

    # Save preprocessed (float64 to keep numerical stability so far)
    raw.save(fif_pre, overwrite=True)
    del raw
    gc.collect()

    # ---------- 2) ICA training copy: 1 Hz high-pass ----------
    print("Preparing ICA training copy (1 Hz high-pass)...")
    raw_ica_train = mne.io.read_raw_fif(fif_pre, preload=True, verbose=False)
    raw_ica_train.filter(1.0, None, verbose=False)
    raw_ica_train.save(fif_tmp, overwrite=True)
    del raw_ica_train
    gc.collect()

    # ---------- 3) Fit ICA and find blink components ----------
    print("Fitting ICA (fastica, n_components=20)...")
    raw_ica = mne.io.read_raw_fif(fif_tmp, preload=True, verbose=False)
    ica = mne.preprocessing.ICA(n_components=20, method="fastica",
                                max_iter="auto", random_state=97)
    ica.fit(raw_ica, picks="eeg")

    eog_inds = []
    for ch in ("Fpz", "Fp1", "Fp2"):
        if ch in raw_ica.ch_names:
            inds, _ = ica.find_bads_eog(raw_ica, ch_name=ch)
            eog_inds.extend(inds)
    ica.exclude = sorted(set(eog_inds))
    print(f"Excluded EOG-related ICs: {ica.exclude}")
    ica.save(fif_ica, overwrite=True)
    del raw_ica
    gc.collect()

    # ---------- 4) Apply ICA to the preprocessed raw ----------
    print("Applying ICA to preprocessed data...")
    raw_pre = mne.io.read_raw_fif(fif_pre, preload=True, verbose=False)
    ica.apply(raw_pre)

    # (Optional) downcast before saving to reduce file size / RAM
    raw_pre._data = raw_pre._data.astype(np.float32, copy=False)
    raw_pre.save(fif_clean, overwrite=True)
    print(f"Saved clean file: {fif_clean}")

    del raw_pre
    gc.collect()

    print("[done]")


if __name__ == "__main__":
    main()
