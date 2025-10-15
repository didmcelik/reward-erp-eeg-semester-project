# notebooks/step02_preprocess.py
from pathlib import Path
import mne

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "ds004147"
DERIV = PROJECT / "derivatives"
DERIV.mkdir(parents=True, exist_ok=True)

SUBJECT = "27"
TASK = "casinos"
vhdr_path = DATA / f"sub-{SUBJECT}" / "eeg" / f"sub-{SUBJECT}_task-{TASK}_eeg.vhdr"

print(f"[INFO] Loading raw: {vhdr_path}")
raw = mne.io.read_raw_brainvision(vhdr_path.as_posix(), preload=True, verbose=True)

# Ensure montage (so topomaps work later)
raw.rename_channels(lambda s: s.strip())
try:
    raw.set_montage("standard_1020", match_case=False, on_missing="warn")
except Exception:
    raw.set_montage("standard_1005", match_case=False, on_missing="ignore")

# ---- BEFORE filtering (QC) ----
raw.compute_psd(fmax=60).plot(picks="eeg", average=False, exclude="bads", show=True, dB=True)

# ---- Preprocessing ----
print("[INFO] Filtering 1–30 Hz (zero-phase FIR)")
raw.filter(l_freq=1.0, h_freq=30.0, fir_design="firwin")

print("[INFO] Average re-reference")
raw.set_eeg_reference("average")

print("[INFO] Resample to 250 Hz")
raw.resample(250, npad="auto")

# ---- AFTER filtering (QC) ----
raw.compute_psd(fmax=60).plot(picks="eeg", average=False, exclude="bads", show=True, dB=True)

# Save preprocessed raw (MNE naming style)
out_fif = DERIV / f"sub-{SUBJECT}_task-{TASK}_preproc_raw.fif"
raw.save(out_fif.as_posix(), overwrite=True)
print("[INFO] Saved:", out_fif)
