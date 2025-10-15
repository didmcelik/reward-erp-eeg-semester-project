# notebooks/step01_load_and_montage.py
from pathlib import Path
import mne

# --- Paths ---
PROJECT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
DATA = PROJECT / "data" / "ds004147"
DERIV = PROJECT / "derivatives"
DERIV.mkdir(parents=True, exist_ok=True)

SUBJECT = "27"          # change to "28", "29", ... if you want
TASK = "casinos"        # confirmed from filenames

vhdr_path = DATA / f"sub-{SUBJECT}" / "eeg" / f"sub-{SUBJECT}_task-{TASK}_eeg.vhdr"
assert vhdr_path.exists(), f"Missing: {vhdr_path}"

print(f"[INFO] Loading: {vhdr_path}")
raw = mne.io.read_raw_brainvision(vhdr_path.as_posix(), preload=True, verbose=True)

# --- Clean channel names and set a standard EEG montage ---
raw.rename_channels(lambda s: s.strip())
# primary attempt: standard 10-20 locations
try:
    raw.set_montage("standard_1020", match_case=False, on_missing="warn")
except Exception as e:
    print("[WARN] standard_1020 montage failed, trying standard_1005:", e)
    raw.set_montage("standard_1005", match_case=False, on_missing="ignore")

print("[INFO] Montage set:", raw.get_montage() is not None)

# --- Quick checks (optional plots) ---
# 1) Power Spectral Density
raw.plot_psd()

# 2) Short raw snippet (10 seconds) to visually inspect channels
raw.plot(duration=10, n_channels=32, scalings="auto", title=f"Raw preview — sub-{SUBJECT}")

# --- Save a copy with montage info (optional) ---
out_fif = DERIV / f"sub-{SUBJECT}_raw_with_montage.fif"
raw.save(out_fif.as_posix(), overwrite=True)
print("[INFO] Saved:", out_fif)
