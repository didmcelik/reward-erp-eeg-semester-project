from pathlib import Path
import mne

PROJECT = Path(__file__).resolve().parents[1]
DERIV = PROJECT / "derivatives"
SUBJECT = "27"
TASK = "casinos"

# Load the preprocessed raw from Step 2
preproc_fif = DERIV / f"sub-{SUBJECT}_task-{TASK}_preproc_raw.fif"
raw = mne.io.read_raw_fif(preproc_fif.as_posix(), preload=True)
print("[INFO] Loaded:", preproc_fif)

# Fit ICA on the filtered, reref, resampled data
ica = mne.preprocessing.ICA(n_components=15, random_state=97)
ica.fit(raw)

# Try to auto-detect EOG-related components (blinks)
try:
    eog_inds, eog_scores = ica.find_bads_eog(raw)
    print(f"[INFO] Auto EOG components: {eog_inds}")
    ica.exclude = list(set(ica.exclude).union(eog_inds))
except Exception as e:
    print("[WARN] Could not run find_bads_eog:", e)
# Visualize components (topographies)
ica.plot_components()  # Inspect spatial patterns

# Browse component time series interactively
ica.plot_sources(raw, show_scrollbars=True)  # works in MNE >= 1.10

# Optionally check power spectra of components
ica.plot_properties(raw, picks=[0, 1, 2, 3])

# Apply ICA to remove selected components (you can update ica.exclude manually)
raw_clean = ica.apply(raw.copy())

# Save cleaned version
out_fif = DERIV / f"sub-{SUBJECT}_task-{TASK}_ica_raw.fif"
raw_clean.save(out_fif.as_posix(), overwrite=True)
print("[INFO] Saved:", out_fif)
