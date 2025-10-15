from pathlib import Path
import re
import argparse
import mne

# ----------------------------
# Defaults (edit if you want)
# ----------------------------
L_FREQ = 1.0
H_FREQ = 30.0
RESAMPLE_HZ = 250
MONTAGES = ("standard_1020", "standard_1005")  # try in this order
TASK_FALLBACK = "casinos"  # used if task cannot be parsed from filename
N_COMPONENTS = 15

def project_root_from_here(this_file: Path) -> Path:
    here = this_file.resolve()
    for p in [here, *here.parents]:
        root = p if p.is_dir() else p.parent
        if (root / "data" / "ds004147").exists():
            return root
    raise RuntimeError("Could not locate project root. Ensure 'data/ds004147' exists.")

def list_subjects(data_root: Path, include=None):
    subs = []
    for p in sorted(data_root.glob("sub-*")):
        m = re.match(r"sub-(\d+)$", p.name)
        if m:
            subs.append(m.group(1))
    if include:
        subs = [s for s in subs if s in set(include)]
    return subs

def find_vhdr(data_root: Path, sub: str) -> Path:
    eeg_dir = data_root / f"sub-{sub}" / "eeg"
    cand = sorted(eeg_dir.glob(f"sub-{sub}_task-*_eeg.vhdr"))
    if cand:
        return cand[0]
    # fallback (rare)
    fb = eeg_dir / f"sub-{sub}_task-{TASK_FALLBACK}_eeg.vhdr"
    if fb.exists():
        return fb
    raise FileNotFoundError(f"No .vhdr for sub-{sub} under {eeg_dir}")

def parse_task(vhdr_name: str) -> str:
    m = re.search(r"_task-([^-_]+)_eeg\.vhdr$", vhdr_name)
    return m.group(1) if m else TASK_FALLBACK

def set_montage_safe(raw: mne.io.BaseRaw):
    raw.rename_channels(lambda s: s.strip())
    for mont in MONTAGES:
        try:
            raw.set_montage(mont, match_case=False, on_missing="warn")
            return
        except Exception:
            continue
    print("[WARN] Could not set a standard montage; plots with topomaps may fail.")

def auto_find_blinks(ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw):
    """Try multiple strategies to locate blink components without EOG channels."""
    # Try automatic
    try:
        eog_inds, _ = ica.find_bads_eog(raw)
        if eog_inds:
            return list(eog_inds)
    except Exception:
        pass
    # Try frontal proxies if present
    for ch in ("Fpz", "Fp2", "Fp1", "AFz"):
        if ch in raw.ch_names:
            try:
                eog_inds, _ = ica.find_bads_eog(raw, ch_name=ch)
                if eog_inds:
                    return list(eog_inds)
            except Exception:
                continue
    return []  # none found

def process_subject(PROJECT: Path, sub: str, do_ica: bool, n_components: int, overwrite: bool):
    DATA = PROJECT / "data" / "ds004147"
    DERIV = PROJECT / "derivatives"

    vhdr = find_vhdr(DATA, sub)
    task = parse_task(vhdr.name)
    print(f"\n=== sub-{sub} | task={task} | {vhdr.name} ===")

    raw = mne.io.read_raw_brainvision(vhdr.as_posix(), preload=True, verbose=False)
    set_montage_safe(raw)

    # ERP-friendly preproc
    print("[INFO] Filtering %.1f–%.1f Hz" % (L_FREQ, H_FREQ))
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin")
    print("[INFO] Average reference")
    raw.set_eeg_reference("average")
    print("[INFO] Resample to %d Hz" % RESAMPLE_HZ)
    raw.resample(RESAMPLE_HZ, npad="auto")

    # Save preproc
    pre_fif = DERIV / f"sub-{sub}_task-{task}_preproc_raw.fif"
    raw.save(pre_fif.as_posix(), overwrite=overwrite)
    print("[INFO] Saved:", pre_fif)

    if not do_ica:
        return

    print("[INFO] ICA: n_components=%d" % n_components)
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97)
    ica.fit(raw)
    eog_inds = auto_find_blinks(ica, raw)
    if eog_inds:
        print(f"[INFO] Auto-marked EOG components: {eog_inds}")
        ica.exclude = sorted(set(ica.exclude).union(eog_inds))
    else:
        print("[INFO] No clear EOG components auto-detected; proceeding with 0 excluded.")

    raw_clean = ica.apply(raw.copy())
    ica_fif = DERIV / f"sub-{sub}_task-{task}_ica_raw.fif"
    raw_clean.save(ica_fif.as_posix(), overwrite=overwrite)
    print("[INFO] Saved:", ica_fif)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch preproc (filter→reref→resample→optional ICA) for ds004147")
    ap.add_argument("--subs", nargs="*", help="Optional subject IDs to process (e.g., 27 28 29). Default: all found.")
    ap.add_argument("--ica", action="store_true", help="Apply ICA and save *_ica_raw.fif")
    ap.add_argument("--ncomp", type=int, default=N_COMPONENTS, help="Number of ICA components (default 15)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing FIF files")
    args = ap.parse_args()

    PROJECT = project_root_from_here(Path(__file__))
    DATA = PROJECT / "data" / "ds004147"

    subs = list_subjects(DATA, include=args.subs)
    if not subs:
        raise SystemExit("No subjects found under data/ds004147.")

    print(f"[INFO] Subjects: {', '.join(subs)}")
    for s in subs:
        process_subject(PROJECT, s, do_ica=args.ica, n_components=args.ncomp, overwrite=args.overwrite)

    print("\n[INFO] Batch preproc finished.")
