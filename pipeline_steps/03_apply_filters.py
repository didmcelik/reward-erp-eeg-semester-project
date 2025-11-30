"""
Step 3: Apply filtering (bandpass and notch filters)
"""

import os
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt
from meegkit import dss

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
TASK = "casinos"

# Filter parameters
L_FREQ = 0.1  # High-pass
H_FREQ = 20.0  # Low-pass
LINE_FREQ = 50.0  # Notch filter for line noise
RESAMPLE_FREQ = 250.0

def load_previous_step(subject_id):
    """Load data from previous step"""
    
    input_file = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 'step02_montage', 
                             f'sub-{subject_id}_task-{TASK}_montage_raw.fif')
    
    raw = mne.io.read_raw_fif(input_file, preload=True)
    return raw

def apply_zapline_filter(raw, line_freq=50.0, nremove=4):
    """
    Apply Zapline filter using DSS to remove line noise
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    line_freq : float
        Line frequency to remove (50 Hz for site 2)
    nremove : int
        Number of line noise components to remove
    
    Returns
    -------
    raw_clean : mne.io.Raw
        Data with line noise removed
    """
    
    print(f"Applying Zapline filter at {line_freq} Hz...")
    
    # Get data and sampling frequency
    data = raw.get_data()  # Shape: (n_channels, n_times)
    sfreq = raw.info['sfreq']
    
    # Apply Zapline using DSS
    # We need to transpose for meegkit (expects n_times, n_channels)
    data_t = data.T  # Shape: (n_times, n_channels)
    
    # Apply DSS to remove line noise
    data_clean, _ = dss.dss_line(
        data_t,
        fline=line_freq,
        sfreq=sfreq,
        nremove=nremove,
        nkeep=None
    )
    
    # Transpose back to MNE format (n_channels, n_times)
    data_clean = data_clean.T
    
    # Create a copy of raw with cleaned data
    raw_clean = raw.copy()
    raw_clean._data = data_clean
    
    print(f"Zapline filter applied - removed {nremove} line noise components")
    
    return raw_clean

def apply_rereferencing(raw):
    """Apply re-referencing to mastoids (TP9, TP10)"""
    
    # mastoid channels
    mastoids = ['TP9', 'TP10']
    available_mastoids = [ch for ch in mastoids if ch in raw.ch_names]
    apply_mastoids = False
    
    if apply_mastoids:
        print(f"Re-referencing to mastoids: {available_mastoids}")
        raw.set_eeg_reference(ref_channels=available_mastoids)
    else:
        print("No mastoid channels found, using common average reference")
        raw.set_eeg_reference(ref_channels='average')
    
    return raw

def apply_filters(raw):
    """Apply bandpass and Zapline filters"""
    
    print(f"Original sampling rate: {raw.info['sfreq']} Hz")
    
    # Store original PSD for comparison
    psd_orig = raw.compute_psd(fmax=100)
    
    # Apply Zapline filter for line noise
    print(f"Applying Zapline filter at {LINE_FREQ} Hz")
    raw = apply_zapline_filter(raw, line_freq=LINE_FREQ, nremove=4)
    
    # Apply bandpass filter
    print(f"Applying bandpass filter: {L_FREQ}-{H_FREQ} Hz")
    raw.filter(L_FREQ, H_FREQ, fir_design='firwin')

    # TEST: Apply notch filter as alternative to Zapline
    # print(f"Applying notch filter at multiples of {LINE_FREQ} Hz")
    # raw.notch_filter(np.arange(LINE_FREQ, raw.info['sfreq'] / 2, LINE_FREQ), filter_length='auto', fir_design='firwin', method='fir', verbose=False)

    raw = apply_rereferencing(raw)
    
    # Resample data
    if raw.info['sfreq'] != RESAMPLE_FREQ:
        print(f"Resampling to {RESAMPLE_FREQ} Hz")
        raw.resample(RESAMPLE_FREQ)
    
    # Set EOG channels
    eog_channels = ['Fp1', 'Fp2']
    available_eog = [ch for ch in eog_channels if ch in raw.ch_names]
    if available_eog:
        raw.set_channel_types({ch: 'eog' for ch in available_eog})
        print(f"Set EOG channels: {available_eog}")
    
    return raw, psd_orig

def visualize_filtering(raw, psd_orig, subject_id, output_dir):
    """Visualize filtering effects"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step03_filters')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Compare PSD before and after filtering
    psd_filt = raw.compute_psd(fmax=100)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original PSD
    psd_orig.plot(axes=axes[0], show=False, average=True)
    axes[0].set_title(f'Sub-{subject_id} - Original PSD')
    
    # Filtered PSD
    psd_filt.plot(axes=axes[1], show=False, average=True)
    axes[1].set_title(f'Sub-{subject_id} - Filtered PSD')
    
    plt.tight_layout()
    plt.savefig(os.path.join(subject_dir, 'psd_comparison.png'), dpi=300)
    plt.close()
    
    # Plot filtered data sample
    fig = raw.plot(duration=10, n_channels=20, scalings='auto',
                   title=f'Sub-{subject_id} Filtered Data', show=False)
    fig.savefig(os.path.join(subject_dir, 'filtered_data_sample.png'), dpi=300)
    plt.close(fig)
    
    print(f"Filtering visualizations saved to: {subject_dir}")

def save_filtered_data(raw, subject_id, output_dir):
    """Save filtered data"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step03_filters')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Save filtered data
    raw_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_filtered_raw.fif')
    raw.save(raw_fname, overwrite=True)
    
    print(f"Filtered data saved to: {raw_fname}")
    
    return raw_fname

def main():
    parser = argparse.ArgumentParser(description='Step 3: Apply filters')
    parser.add_argument('--subject', required=True, help='Subject ID')
    args = parser.parse_args()
    
    subject_id = args.subject
    
    print(f"Step 3: Applying filters for subject {subject_id}")
    
    # Load data from previous step
    raw = load_previous_step(subject_id)
    
    # Apply filters
    raw, psd_orig = apply_filters(raw)
    
    # Create visualizations
    visualize_filtering(raw, psd_orig, subject_id, OUTPUT_DIR)
    
    # Save data
    save_filtered_data(raw, subject_id, OUTPUT_DIR)
    
    print(f"Step 3 completed for subject {subject_id}")

if __name__ == "__main__":
    main()
