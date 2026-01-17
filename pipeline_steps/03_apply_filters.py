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
L_FREQ = 0.1  # Low-pass
H_FREQ = 40.0  # High-pass
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

def add_fz_reference_site2(raw, subject_id):
    """
    Add Fz reference channel for site 2 subjects
    """
    
    site2_subjects = ['27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38'] 
    
    if subject_id in site2_subjects:
        print(f"Subject {subject_id} - adding Fz reference channel")
        
        # Get current data
        data = raw.get_data()
        n_channels, n_times = data.shape
        
        # Create zero-filled Fz channel
        fz_data = np.zeros((1, n_times))
        
        # Concatenate Fz data
        new_data = np.vstack([data, fz_data])
        
        # Create new channel info for Fz
        fz_info = mne.create_info(['Fz'], raw.info['sfreq'], ch_types=['eeg'])
        
        # Create new info with additional channel
        new_info = raw.info.copy()
        new_info = mne.io.meas_info.create_info(
            ch_names=raw.ch_names + ['Fz'],
            sfreq=raw.info['sfreq'],
            ch_types=[raw.get_channel_types()[i] for i in range(len(raw.ch_names))] + ['eeg']
        )
        
        # Copy other info attributes
        for key in ['bads', 'description', 'experimenter', 'line_freq']:
            if key in raw.info:
                new_info[key] = raw.info[key]
        
        # Create new Raw object with Fz channel
        raw_with_fz = mne.io.RawArray(new_data, new_info)
        
        print(f"Added Fz reference channel (zero-filled)")
        return raw_with_fz
    
    else:
        print(f"Subject {subject_id} is not from site 2 - no Fz reference needed")
        return raw

def apply_rereferencing(raw):
    """Apply re-referencing to mastoids (TP9, TP10)"""
    
    # mastoid channels
    mastoids = ['TP9', 'TP10']
    available_mastoids = [ch for ch in mastoids if ch in raw.ch_names]
    
    if len(available_mastoids) == 2:
        print(f"Re-referencing to TP9 and TP10 mastoids: {available_mastoids}")
        raw.set_eeg_reference(ref_channels=available_mastoids)
    elif 'TP9' in available_mastoids:
        print("Re-referencing to TP9 only")
        raw.set_eeg_reference(ref_channels=['TP9'])
    elif 'TP10' in available_mastoids:
        print("Re-referencing to TP10 only")
        raw.set_eeg_reference(ref_channels=['TP10'])
    else:
        print("No mastoid channels found, using common average reference")
        raw.set_eeg_reference(ref_channels='average')
    
    return raw

def detect_bad_channels(raw):
    """Detect bad channels using statistical methods"""
    
    from scipy import stats
    
    print("Detecting bad channels...")
    
    # Get EEG data only
    picks_eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks_eeg)
    ch_names = [raw.ch_names[i] for i in picks_eeg]
    
    # Calculate channel statistics
    ch_std = np.std(data, axis=1)
    ch_range = np.ptp(data, axis=1)
    ch_kurtosis = np.array([stats.kurtosis(ch_data) for ch_data in data])
    
    # Z-score for each metric
    std_zscore = np.abs(stats.zscore(ch_std))
    range_zscore = np.abs(stats.zscore(ch_range))
    kurtosis_zscore = np.abs(stats.zscore(ch_kurtosis))
    
    # Mark as bad if any metric exceeds threshold
    threshold = 3.0  # 3 standard deviations
    
    bad_channels = []
    for i, ch_name in enumerate(ch_names):
        is_bad = False
        reasons = []
        
        if std_zscore[i] > threshold:
            is_bad = True
            reasons.append(f"std_z={std_zscore[i]:.2f}")
        if range_zscore[i] > threshold:
            is_bad = True
            reasons.append(f"range_z={range_zscore[i]:.2f}")
        if kurtosis_zscore[i] > threshold:
            is_bad = True
            reasons.append(f"kurt_z={kurtosis_zscore[i]:.2f}")
        
        if is_bad:
            bad_channels.append(ch_name)
            print(f"  BAD: {ch_name} - {', '.join(reasons)}")
    
    # Mark bad channels in raw
    if bad_channels:
        raw.info['bads'] = bad_channels
        print(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
    else:
        print("No bad channels detected")
    
    return raw, bad_channels


def interpolate_bad_channels(raw):
    """Interpolate bad channels"""
    
    if raw.info['bads']:
        print(f"Interpolating {len(raw.info['bads'])} bad channels: {raw.info['bads']}")
        raw.interpolate_bads(reset_bads=True)
        print("Bad channels interpolated")
    else:
        print("No bad channels to interpolate")
    
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
    
    # Detect and interpolate bad channels before re-referencing
    raw, bad_channels = detect_bad_channels(raw)
    raw = interpolate_bad_channels(raw)
    
    # Apply re-referencing (with clean channels)
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
