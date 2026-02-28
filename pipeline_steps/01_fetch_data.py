"""
Step 1: Load and validate raw EEG data
"""

import os
import argparse
import mne
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BIDS_ROOT = "../data/ds004147"
TASK = "casinos"
OUTPUT_DIR = "../output/derivatives/manual-pipeline"

def load_raw_data(subject_id):
    """Load raw EEG data for one subject"""
    
    # Construct file path
    raw_file = os.path.join(BIDS_ROOT, f'sub-{subject_id}', 'eeg', 
                           f'sub-{subject_id}_task-{TASK}_eeg.vhdr')
    
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")
    
    # Load data
    raw = mne.io.read_raw_brainvision(raw_file, preload=True)
    
    print(f"Loaded data: {raw.info['nchan']} channels, {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    
    return raw

def visualize_raw_data(raw, subject_id, output_dir):
    """Create visualizations of raw data"""
    
    # Create subject output directory
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step01_raw')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Plot data overview
    fig = raw.plot(duration=10, n_channels=30,
                   title=f'Sub-{subject_id} Raw Data Overview', show=False)
    fig.savefig(os.path.join(subject_dir, 'raw_data_overview.png'))
    plt.close(fig)
    
    # Plot power spectral density
    fig = raw.compute_psd(fmax=100).plot(show=False, 
                                        average=True)
    fig.suptitle(f'Sub-{subject_id} PSD') 
    fig.savefig(os.path.join(subject_dir, 'raw_psd.png'), dpi=300)
    plt.close(fig)
    
    # Plot channel locations (if available)
    try:
        fig = raw.plot_sensors(kind='topomap', show_names=True, show=False)
        fig.savefig(os.path.join(subject_dir, 'channel_locations.png'), dpi=300)
        plt.close(fig)
    except:
        print("Could not plot channel locations (no montage set)")
    
    print(f"Raw data visualizations saved to: {subject_dir}")

def save_raw_data(raw, subject_id, output_dir):
    """Save raw data"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step01_raw')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Save raw data
    raw_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_raw.fif')
    raw.save(raw_fname, overwrite=True)
    
    print(f"Raw data saved to: {raw_fname}")
    
    return raw_fname

def main():
    parser = argparse.ArgumentParser(description='Step 1: Fetch raw EEG data')
    parser.add_argument('--subject', required=True, help='Subject ID')
    args = parser.parse_args()
    
    subject_id = args.subject
    
    print(f"Step 1: Loading raw data for subject {subject_id}")
    
    # Load raw data
    raw = load_raw_data(subject_id)
    
    # Create visualizations
    visualize_raw_data(raw, subject_id, OUTPUT_DIR)
    
    # Save data
    save_raw_data(raw, subject_id, OUTPUT_DIR)
    
    print(f"Step 1 completed for subject {subject_id}")

if __name__ == "__main__":
    main()
