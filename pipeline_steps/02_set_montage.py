"""
Step 2: Set electrode montage
"""

import os
import argparse
import mne
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
TASK = "casinos"
LOCS_FILE = "../data/ds004147/site2channellocations.locs"

def load_previous_step(subject_id):
    """Load data from previous step"""
    
    input_file = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 'step01_raw', 
                             f'sub-{subject_id}_task-{TASK}_raw.fif')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    raw = mne.io.read_raw_fif(input_file, preload=True)
    return raw

def set_montage(raw):
    """Set custom montage from electrode locations file"""
    
    if not os.path.exists(LOCS_FILE):
        raise FileNotFoundError(f"Custom montage file not found: {LOCS_FILE}")
    
    # Use MNE's built-in function to read the montage
    print(f"Loading custom montage from: {LOCS_FILE}")
    montage = mne.channels.read_custom_montage(LOCS_FILE)
    
    # Print available channels before setting montage
    print(f"Raw data channels: {raw.ch_names}")
    print(f"Montage channels: {list(montage.ch_names)}")
    
    # Set montage
    raw.set_montage(montage, on_missing='warn')
    
    print(f"Applied custom montage to {len(raw.ch_names)} channels")
    
    # Print channel mapping info
    montage_ch_names = list(montage.ch_names)
    mapped_channels = [ch for ch in raw.ch_names if ch in montage_ch_names]
    unmapped_channels = [ch for ch in raw.ch_names if ch not in montage_ch_names]
    
    print(f"Mapped channels ({len(mapped_channels)}): {mapped_channels}")
    print(f"Unmapped channels ({len(unmapped_channels)}): {unmapped_channels}")
    
    # Print electrode positions for verification
    if raw.info['dig'] is not None:
        eeg_digs = [d for d in raw.info['dig'] if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG]
        print(f"Successfully set {len(eeg_digs)} electrode positions")
    else:
        print("Warning: No electrode positions were set!")
    
    return raw

def visualize_montage(raw, subject_id, output_dir):
    """Visualize electrode montage"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step02_montage')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Plot electrode locations
    try:
        fig = raw.plot_sensors(kind='topomap', show_names=True, show=False,
                              title=f'Sub-{subject_id} Electrode Locations')
        fig.savefig(os.path.join(subject_dir, 'electrode_locations.png'), dpi=300)
        plt.close(fig)
        
        # 3D view
        fig = raw.plot_sensors(kind='3d', show=False)
        fig.savefig(os.path.join(subject_dir, 'electrode_locations_3d.png'), dpi=300)
        plt.close(fig)
        
    except Exception as e:
        print(f"Could not create montage plots: {e}")
    
    print(f"Montage visualizations saved to: {subject_dir}")

def save_montage_data(raw, subject_id, output_dir):
    """Save data with montage"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step02_montage')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Save data with montage
    raw_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_montage_raw.fif')
    raw.save(raw_fname, overwrite=True)
    
    print(f"Data with montage saved to: {raw_fname}")
    
    return raw_fname

def main():
    parser = argparse.ArgumentParser(description='Step 2: Set electrode montage')
    parser.add_argument('--subject', required=True, help='Subject ID')
    args = parser.parse_args()
    
    subject_id = args.subject
    
    print(f"Step 2: Setting montage for subject {subject_id}")
    
    # Load data from previous step
    raw = load_previous_step(subject_id)
    
    # Set montage
    raw = set_montage(raw)
    
    # Create visualizations
    visualize_montage(raw, subject_id, OUTPUT_DIR)
    
    # Save data
    save_montage_data(raw, subject_id, OUTPUT_DIR)
    
    print(f"Step 2 completed for subject {subject_id}")

if __name__ == "__main__":
    main()
    