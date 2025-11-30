"""
Step 4: Run ICA for artifact removal
"""

import os
import argparse
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
TASK = "casinos"

def load_previous_step(subject_id):
    """Load data from previous step"""
    
    input_file = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 'step03_filters', 
                             f'sub-{subject_id}_task-{TASK}_filtered_raw.fif')
    
    raw = mne.io.read_raw_fif(input_file, preload=True)
    return raw


def prepare_ica_data(raw):
    """Prepare data for ICA following ICLabel requirements"""
    
    print("Preparing data for ICA...")
    
    # Create 1-100 Hz filtered copy for ICA (ICLabel requirement)
    print("Creating 1-100 Hz filtered copy for ICA...")
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0, fir_design='firwin')

    print("Setting common average reference...")
    filt_raw.set_eeg_reference('average', projection=False)
    
    return filt_raw

def create_epochs_for_ica(filt_raw):
    """Create epochs for ICA fitting with artifact rejection"""
    
    print("Creating epochs for ICA fitting...")
    
    # Find events
    events, event_id = mne.events_from_annotations(filt_raw)
    
    # Create longer epochs for ICA (better artifact capture)
    epochs = mne.Epochs(
        filt_raw, events, event_id,
        tmin=-0.5, tmax=1.0,  # Longer epochs
        baseline=None,  # No baseline correction for ICA
        reject={'eeg': 150e-6},
        flat={'eeg': 1e-6},
        preload=True,
        proj=False,  # No projectors
        event_repeated='merge'  # Drop duplicate events
    )
    
    print(f"Created {len(epochs)} epochs for ICA fitting")
    
    # Additional artifact rejection - remove extreme epochs
    if len(epochs) > 100:  # Only if we have enough epochs
        data = epochs.get_data()
        
        # Calculate variance per epoch
        epoch_vars = np.var(data, axis=(1, 2))
        
        # Remove top 10% most variable epochs
        var_threshold = np.percentile(epoch_vars, 90)
        good_epochs = epoch_vars < var_threshold
        
        epochs = epochs[good_epochs]
        print(f"After variance-based rejection: {len(epochs)} epochs")
    
    return epochs

def run_ica(filt_raw, epochs):
    """Run ICA decomposition following ICLabel best practices"""
    
    print("Fitting ICA...")
    
    # Setup ICA with ICLabel-compatible parameters
    n_components = min(15, len(mne.pick_types(filt_raw.info, eeg=True)) - 1)
    
    ica = ICA(
        n_components=n_components,
        method='infomax',  # ICLabel requirement
        max_iter='auto',
        random_state=42,
        fit_params=dict(extended=True)  # ICLabel requirement
    )
    
    # Fit ICA on epochs (more stable than continuous data)
    print(f"Fitting ICA with {n_components} components on {len(epochs)} epochs...")
    ica.fit(epochs)
    
    return ica

def classify_components(ica, filt_raw):
    """Classify ICA components using ICLabel and manual methods"""
    
    exclude_components = []
    
    
    try:
        print("Running ICLabel classification...")
        
        ic_labels = label_components(filt_raw, ica, method='iclabel')
        labels = ic_labels['labels']
        
        print("Component classifications:")
        for i, label in enumerate(labels):
            print(f"  IC{i:02d}: {label}")
        
        # Exclude non-brain components (conservative approach)
        exclude_idx = [
            idx for idx, label in enumerate(labels) 
            if label not in ['brain', 'other']
        ]
        
        exclude_components.extend(exclude_idx)
        print(f"ICLabel excluded: {exclude_idx}")
        
    except Exception as e:
        print(f"ICLabel failed: {e}")
    
    # Remove duplicates and sort
    exclude_components = sorted(list(set(exclude_components)))
    
    # Safety check - don't exclude too many components
    # max_exclude = n_components // 3  # Max 1/3 of components
    # if len(exclude_components) > max_exclude:
    #     print(f"Warning: Too many exclusions ({len(exclude_components)}). Limiting to {max_exclude}")
    #     exclude_components = exclude_components[:max_exclude]
    
    ica.exclude = exclude_components
    print(f"Final excluded components: {ica.exclude}")
    
    return ica

def detect_and_interpolate_bad_channels(raw):
    """Detect and interpolate bad channels"""
    
    print("Checking for bad channels...")
    
    # Simple bad channel detection based on variance
    data = raw.get_data()
    channel_vars = np.var(data, axis=1)
    
    # Channels with extremely high or low variance
    median_var = np.median(channel_vars)
    mad = np.median(np.abs(channel_vars - median_var))
    
    # Define outliers (very conservative)
    lower_bound = median_var - 5 * mad
    upper_bound = median_var + 5 * mad
    
    bad_channels = []
    for i, ch_name in enumerate(raw.ch_names):
        if raw.get_channel_types([ch_name])[0] == 'eeg':
            if channel_vars[i] < lower_bound or channel_vars[i] > upper_bound:
                bad_channels.append(ch_name)
    
    if bad_channels:
        print(f"Detected bad channels: {bad_channels}")
        raw.info['bads'] = bad_channels
        raw.interpolate_bads(reset_bads=True)
        print(f"Interpolated {len(bad_channels)} bad channels")
    else:
        print("No bad channels detected")
    
    return raw

def visualize_ica_results(ica, filt_raw, raw_clean, raw_original, subject_id, output_dir):
    """Create essential ICA visualizations"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step04_ica')
    os.makedirs(subject_dir, exist_ok=True)
    
    print("Creating ICA visualizations...")
    
    # 1. Plot ICA components
    try:
        fig = ica.plot_components(inst=filt_raw, show=False)
        fig.savefig(os.path.join(subject_dir, 'ica_components.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not plot components: {e}")
    
    # 2. Plot excluded component properties
    if ica.exclude:
        for comp in ica.exclude[:3]:  # First 3 excluded
            try:
                fig = ica.plot_properties(filt_raw, picks=comp, show=False)
                fig.savefig(os.path.join(subject_dir, f'excluded_IC{comp:02d}.png'), dpi=300)
                plt.close()
            except Exception as e:
                print(f"Could not plot component {comp}: {e}")
    
    # 3. Before/after overlay comparison
    try:
        # Pick some channels for comparison
        picks = mne.pick_channels_regexp(raw_original.ch_names, regexp='F.*|C.*')[:6]
        
        fig = ica.plot_overlay(raw_original, exclude=ica.exclude, picks=picks, show=False)
        fig.savefig(os.path.join(subject_dir, 'before_after_overlay.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not create overlay plot: {e}")
    
    print(f"ICA visualizations saved to: {subject_dir}")

def apply_ica_and_save(ica, raw, subject_id, output_dir):
    """Apply ICA to original data and save results"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step04_ica')
    os.makedirs(subject_dir, exist_ok=True)
    
    print("Applying ICA to original (0.1-40 Hz) data...")
    
    # Apply ICA to original filtered data
    raw_clean = ica.apply(raw.copy())
    
    # Interpolate bad channels after ICA
    raw_clean = detect_and_interpolate_bad_channels(raw_clean)
    
    # Save results
    ica_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_ica.fif')
    raw_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_ica_raw.fif')
    
    ica.save(ica_fname, overwrite=True)
    raw_clean.save(raw_fname, overwrite=True)
    
    print(f"ICA object saved: {ica_fname}")
    print(f"Clean data saved: {raw_fname}")
    
    return raw_clean

def main():
    parser = argparse.ArgumentParser(description='Step 4: ICA with ICLabel')
    parser.add_argument('--subject', required=True, help='Subject ID')
    args = parser.parse_args()
    
    subject_id = args.subject
    
    print(f"Step 4: Running ICA for subject {subject_id}")
    print("="*50)
    
    # Load data from previous step
    raw = load_previous_step(subject_id)
    
    # Prepare ICA data (1-40 Hz, average ref)
    filt_raw = prepare_ica_data(raw)
    
    # Create epochs for ICA
    epochs = create_epochs_for_ica(filt_raw)
    
    # Fit ICA
    ica = run_ica(filt_raw, epochs)
    
    # Classify components
    ica = classify_components(ica, filt_raw)
    
    # Apply ICA and save
    raw_clean = apply_ica_and_save(ica, raw, subject_id, OUTPUT_DIR)
    
    # Create visualizations
    visualize_ica_results(ica, filt_raw, raw_clean, raw, subject_id, OUTPUT_DIR)
    
    print(f"\nStep 4 completed successfully!")
    print(f"Excluded {len(ica.exclude)} components: {ica.exclude}")
    print("="*50)

if __name__ == "__main__":
    main()
