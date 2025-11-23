"""
Step 5: Create epochs from continuous data
"""

import os
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
TASK = "casinos"

# Epoching parameters
EPOCHS_TMIN = -0.2
EPOCHS_TMAX = 0.6
BASELINE = (-0.2, 0)

# Event renaming
RENAME_EVENTS = {
    'Stimulus/S  6': 'low_task_win', 
    'Stimulus/S  7': 'low_task_loss',
    'Stimulus/S 16': 'mid_low_task_win', 
    'Stimulus/S 17': 'mid_low_task_loss',
    'Stimulus/S 26': 'mid_high_task_win', 
    'Stimulus/S 27': 'mid_high_task_loss',
    'Stimulus/S 36': 'high_task_win', 
    'Stimulus/S 37': 'high_task_loss',
    'Stimulus/S  2': 'low_task_cue',
    'Stimulus/S 12': 'mid_low_task_cue',
    'Stimulus/S 22': 'mid_high_task_cue',
    'Stimulus/S 32': 'high_task_cue',
}

def load_previous_step(subject_id):
    """Load data from previous step"""
    
    input_file = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 'step04_ica', 
                             f'sub-{subject_id}_task-{TASK}_ica_raw.fif')
    
    raw = mne.io.read_raw_fif(input_file, preload=True)
    return raw

def create_epochs(raw):
    """Create epochs from continuous data"""
    
    # Find events
    events, event_id = mne.events_from_annotations(raw)
    
    print(f"Found {len(events)} events")
    print(f"Original event IDs: {list(event_id.keys())}")
    
    # Rename events
    new_event_id = {}
    for old_name, new_name in RENAME_EVENTS.items():
        if old_name in event_id:
            new_event_id[new_name] = event_id[old_name]
            print(f"Renamed '{old_name}' to '{new_name}'")
    
    print(f"Final event IDs: {list(new_event_id.keys())}")
    
    # Create epochs
    epochs = mne.Epochs(
        raw, events, new_event_id,
        tmin=EPOCHS_TMIN, tmax=EPOCHS_TMAX,
        baseline=BASELINE,
        reject={'eeg': 100e-6},  # Peak-to-peak amplitude rejection
        flat={'eeg': 1e-6},    # Flat signal rejection
        preload=True
    )
    
    print(f"Created {len(epochs)} epochs")

    print("Applying gradient-based rejection (study criterion: 40 µV/sample)...")
    epochs = apply_gradient_rejection(epochs, threshold=40e-6)

    try:
        drop_stats = epochs.drop_log_stats()
        if isinstance(drop_stats, dict):
            n_dropped = drop_stats.get('n_dropped_total', 0)
        else:
            n_dropped = drop_stats  # If it's a scalar
        print(f"Dropped {n_dropped} epochs due to artifacts")
    except Exception as e:
        # Fallback: count manually
        n_dropped = sum(len(log) > 0 for log in epochs.drop_log)
        print(f"Dropped {n_dropped} epochs due to artifacts (manual count)")
    
    # Print epoch counts per condition
    for condition in epochs.event_id:
        n_epochs = len(epochs[condition])
        print(f"  {condition}: {n_epochs} trials")
    
    return epochs, events, new_event_id


def apply_gradient_rejection(epochs, threshold=40e-6):
    """Apply gradient-based artifact rejection (study criterion)"""
    
    print(f"Applying gradient rejection with threshold: {threshold*1e6:.0f} µV/sample")
    
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    
    # Calculate gradient (difference between consecutive samples)
    gradient = np.abs(np.diff(data, axis=2))  # Shape: (n_epochs, n_channels, n_times-1)
    
    # Find epochs that exceed gradient threshold
    max_gradient_per_epoch = np.max(gradient, axis=(1, 2))  # Max gradient per epoch
    bad_epochs = max_gradient_per_epoch > threshold
    
    n_gradient_rejected = np.sum(bad_epochs)
    print(f"Gradient rejection: {n_gradient_rejected} epochs exceed {threshold*1e6:.0f} µV/sample")
    
    if n_gradient_rejected > 0:
        # Drop bad epochs
        epochs.drop(np.where(bad_epochs)[0], reason='GRADIENT')
        print(f"Remaining epochs after gradient rejection: {len(epochs)}")
    
    return epochs


def create_epoch_overview_plot(epochs, subject_dir, subject_id):
    """Create a simple overview plot of epochs"""
    
    try:
        # Plot first few epochs for visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot epoch count per condition
        conditions = list(epochs.event_id.keys())
        counts = [len(epochs[cond]) for cond in conditions]
        
        axes[0].bar(range(len(conditions)), counts)
        axes[0].set_xticks(range(len(conditions)))
        axes[0].set_xticklabels(conditions, rotation=45, ha='right')
        axes[0].set_ylabel('Number of Epochs')
        axes[0].set_title(f'Sub-{subject_id} Epoch Counts per Condition')
        axes[0].grid(True, alpha=0.3)
        
        # Plot average across all epochs
        if len(epochs) > 0:
            grand_avg = epochs.average()
            grand_avg.plot(axes=axes[1], show=False, titles='Grand Average')
            axes[1].set_title('Grand Average ERP')
        
        plt.tight_layout()
        plt.savefig(os.path.join(subject_dir, 'epoch_overview.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Could not create epoch overview plot: {e}")


def visualize_epochs(epochs, events, raw, subject_id, output_dir):
    """Visualize epoch creation results"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step05_epochs')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Plot events
    fig = mne.viz.plot_events(events, event_id=epochs.event_id, sfreq=raw.info['sfreq'], 
                             first_samp=raw.first_samp, show=False)
    fig.suptitle(f'Sub-{subject_id} Events')
    fig.savefig(os.path.join(subject_dir, 'events.png'), dpi=300)
    plt.close(fig)
    
    # Plot drop log
    fig = epochs.plot_drop_log(show=False)
    fig.savefig(os.path.join(subject_dir, 'drop_log.png'), dpi=300)
    plt.close(fig)
    
    # Plot example epochs for each condition
    conditions = list(epochs.event_id.keys())[:4]  # Plot first 4 conditions
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, condition in enumerate(conditions):
        if condition in epochs.event_id and len(epochs[condition]) > 0:
            epochs[condition].average().plot(axes=axes[i], show=False, titles=condition)
    
    plt.tight_layout()
    fig.suptitle(f'Sub-{subject_id} Average ERPs', y=1.02)
    plt.savefig(os.path.join(subject_dir, 'average_erps_preview.png'), dpi=300)
    plt.close()
    
    create_epoch_overview_plot(epochs, subject_dir, subject_id)
    
    print(f"Epoching visualizations saved to: {subject_dir}")

def save_epochs(epochs, subject_id, output_dir):
    """Save epochs"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step05_epochs')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Save epochs
    epochs_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_epochs.fif')
    epochs.save(epochs_fname, overwrite=True)
    
    print(f"Epochs saved to: {epochs_fname}")
    
    return epochs_fname

def main():
    parser = argparse.ArgumentParser(description='Step 5: Create epochs')
    parser.add_argument('--subject', required=True, help='Subject ID')
    args = parser.parse_args()
    
    subject_id = args.subject
    
    print(f"Step 5: Creating epochs for subject {subject_id}")
    
    # Load data from previous step
    raw = load_previous_step(subject_id)
    
    # Create epochs
    epochs, events, event_id = create_epochs(raw)
    
    # Create visualizations
    visualize_epochs(epochs, events, raw, subject_id, OUTPUT_DIR)
    
    # Save epochs
    save_epochs(epochs, subject_id, OUTPUT_DIR)
    
    print(f"Step 5 completed for subject {subject_id}")

if __name__ == "__main__":
    main()
