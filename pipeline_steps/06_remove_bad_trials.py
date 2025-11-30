"""
Step 6: Remove bad trials using AutoReject
"""

import os
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt
from autoreject import AutoReject, get_rejection_threshold

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
TASK = "casinos"

def load_previous_step(subject_id):
    """Load data from previous step"""
    
    input_file = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 'step05_epochs', 
                             f'sub-{subject_id}_task-{TASK}-epo.fif')
    
    epochs = mne.read_epochs(input_file, preload=True)
    return epochs

def exclude_learning_trials(epochs, n_exclude=5):
    """Exclude first N trials (learning phase)"""
    
    print(f"Excluding first {n_exclude} trials (learning phase)")
    
    # Get trial indices to keep (exclude first n_exclude trials)
    keep_indices = range(n_exclude, len(epochs))
    epochs_clean = epochs[keep_indices]
    
    print(f"Trials: {len(epochs)} → {len(epochs_clean)} (excluded {n_exclude} learning trials)")
    
    return epochs_clean

def remove_bad_trials(epochs):
    """Remove bad trials using AutoReject"""
    
    print(f"Starting with {len(epochs)} epochs")

    # Exclude learning trials
    epochs = exclude_learning_trials(epochs, n_exclude=5)
    
    # AutoReject for interpolation and rejection
    ar = AutoReject(
        n_interpolate=[1, 2, 3, 4], 
        random_state=42, 
        n_jobs=1, 
        verbose=True
    )
    
    print("Fitting AutoReject...")
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    
    # Print rejection statistics
    n_interpolated = np.sum(reject_log.labels == 1)
    n_dropped = np.sum(reject_log.labels == 2)
    n_good = np.sum(reject_log.labels == 0)
    
    print(f"AutoReject results:")
    print(f"  Good epochs: {n_good}")
    print(f"  Interpolated epochs: {n_interpolated}")
    print(f"  Dropped epochs: {n_dropped}")
    print(f"  Final epochs: {len(epochs_clean)}")
    
    # Print per-condition statistics
    print(f"\nPer-condition trial counts:")
    for condition in epochs.event_id:
        if condition in epochs_clean.event_id:
            original_count = len(epochs[condition])
            clean_count = len(epochs_clean[condition])
            print(f"  {condition}: {original_count} → {clean_count} ({clean_count/original_count*100:.1f}%)")
    
    return epochs_clean, reject_log, ar

def visualize_rejection(epochs, epochs_clean, reject_log, ar, subject_id, output_dir):
    """Visualize trial rejection results"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step06_rejection')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Plot rejection log
    fig, ax = plt.subplots(figsize=(12, 6))
    reject_log.plot('horizontal', ax=ax, show=False)
    ax.set_title(f'Sub-{subject_id} AutoReject Results')
    plt.tight_layout()
    plt.savefig(os.path.join(subject_dir, 'autoreject_log.png'), dpi=300)
    plt.close()
    
    # Plot before/after comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data
    epochs.average().plot(axes=axes[0], show=False, titles='Original - Butterfly')
    
    # Clean data 
    epochs_clean.average().plot(axes=axes[1], show=False, titles='Clean - Butterfly')
    
    plt.tight_layout()
    fig.suptitle(f'Sub-{subject_id} Before/After AutoReject - ERPs', y=1.02)
    plt.savefig(os.path.join(subject_dir, 'before_after_erp_comparison.png'), dpi=300)
    plt.close()

    # Original data - image plot
    try:
        fig1 = epochs.plot_image(combine='mean', show=False, title='Original Data')
        fig1.suptitle(f'Sub-{subject_id} Original Epochs')
        fig1.savefig(os.path.join(subject_dir, 'original_epochs_image.png'), dpi=300)
        plt.close(fig1)
    except Exception as e:
        print(f"Could not create original epochs image: {e}")
    
    # Clean data - image plot
    try:
        fig2 = epochs_clean.plot_image(combine='mean', show=False, title='Clean Data')
        fig2.suptitle(f'Sub-{subject_id} Clean Epochs') 
        fig2.savefig(os.path.join(subject_dir, 'clean_epochs_image.png'), dpi=300)
        plt.close(fig2)
    except Exception as e:
        print(f"Could not create clean epochs image: {e}")
    
    # Plot rejection thresholds
    try:
        fig = ar.plot_reject_log(show=False)
        fig.savefig(os.path.join(subject_dir, 'rejection_thresholds.png'), dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Could not create rejection thresholds plot: {e}")
    
    # Plot trial counts per condition
    conditions = list(epochs.event_id.keys())
    original_counts = [len(epochs[cond]) for cond in conditions]
    clean_counts = [len(epochs_clean[cond]) if cond in epochs_clean.event_id else 0 for cond in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, original_counts, width, label='Original', alpha=0.7)
    bars2 = ax.bar(x + width/2, clean_counts, width, label='Clean', alpha=0.7)
    
    ax.set_xlabel('Conditions')
    ax.set_ylabel('Number of Trials')
    ax.set_title(f'Sub-{subject_id} Trial Counts Before/After AutoReject')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend()
    
    # Add percentage labels
    for i, (orig, clean) in enumerate(zip(original_counts, clean_counts)):
        if orig > 0:
            percentage = clean/orig*100
            ax.text(i + width/2, clean + 1, f'{percentage:.1f}%', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(subject_dir, 'trial_counts_comparison.png'), dpi=300)
    plt.close()
    
    create_simple_comparison_plot(epochs, epochs_clean, subject_dir, subject_id)
    
    print(f"Rejection visualizations saved to: {subject_dir}")

def create_simple_comparison_plot(epochs, epochs_clean, subject_dir, subject_id):
    """Create a simple comparison plot without complex subplots"""
    
    try:
        # Create a 2x2 comparison using averages
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Original average at FCz
        epochs.average().plot_topo(axes=axes[0,0], show=False, title='Original - Topo')
        
        # Plot 2: Clean average at FCz  
        epochs_clean.average().plot_topo(axes=axes[0,1], show=False, title='Clean - Topo')
        
        # Plot 3: Original average butterfly
        epochs.average().plot(axes=axes[1,0], show=False, titles='Original - All Channels')
        
        # Plot 4: Clean average butterfly
        epochs_clean.average().plot(axes=axes[1,1], show=False, titles='Clean - All Channels')
        
        plt.tight_layout()
        fig.suptitle(f'Sub-{subject_id} Before/After AutoReject - Detailed Comparison', y=1.02)
        plt.savefig(os.path.join(subject_dir, 'detailed_comparison.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Could not create simple comparison plot: {e}")


def save_clean_epochs(epochs_clean, reject_log, ar, subject_id, output_dir):
    """Save clean epochs and rejection info"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step06_rejection')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Save clean epochs
    epochs_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_clean-epo.fif')
    epochs_clean.save(epochs_fname, overwrite=True)
    
    # Save AutoReject object
    ar_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_autoreject.pkl')
    import pickle
    with open(ar_fname, 'wb') as f:
        pickle.dump(ar, f)
    
    # Save rejection log
    log_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_reject_log.npz')
    np.savez(log_fname, 
             labels=reject_log.labels,
             ch_names=reject_log.ch_names,
             bad_epochs=reject_log.bad_epochs)
    
    print(f"Clean epochs saved to: {epochs_fname}")
    print(f"AutoReject object saved to: {ar_fname}")
    print(f"Rejection log saved to: {log_fname}")
    
    return epochs_fname

def main():
    parser = argparse.ArgumentParser(description='Step 6: Remove bad trials')
    parser.add_argument('--subject', required=True, help='Subject ID')
    args = parser.parse_args()
    
    subject_id = args.subject
    
    print(f"Step 6: Removing bad trials for subject {subject_id}")
    
    # Load data from previous step
    epochs = load_previous_step(subject_id)
    
    # Remove bad trials
    epochs_clean, reject_log, ar = remove_bad_trials(epochs)
    
    # Create visualizations
    visualize_rejection(epochs, epochs_clean, reject_log, ar, subject_id, OUTPUT_DIR)
    
    # Save clean epochs
    save_clean_epochs(epochs_clean, reject_log, ar, subject_id, OUTPUT_DIR)
    
    print(f"Step 6 completed for subject {subject_id}")

if __name__ == "__main__":
    main()
