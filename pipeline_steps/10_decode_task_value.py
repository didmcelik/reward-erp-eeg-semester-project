"""
Step 10: Decode task-value (High vs Mid) from post-feedback EEG
"""

import os
import numpy as np
import mne
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
TASK = "casinos"

def load_epochs_for_decoding(subject_id):
    """Load clean epochs from step 6"""
    
    epochs_file = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 
                              'step06_rejection', 
                              f'sub-{subject_id}_task-{TASK}_clean-epo.fif')
    epochs = mne.read_epochs(epochs_file, preload=True)
    return epochs


def prepare_decoding_data(epochs, conditions):
    """
    Prepare data for binary classification
    
    Parameters
    ----------
    epochs : mne.Epochs
        Clean epochs
    conditions : dict
        {'high': ['high_task_win', 'high_task_loss'],
         'mid': ['mid_high_task_win', 'mid_high_task_loss']}
    
    Returns
    -------
    X : array (n_trials, n_channels, n_times)
    y : array (n_trials,)
        Labels: 0=mid, 1=high
    """
    
    # Select trials
    high_epochs = mne.concatenate_epochs([epochs[cond] for cond in conditions['high'] 
                                          if cond in epochs.event_id])
    mid_epochs = mne.concatenate_epochs([epochs[cond] for cond in conditions['mid'] 
                                         if cond in epochs.event_id])
    
    # Balance classes by downsampling majority class
    n_high = len(high_epochs)
    n_mid = len(mid_epochs)
    n_min = min(n_high, n_mid)
    
    print(f"Original trial counts: High={n_high}, Mid={n_mid}")
    
    # Randomly select equal number of trials
    np.random.seed(42)
    if n_high > n_min:
        high_indices = np.random.choice(n_high, n_min, replace=False)
        high_epochs = high_epochs[high_indices]
    if n_mid > n_min:
        mid_indices = np.random.choice(n_mid, n_min, replace=False)
        mid_epochs = mid_epochs[mid_indices]
    
    # Get data
    X_high = high_epochs.get_data()
    X_mid = mid_epochs.get_data()
    
    # Concatenate
    X = np.concatenate([X_high, X_mid], axis=0)
    y = np.concatenate([np.ones(len(X_high)), np.zeros(len(X_mid))])
    
    print(f"Balanced decoding data: {len(X)} trials")
    print(f"  High-value: {len(X_high)} trials")
    print(f"  Mid-value: {len(X_mid)} trials")
    
    return X, y, high_epochs.times


def decode_task_value(X, y, times, smooth_sigma=12.5):
    """
    Time-resolved decoding of task-value with temporal smoothing
    
    Parameters
    ----------
    smooth_sigma : float
        Gaussian smoothing kernel width in timepoints (default=12.5 = 50ms)
    """
    
    n_trials, n_channels, n_times = X.shape
    scores = np.zeros(n_times)
    
    # Define classifier
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='liblinear', random_state=42)
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Running time-resolved decoding...")
    for t in range(n_times):
        if t % 25 == 0:
            print(f"  Timepoint {t}/{n_times} ({times[t]*1000:.0f}ms)")
        
        # Get data at this timepoint
        X_t = X[:, :, t]
        
        # Cross-validate
        scores[t] = np.mean(cross_val_score(clf, X_t, y, cv=cv, 
                                            scoring='accuracy'))
    
    # Apply temporal smoothing
    scores_smoothed = gaussian_filter1d(scores, sigma=smooth_sigma)
    
    print(f"Applied Gaussian smoothing (σ={smooth_sigma*4:.0f}ms)")
    
    return scores_smoothed


def visualize_decoding_results(scores, times, subject_id, output_dir):
    """Plot decoding accuracy over time"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 
                              'step10_decoding')
    os.makedirs(subject_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot accuracy
    ax.plot(times, scores, linewidth=2, label='Decoding Accuracy (Smoothed)')
    ax.axhline(0.5, color='k', linestyle='--', label='Chance (50%)')
    
    # Highlight RewP window
    ax.axvspan(0.240, 0.340, alpha=0.2, color='red', 
               label='RewP Window (240-340ms)')
    
    # Mark feedback onset
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Feedback Onset')
    
    # Find peak accuracy in POST-FEEDBACK period only
    post_feedback_mask = times >= 0
    post_feedback_scores = scores[post_feedback_mask]
    post_feedback_times = times[post_feedback_mask]
    
    peak_idx = np.argmax(post_feedback_scores)
    peak_time = post_feedback_times[peak_idx]
    peak_acc = post_feedback_scores[peak_idx]
    
    ax.plot(peak_time, peak_acc, 'ro', markersize=10, 
            label=f'Peak: {peak_acc:.2%} at {peak_time*1000:.0f}ms')
    
    # Calculate mean accuracy in RewP window
    rewp_mask = (times >= 0.240) & (times <= 0.340)
    rewp_acc = np.mean(scores[rewp_mask])
    ax.axhline(rewp_acc, xmin=0.55, xmax=0.675, color='red', 
               linewidth=3, alpha=0.7, 
               label=f'RewP Mean: {rewp_acc:.2%}')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title(f'Sub-{subject_id}: Decoding Task-Value (High vs Mid)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.3, 0.7])  # Set reasonable y-axis limits
    
    plt.tight_layout()
    plt.savefig(os.path.join(subject_dir, 'decoding_timecourse.png'), dpi=300)
    plt.close()
    
    print(f"Decoding plot saved to: {subject_dir}")
    print(f"Peak post-feedback accuracy: {peak_acc:.2%} at {peak_time*1000:.0f}ms")
    print(f"Mean RewP window accuracy: {rewp_acc:.2%}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 10: Decode task-value')
    parser.add_argument('--subject', required=True, help='Subject ID')
    args = parser.parse_args()
    
    subject_id = args.subject
    
    print(f"Step 10: Decoding task-value for subject {subject_id}")
    
    # Load data
    epochs = load_epochs_for_decoding(subject_id)
    
    # Define conditions
    conditions = {
        'high': ['high_task_win', 'high_task_loss'],
        'mid': ['mid_high_task_win', 'mid_high_task_loss']
    }
    
    # Prepare data
    X, y, times = prepare_decoding_data(epochs, conditions)
    
    # Decode
    scores = decode_task_value(X, y, times)
    
    # Visualize
    visualize_decoding_results(scores, times, subject_id, OUTPUT_DIR)
    
    # Save results
    save_dir = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 'step10_decoding')
    np.savez(os.path.join(save_dir, f'sub-{subject_id}_decoding_results.npz'),
             scores=scores, times=times, y=y)
    
    print(f"Step 10 completed for subject {subject_id}")


if __name__ == "__main__":
    main()
