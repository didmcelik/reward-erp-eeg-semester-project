"""
Step 11: Group-level decoding analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
SUBJECTS = ['27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']

def load_subject_decoding(subject_id):
    """Load decoding results for one subject"""
    results_file = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 
                               'step10_decoding',
                               f'sub-{subject_id}_decoding_results.npz')
    data = np.load(results_file)
    return data['scores'], data['times']

def group_analysis():
    """Aggregate and analyze decoding across subjects"""
    
    all_scores = []
    times = None
    
    print("Loading subject data...")
    for subject in SUBJECTS:
        try:
            scores, times = load_subject_decoding(subject)
            all_scores.append(scores)
            print(f"  ✓ Sub-{subject}: Peak {np.max(scores[times>=0]):.2%}")
        except FileNotFoundError:
            print(f"  ✗ Sub-{subject}: No data found")
    
    all_scores = np.array(all_scores)  # Shape: (n_subjects, n_times)
    
    # Calculate group statistics
    mean_scores = np.mean(all_scores, axis=0)
    sem_scores = stats.sem(all_scores, axis=0)
    
    # One-sample t-test against chance (50%) at each timepoint
    t_values, p_values = stats.ttest_1samp(all_scores, 0.5, axis=0)
    
    # Find significant timepoints (p < 0.05, uncorrected)
    sig_mask = p_values < 0.05
    
    # Visualize group results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Mean accuracy with SEM
    ax1.plot(times, mean_scores, linewidth=2, label='Group Mean')
    ax1.fill_between(times, mean_scores - sem_scores, mean_scores + sem_scores,
                     alpha=0.3, label='SEM')
    ax1.axhline(0.5, color='k', linestyle='--', label='Chance')
    ax1.axvspan(0.240, 0.340, alpha=0.2, color='red', label='RewP Window')
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Highlight significant periods
    sig_times = times[sig_mask]
    if len(sig_times) > 0:
        ax1.scatter(sig_times, mean_scores[sig_mask], c='red', s=10, 
                   zorder=5, label='p < 0.05')
    
    # Find group peak in RewP window
    rewp_mask = (times >= 0.240) & (times <= 0.340)
    rewp_peak_idx = np.argmax(mean_scores[rewp_mask])
    rewp_peak_time = times[rewp_mask][rewp_peak_idx]
    rewp_peak_acc = mean_scores[rewp_mask][rewp_peak_idx]
    
    ax1.plot(rewp_peak_time, rewp_peak_acc, 'ro', markersize=12,
            label=f'RewP Peak: {rewp_peak_acc:.2%} at {rewp_peak_time*1000:.0f}ms')
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title(f'Group Decoding (N={len(all_scores)}): Task-Value (High vs Mid)', 
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.45, 0.60])
    
    # Plot 2: Individual subjects
    for i, subject in enumerate(SUBJECTS[:len(all_scores)]):
        ax2.plot(times, all_scores[i], alpha=0.5, linewidth=1)
    
    ax2.plot(times, mean_scores, 'k', linewidth=3, label='Group Mean')
    ax2.axhline(0.5, color='k', linestyle='--')
    ax2.axvspan(0.240, 0.340, alpha=0.2, color='red')
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Classification Accuracy', fontsize=12)
    ax2.set_title('Individual Subject Timecourses', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.40, 0.65])
    
    plt.tight_layout()
    
    # Save
    group_dir = os.path.join(OUTPUT_DIR, 'group_analysis')
    os.makedirs(group_dir, exist_ok=True)
    plt.savefig(os.path.join(group_dir, 'group_decoding_results.png'), dpi=300)
    
    print(f"\n{'='*60}")
    print("GROUP RESULTS:")
    print(f"{'='*60}")
    print(f"RewP window peak: {rewp_peak_acc:.2%} at {rewp_peak_time*1000:.0f}ms")
    print(f"RewP window mean: {np.mean(mean_scores[rewp_mask]):.2%}")
    print(f"Significant timepoints: {np.sum(sig_mask)}/{len(times)}")
    print(f"Results saved to: {group_dir}")

if __name__ == "__main__":
    group_analysis()
