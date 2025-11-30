"""
Step 8: Run statistical tests on single subject data
"""

import os
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
from scipy import stats

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
TASK = "casinos"

# Contrasts from config
CONTRASTS = [
    ('low_task_win', 'low_task_loss'),
    ('mid_low_task_win', 'mid_low_task_loss'),
    ('mid_high_task_win', 'mid_high_task_loss'),
    ('high_task_win', 'high_task_loss')
]

def load_previous_step(subject_id):
    """Load data from previous step"""
    
    # Load clean epochs for statistics
    epochs_file = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 'step06_rejection', 
                              f'sub-{subject_id}_task-{TASK}_clean-epo.fif')
    epochs = mne.read_epochs(epochs_file, preload=True)
    
    # Load evoked responses  
    evoked_dir = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 'step07_evoked')
    evokeds = {}
    
    # Load individual evoked files
    for condition in ['low_task_win', 'low_task_loss', 'mid_low_task_win', 'mid_low_task_loss',
                     'mid_high_task_win', 'mid_high_task_loss', 'high_task_win', 'high_task_loss']:
        evoked_file = os.path.join(evoked_dir, f'sub-{subject_id}_task-{TASK}_{condition}_ave.fif')
        if os.path.exists(evoked_file):
            evokeds[condition] = mne.read_evokeds(evoked_file)[0]
    
    return epochs, evokeds


def analyze_rewp_amplitudes(evokeds, subject_id):
    """Extract RewP amplitudes following study methodology"""
    
    rewp_results = {}
    
    # RewP time window and electrode (from study: FCz, 240-340ms)
    time_window = (0.240, 0.340)
    electrode = 'FCz'
    
    print(f"\nAnalyzing RewP at {electrode} electrode, {time_window[0]*1000}-{time_window[1]*1000}ms:")
    
    # Calculate RewP for each condition
    conditions = [
        ('low_task', 'low_task_win', 'low_task_loss'),
        ('mid_low_task', 'mid_low_task_win', 'mid_low_task_loss'),
        ('mid_high_task', 'mid_high_task_win', 'mid_high_task_loss'),
        ('high_task', 'high_task_win', 'high_task_loss')
    ]
    
    for condition_name, win_cond, loss_cond in conditions:
        if win_cond in evokeds and loss_cond in evokeds:
            # Create difference wave (win - loss)
            diff_evoked = mne.combine_evoked([evokeds[win_cond], evokeds[loss_cond]], weights=[1, -1])
            
            # Find FCz channel
            if electrode in diff_evoked.ch_names:
                ch_idx = diff_evoked.ch_names.index(electrode)
                
                # Extract data in time window
                time_mask = (diff_evoked.times >= time_window[0]) & (diff_evoked.times <= time_window[1])
                rewp_data = diff_evoked.data[ch_idx, time_mask] * 1e6  # Convert to µV
                
                # Debug print
                print(f"    {condition_name}: Data range: {rewp_data.min():.3f} to {rewp_data.max():.3f} µV")
                
                # RewP amplitude (most negative for classic RewP)
                rewp_amplitude = float(np.max(rewp_data))  # Most negative
                
                # Mean voltage in window
                mean_amplitude = float(np.mean(rewp_data))
                
                rewp_results[condition_name] = {
                    'rewp_amplitude': rewp_amplitude,
                    'mean_amplitude': mean_amplitude,
                    'peak_time': diff_evoked.times[time_mask][np.argmax(rewp_data)]
                }
                
                print(f"  {condition_name}: RewP = {rewp_amplitude:.2f} µV, Mean = {mean_amplitude:.2f} µV")
            else:
                print(f"    {electrode} not found in {condition_name}")
    
    return rewp_results


def run_permutation_cluster_tests(epochs, contrasts, subject_id):
    """Run permutation cluster tests for contrasts"""
    
    results = {}
    
    print("Running permutation cluster tests:")
    
    for cond1, cond2 in contrasts:
        if cond1 in epochs.event_id and cond2 in epochs.event_id:
            print(f"\n  Testing: {cond1} vs {cond2}")
            
            # Get data for both conditions
            data1 = epochs[cond1].get_data()  # Shape: (n_epochs, n_channels, n_times)
            data2 = epochs[cond2].get_data()
            
            min_trials = min(data1.shape[0], data2.shape[0])
            
            if min_trials > 5:  # Need sufficient trials
                # Take first min_trials from each condition for paired comparison
                data1_matched = data1[:min_trials]
                data2_matched = data2[:min_trials]
                
                # Calculate difference for each trial (now they have same shape)
                diff_data = data1_matched - data2_matched  # Paired difference
                
                print(f"    Using {min_trials} matched trials for comparison")
                
                # Run one-sample t-test against zero (paired comparison)
                T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                    diff_data, 
                    n_permutations=1000,
                    threshold=None,
                    tail=0,  # Two-tailed test
                    n_jobs=1,
                    out_type='mask',
                    verbose=False
                )
                
                # Store results
                results[f"{cond1}_vs_{cond2}"] = {
                    'T_obs': T_obs,
                    'clusters': clusters,
                    'cluster_p_values': cluster_p_values,
                    'H0': H0,
                    'diff_data': diff_data,
                    'n_trials': min_trials
                }
                
                # Print significant clusters
                sig_clusters = np.where(cluster_p_values < 0.05)[0]
                print(f"    Matched trials used: {min_trials}")
                print(f"    Significant clusters: {len(sig_clusters)}")
                
                for i, cluster_idx in enumerate(sig_clusters):
                    p_val = cluster_p_values[cluster_idx]
                    cluster_mask = clusters[cluster_idx]
                    cluster_times = epochs.times[cluster_mask.any(axis=0)]
                    if len(cluster_times) > 0:
                        print(f"      Cluster {i+1}: p={p_val:.4f}, "
                              f"time={cluster_times[0]:.3f}-{cluster_times[-1]:.3f}s")
            else:
                print(f"    Insufficient trials ({min_trials})")
                results[f"{cond1}_vs_{cond2}"] = None
        else:
            print(f"\n  Skipping {cond1} vs {cond2}: missing conditions")
            results[f"{cond1}_vs_{cond2}"] = None
    
    return results

def run_simple_statistics(evokeds):
    """Run simple statistical tests on evoked responses"""
    
    stats_results = {}
    
    print("\nRunning simple statistics on evoked responses:")
    
    # Test each condition against zero at peak time
    for condition, evoked in evokeds.items():
        evoked_data_uv = evoked.data * 1e6  # Convert to µV
        # Find peak amplitude
        peak_idx = np.unravel_index(np.argmax(np.abs(evoked_data_uv)), evoked_data_uv.shape)
        peak_channel = evoked.ch_names[peak_idx[0]]
        peak_time = evoked.times[peak_idx[1]]
        peak_amplitude = evoked_data_uv[peak_idx]
        
        stats_results[condition] = {
            'peak_channel': peak_channel,
            'peak_time': peak_time,
            'peak_amplitude': peak_amplitude,
            'mean_amplitude': np.mean(evoked_data_uv),
            'std_amplitude': np.std(evoked_data_uv)
        }
        
        print(f"  {condition}: peak={peak_amplitude:.2f}µV at {peak_time:.3f}s ({peak_channel})")
    
    return stats_results

def visualize_statistics(cluster_results, epochs, evokeds, rewp_results, subject_id, output_dir):
    """Visualize statistical results"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step08_statistics')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Plot cluster test results
    for contrast_name, result in cluster_results.items():
        if result is not None:
            T_obs = result['T_obs']
            clusters = result['clusters'] 
            cluster_p_values = result['cluster_p_values']
            
            # Find significant clusters
            sig_clusters = np.where(cluster_p_values < 0.05)[0]
            
            if len(sig_clusters) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Plot T-statistic topography at peak time
                peak_time_idx = np.unravel_index(np.argmax(np.abs(T_obs)), T_obs.shape)[1]
                peak_time = epochs.times[peak_time_idx]
                
                im = axes[0,0].imshow(T_obs[:, peak_time_idx:peak_time_idx+1], 
                                     aspect='auto', cmap='RdBu_r')
                axes[0,0].set_title(f'T-statistic at {peak_time:.3f}s')
                plt.colorbar(im, ax=axes[0,0])
                
                # Plot significant cluster mask
                cluster_mask = clusters[sig_clusters[0]]
                mean_T = np.mean(T_obs, axis=0)
                axes[1,0].plot(epochs.times, mean_T)
                axes[1,0].set_xlabel('Time (s)')
                axes[1,0].set_ylabel('T-statistic')
                axes[1,0].set_title('Mean T-statistic over time')
                
                axes[1,0].axhline(0, color='k', linestyle='--', alpha=0.5)
                axes[1,0].grid(True, alpha=0.3)
                
                # Plot difference wave
                if len(contrast_name.split('_vs_')) == 2:
                    cond1, cond2 = contrast_name.split('_vs_')
                    if cond1 in evokeds and cond2 in evokeds:
                        diff_evoked = evokeds[cond1].copy()
                        diff_evoked.data = evokeds[cond1].data - evokeds[cond2].data
                        
                        # Plot butterfly plot of difference
                        diff_evoked.plot(axes=axes[1,1], show=False, 
                                       titles=f'{cond1} - {cond2}')
                
                plt.tight_layout()
                fig.suptitle(f'Sub-{subject_id} Cluster Test: {contrast_name}', y=1.02)
                plt.savefig(os.path.join(subject_dir, f'cluster_test_{contrast_name}.png'), dpi=300)
                plt.close()
    
    # Create summary statistics plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot RewP results
    if rewp_results:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot RewP amplitudes
        conditions = list(rewp_results.keys())
        rewp_amps = [results['rewp_amplitude'] for results in rewp_results.values()]
        mean_amps = [results['mean_amplitude'] for results in rewp_results.values()]
        
        x = np.arange(len(conditions))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, rewp_amps, width, label='Peak RewP', alpha=0.7)
        bars2 = axes[0].bar(x + width/2, mean_amps, width, label='Mean RewP', alpha=0.7)
        
        axes[0].set_xlabel('Conditions')
        axes[0].set_ylabel('RewP Amplitude (µV)')
        axes[0].set_title(f'Sub-{subject_id} RewP Amplitudes (FCz, 240-340ms)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(conditions, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot peak times
        peak_times = [results['peak_time']*1000 for results in rewp_results.values()]
        bars = axes[1].bar(x, peak_times, alpha=0.7, color='orange')
        axes[1].set_xlabel('Conditions')
        axes[1].set_ylabel('Peak Time (ms)')
        axes[1].set_title(f'Sub-{subject_id} RewP Peak Times')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(conditions, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        # Add horizontal line at expected RewP time (~290ms)
        axes[1].axhline(290, color='red', linestyle='--', alpha=0.7, label='Expected RewP (~290ms)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(subject_dir, 'rewp_analysis.png'), dpi=300)
        plt.close()
    
    # Plot peak amplitudes
    conditions = list(evokeds.keys())
    peak_amps = [np.max(np.abs(evoked.data)) for evoked in evokeds.values()]
    peak_times = [evoked.times[np.argmax(np.abs(evoked.data.mean(axis=0)))] 
                  for evoked in evokeds.values()]
    
    x = np.arange(len(conditions))
    bars = axes[0].bar(x, peak_amps, alpha=0.7)
    axes[0].set_xlabel('Conditions')
    axes[0].set_ylabel('Peak Amplitude (µV)')
    axes[0].set_title(f'Sub-{subject_id} Peak Amplitudes')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conditions, rotation=45, ha='right')
    
    # Plot peak times
    bars = axes[1].bar(x, [t*1000 for t in peak_times], alpha=0.7)
    axes[1].set_xlabel('Conditions')
    axes[1].set_ylabel('Peak Time (ms)')
    axes[1].set_title(f'Sub-{subject_id} Peak Times')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conditions, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(subject_dir, 'peak_statistics.png'), dpi=300)
    plt.close()
    
    print(f"Statistical visualizations saved to: {subject_dir}")

def save_statistics(cluster_results, simple_stats, rewp_results, subject_id, output_dir):
    """Save statistical results"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step08_statistics')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Save cluster results
    cluster_fname = os.path.join(subject_dir, f'sub-{subject_id}_cluster_results.npz')
    cluster_data = {}
    
    for contrast, result in cluster_results.items():
        if result is not None:
            cluster_data[f'{contrast}_T_obs'] = result['T_obs']
            cluster_data[f'{contrast}_cluster_p_values'] = result['cluster_p_values']
            cluster_data[f'{contrast}_n_trials'] = result['n_trials']
    
    if cluster_data:
        np.savez(cluster_fname, **cluster_data)
        print(f"Cluster results saved to: {cluster_fname}")
    
    # Save simple statistics as text
    stats_fname = os.path.join(subject_dir, f'sub-{subject_id}_simple_stats.txt')
    with open(stats_fname, 'w') as f:
        f.write(f"Simple Statistics for Subject {subject_id}\n")
        f.write("="*50 + "\n\n")
        
        for condition, stats in simple_stats.items():
            f.write(f"{condition}:\n")
            f.write(f"  Peak channel: {stats['peak_channel']}\n")
            f.write(f"  Peak time: {stats['peak_time']:.3f} s\n")
            f.write(f"  Peak amplitude: {stats['peak_amplitude']:.2f} µV\n")
            f.write(f"  Mean amplitude: {stats['mean_amplitude']:.2f} µV\n")
            f.write(f"  Std amplitude: {stats['std_amplitude']:.2f} µV\n\n")
    
    print(f"Simple statistics saved to: {stats_fname}")

    if rewp_results:
        rewp_fname = os.path.join(subject_dir, f'sub-{subject_id}_rewp_results.txt')
        with open(rewp_fname, 'w') as f:
            f.write(f"RewP Analysis Results for Subject {subject_id}\n")
            f.write("="*50 + "\n")
            f.write("Following Sambrook & Goslin (2015): FCz electrode, 240-340ms window\n\n")
            
            for condition, results in rewp_results.items():
                f.write(f"{condition}:\n")
                f.write(f"  RewP Amplitude (max): {results['rewp_amplitude']:.2f} µV\n")
                f.write(f"  Mean Amplitude: {results['mean_amplitude']:.2f} µV\n")
                f.write(f"  Peak Time: {results['peak_time']:.3f} s\n\n")
        
        print(f"RewP results saved to: {rewp_fname}")
    
    return subject_dir


def main():
    parser = argparse.ArgumentParser(description='Step 8: Run statistical tests')
    parser.add_argument('--subject', required=True, help='Subject ID')
    args = parser.parse_args()
    
    subject_id = args.subject
    
    print(f"Step 8: Running statistical tests for subject {subject_id}")
    
    # Load data from previous steps
    epochs, evokeds = load_previous_step(subject_id)

    # Analyze RewP amplitudes
    rewp_results = analyze_rewp_amplitudes(evokeds, subject_id)
    
    # Run cluster permutation tests
    cluster_results = run_permutation_cluster_tests(epochs, CONTRASTS, subject_id)
    
    # Run simple statistics
    simple_stats = run_simple_statistics(evokeds)
    
    # Create visualizations
    visualize_statistics(cluster_results, epochs, evokeds, rewp_results, subject_id, OUTPUT_DIR)
    
    # Save results
    save_statistics(cluster_results, simple_stats, rewp_results, subject_id, OUTPUT_DIR)
    
    print(f"Step 8 completed for subject {subject_id}")

if __name__ == "__main__":
    main()
