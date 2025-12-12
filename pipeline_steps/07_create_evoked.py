"""
Step 7: Create evoked responses for each condition
"""

import os
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
TASK = "casinos"

# Conditions from config
CONDITIONS = [
    'low_task_win', 'low_task_loss',
    'mid_low_task_win', 'mid_low_task_loss',
    'mid_high_task_win', 'mid_high_task_loss',
    'high_task_win', 'high_task_loss'
]

def load_previous_step(subject_id):
    """Load data from previous step"""
    
    input_file = os.path.join(OUTPUT_DIR, f'sub-{subject_id}', 'step06_rejection', 
                             f'sub-{subject_id}_task-{TASK}_clean-epo.fif')
    
    epochs = mne.read_epochs(input_file, preload=True)
    return epochs

def create_evoked_responses(epochs):
    """Create evoked responses for each condition"""
    
    evokeds = {}
    
    print("Creating evoked responses:")
    for condition in CONDITIONS:
        if condition in epochs.event_id:
            n_trials = len(epochs[condition])
            if n_trials > 0:
                evoked = epochs[condition].average()

                # Apply baseline correction (-200 to 0 ms)
                evoked.apply_baseline(baseline=(-0.2, 0))

                if 'FCz' in evoked.ch_names:
                    fcz_idx = evoked.ch_names.index('FCz')
                    fcz_data = evoked.data[fcz_idx, :]
                    
                    # Check FCz in the P300/RewP window (200-400ms)
                    time_mask = (evoked.times >= 0.2) & (evoked.times <= 0.4)
                    if np.any(time_mask):
                        windowed_data = fcz_data[time_mask]
                        peak_val = windowed_data[np.argmax(np.abs(windowed_data))]
                        
                        # Diagnostic info
                        print(f"  {condition}: {n_trials} trials, FCz peak in 200-400ms: {peak_val*1e6:.2f}µV")
                
                # Ensure data is in Volts
                data_range = np.max(np.abs(evoked.data))
                if data_range > 1e-3:  # Data is in µV, convert to V
                    evoked.data = evoked.data / 1e6
                    print(f"  {condition}: {n_trials} trials (converted µV→V)")
                else:
                    print(f"  {condition}: {n_trials} trials (already in V)")
                
                evokeds[condition] = evoked
            else:
                print(f"  {condition}: 0 trials - skipping")
        else:
            print(f"  {condition}: not found in epochs")
    
    return evokeds

def create_contrast_evoked(evokeds):
    """Create contrast evoked responses (win - loss) - RewP difference waves"""
    
    contrasts = {}
    
    # Define contrast pairs
    contrast_pairs = [
        ('low_task_win', 'low_task_loss', 'low_win_rewp'),
        ('mid_low_task_win', 'mid_low_task_loss', 'mid_low_rewp'),
        ('mid_high_task_win', 'mid_high_task_loss', 'mid_high_rewp'),
        ('high_task_win', 'high_task_loss', 'high_rewp')
    ]
    
    print("\nCreating contrast evoked responses:")
    for win_cond, loss_cond, contrast_name in contrast_pairs:
        if win_cond in evokeds and loss_cond in evokeds:
            contrast_evoked = mne.combine_evoked([evokeds[win_cond], evokeds[loss_cond]], 
                                               weights=[1, -1])
            contrast_evoked.comment = contrast_name
            contrasts[contrast_name] = contrast_evoked
            print(f"  {contrast_name}: {win_cond} - {loss_cond}")
        else:
            print(f"  {contrast_name}: missing conditions - skipping")
    
    return contrasts

def create_study_replication_plots(evokeds, contrasts, subject_id, subject_dir):
    """Create plots matching the study figures"""
    
    # Plot similar to Figure 3a - Feedback-locked waveforms at FCz
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Win conditions
    win_conditions = [
        ('low_task_win', 'Low-Task Win'),
        ('mid_low_task_win', 'Mid-Task Low-Cue Win'),
        ('mid_high_task_win', 'Mid-Task High-Cue Win'),
        ('high_task_win', 'High-Task Win')
    ]
    
    # Loss conditions  
    loss_conditions = [
        ('low_task_loss', 'Low-Task Loss'),
        ('mid_low_task_loss', 'Mid-Task Low-Cue Loss'),
        ('mid_high_task_loss', 'Mid-Task High-Cue Loss'),
        ('high_task_loss', 'High-Task Loss')
    ]
    
    # Plot win conditions
    for cond, label in win_conditions:
        if cond in evokeds and 'FCz' in evokeds[cond].ch_names:
            evokeds[cond].plot(picks=['FCz'], axes=axes[0,0], show=False, 
                             titles='Win Conditions at FCz')
    
    # Plot loss conditions
    for cond, label in loss_conditions:
        if cond in evokeds and 'FCz' in evokeds[cond].ch_names:
            evokeds[cond].plot(picks=['FCz'], axes=axes[0,1], show=False,
                             titles='Loss Conditions at FCz')
    
    # Plot RewP difference waves (similar to Figure 3c)
    rewp_conditions = [
        ('mid_high_rewp', 'Mid-Task High-Cue'),
        ('high_rewp', 'High-Task')
    ]
    
    for cond, label in rewp_conditions:
        if cond in contrasts and 'FCz' in contrasts[cond].ch_names:
            contrasts[cond].plot(picks=['FCz'], axes=axes[1,0], show=False,
                               titles='RewP Difference Waves at FCz')
    
    plt.tight_layout()
    fig.suptitle(f'Sub-{subject_id} - Study Replication Analysis', y=1.02)
    plt.savefig(os.path.join(subject_dir, 'study_replication_plots.png'), dpi=300)
    plt.close()

def visualize_evoked(evokeds, contrasts, subject_id, output_dir):
    """Visualize evoked responses"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step07_evoked')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Plot all evoked responses
    n_conditions = len(evokeds)
    if n_conditions > 0:
        n_cols = 2
        n_rows = int(np.ceil(n_conditions / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (condition, evoked) in enumerate(evokeds.items()):
            row, col = divmod(i, n_cols)
            evoked.plot(axes=axes[row, col], show=False, titles=condition)
        
        # Hide empty subplots
        for i in range(n_conditions, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        fig.suptitle(f'Sub-{subject_id} Evoked Responses', y=1.02)
        plt.savefig(os.path.join(subject_dir, 'all_evoked_responses.png'), dpi=300)
        plt.close()
    
    # Plot contrast evoked responses
    if contrasts:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (contrast_name, contrast_evoked) in enumerate(contrasts.items()):
            if i < 4:
                contrast_evoked.plot(axes=axes[i], show=False, titles=contrast_name)
        
        plt.tight_layout()
        fig.suptitle(f'Sub-{subject_id} Contrast Evoked (Win - Loss)', y=1.02)
        plt.savefig(os.path.join(subject_dir, 'contrast_evoked_responses.png'), dpi=300)
        plt.close()
    
    # Create topoplots at different time points
    if evokeds:
        # Pick a representative evoked response
        representative_evoked = list(evokeds.values())[0]
        
        # Time points of interest (in seconds)
        time_points = [0.1, 0.2, 0.3, 0.4]
        
        fig, axes = plt.subplots(1, len(time_points), figsize=(15, 4))
        
        for i, time_point in enumerate(time_points):
            representative_evoked.plot_topomap(times=time_point, axes=axes[i], 
                                             show=False, colorbar=False)
            axes[i].set_title(f'{time_point*1000:.0f} ms')
        
        plt.tight_layout()
        fig.suptitle(f'Sub-{subject_id} Topography Over Time', y=1.1)
        plt.savefig(os.path.join(subject_dir, 'topography_time_course.png'), dpi=300)
        plt.close()
    
    # Plot global field power
    if evokeds:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for condition, evoked in evokeds.items():
            # Calculate global field power
            gfp = np.std(evoked.data, axis=0)
            ax.plot(evoked.times, gfp, label=condition, linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Global Field Power (µV)')
        ax.set_title(f'Sub-{subject_id} Global Field Power')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(subject_dir, 'global_field_power.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    create_study_replication_plots(evokeds, contrasts, subject_id, subject_dir)
    
    print(f"Evoked visualizations saved to: {subject_dir}")

def save_evoked_responses(evokeds, contrasts, subject_id, output_dir):
    """Save evoked responses"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step07_evoked')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Save individual evoked responses
    for condition, evoked in evokeds.items():
        evoked_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_{condition}_ave.fif')
        evoked.save(evoked_fname, overwrite=True)
        print(f"Saved {condition} evoked to: {evoked_fname}")
    
    # Save contrast evoked responses
    for contrast_name, contrast_evoked in contrasts.items():
        contrast_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_{contrast_name}_ave.fif')
        contrast_evoked.save(contrast_fname, overwrite=True)
        print(f"Saved {contrast_name} contrast to: {contrast_fname}")
    
    # Save all evoked in one file
    all_evoked = list(evokeds.values()) + list(contrasts.values())
    if all_evoked:
        all_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_all_ave.fif')
        mne.write_evokeds(all_fname, all_evoked, overwrite=True)
        print(f"Saved all evoked responses to: {all_fname}")
    
    return subject_dir

def main():
    parser = argparse.ArgumentParser(description='Step 7: Create evoked responses')
    parser.add_argument('--subject', required=True, help='Subject ID')
    args = parser.parse_args()
    
    subject_id = args.subject
    
    print(f"Step 7: Creating evoked responses for subject {subject_id}")
    
    # Load data from previous step
    epochs = load_previous_step(subject_id)
    
    # Create evoked responses
    evokeds = create_evoked_responses(epochs)
    
    # Create contrast evoked
    contrasts = create_contrast_evoked(evokeds)
    
    # Create visualizations
    visualize_evoked(evokeds, contrasts, subject_id, OUTPUT_DIR)
    
    # Save evoked responses
    save_evoked_responses(evokeds, contrasts, subject_id, OUTPUT_DIR)
    
    print(f"Step 7 completed for subject {subject_id}")

if __name__ == "__main__":
    main()
