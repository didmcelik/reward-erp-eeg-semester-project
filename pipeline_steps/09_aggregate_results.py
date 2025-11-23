"""
Step 9: Aggregate results across subjects (when multiple subjects are available)
"""

import os
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
TASK = "casinos"

# All possible subjects (extend as needed)
ALL_SUBJECTS = ['27']  # Add more as you process them

def find_processed_subjects():
    """Find all subjects that have completed processing"""
    
    processed_subjects = []
    
    for subject in ALL_SUBJECTS:
        # Check if subject has completed step 8
        stats_dir = os.path.join(OUTPUT_DIR, f'sub-{subject}', 'step08_statistics')
        if os.path.exists(stats_dir):
            processed_subjects.append(subject)
    
    print(f"Found {len(processed_subjects)} processed subjects: {processed_subjects}")
    return processed_subjects

def aggregate_evoked_responses(subjects):
    """Aggregate evoked responses across subjects"""
    
    # Dictionary to store evoked responses by condition
    all_evokeds = {}
    
    print("Loading evoked responses from all subjects...")
    
    for subject in subjects:
        evoked_dir = os.path.join(OUTPUT_DIR, f'sub-{subject}', 'step07_evoked')
        
        # Load all evoked file
        all_evoked_file = os.path.join(evoked_dir, f'sub-{subject}_task-{TASK}_all_ave.fif')
        
        if os.path.exists(all_evoked_file):
            evokeds_list = mne.read_evokeds(all_evoked_file)
            
            for evoked in evokeds_list:
                condition = evoked.comment
                if condition not in all_evokeds:
                    all_evokeds[condition] = []
                all_evokeds[condition].append(evoked)
                
            print(f"  Loaded {len(evokeds_list)} evoked responses from subject {subject}")
    
    # Calculate grand averages
    grand_averages = {}
    for condition, evokeds_list in all_evokeds.items():
        if evokeds_list:
            grand_avg = mne.grand_average(evokeds_list)
            grand_avg.comment = f'Grand_avg_{condition}'
            grand_averages[condition] = grand_avg
            print(f"  Grand average for {condition}: {len(evokeds_list)} subjects")
    
    return grand_averages, all_evokeds

def aggregate_statistics(subjects):
    """Aggregate statistical results across subjects"""
    
    # Collect peak statistics
    all_stats = []
    
    print("Aggregating statistical results...")
    
    for subject in subjects:
        stats_file = os.path.join(OUTPUT_DIR, f'sub-{subject}', 'step08_statistics', 
                                 f'sub-{subject}_simple_stats.txt')
        
        if os.path.exists(stats_file):
            # Parse simple statistics file
            with open(stats_file, 'r') as f:
                lines = f.readlines()
            
            current_condition = None
            subject_stats = {'subject': subject}
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('=') and not line.startswith('Simple'):
                    if line.endswith(':') and not line.startswith(' '):
                        current_condition = line[:-1]
                    elif current_condition and line.startswith('  Peak amplitude:'):
                        amplitude = float(line.split(':')[1].split()[0])
                        subject_stats[f'{current_condition}_peak_amp'] = amplitude
                    elif current_condition and line.startswith('  Peak time:'):
                        time = float(line.split(':')[1].split()[0])
                        subject_stats[f'{current_condition}_peak_time'] = time
            
            all_stats.append(subject_stats)
    
    # Convert to DataFrame
    if all_stats:
        df_stats = pd.DataFrame(all_stats)
        print(f"  Collected statistics from {len(all_stats)} subjects")
        return df_stats
    else:
        print("  No statistics found")
        return None
    

def create_study_replication_plots(grand_averages, df_stats, output_dir):
    """Create plots matching Figure 3 from the study"""
    
    group_dir = os.path.join(output_dir, 'group_analysis')
    os.makedirs(group_dir, exist_ok=True)
    
    print("Creating study replication plots (Figure 3 style)...")
    
    # Figure 3a: Win/loss waveforms by task and cue value
    create_figure_3a(grand_averages, group_dir)
    
    # Figure 3b: Scalp distribution of win-loss difference
    create_figure_3b(grand_averages, group_dir)
    
    # Figure 3c: Difference waveforms (RewP)
    create_figure_3c(grand_averages, group_dir)
    
    # Figure 3d: RewP scores by condition
    create_figure_3d(df_stats, group_dir)

def create_figure_3a(grand_averages, output_dir):
    """Create Figure 3a: Win/loss waveforms by task and cue value"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors and styles matching the study
    colors = {
        'low_low': '#FF6B6B',      # Light red for low-low
        'mid_low': '#66B2FF',      # Light blue for mid-low  
        'mid_high': '#FF6666',     # Red for mid-high
        'high_high': '#3366FF'     # Blue for high-high
    }

    line_styles = {'win': '-', 'loss': '--'}
    
    # Plot conditions matching study design
    conditions_to_plot = [
        ('low_task_win', 'low_task_loss', 'low_low', 'Low-Low'),
        ('mid_low_task_win', 'mid_low_task_loss', 'mid_low', 'Mid-Low'),
        ('mid_high_task_win', 'mid_high_task_loss', 'mid_high', 'Mid-High'),
        ('high_task_win', 'high_task_loss', 'high_high', 'High-High')
    ]
    
    for win_cond, loss_cond, color_key, label in conditions_to_plot:
        if win_cond in grand_averages and loss_cond in grand_averages:
            win_evoked = grand_averages[win_cond]
            loss_evoked = grand_averages[loss_cond]
            
            # Get FCz data
            if 'FCz' in win_evoked.ch_names:
                ch_idx = win_evoked.ch_names.index('FCz')
                
                # Conversion to µV
                win_data = win_evoked.data[ch_idx] * 1e6  # Convert to µV
                loss_data = loss_evoked.data[ch_idx] * 1e6  # Convert to µV
                times = win_evoked.times
                
                # Plot win and loss
                ax.plot(times, win_data, color=colors[color_key], 
                       linestyle=line_styles['win'], linewidth=2, 
                       label=f'{label} Win')
                ax.plot(times, loss_data, color=colors[color_key], 
                       linestyle=line_styles['loss'], linewidth=2,
                       label=f'{label} Loss')
    
    # Formatting to match study exactly
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Voltage (µV)', fontsize=12)
    ax.set_title('FCz', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, 0.6)
    
    # Add legend in style of study
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_3a_win_loss_waveforms.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_figure_3b(grand_averages, output_dir):
    """Create Figure 3b: Scalp distribution of win-loss difference"""
    
    # Calculate grand average difference wave across all conditions
    all_diff_evokeds = []
    
    contrast_pairs = [
        ('low_task_win', 'low_task_loss'),
        ('mid_low_task_win', 'mid_low_task_loss'),
        ('mid_high_task_win', 'mid_high_task_loss'),
        ('high_task_win', 'high_task_loss')
    ]
    
    for win_cond, loss_cond in contrast_pairs:
        if win_cond in grand_averages and loss_cond in grand_averages:
            diff_evoked = mne.combine_evoked([grand_averages[win_cond], 
                                            grand_averages[loss_cond]], 
                                           weights=[1, -1])
            all_diff_evokeds.append(diff_evoked)
    
    if all_diff_evokeds:
        # Average across all difference waves
        grand_diff = mne.grand_average(all_diff_evokeds)
        
        # Create topography plot for RewP window (240-340ms)
        fig = grand_diff.plot_topomap(times=[0.29], # Peak of RewP window
                                     show=False, 
                                     cmap='RdBu_r',
                                     size=4)
        fig.suptitle('Win-Loss Difference', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(output_dir, 'figure_3b_scalp_topography.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_figure_3c(grand_averages, output_dir):
    """Create Figure 3c: Difference waveforms (RewP)"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors for difference waves
    colors = {
        'low_task': '#E74C3C',      # Red
        'mid_low_task': '#3498DB',  # Blue  
        'mid_high_task': '#9B59B6', # Purple
        'high_task': '#2ECC71'      # Green
    }
    
    # Create and plot difference waves
    contrast_pairs = [
        ('low_task_win', 'low_task_loss', 'low_task', 'Low Task, Low Cue'),
        ('mid_low_task_win', 'mid_low_task_loss', 'mid_low_task', 'Mid Task, Low Cue'),
        ('mid_high_task_win', 'mid_high_task_loss', 'mid_high_task', 'Mid Task, High Cue'),
        ('high_task_win', 'high_task_loss', 'high_task', 'High Task, High Cue')
    ]
    
    for win_cond, loss_cond, color_key, label in contrast_pairs:
        if win_cond in grand_averages and loss_cond in grand_averages:
            # Create difference wave
            diff_evoked = mne.combine_evoked([grand_averages[win_cond], 
                                            grand_averages[loss_cond]], 
                                           weights=[1, -1])
            
            # Get FCz data
            if 'FCz' in diff_evoked.ch_names:
                ch_idx = diff_evoked.ch_names.index('FCz')
                diff_data = diff_evoked.data[ch_idx] * 1e6  # Convert to µV
                times = diff_evoked.times
                
                # Plot difference wave
                ax.plot(times, diff_data, color=colors[color_key], 
                       linewidth=3, label=label)
    
    # Add shaded region for analysis window (240-340ms)
    ax.axvspan(0.240, 0.340, alpha=0.2, color='gray', 
               label='Analysis Window (240-340ms)')
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Voltage (µV)', fontsize=12)
    ax.set_title('FCz', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, 0.6)
    ax.set_ylim(-10, 10)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_3c_difference_waveforms.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_figure_3d(df_stats, output_dir):
    """Create Figure 3d: RewP scores by condition"""
    
    if df_stats is None or len(df_stats) == 0:
        print("No statistics available for Figure 3d")
        return
    
    # Load RewP results for proper values
    rewp_data = load_rewp_results(df_stats)
    
    if not rewp_data:
        print("No RewP data available")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Prepare data for plotting
    conditions = ['Low-Low', 'Mid-Low', 'Mid-High', 'High-High']
    rewp_values = []
    
    # Extract RewP values (you'll need to modify this based on your data structure)
    condition_mapping = {
        'Low-Low': 'low_task',
        'Mid-Low': 'mid_low_task', 
        'Mid-High': 'mid_high_task',
        'High-High': 'high_task'
    }
    
    for condition in conditions:
        key = condition_mapping.get(condition)
        if key in rewp_data:
            rewp_values.append(abs(rewp_data[key]))  # Use absolute value for display
        else:
            rewp_values.append(0)
    
    # Create box plot
    x_pos = np.arange(len(conditions))
    
    # Plot bars (since you have single subject, show as bars with individual points)
    bars = ax.bar(x_pos, rewp_values, alpha=0.7, color=['#E74C3C', '#3498DB', '#9B59B6', '#2ECC71'])
    
    # Add individual data points (dots for each subject)
    for i, value in enumerate(rewp_values):
        ax.scatter(i, value, color='black', s=50, zorder=10)
    
    # Formatting
    ax.set_xlabel('Task-Cue Combination', fontsize=12)
    ax.set_ylabel('Voltage (µV)', fontsize=12)
    ax.set_title('RewP Amplitude by Condition', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(rewp_values) * 1.2 if rewp_values else 10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_3d_rewp_scores.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def load_rewp_results(df_stats):
    """Load RewP results from statistics files"""
    
    rewp_data = {}
    
    # Try to load from the first subject's RewP results
    if len(df_stats) > 0:
        subject = df_stats.iloc[0]['subject']
        rewp_file = os.path.join(OUTPUT_DIR, f'sub-{subject}', 'step08_statistics', 
                                f'sub-{subject}_rewp_results.txt')
        
        if os.path.exists(rewp_file):
            with open(rewp_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if ':' in line and 'RewP' in line:
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        condition = parts[0].strip()
                        try:
                            value = float(parts[1].split('µV')[0].strip())
                            rewp_data[condition] = value
                        except:
                            pass
    
    return rewp_data


def create_group_visualizations(grand_averages, df_stats, output_dir):
    """Create group-level visualizations"""
    
    group_dir = os.path.join(output_dir, 'group_analysis')
    os.makedirs(group_dir, exist_ok=True)
    
    print("Creating group visualizations...")
    
    # Plot grand average evoked responses
    if grand_averages:
        # Plot all grand averages
        n_conditions = len(grand_averages)
        if n_conditions > 0:
            n_cols = 2
            n_rows = int(np.ceil(n_conditions / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (condition, grand_avg) in enumerate(grand_averages.items()):
                row, col = divmod(i, n_cols)
                grand_avg.plot(axes=axes[row, col], show=False, titles=f'Grand Avg: {condition}')
            
            # Hide empty subplots
            for i in range(n_conditions, n_rows * n_cols):
                row, col = divmod(i, n_cols)
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            fig.suptitle('Group Grand Average Evoked Responses', y=1.02)
            plt.savefig(os.path.join(group_dir, 'grand_average_evoked.png'), dpi=300)
            plt.close()
        
        # Plot comparison of win vs loss conditions
        win_conditions = [cond for cond in grand_averages.keys() if 'win' in cond and 'minus' not in cond]
        loss_conditions = [cond for cond in grand_averages.keys() if 'loss' in cond]
        
        if win_conditions and loss_conditions:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            task_levels = ['low_task', 'mid_low_task', 'mid_high_task', 'high_task']
            
            for i, level in enumerate(task_levels):
                win_cond = f'{level}_win'
                loss_cond = f'{level}_loss'
                
                if win_cond in grand_averages and loss_cond in grand_averages and i < 4:
                    # FIXED: Plot manually to control colors and styles
                    win_evoked = grand_averages[win_cond]
                    loss_evoked = grand_averages[loss_cond]
                    
                    # Plot win condition (blue, solid line)
                    for ch_idx, ch_name in enumerate(win_evoked.ch_names):
                        if ch_name == 'FCz':  # Focus on FCz for clarity
                            axes[i].plot(win_evoked.times, win_evoked.data[ch_idx], 
                                       color='blue', linewidth=2, label='Win')
                            axes[i].plot(loss_evoked.times, loss_evoked.data[ch_idx], 
                                       color='red', linewidth=2, linestyle='--', label='Loss')
                            break
                    
                    axes[i].set_xlabel('Time (s)')
                    axes[i].set_ylabel('Amplitude (µV)')
                    axes[i].set_title(f'{level.replace("_", " ").title()}: Win vs Loss (FCz)')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                    axes[i].axhline(0, color='black', linewidth=0.5)
                    axes[i].axvline(0, color='black', linewidth=0.5)
            
            plt.tight_layout()
            fig.suptitle('Group Comparison: Win vs Loss by Task Level (FCz electrode)', y=1.02)
            plt.savefig(os.path.join(group_dir, 'win_vs_loss_comparison.png'), dpi=300)
            plt.close()
    
    # Plot statistical summaries
    if df_stats is not None and len(df_stats) > 1:
        # Plot peak amplitudes across subjects
        win_loss_pairs = [
            ('low_task_win', 'low_task_loss'),
            ('mid_low_task_win', 'mid_low_task_loss'),
            ('mid_high_task_win', 'mid_high_task_loss'),
            ('high_task_win', 'high_task_loss')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (win_cond, loss_cond) in enumerate(win_loss_pairs):
            win_col = f'{win_cond}_peak_amp'
            loss_col = f'{loss_cond}_peak_amp'
            
            if win_col in df_stats.columns and loss_col in df_stats.columns:
                x = np.arange(len(df_stats))
                width = 0.35
                
                axes[i].bar(x - width/2, df_stats[win_col], width, label='Win', alpha=0.7)
                axes[i].bar(x + width/2, df_stats[loss_col], width, label='Loss', alpha=0.7)
                
                axes[i].set_xlabel('Subject')
                axes[i].set_ylabel('Peak Amplitude (µV)')
                axes[i].set_title(f'{win_cond.replace("_", " ").title()}')
                axes[i].set_xticks(x)
                axes[i].set_xticklabels([f'Sub-{s}' for s in df_stats['subject']])
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.suptitle('Peak Amplitudes Across Subjects', y=1.02)
        plt.savefig(os.path.join(group_dir, 'peak_amplitudes_across_subjects.png'), dpi=300)
        plt.close()
        
        # Summary statistics table
        summary_stats = df_stats.describe()
        
        # Save summary statistics
        summary_file = os.path.join(group_dir, 'group_summary_statistics.csv')
        summary_stats.to_csv(summary_file)
        print(f"  Summary statistics saved to: {summary_file}")

    create_study_replication_plots(grand_averages, df_stats, output_dir)
    
    print(f"Group visualizations saved to: {group_dir}")

def save_group_results(grand_averages, df_stats, output_dir):
    """Save group-level results"""
    
    group_dir = os.path.join(output_dir, 'group_analysis')
    os.makedirs(group_dir, exist_ok=True)
    
    # Save grand averages
    if grand_averages:
        grand_avg_list = list(grand_averages.values())
        grand_avg_file = os.path.join(group_dir, f'group_task-{TASK}_grand_avg.fif')
        mne.write_evokeds(grand_avg_file, grand_avg_list, overwrite=True)
        print(f"Grand averages saved to: {grand_avg_file}")
    
    # Save aggregated statistics
    if df_stats is not None:
        stats_file = os.path.join(group_dir, 'group_statistics.csv')
        df_stats.to_csv(stats_file, index=False)
        print(f"Group statistics saved to: {stats_file}")
    
    # Create processing report
    report_file = os.path.join(group_dir, 'processing_report.txt')
    with open(report_file, 'w') as f:
        f.write(f"EEG Processing Pipeline Report\n")
        f.write(f"Task: {TASK}\n")
        f.write(f"Processing date: {pd.Timestamp.now()}\n")
        f.write(f"="*50 + "\n\n")
        
        if df_stats is not None:
            f.write(f"Number of subjects processed: {len(df_stats)}\n")
            f.write(f"Subjects: {list(df_stats['subject'])}\n\n")
        
        if grand_averages:
            f.write(f"Conditions with grand averages: {len(grand_averages)}\n")
            f.write(f"Conditions: {list(grand_averages.keys())}\n\n")
        
        f.write("Processing steps completed:\n")
        f.write("  1. Raw data loading\n")
        f.write("  2. Montage setting\n")
        f.write("  3. Filtering\n")
        f.write("  4. ICA artifact removal\n")
        f.write("  5. Epoching\n")
        f.write("  6. Bad trial removal\n")
        f.write("  7. Evoked response creation\n")
        f.write("  8. Statistical testing\n")
        f.write("  9. Group aggregation\n")
    
    print(f"Processing report saved to: {report_file}")
    
    return group_dir

def main():
    parser = argparse.ArgumentParser(description='Step 9: Aggregate results across subjects')
    parser.add_argument('--subject', required=True, help='Subject ID (used for compatibility)')
    args = parser.parse_args()
    
    print(f"Step 9: Aggregating results across all subjects")
    
    # Find all processed subjects
    subjects = find_processed_subjects()
    
    if len(subjects) == 0:
        print("No processed subjects found!")
        return
    
    # Aggregate evoked responses
    grand_averages, all_evokeds = aggregate_evoked_responses(subjects)
    
    # Aggregate statistics
    df_stats = aggregate_statistics(subjects)
    
    # Create group visualizations
    create_group_visualizations(grand_averages, df_stats, OUTPUT_DIR)
    
    # Save group results
    save_group_results(grand_averages, df_stats, OUTPUT_DIR)
    
    print(f"Step 9 completed - group analysis finished!")
    print(f"Results available in: {os.path.join(OUTPUT_DIR, 'group_analysis')}")

if __name__ == "__main__":
    main()
