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
