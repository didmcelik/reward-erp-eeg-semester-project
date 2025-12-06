"""
Master script to run the entire EEG analysis pipeline
Each step processes one subject at a time and saves intermediate results
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
# SUBJECTS = ['27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38'] # List of subject IDs to process

# Exclude subject 28 (91.2% of epochs dropped, very noisy)
# SUBJECTS = ['27', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38'] 

SUBJECTS = ['27'] # For testing, process only one subject
PIPELINE_STEPS = [
    '01_fetch_data.py',
    '02_set_montage.py', 
    '03_apply_filters.py',
    '04_run_ica.py',
    '05_create_epochs.py',
    '06_remove_bad_trials.py',
    '07_create_evoked.py',
    '08_run_statistics.py',
    '09_aggregate_results.py'
]

def run_pipeline_step(script_name, subject_id):
    """Run a single pipeline step for one subject"""
    print(f"\n{'='*60}")
    print(f"Running {script_name} for subject {subject_id}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, script_name, '--subject', subject_id
        ], check=True, capture_output=True, text=True)
        
        print(f"✓ {script_name} completed successfully for subject {subject_id}")
        if result.stdout:
            print(f"Output: {result.stdout}")
            
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed for subject {subject_id}")
        print(f"Error: {e.stderr}")
        return False
    
    return True

def main():
    """Run the complete pipeline for all subjects"""
    print("Starting EEG Analysis Pipeline")
    print(f"Subjects to process: {SUBJECTS}")
    
    # Process each subject through all steps
    for subject in SUBJECTS:
        print(f"\n Processing Subject {subject}")
        
        for step in PIPELINE_STEPS:
            if not run_pipeline_step(step, subject):
                print(f"Pipeline failed at {step} for subject {subject}")
                return False
    
    print("\n🎉 Pipeline completed successfully for all subjects!")
    return True

if __name__ == "__main__":
    main()
