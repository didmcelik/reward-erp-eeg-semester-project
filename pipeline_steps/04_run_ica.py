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


def run_ica_classification(ica, raw):
    """Classify ICA components using ICLabel"""
    
    try:
        # Filter for ICLabel (1-100 Hz)
        raw_for_iclabel = raw.copy().filter(1.0, 100.0)
        
        print("Running ICLabel classification...")

        # Use ICLabel to classify components
        ic_labels = label_components(raw_for_iclabel, ica, method='iclabel')
        
        # Get component labels and probabilities
        labels = ic_labels['labels']
        y_pred_proba = ic_labels['y_pred_proba']

        print(f"ICLabel found these component types: {set(labels)}")
        
        # Get brain probability for each component (more reliable approach)
        exclude_components = []
        for i in range(len(labels)):
            # Get probabilities for this component
            probs = y_pred_proba[i]
            brain_prob = probs[0]  # Brain is typically index 0
            predicted_label = labels[i]
            max_prob = np.max(probs)
            
            print(f"Component {i}: {predicted_label} (brain: {brain_prob:.3f}, max: {max_prob:.3f})")
            
            # Exclude components with LOW brain probability
            if brain_prob < 0.3:  # Threshold for exclusion
                exclude_components.append(i)
                print(f"  -> Excluding component {i} (low brain prob: {brain_prob:.3f})")
        
        print(f"ICLabel identified {len(exclude_components)} artifact components: {exclude_components}")
        
        return exclude_components, ic_labels
        
    except ImportError:
        print("mne-icalabel not available, using manual EOG detection")
        return [], None
    except Exception as e:
        print(f"ICLabel classification failed: {e}")
        return [], None
    

def find_eog_components(ica, raw):
    """Find EOG-related components using correlation"""
    
    eog_indices = []
    
    # Find EOG-related components
    eog_channels = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eog']
    if eog_channels:
        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels[0], threshold=3.0)
            print(f"Found {len(eog_indices)} EOG-related components: {eog_indices}")
        except Exception as e:
            print(f"Could not auto-detect EOG components: {e}")
    else:
        # Try using frontal channels as EOG proxies
        frontal_channels = [ch for ch in ['Fp1', 'Fp2', 'AF3', 'AF4'] if ch in raw.ch_names]
        if frontal_channels:
            try:
                eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=frontal_channels[0], threshold=3.0)
                print(f"Found {len(eog_indices)} EOG-related components using {frontal_channels[0]}: {eog_indices}")
            except Exception as e:
                print(f"Could not detect EOG components using frontal channels: {e}")
    
    return eog_indices


def run_ica(raw):
    """Run ICA decomposition with IC Labeling """
    
    # Create copy with higher high-pass for ICA
    raw_ica = raw.copy().filter(1.0, None)
    
    # Setup ICA
    n_components = min(25, len([ch for ch in raw.ch_names if 'eeg' in raw.get_channel_types([ch])[0]]))
    ica = ICA(n_components=n_components, method='picard', max_iter=300, random_state=42)
    
    print(f"Fitting ICA with {n_components} components...")
    ica.fit(raw_ica)
    
    # STEP 1: Run IC labeling AFTER fitting ICA
    auto_exclude, ic_labels = run_ica_classification(ica, raw_ica)
    
    # STEP 2: Find EOG-based exclusion
    eog_exclude = find_eog_components(ica, raw_ica)
    
    # STEP 3: Combine all exclusions
    all_exclude = list(set(auto_exclude + eog_exclude))
    ica.exclude = all_exclude
    
    print(f"Total excluded components: {len(all_exclude)} = ICLabel: {len(auto_exclude)} + EOG: {len(eog_exclude)}")
    print(f"Final excluded components: {ica.exclude}")
    
    return ica, raw_ica

def visualize_ica(ica, raw_ica, raw_original, subject_id, output_dir):
    """Visualize ICA results"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step04_ica')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Plot ICA components
    figs = ica.plot_components(inst=raw_ica, show=False)
    if isinstance(figs, list):
        for i, fig in enumerate(figs):
            fig.savefig(os.path.join(subject_dir, f'ica_components_page_{i+1}.png'), dpi=300)
            plt.close(fig)
    else:
        figs.savefig(os.path.join(subject_dir, 'ica_components.png'), dpi=300)
        plt.close(figs)
    
    # Plot excluded components
    if ica.exclude:
        for i, comp in enumerate(ica.exclude[:4]):  # Plot first 4 excluded
            try:
                fig = ica.plot_properties(raw_ica, picks=comp, show=False)
                fig.savefig(os.path.join(subject_dir, f'excluded_component_{comp}.png'), dpi=300)
                plt.close(fig)
            except:
                pass
    
    # Compare data before and after ICA
    raw_corrected = ica.apply(raw_original.copy())
    
    # Create separate plots for before/after comparison
    fig1 = raw_original.plot(duration=10, n_channels=10, scalings='auto', 
                           title=f'Sub-{subject_id} Before ICA', show=False)
    fig1.savefig(os.path.join(subject_dir, 'before_ica.png'), dpi=300)
    plt.close(fig1)
    
    fig2 = raw_corrected.plot(duration=10, n_channels=10, scalings='auto',
                            title=f'Sub-{subject_id} After ICA', show=False)
    fig2.savefig(os.path.join(subject_dir, 'after_ica.png'), dpi=300)
    plt.close(fig2)
    
    print(f"ICA visualizations saved to: {subject_dir}")

def apply_ica_and_save(ica, raw, subject_id, output_dir):
    """Apply ICA and save results"""
    
    subject_dir = os.path.join(output_dir, f'sub-{subject_id}', 'step04_ica')
    os.makedirs(subject_dir, exist_ok=True)
    
    # Apply ICA to remove artifacts
    raw_clean = ica.apply(raw.copy())
    
    # Save ICA object
    ica_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_ica.fif')
    ica.save(ica_fname, overwrite=True)
    
    # Save cleaned data
    raw_fname = os.path.join(subject_dir, f'sub-{subject_id}_task-{TASK}_ica_raw.fif')
    raw_clean.save(raw_fname, overwrite=True)
    
    print(f"ICA object saved to: {ica_fname}")
    print(f"ICA-cleaned data saved to: {raw_fname}")
    
    return raw_fname, ica_fname

def main():
    parser = argparse.ArgumentParser(description='Step 4: Run ICA')
    parser.add_argument('--subject', required=True, help='Subject ID')
    args = parser.parse_args()
    
    subject_id = args.subject
    
    print(f"Step 4: Running ICA for subject {subject_id}")
    
    # Load data from previous step
    raw = load_previous_step(subject_id)
    
    # Run ICA
    ica, raw_ica = run_ica(raw)
    
    # Create visualizations
    visualize_ica(ica, raw_ica, raw, subject_id, OUTPUT_DIR)
    
    # Apply ICA and save
    apply_ica_and_save(ica, raw, subject_id, OUTPUT_DIR)
    
    print(f"Step 4 completed for subject {subject_id}")

if __name__ == "__main__":
    main()
