# 🧩 EEG Semester Project — ds004147 (Reward ERP / RewP)

## Overview

This pipeline processes raw EEG data through 9 sequential steps to extract and analyze reward-related event-related potentials (ERPs). It implements automated artifact removal, statistical testing, and generates publication-quality visualizations.

**Key Features:**
- Automated ICA-based artifact removal with ICLabel classifier
- Data-driven bad channel detection and interpolation
- Cluster-based permutation testing for statistical inference
- Comprehensive quality control at each step
- BIDS-compatible data organization

## Pipeline Steps

1. **Data Fetching** - Load raw BrainVision EEG files
2. **Montage Setup** - Apply the site2channellocations.locs montage for correct spatial layout.
3. **Filtering & Preprocessing** - Zapline (50 Hz), bandpass (0.1-40 Hz), bad channel detection/interpolation, re-referencing to mastoids
4. **ICA** - Remove eye, muscle, and cardiac artifacts using ICLabel
5. **Epoch Creation** - Extract feedback-locked epochs (-200 to 600 ms)
6. **Bad Trial Removal** - Automated rejection with Autoreject
7. **Evoked Responses** - Average trials and compute win-loss difference waves
8. **Statistical Analysis** - RewP quantification and cluster-based permutation tests
9. **Results Aggregation** - Generate group-level figures and statistics

## Installation

### Prerequisites
- Python 3.1+

### Install Dependencies

```bash
pip install mne mne-icalabel meegkit autoreject numpy scipy matplotlib pandas
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```


# To run the complete pipeline:
```bash
cd pipeline_steps
python master_script.py
```

# Or run individual steps:
```bash
python 01_fetch_data.py --subject 27
python 02_set_montage.py --subject 27

etc.
```

### Output Structure 
 ```
output/derivatives/manual-pipeline/
├── sub-27/
│   ├── step01_raw/
│   ├── step02_montage/
│   ├── step03_filters/
│   ├── step04_ica/
│   ├── step05_epochs/
│   ├── step06_clean_epochs/
│   ├── step07_evoked/
│   └── step08_statistics/
└── group_analysis/
    ├── figure_3a_win_loss_waveforms.png
    ├── figure_3b_scalp_topography.png
    ├── figure_3c_difference_waveforms.png
    └── figure_3d_rewp_scores.png
 ```
## Analysis details

### Experimental Design
- 4 expectancy levels: Low-Low, Mid-Low, Mid-High, High-High
- 2 outcomes: Win, Loss
- 8 total conditions: 4 expectancy × 2 outcomes

### RewP Quantification
- Electrode: FCz
- Time window: 240-340 ms post-feedback
- Metric: Mean amplitude of win-loss difference

### Statistical Testing
- Cluster-based permutation tests (1000 permutations)
- Tests win vs. loss within each expectancy level
- Cluster-corrected p-values across space and time

## Key Results
The pipeline extracts the RewP component with:

- Peak latency: ~250-350 ms post-feedback
- Topography: Frontocentral (FCz, Fz, Cz)
- Effect: More positive for wins than losses

### Processing Time
~5 minutes per subject (full pipeline on standard hardware)

### Quality Control
Each step generates diagnostic plots:

- Power spectral density comparisons
- ICA component classifications
- Before/after artifact removal overlays
- Epoch rejection summaries
- Condition-averaged ERPs

### Troubleshooting
- Missing channels: Pipeline automatically handles missing mastoid references by falling back to average reference.

- Site 2 subjects: Fz reference channel automatically added for subjects 27-38.

- Memory issues: Reduce number of ICA components or process subjects sequentially.

## Acknowledgments

This pipeline was developed to replicate and extend the analysis methods described in:

**Hassall, C. D., Hunt, L. T., & Holroyd, C. B. (2022).** Task-level value affects trial-level reward processing. *NeuroImage*, 260, 119456. https://doi.org/10.1016/j.neuroimage.2022.119456

We acknowledge the authors for making their data publicly available and for their detailed methodological descriptions, which informed the design of this preprocessing pipeline. Their work on how average task value modulates the reward positivity (RewP) in anterior cingulate cortex provided the theoretical foundation for our analysis approach.

We also acknowledge the developers of the open-source software packages that made this pipeline possible: MNE-Python, ICLabel, Autoreject, and the broader Python scientific computing ecosystem.

Pipeline developed as part of the Signal processing and Analysis of human brain potentials (EEG) Semester Project at the University of Stuttgart, 2026.