# EEG Semester Project — Reward ERP Dataset (ds004147)

This project analyzes the Reward ERP EEG dataset (ds004147) using MNE-Python and MNE-BIDS-Pipeline.
The aim is to reproduce the original findings with a clean and reproducible pipeline.

## Setup

### Requirements
- Python 3.10 or 3.11
- At least 10 GB of free disk space

### Install dependencies
Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install mne==1.10.2 mne-bids==0.15 mne-bids-pipeline==1.9 autoreject pingouin numpy scipy pandas matplotlib scikit-learn
```

## Project structure

```
project/
├── data/
│   └── ds004147/         # EEG dataset (BIDS format)
│       ├── sub-27/
│       ├── sub-28/
│       ├── ...
│       ├── dataset_description.json
│       ├── participants.tsv
│       └── README
├── notebooks/
│   └── test.py            # test or analysis scripts
├── derivatives/
├── requirements.txt
└── README.md

```
## Dataset Download (ds004147)
The dataset is publicly available on OpenNeuro:
🔗 https://openneuro.org/datasets/ds004147

Extract the ZIP file and move the folder into your project:
    ```
    project/data/ds004147/
    ```

### QC reports
python notebooks/step07_qc_report.py
# -> derivatives/qc/sub-XX_report.html + index.html

### Robustness sweep (small subset)
python notebooks/step08_robustness_sweep.py
# -> derivatives/group/robustness.csv

