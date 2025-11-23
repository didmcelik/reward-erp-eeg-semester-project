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

# Outcome conditions
CONDITIONS = [
    "low_task_win", "low_task_loss",
    "mid_low_task_win", "mid_low_task_loss",
    "mid_high_task_win", "mid_high_task_loss",
    "high_task_win", "high_task_loss",
]


def load_previous_step(subject_id: str) -> mne.Epochs:
    """Load clean epochs from Step 6."""
    input_file = os.path.join(
        OUTPUT_DIR,
        f"sub-{subject_id}",
        "step06_rejection",
        f"sub-{subject_id}_task-{TASK}_clean_epochs.fif",
    )
    epochs = mne.read_epochs(input_file, preload=True)
    return epochs


def create_evoked_responses(epochs: mne.Epochs) -> dict:
    """Create evoked responses for each outcome condition."""
    evokeds = {}

    print("Creating evoked responses:")
    for condition in CONDITIONS:
        if condition in epochs.event_id:
            n_trials = len(epochs[condition])
            if n_trials > 0:
                evoked = epochs[condition].average()
                evoked.comment = condition
                evokeds[condition] = evoked
                print(f"  {condition}: {n_trials} trials")
            else:
                print(f"  {condition}: 0 trials - skipping")
        else:
            print(f"  {condition}: not found in epochs")

    return evokeds


def create_contrast_evoked(evokeds: dict) -> dict:
    """
    Create contrast evoked responses (win - loss) = RewP difference waves.
    Names are *_win_minus_loss for consistency with the paper and later analyses.
    """
    contrasts = {}

    contrast_pairs = [
        ("low_task_win", "low_task_loss", "low_win_minus_loss"),
        ("mid_low_task_win", "mid_low_task_loss", "mid_low_win_minus_loss"),
        ("mid_high_task_win", "mid_high_task_loss", "mid_high_win_minus_loss"),
        ("high_task_win", "high_task_loss", "high_win_minus_loss"),
    ]

    print("\nCreating contrast evoked responses (win - loss):")
    for win_cond, loss_cond, contrast_name in contrast_pairs:
        if win_cond in evokeds and loss_cond in evokeds:
            contrast_evoked = mne.combine_evoked(
                [evokeds[win_cond], evokeds[loss_cond]], weights=[1, -1]
            )
            contrast_evoked.comment = contrast_name
            contrasts[contrast_name] = contrast_evoked
            print(f"  {contrast_name}: {win_cond} - {loss_cond}")
        else:
            print(f"  {contrast_name}: missing conditions - skipping")

    return contrasts


def create_study_replication_plots(
    evokeds: dict, contrasts: dict, subject_id: str, subject_dir: str
) -> None:
    """
    Create plots approximating the figures in the original study.

    1) 2×2 FCz plots: Win vs Loss for each task level (low, mid-low, mid-high, high)
    2) RewP difference waves at FCz for mid-high and high conditions.
    """
    # ---------- Figure 1: Win vs Loss at FCz for each task level ----------
    levels = [
        ("low_task_win", "low_task_loss", "Low Task"),
        ("mid_low_task_win", "mid_low_task_loss", "Mid Low Task"),
        ("mid_high_task_win", "mid_high_task_loss", "Mid High Task"),
        ("high_task_win", "high_task_loss", "High Task"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (win_cond, loss_cond, title) in zip(axes, levels):
        if win_cond in evokeds and loss_cond in evokeds:
            ev_win = evokeds[win_cond]
            ev_loss = evokeds[loss_cond]

            if "FCz" in ev_win.ch_names:
                ch_idx = ev_win.ch_names.index("FCz")
                times = ev_win.times
                win_data = ev_win.data[ch_idx] * 1e6  # µV
                loss_data = ev_loss.data[ch_idx] * 1e6  # µV

                ax.plot(times, win_data, label="Win", linewidth=2)
                ax.plot(times, loss_data, label="Loss", linestyle="--", linewidth=2)
                ax.axvline(0, color="k", linewidth=0.8)
                ax.axhline(0, color="k", linewidth=0.8)
                ax.set_title(f"{title}: Win vs Loss (FCz)")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude (µV)")
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.set_title(f"{title}: FCz not found")
        else:
            ax.set_title(f"{title}: missing data")

    plt.tight_layout()
    fig.suptitle(f"Sub-{subject_id} – Win vs Loss by Task Level (FCz)", y=1.02)
    fig.savefig(
        os.path.join(subject_dir, "fcz_win_vs_loss_by_task.png"), dpi=300
    )
    plt.close(fig)

    # ---------- Figure 2: RewP difference waves (win - loss) at FCz ----------
    rewp_conditions = [
        ("mid_high_win_minus_loss", "Mid-Task High-Cue"),
        ("high_win_minus_loss", "High-Task"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for contrast_name, label in rewp_conditions:
        if contrast_name in contrasts:
            ev = contrasts[contrast_name]
            if "FCz" in ev.ch_names:
                ch_idx = ev.ch_names.index("FCz")
                times = ev.times
                data = ev.data[ch_idx] * 1e6  # µV
                ax.plot(times, data, label=label, linewidth=2)

    ax.axvline(0, color="k", linewidth=0.8)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("RewP Difference Waves (Win − Loss) at FCz")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(subject_dir, "fcz_rewp_difference_waves.png"), dpi=300)
    plt.close(fig)


def visualize_evoked(evokeds: dict, contrasts: dict, subject_id: str, output_dir: str):
    """Create a variety of evoked-response visualizations for one subject."""
    subject_dir = os.path.join(output_dir, f"sub-{subject_id}", "step07_evoked")
    os.makedirs(subject_dir, exist_ok=True)

    # --------- All evoked responses (per condition) ----------
    n_conditions = len(evokeds)
    if n_conditions > 0:
        n_cols = 2
        n_rows = int(np.ceil(n_conditions / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
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
        fig.suptitle(f"Sub-{subject_id} Evoked Responses", y=1.02)
        plt.savefig(os.path.join(subject_dir, "all_evoked_responses.png"), dpi=300)
        plt.close(fig)

    # --------- Contrast evokeds (win − loss) ----------
    if contrasts:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, (contrast_name, contrast_evoked) in enumerate(contrasts.items()):
            if i < 4:
                contrast_evoked.plot(axes=axes[i], show=False, titles=contrast_name)

        plt.tight_layout()
        fig.suptitle(f"Sub-{subject_id} Contrast Evoked (Win − Loss)", y=1.02)
        plt.savefig(
            os.path.join(subject_dir, "contrast_evoked_responses.png"), dpi=300
        )
        plt.close(fig)

    # --------- Topographies over time (one representative condition) ----------
    if evokeds:
        representative_evoked = list(evokeds.values())[0]
        time_points = [0.1, 0.2, 0.3, 0.4]  # seconds

        fig, axes = plt.subplots(1, len(time_points), figsize=(15, 4))
        for i, t in enumerate(time_points):
            representative_evoked.plot_topomap(
                times=t, axes=axes[i], show=False, colorbar=False
            )
            axes[i].set_title(f"{int(t * 1000)} ms")

        plt.tight_layout()
        fig.suptitle(f"Sub-{subject_id} Topography Over Time", y=1.08)
        plt.savefig(os.path.join(subject_dir, "topography_time_course.png"), dpi=300)
        plt.close(fig)

    # --------- Global Field Power (GFP) ----------
    if evokeds:
        fig, ax = plt.subplots(figsize=(12, 6))
        for condition, evoked in evokeds.items():
            gfp = np.std(evoked.data, axis=0)
            ax.plot(evoked.times, gfp, label=condition, linewidth=1.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Global Field Power (V)")
        ax.set_title(f"Sub-{subject_id} Global Field Power")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(subject_dir, "global_field_power.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # --------- Study-style FCz plots ----------
    create_study_replication_plots(evokeds, contrasts, subject_id, subject_dir)

    print(f"Evoked visualizations saved to: {subject_dir}")


def save_evoked_responses(
    evokeds: dict, contrasts: dict, subject_id: str, output_dir: str
) -> str:
    """Save individual and contrast evoked responses to disk."""
    subject_dir = os.path.join(output_dir, f"sub-{subject_id}", "step07_evoked")
    os.makedirs(subject_dir, exist_ok=True)

    # Individual evoked responses
    for condition, evoked in evokeds.items():
        fname = os.path.join(
            subject_dir, f"sub-{subject_id}_task-{TASK}_{condition}_ave.fif"
        )
        evoked.save(fname, overwrite=True)
        print(f"Saved {condition} evoked to: {fname}")

    # Contrast evoked responses
    for contrast_name, contrast_evoked in contrasts.items():
        fname = os.path.join(
            subject_dir, f"sub-{subject_id}_task-{TASK}_{contrast_name}_ave.fif"
        )
        contrast_evoked.save(fname, overwrite=True)
        print(f"Saved {contrast_name} contrast to: {fname}")

    # Combined file used by Step 9
    all_evoked = list(evokeds.values()) + list(contrasts.values())
    if all_evoked:
        all_fname = os.path.join(
            subject_dir, f"sub-{subject_id}_task-{TASK}_all_ave.fif"
        )
        mne.write_evokeds(all_fname, all_evoked, overwrite=True)
        print(f"Saved all evoked responses to: {all_fname}")

    return subject_dir


def main():
    parser = argparse.ArgumentParser(description="Step 7: Create evoked responses")
    parser.add_argument("--subject", required=True, help="Subject ID")
    args = parser.parse_args()

    subject_id = args.subject
    print(f"Step 7: Creating evoked responses for subject {subject_id}")

    # Load data from previous step
    epochs = load_previous_step(subject_id)

    # Create evoked responses
    evokeds = create_evoked_responses(epochs)

    # Create contrast evoked (Win − Loss)
    contrasts = create_contrast_evoked(evokeds)

    # Visualizations
    visualize_evoked(evokeds, contrasts, subject_id, OUTPUT_DIR)

    # Save evoked responses
    save_evoked_responses(evokeds, contrasts, subject_id, OUTPUT_DIR)

    print(f"Step 7 completed for subject {subject_id}")


if __name__ == "__main__":
    main()
