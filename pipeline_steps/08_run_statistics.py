"""
Step 8: Run statistical tests on single subject data
"""

import os
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test

OUTPUT_DIR = "../output/derivatives/manual-pipeline"
TASK = "casinos"

# Contrasts (conditions) for permutation tests: Win vs Loss
CONTRASTS = [
    ("low_task_win", "low_task_loss"),
    ("mid_low_task_win", "mid_low_task_loss"),
    ("mid_high_task_win", "mid_high_task_loss"),
    ("high_task_win", "high_task_loss"),
]


def load_previous_step(subject_id):
    """Load clean epochs and evoked responses from previous steps."""
    # Clean epochs (Step 6)
    epochs_file = os.path.join(
        OUTPUT_DIR,
        f"sub-{subject_id}",
        "step06_rejection",
        f"sub-{subject_id}_task-{TASK}_clean_epochs.fif",
    )
    epochs = mne.read_epochs(epochs_file, preload=True)

    # Evoked responses (Step 7)
    evoked_dir = os.path.join(OUTPUT_DIR, f"sub-{subject_id}", "step07_evoked")
    evokeds = {}

    # Outcome conditions
    for condition in [
        "low_task_win",
        "low_task_loss",
        "mid_low_task_win",
        "mid_low_task_loss",
        "mid_high_task_win",
        "mid_high_task_loss",
        "high_task_win",
        "high_task_loss",
    ]:
        evoked_file = os.path.join(
            evoked_dir, f"sub-{subject_id}_task-{TASK}_{condition}_ave.fif"
        )
        if os.path.exists(evoked_file):
            evokeds[condition] = mne.read_evokeds(evoked_file)[0]

    return epochs, evokeds


def analyze_rewp_amplitudes(evokeds, subject_id):
    """
    Extract RewP amplitudes following study methodology.

    - FCz electrode
    - 240–340 ms time window
    - Difference wave: Win − Loss
    - All amplitudes reported in µV
    """
    rewp_results = {}

    time_window = (0.240, 0.340)  # seconds
    electrode = "FCz"

    print(
        f"\nAnalyzing RewP at {electrode} electrode, "
        f"{time_window[0] * 1000:.0f}-{time_window[1] * 1000:.0f} ms:"
    )

    conditions = [
        ("low_task_low_cue", "low_task_win", "low_task_loss"),
        ("mid_task_low_cue", "mid_low_task_win", "mid_low_task_loss"),
        ("mid_task_high_cue", "mid_high_task_win", "mid_high_task_loss"),
        ("high_task_low_cue", "high_task_win", "high_task_loss"),
    ]

    for condition_name, win_cond, loss_cond in conditions:
        if win_cond in evokeds and loss_cond in evokeds:
            # Win − Loss difference wave
            diff_evoked = mne.combine_evoked(
                [evokeds[win_cond], evokeds[loss_cond]], weights=[1, -1]
            )

            if electrode in diff_evoked.ch_names:
                ch_idx = diff_evoked.ch_names.index(electrode)
                time_mask = (diff_evoked.times >= time_window[0]) & (
                    diff_evoked.times <= time_window[1]
                )
                rewp_data = diff_evoked.data[ch_idx, time_mask]  # in Volts

                # Convert to µV
                rewp_data_uv = rewp_data * 1e6

                rewp_amplitude = float(np.max(rewp_data_uv))
                mean_amplitude = float(np.mean(rewp_data_uv))
                peak_time = float(
                    diff_evoked.times[time_mask][np.argmax(rewp_data_uv)]
                )

                rewp_results[condition_name] = {
                    "rewp_amplitude": rewp_amplitude,
                    "mean_amplitude": mean_amplitude,
                    "peak_time": peak_time,
                }

                print(
                    f"  {condition_name}: RewP = {rewp_amplitude:.2f} µV, "
                    f"Mean = {mean_amplitude:.2f} µV"
                )

    return rewp_results


def run_permutation_cluster_tests(epochs, contrasts, subject_id):
    """
    Run permutation cluster tests for Win vs Loss contrasts.

    For each contrast:
    - Match number of trials across conditions
    - Compute trial-wise difference (Win − Loss)
    - Run one-sample cluster permutation test against 0
    """
    results = {}

    print("Running permutation cluster tests:")

    for cond1, cond2 in contrasts:
        key = f"{cond1}_vs_{cond2}"
        if cond1 in epochs.event_id and cond2 in epochs.event_id:
            print(f"\n  Testing: {cond1} vs {cond2}")

            data1 = epochs[cond1].get_data()  # (n_epochs, n_channels, n_times)
            data2 = epochs[cond2].get_data()

            min_trials = min(data1.shape[0], data2.shape[0])

            if min_trials > 5:
                data1_matched = data1[:min_trials]
                data2_matched = data2[:min_trials]

                # Paired differences (Win − Loss)
                diff_data = data1_matched - data2_matched

                print(f"    Using {min_trials} matched trials for comparison")

                T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                    diff_data,
                    n_permutations=1000,
                    threshold=None,
                    tail=0,
                    n_jobs=1,
                    out_type="mask",
                    verbose=False,
                )

                results[key] = {
                    "T_obs": T_obs,
                    "clusters": clusters,
                    "cluster_p_values": cluster_p_values,
                    "H0": H0,
                    "diff_data": diff_data,
                    "n_trials": min_trials,
                }

                sig_clusters = np.where(cluster_p_values < 0.05)[0]
                print(f"    Matched trials used: {min_trials}")
                print(f"    Significant clusters: {len(sig_clusters)}")

                for i, cluster_idx in enumerate(sig_clusters):
                    p_val = float(cluster_p_values[cluster_idx])
                    cluster_mask = clusters[cluster_idx]
                    cluster_times = epochs.times[cluster_mask.any(axis=0)]
                    if len(cluster_times) > 0:
                        print(
                            f"      Cluster {i+1}: p={p_val:.4f}, "
                            f"time={cluster_times[0]:.3f}-{cluster_times[-1]:.3f}s"
                        )
            else:
                print(f"    Insufficient trials ({min_trials})")
                results[key] = None
        else:
            print(f"\n  Skipping {cond1} vs {cond2}: missing conditions")
            results[key] = None

    return results


def run_simple_statistics(evokeds):
    """
    Simple descriptive statistics on each evoked response.

    - Find global peak amplitude (in µV)
    - Peak time
    - Mean and std of all samples (in µV)
    """
    stats_results = {}

    print("\nRunning simple statistics on evoked responses:")

    for condition, evoked in evokeds.items():
        # evoked.data: (n_channels, n_times), in Volts
        abs_data = np.abs(evoked.data)
        peak_idx = np.unravel_index(np.argmax(abs_data), abs_data.shape)
        peak_channel = evoked.ch_names[peak_idx[0]]
        peak_time = float(evoked.times[peak_idx[1]])
        peak_amplitude_uv = float(evoked.data[peak_idx] * 1e6)  # V → µV

        mean_amplitude_uv = float(np.mean(evoked.data) * 1e6)
        std_amplitude_uv = float(np.std(evoked.data) * 1e6)

        stats_results[condition] = {
            "peak_channel": peak_channel,
            "peak_time": peak_time,
            "peak_amplitude": peak_amplitude_uv,
            "mean_amplitude": mean_amplitude_uv,
            "std_amplitude": std_amplitude_uv,
        }

        print(
            f"  {condition}: peak={peak_amplitude_uv:.2f} µV "
            f"at {peak_time:.3f}s ({peak_channel})"
        )

    return stats_results


def visualize_statistics(cluster_results, epochs, evokeds, rewp_results, subject_id, output_dir):
    """Create figures for cluster tests, RewP bars, and peak statistics."""
    subject_dir = os.path.join(output_dir, f"sub-{subject_id}", "step08_statistics")
    os.makedirs(subject_dir, exist_ok=True)

    # -------- Cluster-test visualizations --------
    for contrast_name, result in cluster_results.items():
        if result is None:
            continue

        T_obs = result["T_obs"]
        clusters = result["clusters"]
        cluster_p_values = result["cluster_p_values"]

        sig_clusters = np.where(cluster_p_values < 0.05)[0]
        if len(sig_clusters) == 0:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Peak T-stat time index
        peak_time_idx = np.unravel_index(np.argmax(np.abs(T_obs)), T_obs.shape)[1]
        peak_time = epochs.times[peak_time_idx]

        im = axes[0, 0].imshow(
            T_obs[:, peak_time_idx : peak_time_idx + 1],
            aspect="auto",
            cmap="RdBu_r",
        )
        axes[0, 0].set_title(f"T-statistic at {peak_time:.3f}s")
        plt.colorbar(im, ax=axes[0, 0])

        cluster_mask = clusters[sig_clusters[0]]
        im = axes[0, 1].imshow(
            cluster_mask.astype(float), aspect="auto", cmap="RdBu_r"
        )
        axes[0, 1].set_title(
            f"Significant cluster (p={cluster_p_values[sig_clusters[0]]:.4f})"
        )

        # Time course of cluster
        cluster_timecourse = np.mean(T_obs[cluster_mask], axis=0)
        axes[1, 0].plot(epochs.times, cluster_timecourse)
        axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("T-statistic")
        axes[1, 0].set_title("Cluster time course")
        axes[1, 0].grid(True, alpha=0.3)

        # Difference wave butterfly plot (if both evokeds exist)
        if len(contrast_name.split("_vs_")) == 2:
            cond1, cond2 = contrast_name.split("_vs_")
            if cond1 in evokeds and cond2 in evokeds:
                diff_evoked = evokeds[cond1].copy()
                diff_evoked.data = evokeds[cond1].data - evokeds[cond2].data
                diff_evoked.plot(
                    axes=axes[1, 1], show=False, titles=f"{cond1} - {cond2}"
                )

        plt.tight_layout()
        fig.suptitle(f"Sub-{subject_id} Cluster Test: {contrast_name}", y=1.02)
        fig.savefig(
            os.path.join(subject_dir, f"cluster_test_{contrast_name}.png"), dpi=300
        )
        plt.close(fig)

    # -------- RewP bar plots (separate figure) --------
    if rewp_results:
        fig_rewp, axes_rewp = plt.subplots(1, 2, figsize=(15, 6))

        conditions = list(rewp_results.keys())
        rewp_amps = [r["rewp_amplitude"] for r in rewp_results.values()]
        mean_amps = [r["mean_amplitude"] for r in rewp_results.values()]
        peak_times_ms = [r["peak_time"] * 1000 for r in rewp_results.values()]

        x = np.arange(len(conditions))
        width = 0.35

        # Amplitudes
        axes_rewp[0].bar(x - width / 2, rewp_amps, width, label="Peak RewP", alpha=0.7)
        axes_rewp[0].bar(x + width / 2, mean_amps, width, label="Mean RewP", alpha=0.7)
        axes_rewp[0].set_xlabel("Conditions")
        axes_rewp[0].set_ylabel("RewP Amplitude (µV)")
        axes_rewp[0].set_title(
            f"Sub-{subject_id} RewP Amplitudes (FCz, 240–340 ms)"
        )
        axes_rewp[0].set_xticks(x)
        axes_rewp[0].set_xticklabels(conditions, rotation=45, ha="right")
        axes_rewp[0].legend()
        axes_rewp[0].grid(True, alpha=0.3)

        # Peak times
        axes_rewp[1].bar(x, peak_times_ms, alpha=0.7, color="orange")
        axes_rewp[1].set_xlabel("Conditions")
        axes_rewp[1].set_ylabel("Peak Time (ms)")
        axes_rewp[1].set_title(f"Sub-{subject_id} RewP Peak Times")
        axes_rewp[1].set_xticks(x)
        axes_rewp[1].set_xticklabels(conditions, rotation=45, ha="right")
        axes_rewp[1].grid(True, alpha=0.3)
        axes_rewp[1].axhline(
            290, color="red", linestyle="--", alpha=0.7, label="Expected RewP (~290 ms)"
        )
        axes_rewp[1].legend()

        plt.tight_layout()
        fig_rewp.savefig(
            os.path.join(subject_dir, "rewp_analysis.png"), dpi=300
        )
        plt.close(fig_rewp)

    # -------- Peak amplitude & time (all conditions) --------
    conditions = list(evokeds.keys())
    peak_amps_uv = [np.max(np.abs(ev.data)) * 1e6 for ev in evokeds.values()]
    peak_times_ms = [
        ev.times[np.argmax(np.abs(ev.data.mean(axis=0)))] * 1000
        for ev in evokeds.values()
    ]

    fig_pk, axes_pk = plt.subplots(2, 1, figsize=(12, 10))

    x = np.arange(len(conditions))

    # Peak amplitudes
    axes_pk[0].bar(x, peak_amps_uv, alpha=0.7)
    axes_pk[0].set_xlabel("Conditions")
    axes_pk[0].set_ylabel("Peak Amplitude (µV)")
    axes_pk[0].set_title(f"Sub-{subject_id} Peak Amplitudes")
    axes_pk[0].set_xticks(x)
    axes_pk[0].set_xticklabels(conditions, rotation=45, ha="right")

    # Peak times
    axes_pk[1].bar(x, peak_times_ms, alpha=0.7)
    axes_pk[1].set_xlabel("Conditions")
    axes_pk[1].set_ylabel("Peak Time (ms)")
    axes_pk[1].set_title(f"Sub-{subject_id} Peak Times")
    axes_pk[1].set_xticks(x)
    axes_pk[1].set_xticklabels(conditions, rotation=45, ha="right")

    plt.tight_layout()
    fig_pk.savefig(os.path.join(subject_dir, "peak_statistics.png"), dpi=300)
    plt.close(fig_pk)

    print(f"Statistical visualizations saved to: {subject_dir}")


def save_statistics(cluster_results, simple_stats, rewp_results, subject_id, output_dir):
    """Save numerical statistics to disk (NPZ + text files)."""
    subject_dir = os.path.join(output_dir, f"sub-{subject_id}", "step08_statistics")
    os.makedirs(subject_dir, exist_ok=True)

    # Cluster results
    cluster_fname = os.path.join(subject_dir, f"sub-{subject_id}_cluster_results.npz")
    cluster_data = {}

    for contrast, result in cluster_results.items():
        if result is not None:
            cluster_data[f"{contrast}_T_obs"] = result["T_obs"]
            cluster_data[f"{contrast}_cluster_p_values"] = result["cluster_p_values"]
            cluster_data[f"{contrast}_n_trials"] = result["n_trials"]

    if cluster_data:
        np.savez(cluster_fname, **cluster_data)
        print(f"Cluster results saved to: {cluster_fname}")

    # Simple stats
    stats_fname = os.path.join(subject_dir, f"sub-{subject_id}_simple_stats.txt")
    with open(stats_fname, "w") as f:
        f.write(f"Simple Statistics for Subject {subject_id}\n")
        f.write("=" * 50 + "\n\n")

        for condition, stats in simple_stats.items():
            f.write(f"{condition}:\n")
            f.write(f"  Peak channel: {stats['peak_channel']}\n")
            f.write(f"  Peak time: {stats['peak_time']:.3f} s\n")
            f.write(f"  Peak amplitude: {stats['peak_amplitude']:.2f} µV\n")
            f.write(f"  Mean amplitude: {stats['mean_amplitude']:.2f} µV\n")
            f.write(f"  Std amplitude: {stats['std_amplitude']:.2f} µV\n\n")

    print(f"Simple statistics saved to: {stats_fname}")

    # RewP results
    if rewp_results:
        rewp_fname = os.path.join(subject_dir, f"sub-{subject_id}_rewp_results.txt")
        with open(rewp_fname, "w") as f:
            f.write(f"RewP Analysis Results for Subject {subject_id}\n")
            f.write("=" * 50 + "\n")
            f.write(
                "Following Sambrook & Goslin (2015): FCz electrode, 240–340 ms window\n\n"
            )

            for condition, results in rewp_results.items():
                f.write(f"{condition}:\n")
                f.write(f"  RewP Amplitude (max): {results['rewp_amplitude']:.2f} µV\n")
                f.write(f"  Mean Amplitude: {results['mean_amplitude']:.2f} µV\n")
                f.write(f"  Peak Time: {results['peak_time']:.3f} s\n\n")

        print(f"RewP results saved to: {rewp_fname}")

    return subject_dir


def main():
    parser = argparse.ArgumentParser(description="Step 8: Run statistical tests")
    parser.add_argument("--subject", required=True, help="Subject ID")
    args = parser.parse_args()

    subject_id = args.subject

    print(f"Step 8: Running statistical tests for subject {subject_id}")

    # Load data
    epochs, evokeds = load_previous_step(subject_id)

    # RewP analysis (FCz, 240–340 ms, Win − Loss)
    rewp_results = analyze_rewp_amplitudes(evokeds, subject_id)

    # Cluster permutation tests (Win vs Loss)
    cluster_results = run_permutation_cluster_tests(epochs, CONTRASTS, subject_id)

    # Simple descriptive statistics
    simple_stats = run_simple_statistics(evokeds)

    # Visualizations
    visualize_statistics(cluster_results, epochs, evokeds, rewp_results, subject_id, OUTPUT_DIR)

    # Save stats
    save_statistics(cluster_results, simple_stats, rewp_results, subject_id, OUTPUT_DIR)

    print(f"Step 8 completed for subject {subject_id}")


if __name__ == "__main__":
    main()
