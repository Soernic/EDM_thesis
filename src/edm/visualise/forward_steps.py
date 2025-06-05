# visualize_schedule_timelines.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from edm.diffusion import LinearNoiseSchedule, CosineNoiseSchedule

def get_alpha_positions(schedule, T=1000, interval=10):
    """
    Returns normalized positions of 1 - alpha(t) for visualization.
    """
    indices = np.arange(0, T, interval)
    with torch.no_grad():
        alphas = schedule.alphas.cpu().numpy()[indices]
    positions = 1 - alphas
    positions = (positions - positions.min()) / (positions.max() - positions.min())
    return positions, indices, alphas

def plot_schedule_timeline(ax, positions, indices, palette, name, label_cutoff=None, label_tail=None, exclude_label=None):
    """
    Draw a single timeline plot on the provided axis.
    """
    ax.hlines(0, 0, 1, color='lightgray', linewidth=3)
    for i, (pos, t) in enumerate(zip(positions, indices)):
        ax.vlines(pos, -0.1, 0.1, color=palette[i], linewidth=2)
        if exclude_label is not None and t == exclude_label:
            continue
        if label_cutoff and t % 100 == 0 and t <= label_cutoff:
            ax.text(pos, 0.12, f"{t}", ha='center', fontsize=9)
        elif label_tail and t in label_tail:
            ax.text(pos, 0.12, f"{t}", ha='center', fontsize=9)
    ax.text(positions[0], -0.15, r"$\alpha_t=1$", ha='center', fontsize=10)
    ax.text(positions[-1], -0.15, r"$\alpha_t=0$", ha='center', fontsize=10)
    ax.set_title(name, fontsize=13, pad=20)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)

def main():
    T = 1000
    interval = 10
    palette = sns.color_palette("rocket", T // interval)

    # Linear schedule
    linear_schedule = LinearNoiseSchedule(T=T)
    pos_linear, idx_linear, _ = get_alpha_positions(linear_schedule, T, interval)

    # Cosine schedule
    cosine_schedule = CosineNoiseSchedule(T=T)
    pos_cosine, idx_cosine, _ = get_alpha_positions(cosine_schedule, T, interval)

    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 3.3), sharex=True, height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.7)

    plot_schedule_timeline(
        axs[0], pos_linear, idx_linear, palette, name="Linear schedule",
        label_cutoff=700
    )
    axs[0].text(pos_linear[-1], 0.12, "800+", ha='center', fontsize=9)

    plot_schedule_timeline(
        axs[1], pos_cosine, idx_cosine, palette, name="Cosine schedule",
        label_cutoff=700, label_tail=[800, 900, 1000], exclude_label=0
    )

    plt.tight_layout()
    os.makedirs("plots/forward", exist_ok=True)
    output_path = os.path.join("plots", "forward", "combined_ticks.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()
