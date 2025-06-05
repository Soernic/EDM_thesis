
# visualize_forward_comparison.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from edm.diffusion import CosineNoiseSchedule, LinearNoiseSchedule

def bimodal_p_x(x):
    return 0.5 * torch.distributions.Normal(-2, 0.5).log_prob(x).exp() + \
           0.5 * torch.distributions.Normal(2, 0.7).log_prob(x).exp()

def estimate_q_xt(samples, alpha_t, bins=200, x_range=(-6, 6)):
    mean = alpha_t * samples
    std = torch.sqrt(1 - alpha_t**2)
    noised_samples = mean + torch.randn_like(mean) * std
    hist, bin_edges = np.histogram(noised_samples.numpy(), bins=bins, range=x_range, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, hist

def plot_forward_side_by_side(ax, alphas, x_vals, px_vals, samples, steps, palette, title):
    for i, t in enumerate(steps):
        if t == 0:
            ax.plot(x_vals.numpy(), px_vals, label=r"$p(x)$", color=palette[i], lw=2)
        else:
            alpha_t = alphas[t]
            z, q_vals = estimate_q_xt(samples, alpha_t)
            ax.plot(z, q_vals, label=rf"$q(z_{{{t}}} \mid x)$", color=palette[i], lw=2)
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("Density")
    ax.grid(False)

def main():
    # Settings
    T = 1000
    steps = [0, 200, 400, 600, 800, 1000]
    num_samples = 200_000
    palette = sns.color_palette("rocket", len(steps))
    # palette = sns.color_palette("icefire", len(steps))


    save_folder = os.path.join('plots', 'forward')
    os.makedirs(save_folder, exist_ok=True)

    # Sample from p(x)
    samples = torch.cat([
        torch.normal(-2, 0.5, size=(num_samples // 2,)),
        torch.normal(2, 0.7, size=(num_samples // 2,))
    ])
    x_vals = torch.linspace(-6, 6, 1000)
    px_vals = bimodal_p_x(x_vals).numpy()

    # Schedules
    cosine_schedule = CosineNoiseSchedule(T=T)
    linear_schedule = LinearNoiseSchedule(T=T)
    cosine_alphas = cosine_schedule.alphas.cpu()
    linear_alphas = linear_schedule.alphas.cpu()

    # Plot side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    plot_forward_side_by_side(
        axs[0], linear_alphas, x_vals, px_vals, samples, steps, palette,
        "Linear schedule"
    )
    plot_forward_side_by_side(
        axs[1], cosine_alphas, x_vals, px_vals, samples, steps, palette,
        "Cosine schedule"
    )


    # Bigger, more readable legend slightly higher up
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=len(steps),
        fontsize=12,       # <- increased from 9 or 11 to 12
        frameon=False,
        bbox_to_anchor=(0.5, -0.01) 
    )

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0.06, 1, 1]) 

    output_path = os.path.join(save_folder, 'forward_schedule_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved side-by-side comparison to {output_path}")

if __name__ == "__main__":
    main()





# # visualize_forward_noise_schedules.py

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# from edm.diffusion import CosineNoiseSchedule, LinearNoiseSchedule

# def bimodal_p_x(x):
#     return 0.5 * torch.distributions.Normal(-2, 0.5).log_prob(x).exp() + \
#            0.5 * torch.distributions.Normal(2, 0.7).log_prob(x).exp()

# def estimate_q_xt(samples, alpha_t, bins=200, x_range=(-6, 6)):
#     mean = alpha_t * samples
#     std = torch.sqrt(1 - alpha_t**2)
#     noised_samples = mean + torch.randn_like(mean) * std
#     hist, bin_edges = np.histogram(noised_samples.numpy(), bins=bins, range=x_range, density=True)
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#     return bin_centers, hist

# def plot_forward_diffusion(alphas, x_vals, px_vals, samples, steps, palette, save_path, title):
#     plt.figure(figsize=(10, 4))
#     for i, t in enumerate(steps):
#         if t == 0:
#             plt.plot(x_vals.numpy(), px_vals, label=r"$p(x)$", color=palette[i], lw=2)
#         else:
#             alpha_t = alphas[t]
#             z, q_vals = estimate_q_xt(samples, alpha_t)
#             plt.plot(z, q_vals, label=rf"$q(z_{{{t}}} \mid x)$", color=palette[i], lw=2)

#     plt.title(title)
#     plt.xlabel("z")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.grid(False)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     print(f"Saved plot to {save_path}")

# def main():
#     # Settings
#     T = 1000
#     steps = [0, 200, 400, 600, 800, 1000]
#     num_samples = 200_000
#     palette = sns.color_palette("crest", len(steps))

#     save_folder = os.path.join('plots', 'forward')
#     os.makedirs(save_folder, exist_ok=True)

#     # Prepare bimodal data samples
#     samples = torch.cat([
#         torch.normal(-2, 0.5, size=(num_samples // 2,)),
#         torch.normal(2, 0.7, size=(num_samples // 2,))
#     ])

#     # Target density
#     x_vals = torch.linspace(-6, 6, 1000)
#     px_vals = bimodal_p_x(x_vals).numpy()

#     # Plot for Cosine Schedule
#     cosine_schedule = CosineNoiseSchedule(T=T)
#     cosine_alphas = cosine_schedule.alphas.cpu()
#     cosine_path = os.path.join(save_folder, 'cosine_forward.png')
#     plot_forward_diffusion(
#         cosine_alphas, x_vals, px_vals, samples, steps, palette,
#         cosine_path, "Forward diffusion process (cosine, T=1000)"
#     )

#     # Plot for Linear Schedule
#     linear_schedule = LinearNoiseSchedule(T=T)
#     linear_alphas = linear_schedule.alphas.cpu()
#     linear_path = os.path.join(save_folder, 'linear_forward.png')
#     plot_forward_diffusion(
#         linear_alphas, x_vals, px_vals, samples, steps, palette,
#         linear_path, "Forward diffusion process (linear, T=1000)"
#     )

# if __name__ == "__main__":
#     main()


