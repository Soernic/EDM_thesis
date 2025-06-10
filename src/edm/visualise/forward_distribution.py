# visualize_forward_time_comparison.py

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

def plot_rotated_comparison(ax, linear_alphas, cosine_alphas, x_vals, px_vals, samples, t, color_linear, color_cosine):
    if t == 0:
        ax.plot(px_vals, x_vals, label=f"t={t}", color=color_linear, lw=2)
    else:
        z_lin, q_lin = estimate_q_xt(samples, linear_alphas[t])
        z_cos, q_cos = estimate_q_xt(samples, cosine_alphas[t])

        ax.plot(q_lin, z_lin, label=f"Linear (t={t})", color=color_linear, lw=2, linestyle='-')
        ax.plot(q_cos, z_cos, label=f"Cosine (t={t})", color=color_cosine, lw=2, linestyle='--')

    ax.set_xlabel("Density")
    ax.set_ylim(-6, 6)
    ax.grid(False)
    ax.set_title(f"t = {t}")

def main():
    T = 1000
    steps = [0, 200, 400, 600, 800, 1000]
    num_samples = 200_000

    save_folder = os.path.join('plots', 'forward')
    os.makedirs(save_folder, exist_ok=True)

    # Sample from p(x)
    samples = torch.cat([
        torch.normal(-2, 0.5, size=(num_samples // 2,)),
        torch.normal(2, 0.7, size=(num_samples // 2,))
    ])
    x_vals = torch.linspace(-6, 6, 1000)
    px_vals = bimodal_p_x(x_vals).numpy()

    # Noise schedules
    cosine_schedule = CosineNoiseSchedule(T=T)
    linear_schedule = LinearNoiseSchedule(T=T)
    cosine_alphas = cosine_schedule.alphas.cpu()
    linear_alphas = linear_schedule.alphas.cpu()

    fig, axs = plt.subplots(1, len(steps), figsize=(10, 4), sharey=True)

    palette_linear = sns.color_palette("rocket", len(steps))
    palette_cosine = sns.color_palette("mako", len(steps))

    for i, t in enumerate(steps):
        plot_rotated_comparison(
            axs[i], linear_alphas, cosine_alphas,
            x_vals.numpy(), px_vals, samples, t,
            color_linear=palette_linear[i],
            color_cosine=palette_cosine[i]
        )

    axs[0].set_ylabel("z")

    # Add a legend
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = os.path.join(save_folder, 'forward_schedule_time_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved rotated time comparison to {output_path}")

if __name__ == "__main__":
    main()







# # visualize_forward_comparison.py

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

# def plot_forward_side_by_side(ax, alphas, x_vals, px_vals, samples, steps, palette, title):
#     for i, t in enumerate(steps):
#         if t == 0:
#             ax.plot(x_vals.numpy(), px_vals, label=r"$p(x)$", color=palette[i], lw=2)
#         else:
#             alpha_t = alphas[t]
#             z, q_vals = estimate_q_xt(samples, alpha_t)
#             ax.plot(z, q_vals, label=rf"$q(z_{{{t}}} \mid x)$", color=palette[i], lw=2)
#     ax.set_title(title)
#     ax.set_xlabel("z")
#     ax.set_ylabel("Density")
#     ax.grid(False)

# def main():
#     # Settings
#     T = 1000
#     steps = [0, 200, 400, 600, 800, 1000]
#     num_samples = 200_000
#     palette = sns.color_palette("rocket", len(steps))
#     # palette = sns.color_palette("icefire", len(steps))


#     save_folder = os.path.join('plots', 'forward')
#     os.makedirs(save_folder, exist_ok=True)

#     # Sample from p(x)
#     samples = torch.cat([
#         torch.normal(-2, 0.5, size=(num_samples // 2,)),
#         torch.normal(2, 0.7, size=(num_samples // 2,))
#     ])
#     x_vals = torch.linspace(-6, 6, 1000)
#     px_vals = bimodal_p_x(x_vals).numpy()

#     # Schedules
#     cosine_schedule = CosineNoiseSchedule(T=T)
#     linear_schedule = LinearNoiseSchedule(T=T)
#     cosine_alphas = cosine_schedule.alphas.cpu()
#     linear_alphas = linear_schedule.alphas.cpu()

#     # Plot side by side
#     fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
#     plot_forward_side_by_side(
#         axs[0], linear_alphas, x_vals, px_vals, samples, steps, palette,
#         "Linear schedule"
#     )
#     plot_forward_side_by_side(
#         axs[1], cosine_alphas, x_vals, px_vals, samples, steps, palette,
#         "Cosine schedule"
#     )


#     # Bigger, more readable legend slightly higher up
#     handles, labels = axs[1].get_legend_handles_labels()
#     fig.legend(
#         handles, labels,
#         loc='lower center',
#         ncol=len(steps),
#         fontsize=12,       # <- increased from 9 or 11 to 12
#         frameon=False,
#         bbox_to_anchor=(0.5, -0.01) 
#     )

#     # Adjust layout to make room for the legend
#     plt.tight_layout(rect=[0, 0.06, 1, 1]) 

#     output_path = os.path.join(save_folder, 'forward_schedule_comparison.png')
#     plt.savefig(output_path, dpi=300)
#     print(f"Saved side-by-side comparison to {output_path}")

# if __name__ == "__main__":
#     main()



