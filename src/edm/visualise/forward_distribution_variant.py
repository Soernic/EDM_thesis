import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from edm.diffusion import CosineNoiseSchedule, LinearNoiseSchedule

from pdb import set_trace

def bimodal_samples(num_samples):
    # half from N(-2, 0.5²), half from N(2, 0.7²)
    return torch.cat([
        torch.normal(-1.6, 0.5, size=(num_samples // 2,)),
        torch.normal( 1.6, 0.7, size=(num_samples // 2,))
    ])

def generate_noisy_samples(samples, alpha_t, num_samples=2000):
    mean  = alpha_t * samples
    std   = torch.sqrt(1 - alpha_t ** 2)
    noisy = mean + torch.randn_like(mean) * std
    idx   = torch.randperm(noisy.size(0))[:num_samples]
    return noisy[idx].numpy()

def ridge_plot_schedule(schedule_alphas, samples, steps, title, palette):
    df_list = []
    for t in steps:
        alpha_t = schedule_alphas[t]
        noisy   = generate_noisy_samples(samples, alpha_t, num_samples=10000)
        noisy   = np.clip(noisy, -3.5, 4)
        df_list.append(pd.DataFrame({
            'x': noisy,
            'time': str(t)
        }))
    df = pd.concat(df_list, ignore_index=True)

    g = sns.FacetGrid(
        df, row="time", hue="time",
        aspect=15, height=0.6, palette=palette
    )
    g.map(sns.kdeplot, "x",
          bw_adjust=0.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "x",
          clip_on=False, color="w", lw=2, bw_adjust=0.5)
    g.refline(y=0, linewidth=1, linestyle="-", clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(-4, 0, label,
                fontweight="bold", color=color,
                ha="right", va="center")
    g.map(label, "x")

    # tighten vertical spacing
    g.figure.subplots_adjust(hspace=-0.4)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(xlim=(-3.5, 4))
    g.despine(bottom=True, left=True)

    # make the whole figure taller (width=6", height=9")
    # g.figure.set_size_inches(6, 9)

    # g.figure.suptitle(title, fontsize=16, fontweight='bold', x=0.55, y=0.98)
    g.figure.suptitle(title, fontsize=16, fontweight='bold', x=0.50, y=0.98)
    return g

def main():
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    T           = 1000
    steps       = [0, 200, 400, 600, 800, 1000]
    # steps       = [0, 125, 250, 375, 500, 625, 750, 875, 1000]
    # steps       = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    palette     = sns.color_palette("rocket", len(steps))
    samples     = bimodal_samples(200_000)

    linear_sched = LinearNoiseSchedule(T=T).alphas.cpu()
    cosine_sched = CosineNoiseSchedule(T=T).alphas.cpu()

    g1 = ridge_plot_schedule(
        linear_sched, samples, steps,
        "Linear Schedule", palette
    )
    g1.savefig('plots/forward/ridge_diffusion_linear_long.png',
               dpi=300, bbox_inches='tight')
    plt.close()

    g2 = ridge_plot_schedule(
        cosine_sched, samples, steps,
        "Cosine Schedule", palette
    )
    
    # g2 = ridge_plot_schedule(
    #     cosine_sched, samples, steps,
    #     "Forward process (cosine)", palette
    # )

    g2.savefig('plots/forward/ridge_diffusion_cosine_long.png',
               dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()






# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import torch
# from edm.diffusion import CosineNoiseSchedule, LinearNoiseSchedule

# def bimodal_samples(num_samples):
#     # half from N(-2, 0.5²), half from N(2, 0.7²)
#     return torch.cat([
#         torch.normal(-2, 0.5, size=(num_samples // 2,)),
#         torch.normal( 2, 0.7, size=(num_samples // 2,))
#     ])

# def generate_noisy_samples(samples, alpha_t, num_samples=2000):
#     # produce mean and std of the noised distribution
#     mean = alpha_t * samples
#     std  = torch.sqrt(1 - alpha_t ** 2)
#     # full noised tensor
#     noisy = mean + torch.randn_like(mean) * std
#     # randomly pick num_samples indices across the whole tensor
#     idx = torch.randperm(noisy.size(0))[:num_samples]
#     return noisy[idx].numpy()

# def ridge_plot_schedule(schedule_alphas, samples, steps, title, palette):
#     df_list = []
#     for t in steps:
#         alpha_t = schedule_alphas[t]
#         noisy   = generate_noisy_samples(samples, alpha_t)
#         # clamp extremes to [−4, 4]
#         noisy   = np.clip(noisy, -4, 4)
#         df_temp = pd.DataFrame({
#             'x': noisy,
#             'time': [f"{t}" for _ in range(len(noisy))]
#         })
#         df_list.append(df_temp)
#     df = pd.concat(df_list, ignore_index=True)

#     # make the ridge plot
#     g = sns.FacetGrid(
#         df, row="time", hue="time",
#         aspect=15, height=0.6, palette=palette
#     )
#     g.map(
#         sns.kdeplot, "x",
#         bw_adjust=0.5, clip_on=False,
#         fill=True, alpha=1, linewidth=1.5
#     )
#     g.map(
#         sns.kdeplot, "x",
#         clip_on=False, color="w", lw=2, bw_adjust=0.5
#     )
#     g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

#     # add a label to each row
#     def label(x, color, label):
#         ax = plt.gca()
#         ax.text(-4, 0, label, fontweight="bold", color=color,
#                 ha="right", va="center")
#     g.map(label, "x")

#     # tighten things up and set the x‐axis limits
#     g.figure.subplots_adjust(hspace=-0.4)
#     g.set_titles("")
#     g.set(yticks=[], ylabel="")
#     g.set(xlim=(-4, 4))
#     g.despine(bottom=True, left=True)
#     g.fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
#     return g

# def main():
#     sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
#     palette     = sns.color_palette("rocket", 6)
#     T           = 1000
#     steps       = [0, 200, 400, 600, 800, 1000]
#     num_samples = 200_000
#     samples     = bimodal_samples(num_samples)

#     cosine_schedule = CosineNoiseSchedule(T=T).alphas.cpu()
#     linear_schedule = LinearNoiseSchedule(T=T).alphas.cpu()

#     # Linear schedule plot
#     g1 = ridge_plot_schedule(
#         linear_schedule, samples, steps,
#         "Linear Schedule", palette
#     )
#     plt.savefig('plots/forward/ridge_diffusion_linear.png',
#                 dpi=300, bbox_inches='tight')
#     plt.close()

#     # Cosine schedule plot
#     g2 = ridge_plot_schedule(
#         cosine_schedule, samples, steps,
#         "Cosine Schedule", palette
#     )
#     plt.savefig('plots/forward/ridge_diffusion_cosine.png',
#                 dpi=300, bbox_inches='tight')
#     plt.close()

# if __name__ == "__main__":
#     main()

