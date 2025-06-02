import matplotlib.pyplot as plt
import os
import torch

from edm.diffusion import LinearNoiseSchedule, CosineNoiseSchedule

class NoiseSchedulePlotter:
    """
    Methods to 
    - plot noise schedules as scalar graphs
    - plot noising processes for molecules
    """
    def __init__(self, folder='plots/noise_schedule'):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    
    def plot_schedule(self, schedule, label: str | None = None):
        t = torch.arange(len(schedule.alphas))

        plt.figure(figsize=(6, 4))
        plt.plot(t, schedule.alphas.cpu(), label=r"$\alpha_t$" if label is None else f"{label}: α_t")
        plt.plot(t, schedule.sigmas.cpu(), label=r"$\sigma_t$" if label is None else f"{label}: σ_t")
        plt.xlabel("t")
        plt.legend()
        plt.title("Noise schedule")
        plt.tight_layout()
        plt.savefig(label + '.png', dpi=300)


    def compare_schedules(self, schedules: list, labels: list[str] | None = None):
        plt.figure(figsize=(6, 4))
        for i, sched in enumerate(schedules):
            lab = None if labels is None else labels[i]
            plt.plot(sched.alphas.cpu(), label=f"{lab}: α_t" if lab else r"$\alpha_t$")
            plt.plot(sched.sigmas.cpu(), linestyle="--", label=f"{lab}: σ_t" if lab else r"$\sigma_t$")
        plt.xlabel("t")
        plt.legend()
        plt.title("Noise schedules comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, 'noise_schedules.png'), dpi=300)


if __name__ == '__main__':
    
    cos = CosineNoiseSchedule()
    lin = LinearNoiseSchedule()

    NoiseSchedulePlotter().compare_schedules([cos, lin], ['cosine', 'linear'])