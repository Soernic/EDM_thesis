import torch
import argparse
import math

from edm.utils import load_model
from edm.benchmark import Benchmarks
from edm.diffusion import EDMSampler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark EDM model'
    )

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Compute device')
    parser.add_argument('--path', type=str, default='models/edm_cosine.pt', help='path to .pt file containing model weights')
    parser.add_argument('--samples', type=int, default=500, help='how many molecules to sample')

    return parser.parse_args()


def sample_in_chunks(sampler, total_samples):
    """Sample in one or two passes so no call exceeds 5 000 molecules."""
    if total_samples <= 5000:
        return sampler.sample(total_samples)

    print(f'[benchmark] More than 5k samples requested - GPU memory does not like that')
    print(f'[benchmark] Splitting in two separate sampling paths.. ')

    # Split into two halves, each ≤ 5 000
    first_half = math.ceil(total_samples / 2)      # rounds up
    second_half = total_samples - first_half       # rounds down
    assert first_half <= 5_000 and second_half <= 5_000

    samples = sampler.sample(first_half)
    sampler.seed += 1                               # or use any new integer

    # Free GPU memory from the first batch before the second pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    samples += sampler.sample(second_half)         # concatenate lists
    return samples



def main():
    args = parse_args()

    model, noise, cfg = load_model(args.path, args.device)
    sampler = EDMSampler(model, noise, cfg, args)
    bench = Benchmarks()

    samples = sample_in_chunks(sampler, args.samples)
    results = bench.run_all(samples, args.samples)

    return results

if __name__ == '__main__':
    main()