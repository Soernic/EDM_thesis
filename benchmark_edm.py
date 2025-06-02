import torch
import argparse

from edm.utils import load_model
from edm.benchmark import Benchmarks
from edm.diffusion import EDMSampler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark EDM model'
    )

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Compute device')
    parser.add_argument('--path', type=str, default='models/edm.pt', help='path to .pt file containing model weights')
    parser.add_argument('--samples', type=int, default=500, help='how many molecules to sample')

    return parser.parse_args()


def main():
    args = parse_args()

    model, noise, cfg = load_model(args.path, args.device)
    sampler = EDMSampler(model, noise, cfg, args)
    bench = Benchmarks()

    samples = sampler.sample(args.samples)
    results = bench.run_all(samples, args.samples)

    return results

if __name__ == '__main__':
    main()