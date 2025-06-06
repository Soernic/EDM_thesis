# src/edm/visualise/atom_scale_sweep.py

import os
import re
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from edm.utils import load_model
from edm.benchmark import Benchmarks
from edm.diffusion import EDMSampler

from pdb import set_trace

sns.set(style='whitegrid')


def extract_scale(filename):
    """
    Extracts atom scale as XXX / 100.0 from filenames like EDM_5_XXX.pt
    """
    match = re.search(r'EDM5_(\d{3})\.pt', filename)
    if match:
        return int(match.group(1)) / 100.0
    return None


def benchmark_model(model_path, device, samples):
    model, noise, cfg = load_model(model_path, device)
    sampler = EDMSampler(model, noise, cfg, argparse.Namespace(device=device, samples=samples))
    bench = Benchmarks()
    mols = sampler.sample(samples)
    _, mol_stab = bench.stability(mols, samples, q=True)  # Only molecule-level stability
    return mol_stab


def main():
    parser = argparse.ArgumentParser(description="Run atom scale sweep benchmark")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--samples', type=int, default=200, help='Number of molecules to sample per model')
    parser.add_argument('--model_dir', type=str, default='models/atom_scale_sweep/', help='Folder with .pt model files')

    args = parser.parse_args()

    # Make sure output directory exists
    out_dir = 'plots/atom_scale/'
    out_fname = f"{args.samples:05}_atom_scale_sweep.png"
    out_path = os.path.join(out_dir, out_fname)
    os.makedirs(out_dir, exist_ok=True)


    results = []
    for fname in sorted(os.listdir(args.model_dir)):
        # set_trace()
        if fname.endswith('.pt'):
            scale = extract_scale(fname)
            if scale is None:
                continue
            model_path = os.path.join(args.model_dir, fname)
            print(f"[run] Benchmarking model {fname} (scale = {scale})")
            try:
                mol_stab = benchmark_model(model_path, args.device, args.samples)
                results.append((scale, mol_stab))
            except Exception as e:
                print(f"[warn] Failed to benchmark {fname}: {e}")

    if not results:
        print("[error] No valid models found.")
        return

    # Sort results by scale
    results.sort()
    scales, stabilities = zip(*results)

    # Plotting
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=scales, y=stabilities, marker='o')
    plt.xlabel("Atom scale (relative h weighting)", fontsize=12)
    plt.ylabel("Molecule stability", fontsize=12)
    plt.title(f"Stability vs. Atom Scale Sweep ({args.samples} samples/model)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


    import pandas as pd  # Put this at the top of your script

    # Save raw data as CSV
    csv_path = out_path.replace('.png', '.csv')
    df = pd.DataFrame({'atom_scale': scales, 'molecule_stability': stabilities})
    df.to_csv(csv_path, index=False)
    print(f"[done] Plot saved to {out_path}")
    print(f"[done] CSV saved to {csv_path}")



if __name__ == '__main__':
    main()
