# src/edm/visualise/trajectory.py
import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from edm.utils import load_model
from edm.diffusion.edm_sampler import EDMTrajectorySampler
from edm.visualise import save_molecule_png


def _parse_int_list(s: str):
    """Turn '1000,900,800,0' into [1000, 900, 800, 0]."""
    if not s:
        return None
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError("checkpoints must be comma-separated integers") from e


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample EDM trajectories and render each time-step to PNGs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--path", type=str, default="models/edm_cosine.pt",
                   help="Path to .pt file with model weights")
    p.add_argument("--samples", type=int, default=4,
                   help="How many trajectories (molecules) to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # checkpoint selection (mutually exclusive)
    p.add_argument("--checkpoints", type=_parse_int_list, default=None,
                   metavar="t1,t2,...",
                   help="Explicit list of diffusion steps to save, "
                        "e.g. 1000,900,500,0")
    p.add_argument("--num_checkpoints", type=int, default=11, dest="num_ckpt",
                   help="Evenly spaced number of checkpoints between T and 0 "
                        "(ignored if --checkpoints is given)")

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---------- model & sampler ------------------------------------------------
    model, noise_sched, cfg = load_model(args.path, args.device)

    sampler_kwargs = {}
    if args.checkpoints is not None:
        sampler_kwargs["checkpoints"] = args.checkpoints
    else:
        sampler_kwargs["num_checkpoints"] = args.num_ckpt

    sampler = EDMTrajectorySampler(model, noise_sched, cfg,
                                   args=args, **sampler_kwargs)

    trajectories = sampler.sample(n_samples=args.samples)

    # ---------- output set-up --------------------------------------------------
    root = Path("plots") / "trajectory"
    root.mkdir(parents=True, exist_ok=True)

    checkpoints = sampler.checkpoints        # resolved list from the sampler
    pad = max(2, len(str(max(checkpoints)))) # e.g. 4 for up to 1000

    print(f"[trajectory] Saving PNGs to {root}/")
    for i, traj in tqdm(list(enumerate(trajectories)), unit="mol"):
        sample_dir = root / f"sample_{i:02}"
        sample_dir.mkdir(exist_ok=True)

        # Each traj element corresponds to checkpoints[j]
        for t, mol in zip(checkpoints, traj):
            fname = sample_dir / f"{t:0{pad}d}.png"
            save_molecule_png(mol, fname)

    print("[trajectory] Done.")


if __name__ == "__main__":
    main()
