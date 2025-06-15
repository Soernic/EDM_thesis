import argparse
import torch
import os

from edm.utils import load_model
from edm.benchmark import Benchmarks
from edm.diffusion import EDMSampler
from edm.visualise import save_molecule_png
from edm.qm9 import QM9Dataset

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualise data from dataset or EDM'
    )
    
    parser.add_argument('--task', type=str, choices=['dataset', 'edm', 'both'], default='dataset',
                        help='What do you want to visualise?')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Compute device')

    parser.add_argument('--path', type=str, default='models/edm_cosine.pt', help='path to .pt file containing model weights')
    parser.add_argument('--samples', type=int, default=8, 
                        help='how many molecules to sample? If --task = both, sample --samples of both')
    
    parser.add_argument('--edm_save_folder', type=str, default='plots/edm', help='Folder to which the edm samples will be saved')
    parser.add_argument('--qm9_save_folder', type=str, default='plots/qm9', help='Folder to which the qm9 samples are saved')
    return parser.parse_args()

  

if __name__ == '__main__':
    args = parse_args()
    
    if args.task in ['edm', 'both']:
        # Load in model relevant stuff
        model, noise, cfg = load_model(args.path, args.device)

        try: 
            cd = cfg['categorical_distribution']
        except KeyError:
            cd = None

        # Before saving, we check stability and validity, hence the benchmarks
        bench = Benchmarks()
        sampler = EDMSampler(model, noise, cfg, args)
        mols = sampler.sample(args.samples)

        # Make sure folder exist
        # os.makedirs(os.path.join('plots', 'edm'), exist_ok=True)
        os.makedirs(args.edm_save_folder, exist_ok=True)

        # Saving to disc
        print(f'[visualisation] Rendering {args.samples} EDM samples..\n')
        for idx, mol in tqdm(enumerate(mols)):
            res = bench.run_all([mol], requested=1, q=True)
            if res['stability'] == (1.0, 1.0) and res['validity'] == 1.0:
                print(f'[visualisation] Mol {idx} stable, saving..')
                save_molecule_png(mol, f'{args.edm_save_folder}/edm_{idx:02}.png')
            else:
                print(f'[visualisation] Mol {idx} unstable, skipping..')


    if args.task in ['dataset', 'both']:

        # Make sure folder exists
        # os.makedirs(os.path.join('plots', 'qm9'), exist_ok=True)
        os.makedirs(args.qm9_save_folder, exist_ok=True)

        # Load in data
        # batch size 1 for visualising
        data = QM9Dataset(p=0.001, device='cpu', batch_size=1) 

        # Saving to disc
        print(f'[visualisation] Rendering {args.samples} dataset samples..\n')
        random_indices = torch.randint(0, len(data.dataset), size=(args.samples,))
        for idx, mol in tqdm(enumerate(data.dataset[random_indices])):
            save_molecule_png(mol, f'{args.qm9_save_folder}/qm9_{idx:02}.png')
        

        