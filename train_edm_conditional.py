import argparse
import torch

from edm.trainers import ConditionalEDMTrainer
from edm.utils.model_utils import args_to_config

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the EDM model with configurable hyperparameters')
    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--T', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--p', type=float, default=0.01, help='Fraction of dataset to use (0-1]')
    parser.add_argument('--target_idx', type=int, default=1, help='Which regression target to use as conditional guidance')
    parser.add_argument('--resolution', type=int, default=100, 
                        help='Number of bins in joint distribution over molecules and regression target values')
    
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Compute device')
    # Optimization
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--factor', type=float, default=0.5, help='LR scheduler factor')
    parser.add_argument('--patience', type=int, default=20, help='LR scheduler patience')
    parser.add_argument('--alpha', type=float, default=0.99, help='EMA weight for model parameters')
    parser.add_argument('--sampling_model', type=str, choices=['ema', 'model'], default='ema',
                        help='Which model to use for sampling: ema or model')

    # Architecture
    parser.add_argument('--num_rounds', type=int, default=9, help='Number of interaction rounds')
    parser.add_argument('--state_dim', type=int, default=192, help='Dimensionality of node embeddings')
    parser.add_argument('--cutoff_preprocessing', type=float, default=None, 
                        help='Edge cutoff distance, should be None for diffusion, ~5 for property prediction')
    parser.add_argument('--cutoff_painn', type=float, default=5,
                        help='r_cut for cosine cutoff scaling in PaiNN model')
    parser.add_argument('--edge_dim', type=int, default=64, help='Dimensionality of edge features')
    # Data & training
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--atom_scale', type=float, default=0.25,
                        help='Scaling factor for atom logits during sampling, 0.25 seems to work best')
    parser.add_argument('--noise_schedule', type=str, choices=['linear', 'cosine'], default='cosine',
                        help='Type of noise schedule')
    parser.add_argument('--benchmark_every', type=int, default=50,
                        help='Run benchmark every N epochs')
    parser.add_argument('--benchmark_mols', type=int, default=512,
                        help='How many molecules to generate pr. benchmark')
    # Logging & output
    parser.add_argument('--use_tensorboard', type=bool, default=True, help='Enable TensorBoard logging')

    # Run control
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')

    parser.add_argument('--name', type=str, default="edm_conditional", help="The experiment name")
    return parser.parse_args()


def main():
    args = parse_args()
    config = args_to_config(args)

    trainer = ConditionalEDMTrainer(config)
    trainer.sample(2)
    trainer.train(args.epochs)
    atom_stab, mol_stab = trainer.benchmark()
    print(f'Final benchmark — atom stability: {atom_stab:.3f}, molecule stability: {mol_stab:.3f}')


if __name__ == '__main__':
    main()
