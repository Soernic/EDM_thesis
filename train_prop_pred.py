import argparse
import torch

from edm.trainers import PropertyPredictionTrainer
from edm.utils import args_to_config

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a property prediction model for molecular data with configurable hyperparameters'
    )

    # Data
    parser.add_argument('--target_idx', type=int, default=7, help='QM9 target to train on?')
    parser.add_argument('--p', type=float, default=0.01, help='Fraction of dataset to train on')
    parser.add_argument('--cutoff_data', type=float, default=5.0, help='Cutoff threshold for graph edges in molecule')
    
    # LR and optimiser
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for model trianing')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--patience', type=int, default=20, help='ReduceLROnPlateau patience')
    parser.add_argument('--factor', type=float, default=0.5, help='ReduceLROnPlateau reduction factor')
    
    # Model configuration
    parser.add_argument('--num_rounds', type=int, default=3, help='Number of message passing rounds')
    parser.add_argument('--state_dim', type=int, default=128, help='State dimension for both scalar and vector states')
    parser.add_argument('--cutoff_painn', type=float, default=5.0, help='Cosine cutoff r_cut threshold in PaiNN')
    parser.add_argument('--edge_dim', type=int, default=20, help='Dimensionality of RBF basis during invariant part of forward pass')

    # EMA
    parser.add_argument('--ema_alpha', type=float, default=0.9, help='ema for validation loss, used for reducing learning rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=2000, help='Epochs to train for')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--benchmark_every', type=int, default=10, help='How often to run a benchmark during trianing')
    parser.add_argument('--model_save_path', type=str, default='best_model', help='What to call the model in the run directory')
    parser.add_argument('--use_tensorboard', type=bool, default=True)

    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', type=str, default='prop', help='Experiment name')

    return parser.parse_args()


def main():
    args = parse_args()
    config = args_to_config(args)
    trainer = PropertyPredictionTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
