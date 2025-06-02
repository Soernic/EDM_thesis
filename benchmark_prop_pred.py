import os
import torch
import argparse

from edm.utils import load_prop_pred, args_to_config
from edm.benchmark import PropertyPredictionBenchmarks
from edm.models import PaiNNPropertyPredictor
from edm.qm9 import QM9Dataset
from edm.visualise import plot_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark property prediction model'
    )

    # Seed that controls train/val/test split is stored in model state dict
    # If overwriting here, it would contaminate data splits

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Compute device')
    parser.add_argument('--path', type=str, default='models/prop_pred.pt', help='path to .pt file containing model weights')
    parser.add_argument('--plot_folder', type=str, default='plots/prop_pred', help='where to save performance plot')
    parser.add_argument('--plot_name', type=str, default='test_mae', help='what to call the performance plot')

    return parser.parse_args()


def main():
    args = parse_args()
    model, config = load_prop_pred(args.path, args.device)

    # Generator
    torch.manual_seed(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    data = QM9Dataset(
        config['p'],
        generator,
        args.device,
        config['batch_size'],
        1, # atom scale always 1 for property prediction training
        config['cutoff_data'],
        config['target_idx']
    )
    data.get_data()
    data.compute_statistics_and_normalise() # uses train set
    _, _, test_loader = data.make_dataloaders()

    bench = PropertyPredictionBenchmarks(
        model,
        data.denormalise,
        config['target_idx'],
        args.device
    )

    metrics = bench.evaluate(test_loader)
    print(f'[benchmark | property {config["target_idx"]}] MAE {metrics["mae"]:.5f}')

    # Do a plot of the training run in the specified flder
    os.makedirs(args.plot_folder, exist_ok=True)
    plot_predictions(config['target_idx'], metrics, os.path.join(args.plot_folder, args.plot_name))



if __name__ == '__main__':
    main()