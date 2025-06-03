import argparse
import torch

from tqdm import tqdm
from pdb import set_trace

from edm.utils.model_utils import args_to_config, load_model, load_prop_pred
from edm.diffusion import EDMSampler
from edm.benchmark import Benchmarks
from edm.qm9 import QM9Dataset

class HistogramPlotter:
    """
    Takes in a torch property prediction checkpoint, and an EDM checkpoint. 
    - 
    """
    def __init__(self, prop_pred_model, edm_model):
        self.prop_pred_model = prop_pred_model
        self.edm_model = edm_model


    def plot_hist(mols):
        pass

    def plot_double_hist(mols1, mols2, label1, label2):
        """
        Take mols1 from dataset and mols2 from EDM model. 
        Use property predictor to determine the rough distribution

        """


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualise histogram of properties on molecules, from dataset or generated'
    )

    parser.add_argument('--prop_pred_path', type=str, default='models/mu.pt', help='Path to property prediction model')
    parser.add_argument('--edm_path', type=str, default='models/edm.pt', help='Path to EDM sampler model')

    parser.add_argument('--samples', type=int, default=500, help='How many samples to generate with EDM')
    parser.add_argument('--resolution', type=int, default=50, help='How many bins there should be in the histogram')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for sampling')

    return parser.parse_args


if __name__ == '__main__':
    """
    Load in property prediction model from 
    """

    args = parse_args()

    prop_pred, prop_pred_cfg = load_prop_pred(args.prop_pred_path, args.device)
    edm, noise, edm_cfg = load_model(args.edm_path, args.device)

    set_trace()

    sampler = EDMSampler(edm, noise, edm_cfg, args)
    bench = Benchmarks() # for EDM stability checks

    set_trace()

    # Grab samples from EDM model
    samples = sampler.sample(args.samples) 
    clean_samples = []

    set_trace()

    # Check for stability and validity
    for idx, mol in tqdm(enumerate(samples)):
        res = bench.run_all([mol], requested=1, q=True)
        if res['stability'] == (1.0, 1.0) and res['validity'] == 1.0:
            clean_samples.append(mol)

    set_trace()

    # This is how many we're gonna grab from the QM9 Dataset
    number_of_samples = len(clean_samples)

    set_trace()

    data = QM9Dataset(p=number_of_samples / 100_000, device='cpu', batch_size=number_of_samples)
    data.get_data()
    visualisation_set = data.train_data # Should be exactly the same number of mols as #samples

    set_trace()






