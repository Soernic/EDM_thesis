import torch
from edm.diffusion import LinearNoiseSchedule, CosineNoiseSchedule, EDM
from edm.models import PaiNNPropertyPredictor


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt['config']
    noise = (LinearNoiseSchedule if cfg['noise_schedule'] == 'linear' 
             else CosineNoiseSchedule)(cfg['T'], device=device)
    model = EDM(
        noise, 
        cfg['num_rounds'],
        cfg['state_dim'],
        cfg['cutoff_painn'],
        cfg['edge_dim'],
        ).to(device)
    model.load_state_dict(ckpt['ema_state'])
    model.eval()
    return model, noise, cfg


def load_prop_pred(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt['config']

    model = PaiNNPropertyPredictor(
        cfg['num_rounds'],
        cfg['state_dim'],
        cfg['cutoff_painn'],
        cfg['edge_dim'],
        cfg['property']
    ).to(device)

    model.load_state_dict(ckpt['state'])
    model.eval()
    return model, cfg


def args_to_config(args):
    return vars(args)