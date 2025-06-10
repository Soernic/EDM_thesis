import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from pdb import set_trace

class PropertyPredictionBenchmarks:
    def __init__(self, model, denormalise_fn=None, target_idx=0, device='cpu'):
        """
        Arguments:
            model: a trained model
            denormalise_fn: function to undo normalisation (can be None if not needed)
            target_idx: index of the property to evaluate
            device: torch device
        """
        self.model = model.to(device)
        self.denormalise = denormalise_fn
        self.target_idx = target_idx
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        mae_total = 0.0
        num_samples = 0

        data_list = []
        all_preds = []
        all_targets = []

        for data in dataloader:
            data = data.to(self.device)
            out = self.model(data)
            target = data.y[:, self.target_idx]

            data_list.extend(Batch.to_data_list(data))
            all_preds.append(out)
            all_targets.append(target)
            
            # mae_total += F.l1_loss(out, target).item() * data.y.size(0)
            mae_total += F.l1_loss(out, target, reduction='sum').item()
            num_samples += target.size(0)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = {}
        metrics['mae_normalised'] = mae_total / num_samples

        if self.denormalise:
            preds_original = self.denormalise(all_preds, data_list)
            targets_original = self.denormalise(all_targets, data_list)
            metrics['mae'] = F.l1_loss(preds_original, targets_original).item()
            metrics['preds_original'] = preds_original.cpu().numpy().flatten().tolist()
            metrics['targets_original'] = targets_original.cpu().numpy().flatten().tolist()
        else:
            metrics['mae'] = metrics['mae_normalised']

        return metrics
