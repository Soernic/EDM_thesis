import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from edm.models import PaiNNPropertyPredictor
from edm.qm9 import QM9Dataset
from edm.visualise import plot_predictions
from edm.benchmark import PropertyPredictionBenchmarks

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class PropertyPredictionTrainer:
    """
    Uses the PaiNN backbone with a regressor head to predict 
    molecular properties like energy, dipole moments, or 
    polarizability tensors. Used with the train_prop_pred.py file.
    """
    def __init__(self, config):
        print(config)

        self.config = config
        
        self.target_idx =       config.get('target_idx')
        self.p =                config.get('p')
        self.cutoff_data =      config.get('cutoff_data')

        self.lr =               config.get('lr')
        self.weight_decay =     config.get('weight_decay')
        self.patience =         config.get('patience')
        self.factor =           config.get('factor')

        self.num_rounds =       config.get('num_rounds')
        self.state_dim =        config.get('state_dim')
        self.cutoff_painn =     config.get('cutoff_painn')
        self.edge_dim =         config.get('edge_dim')
        
        self.ema_alpha =        config.get('ema_alpha')

        self.epochs =           config.get('epochs')
        self.batch_size =       config.get('batch_size')
        self.benchmark_every =  config.get('benchmark_every')
        self.model_save_path =  config.get('model_save_path')
        self.use_tensorboard =  config.get('use_tensorboard')

        self.device =           config.get('device')
        self.seed =             config.get('seed')
        self.name =             config.get('name')

        self.config['property'] = self._property_type()
        self.property = self.config['property']
        self.config['atom_scale'] = 1

        # Generator
        torch.manual_seed(self.seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        
        self.model = PaiNNPropertyPredictor(
            num_rounds=self.num_rounds,
            state_dim=self.state_dim,
            cutoff=self.cutoff_painn, # TODO: Change so its clear its PaiNN Cutoff?
            edge_dim=self.edge_dim,
            property=self._property_type() # check if its inv or equiv.
        ).to(self.device)

        self.data = QM9Dataset(
            p = self.p,
            generator=self.generator,
            device=self.device,
            batch_size=self.batch_size,
            atom_scale=1, # locked in.
            cutoff_preprocessing=self.cutoff_data,
            target_idx=self.target_idx
        )

        self.bench = PropertyPredictionBenchmarks(
            model=self.model,
            denormalise_fn=self.data.denormalise,
            target_idx=self.target_idx,
            device=self.device
        )

        self.data.get_data()
        self.data.compute_statistics_and_normalise() # uses train set
        self.train_loader, self.val_loader, self.test_loader = self.data.make_dataloaders()

        self.setup_learning()
        self.setup_dir()
        self.setup_writer()


    def setup_learning(self):
        #self.optimiser = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0, amsgrad=True)
        self.optimiser = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimiser, mode='min', factor=self.factor, patience=self.patience)


    def setup_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"runs/{self.name}_{timestamp}" # TODO: Set default to "prop" or something
        os.makedirs(self.run_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.run_dir, f'{self.model_save_path}.pt') 

    def setup_writer(self):
        if self.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.run_dir)

            def filter_hparams(config):
                valid_types = (int, float, str, bool, torch.Tensor)
                return {k: v for k, v in config.items() if isinstance(v, valid_types)}

            self.writer.add_hparams(filter_hparams(self.config), {'dummy': 0})

        else:
            print(f'Not using tensorboard..')        
    

    def train_epoch(self):
        self.model.train()
        total_mse_loss = 0.0
        total_mae_loss = 0.0
        num_samples = 0

        for data in self.train_loader:
            self.optimiser.zero_grad()
            out = self.model(data)
            target = data.y[:, self.target_idx]
            mse_loss = F.mse_loss(out, target)
            mse_loss.backward()
            self.optimiser.step()

            # Also calculate mae_loss for comparison
            with torch.no_grad():
                mae_loss = F.l1_loss(out, target)
            
            batch_size = data.y.size(0)
            total_mse_loss += mse_loss.item() * batch_size
            total_mae_loss += mae_loss.item() * batch_size
            num_samples += batch_size

        mse_out = total_mse_loss / num_samples
        mae_out = total_mae_loss / num_samples
        return mse_out, mae_out
    

    # def evaluate(self, dataloader_type: str):

    #     # Pick dataloader
    #     dataloader = None
    #     if dataloader_type == 'validation':
    #         dataloader = self.val_loader
    #     elif dataloader_type == 'test':
    #         dataloader = self.test_loader
    #     else: 
    #         raise RuntimeError(f'Dataloader type not supported. (validation, test) are valid')

    #     # Prep for eval
    #     self.model.eval()
    #     mae_total = 0.0
    #     num_samples = 0

    #     all_preds = []
    #     all_targets = []

    #     # Inference
    #     with torch.no_grad():
    #         for data in dataloader:
    #             out = self.model(data)
    #             target = data.y[:, self.target_idx]

    #             all_preds.append(out)
    #             all_targets.append(target)

    #             mae_total = F.l1_loss(out, target).item() * data.y.size(0)
    #             num_samples += data.y.size(0)

    #     all_preds = torch.cat(all_preds, dim=0)
    #     all_targets = torch.cat(all_targets, dim=0)

    #     # Inverse normalisation
    #     preds_original = self.data.denormalise(all_preds)
    #     targets_original = self.data.denormalise(all_targets)

    #     # Populate metrics
    #     metrics = {}
    #     metrics['mae_normalised'] = mae_total / num_samples
    #     metrics['mae'] = F.l1_loss(preds_original, targets_original).item()
    #     metrics['preds_original'] = preds_original.cpu().numpy().flatten().tolist()
    #     metrics['targets_original'] = targets_original.cpu().numpy().flatten().tolist()

    #     return metrics
        
    def evaluate(self, dataloader_type: str):
        dataloader = self.val_loader if dataloader_type == 'validation' else self.test_loader
        results = self.bench.evaluate(dataloader)
        return results
    
    def train(self):
        best_val_ema_mae = float('inf')
        val_ema_mae = None # initialise EMA trakcing
        latest_test_mae = None # for pbar postfix

        results = {
            'train_mse': [],
            'train_mae': [],
            'val_mae': [],
            'val_ema_mae': [],
            'learning_rates': [],
            'best_epoch': 0,
            'best_val_ema_mae': float('inf'),
            'best_val_mae_unnormalised': float('inf'),
            'test_metrics':  {}
        }

        print(f'[training] Starting training for target property: {self.target_idx}')
        print(f'[training] Model checkpoints and logs will be saved to: {self.run_dir}')
        print(f'[training] Using EMA validation tracking with alpha:{self.ema_alpha}')

        with trange(1, self.epochs+1, desc='Training', leave=True) as pbar:
            for epoch in pbar:
                train_mse, train_mae = self.train_epoch()

                # For EMA update. 
                val_metrics = self.evaluate('validation') 
                val_mae = val_metrics['mae_normalised'] 

                # EMA handling
                if val_ema_mae is None: 
                    val_ema_mae = val_mae
                else: 
                    val_ema_mae = self.ema_alpha * val_ema_mae + (1 - self.ema_alpha)*val_mae

                self.scheduler.step(val_ema_mae)
                current_lr = self.optimiser.param_groups[0]['lr'] # grab this

                results['train_mse'].append(train_mse) # MSE
                results['train_mae'].append(train_mae) # MAE
                results['val_mae'].append(val_mae)     # MAE
                results['val_ema_mae'].append(val_ema_mae)
                results['learning_rates'].append(current_lr)

                if self.writer:
                    self.writer.add_scalar('MAE/train', train_mae, epoch)
                    self.writer.add_scalar('MAE/val', val_mae, epoch)
                    self.writer.add_scalar('MAE/val_ema', val_ema_mae, epoch)
                    self.writer.add_scalar('MAE/val_denormalised', val_metrics.get('mae'), epoch)
                    self.writer.add_scalar('LearningRate', current_lr, epoch)

                # print(f'Epoch: {epoch:4d}, Train Loss: {train_mae:.8f}, '
                #     f'Val MAE: {val_mae:.8f}, Val EMA: {val_ema_mae:.8f}, LR: {current_lr:.6f}')

                # Keep track of best EMA performance
                new_best = False
                if val_ema_mae < best_val_ema_mae:
                    best_val_ema_mae = val_ema_mae
                    results['best_epoch'] = epoch
                    results['best_val_ema_mae'] = val_ema_mae
                    results['best_val_mae_unnormalised'] = val_metrics.get('mae')

                    torch.save(
                        {
                            'config': self.config,
                            'state': self.model.state_dict(), 
                        },
                        self.checkpoint_path
                    )
                    new_best = True
                

                # Test evaluation every self.benchmark_every
                if epoch % self.benchmark_every == 0 or epoch == self.epochs - 1: 
                    test_metrics = self.evaluate('test')
                    latest_test_mae = test_metrics.get('mae', 0) 
                    results['test_metrics'][f'epoch_{epoch}'] = {
                        'mae': test_metrics.get('mae', 0),
                        'mae_normalised': test_metrics['mae_normalised']                    
                    }

                    plot_path = f"{self.run_dir}/{epoch:04}_preds.png"
                    self._plot_predictions(test_metrics, plot_path)

                    if self.writer:
                        self.writer.add_scalar('MAE/test_denormalised', latest_test_mae, epoch)                    

                postfix = {
                    'train': f'{train_mae:.6f}',
                    'val':f'{val_mae:.6f}',
                    'lr':f"{self.optimiser.param_groups[0]['lr']:.2e}"
                }

                if new_best:
                    best = val_metrics.get('mae')
                    postfix['denormalised_val_best'] = f'{best:.6f}'
                
                if latest_test_mae is not None: 
                    postfix['denormalised_test_latest'] = f'{latest_test_mae:.6f}'

                pbar.set_postfix(**postfix)


            # Close tensorboard writer if used
            if self.writer:
                self.writer.close()
            
            return results
        

    def _property_type(self):
        """
        Decide what kind of regressor head to use. I honestly don't know
        anything beyond that energy is invariant and dipole moment and 
        polarizability tensors are vector properties. Take a closer look
        if you need to use others. 
        """
        if self.target_idx == 0: # [0: mu | 1: alpha]
            return 'equivariant'
        
        elif self.target_idx == 1:
            return 'alpha_equivariant'
        
        else: # rest is assumed invariant
            return 'invariant'
            
        
    def _plot_predictions(self, test_metrics, plot_path):
        plot_predictions(self.target_idx, test_metrics, plot_path)
        return

        
