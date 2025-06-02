import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import copy
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from edm.diffusion import EDM, CosineNoiseSchedule, LinearNoiseSchedule, EDMSampler
from edm.qm9 import QM9Dataset
from edm.benchmark import Benchmarks
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False




class EDMTrainer:
    def __init__(self, config):
        
        # Config
        self.config =           config
        self.seed =             config.get('seed', 42)
        self.T =                config.get('T', 1000)
        self.device =           config.get('device', torch.device('cpu'))
        self.p =                config.get('p', 1.0)
        self.batch_size =       config.get('batch_size', 100)
        self.atom_scale =       config.get('atom_scale', 0.25)
        self.lr =               config.get('lr', 5e-4)
        self.alpha =            config.get('alpha', 0.99) # for model weights
        self.sampling_model =   config.get('sampling_model', 'ema')
        self.factor =           config.get('factor', 0.5) # ReduceLROnPlateau
        self.patience =         config.get('patience', 10) # ReduceLROnPlateau
        self.use_tensorboard =  config.get('use_tensorboard', True)
        self.noise_schedule =   config.get('noise_schedule', 'cosine')
        self.benchmark_every =  config.get('benchmark_every', 50) # epochs
        self.benchmark_mols =   config.get('benchmark_mols', 500) # molecules to benchmark
        self.name =             config.get('name', "")

        # Generator
        torch.manual_seed(self.seed)
        self.generator = torch.Generator().manual_seed(self.seed)

        # Noise schedule and model
        if self.noise_schedule == 'linear': 
            self.noise = LinearNoiseSchedule(self.T, device=self.device)
        else: 
            self.noise = CosineNoiseSchedule(device=self.device)

        # Register EDM model (includes PaiNN EGNN backbone)
        self.model = EDM(
            noise_schedule=self.noise,
            num_rounds=config.get('num_rounds', 9),
            state_dim=config.get('state_dim', 256),
            cutoff_painn=config.get('cutoff_painn', 5), # this cutoff is different from data preprocessing cutoff
            edge_dim=config.get('edge_dim', 64)            
        ).to(self.device)

        # EMA model for nice sampling
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)


        # Initialise QM9Dataset instance and and download data
        self.data = QM9Dataset(
            p=self.p,
            generator=self.generator,
            device=self.device,
            batch_size=self.batch_size,
            atom_scale=self.atom_scale
        )

        # Ready train, val, test split, create loaders, and push everything to GPU all at once. 
        self.data.get_data()
        self.categorical_distribution = self.data.compute_categorical_distribution()
        self.config['categorical_distribution'] = self.categorical_distribution
        self.train_loader, self.val_loader, self.test_loader = self.data.make_dataloaders() # on GPU. 

        # Initialising after categorical distribution is defined and in config. 
        # The sampler needs it.
        self.sampler = EDMSampler(model=self.ema_model, noise_schedule=self.noise, cfg=self.config)
        self.bench = Benchmarks()

        # Optimiser, scheduler, directories, logging...
        self.setup_learning()
        self.setup_dir()
        self.setup_writer()

    def setup_learning(self):
        self.optimiser = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-6, amsgrad=True)
        self.scheduler = ReduceLROnPlateau(self.optimiser, mode='min', factor=self.factor, patience=self.patience)

    def setup_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"runs/{self.name}_{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.run_dir, 'best_model.pt')
        
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
        """
        Train for one epoch. Pushing to device is purposefully absent
        because of specific loading code where all of the data is 
        pushed to the GPU at once in the beginning. It never leaves.
        """
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:

            # Skip last batch for stablity reasons
            if batch.num_graphs < self.batch_size:
                continue
            
            self.optimiser.zero_grad()
        
            loss = self.model.loss_fn(batch)
            if torch.isnan(loss) or not torch.isfinite(loss):
                print(f'NaN or inf loss detected. Zeroing gradients and skipping batch.')
                self.optimiser.zero_grad()
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)
            self.optimiser.step()

            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)


    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.model.loss_fn(batch)
                total_loss += loss.item() # / batch.num_graphs
        return total_loss / len(self.val_loader)
    

    def train(self, epochs):
        best_atom_s = 0 # track best atom stability so far
        best_mol_s = 0 

        with trange(1, epochs + 1, desc='Training', leave=True) as pbar:
            for epoch in pbar: 

                # Will be updated if model is good
                save_model = False                

                # One epoch
                train_loss = self.train_epoch()
                val_loss = self.evaluate()
                if epoch % self.benchmark_every == 0:

                    # When over 80% molecule stability, it gets hard to tell the difference 
                    # from small samples. We incrase the number of samples for this to 2000
                    # which should be enough to distinguish even relatively small differences
                    if best_mol_s < 0.8:
                        atom_s, mol_s = self.benchmark(self.benchmark_mols)
                    else:
                        print(f'[info] Best molecule stability above 80%, defaulting to sample size of 2k molecules to detect differences with reasonable certainty..')
                        atom_s, mol_s = self.benchmark(2000)
               
                    # Check if new model is any good
                    if mol_s > best_mol_s:
                        print(f'[info] New best model detected at {mol_s*100:5.2f}% molecule stability | previous best was {best_mol_s*100:5.2f}%')
                        print(f'[info] Saving new model..')
                        best_mol_s = mol_s
                        save_model = True      

                    if atom_s > best_atom_s:
                        best_atom_s = atom_s 

                    if self.use_tensorboard and TENSORBOARD_AVAILABLE:
                        self.writer.add_scalar('Benchmark/atom_stability', atom_s, epoch)
                        self.writer.add_scalar('Benchmark/mol_stability',  mol_s, epoch)            

                        # best ones
                        self.writer.add_scalar('Benchmark/best_atom_stability', best_atom_s, epoch)
                        self.writer.add_scalar('Benchmark/best_mol_stability',  best_mol_s, epoch)                          

                # Step on the validation loss with patience
                self.scheduler.step(val_loss)

                # Update EMA version of the model weights for sampling
                for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
                    ema_p.data.mul_(self.alpha).add_(p.data, alpha=1 - self.alpha)

                # Save model only if it is actually better
                if save_model: 
                    torch.save(
                        {
                            "config": self.config,
                            "ema_state": self.ema_model.state_dict(),
                            "raw_state": self.model.state_dict()
                        },
                        self.checkpoint_path
                    )

                # Tensorboard logging
                if self.use_tensorboard and TENSORBOARD_AVAILABLE:
                    self.writer.add_scalar('Loss/train', train_loss, epoch)
                    self.writer.add_scalar('Loss/val', val_loss, epoch)
                    current_lr = self.optimiser.param_groups[0]['lr']
                    self.writer.add_scalar('Learning_rate', current_lr, epoch)
                    self.writer.flush()                
                
                pbar.set_postfix(
                    train=f"{train_loss:.4f}",
                    val=f"{val_loss:.4f}",
                    lr=f"{self.optimiser.param_groups[0]['lr']:.2e}"
                )

        tqdm.write("Training complete.")
        if self.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer.close()    


    @torch.no_grad()
    def sample(self, n_samples: int = 16):
        return self.sampler.sample(n_samples)
    

    @torch.no_grad()
    def benchmark(self, n_samples):
        # Use EDMSampler to sample, and Benchmarks class for metrics
        samples = self.sample(n_samples)
        metrics = self.bench.run_all(samples, n_samples)
        stability = metrics['stability']
        atom_stable, molecule_stable = stability

        return atom_stable, molecule_stable # stability is used for deciding on best model
        