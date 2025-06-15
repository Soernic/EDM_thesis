import torch
import torch.nn.functional as F
from edm.qm9 import QM9Dataset
from tqdm import tqdm

from pdb import set_trace


class EDMSampler:
    def __init__(
            self,
            model,
            noise_schedule,
            cfg, 
            args=None
            ):
        
        # TODO: Potential overwrite some of the above with arguments from argparse?
        if args is not None:
            if hasattr(args, 'seed'):
                self.seed = args.seed
                cfg['seed'] = args.seed

            if hasattr(args, 'device'):
                self.device = args.device
                cfg['device'] = args.device


        self.model = model
        self.noise = noise_schedule
        self.cfg = cfg
        self.device = cfg['device']
        self.seed = cfg['seed']
        self.T = cfg['T']
        self.atom_scale = cfg['atom_scale']
        self.data = QM9Dataset(p=cfg['p'], device=cfg['device'], atom_scale=cfg['atom_scale'])

        # If there's a graph library, load that in. The presence of a graph library means that
        # the diffusion model is trained with an edge cutoff, and we should sample accordingly
        # This means, when we initialise the graphs, we use partially connected graphs like we did
        # during training - otherwise EDM will assume that "since everythign is connected, then
        # everything must be close to everything" --> very round structures and lower uniqueness
        self.graph_library = cfg.get('graph_library', None)
        self.atom_types_reverse = cfg.get('atom_types_reverse',
                                        torch.tensor([1,6,7,8,9], device=self.device))
        
        self.categorical_distribution = cfg['categorical_distribution']

        if 'joint_distribution' in cfg:
            self.joint_distribution = cfg['joint_distribution']
            self.type = 'conditional'
        else:
            self.type = 'unconditional'



    @torch.no_grad()
    def sample(self, n_samples: int = 16):
        """
        Draw n_samples molecules from the model.
        When sampling, we must construct the samples as PyG objects for the 
        PaiNN model to work with them. 
        """

        from torch_geometric.data import Data
        device = self.device
        T = self.T
        sched = self.noise
        g = torch.Generator(device=device)
        g.manual_seed(self.seed) # can be overwritten by benchmark_edm.py
        net = self.model
        net.eval()


        if self.type == 'conditional':
            # Flatten the distribution to a vector of probabilities
            joint_flat = self.joint_distribution.flatten()

            # Sample using those probabilities
            joint_idx = torch.multinomial(joint_flat, n_samples, replacement=True, generator=g).to(device)
            
            #
            num_bins, max_Mp1 = self.joint_distribution.shape
            c_bin_idx = joint_idx // max_Mp1
            M_vec = joint_idx % max_Mp1

            # Convert bin indices to values in [0, 1] (midpoint of bin)
            c_vals = ((c_bin_idx + 0.5) / num_bins).to(device)
        else:
            M_vec = torch.multinomial(self.categorical_distribution, n_samples, replacement=True, generator=g).to(device)
            c_vals = None
        total_atoms = int(M_vec.sum())
        batch = torch.repeat_interleave(torch.arange(n_samples, device=device), M_vec).long()

        # set_trace()

        # Sampling procedure using either the graph library if it exists and 
        # otherwise defaults to standard dense graph initialisation
        import random
        edge_chunks = []
        start = 0
        for Mi in M_vec.tolist():
            if self.graph_library and Mi in self.graph_library:
                # pick a random template, shift indices, push to device
                template = random.choice(self.graph_library[Mi]).to(device)
                edges = template + start
            else:
                # fall back to full clique
                idx = torch.arange(start, start + Mi, device=device)
                pairs = torch.combinations(idx, r=2, with_replacement=False)
                edges = torch.cat([pairs, pairs.flip(1)], dim=0).T
            edge_chunks.append(edges)
            start += Mi
        edge_index = torch.cat(edge_chunks, dim=1)

        # set_trace()

        # # Generate fully connected graphs
        # edge_chunks = []
        # start = 0
        # for Mi in M_vec.tolist():
        #     idx = torch.arange(start, start + Mi, device=device)
        #     pairs = torch.combinations(idx, r=2, with_replacement=False)
        #     edges = torch.cat([pairs, pairs.flip(1)], dim=0).T
        #     edge_chunks.append(edges)
        #     start += Mi
        # edge_index = torch.cat(edge_chunks, dim=1)  

        # Draw positional and invariant features and construct Data object. 
        pos = torch.randn(total_atoms, 3, device=device, generator=g)
        pos = net.subtract_CoG(batch, pos)
        h = torch.randn(total_atoms, len(self.data.atom_types), device=device, generator=g)
        data = Data(pos=pos, h=h, edge_index=edge_index, batch=batch, c=c_vals)

        # Sampling loop
        print(f'Sampling {n_samples} molecules...')
        for t in tqdm(range(T, 0, -1)):
            t_full = torch.full((total_atoms, ), t, device=device)
            s_full = t_full - 1
            t_norm = t_full / T

            # From 3.2 in EDM
            # [eps_x, eps_h] = phi(z_x, z_h, t)
            # phi(z_x, z_h, t) = model(z_x, z_h, t/T) - [z_x, 0]
            # Subtract CoG from eps_x to project it to zero-CoG subspace
            # This happens within the forward call of the model self.model(data, t_norm)
            eps_pred_x, eps_pred_h = net(data, t_norm)
            
            # Noise schedule values
            alpha_ts = sched.alpha_t_given_s(t_full, s_full)
            sigma_ts = sched.sigma_t_given_s(t_full, s_full)
            sigma_t = sched.sigma(t_full)

            # For cosine, noise schedule clipping happens in CosineNoiseSchedule class
            c1 = 1 / alpha_ts
            c2 = -(sigma_ts ** 2) / (alpha_ts * sigma_t)
            c3 = sched.sigma_t_to_s(t_full, s_full)

            c1 = torch.clip(c1, max=15) # would be 30 at T = 1000 otherwise
            c2 = torch.clip(c2, min=-15) # would be -30 at T = 1000 otherwise

            # Broadcast to feature dimension
            c1 = c1[:, None]
            c2 = c2[:, None]
            c3 = c3[:, None]

            # Draw noise and center it
            # eps_x = torch.randn_like(data.pos, generator=g) # wtf, randn_like doesn't have generator support?
            eps_x = torch.randn(data.pos.shape, generator=g, device=device)
            eps_x = net.subtract_CoG(batch, eps_x)
            # eps_h = torch.randn_like(data.h, generator=g) 
            eps_h = torch.randn(data.h.shape, generator=g, device=device)

            # Overwrite data object attributes so we can use it again
            data.pos = c1*data.pos + c2*eps_pred_x + c3*eps_x
            data.h = c1*data.h + c2*eps_pred_h + c3*eps_h

            # Final CoG centering to be certain since scaling might affect it
            data.pos = net.subtract_CoG(batch, data.pos)

        #  Mark every graph that contains a NaN (so trianing loops doesn't break)
        bad_graphs = data.batch[ data.pos.isnan().any(dim=1)
                               | data.h.isnan().any(dim=1) ].unique()

        if len(bad_graphs):
            keep = ~data.batch.unsqueeze(0).eq(bad_graphs[:, None]).any(0)
            data.pos = data.pos[keep]
            data.h   = data.h[keep]
            data.batch = data.batch[keep]
            # re-index batches so they stay contiguous
            old2new = {old: new for new, old in enumerate(data.batch.unique().tolist())}
            data.batch = torch.tensor([old2new[b.item()] for b in data.batch],
                                      device=data.batch.device)


        # early exit (if all graphs are unstable)
        if data.batch.numel() == 0: # all samples were bad
            return [] # so .benchmark survives
        
        # Convert h to probs and save nuclear charges z in data object
        probs = torch.softmax(data.h / self.atom_scale, dim=1)
        type_idx = probs.argmax(dim=1)
        data.z = self.data.atom_types_reverse[type_idx]

        # Reset data.h for PaiNNPropertyPredictor
        data.h = F.one_hot(type_idx, num_classes=len(self.data.atom_types)).float()
        # set_trace()

        # Convert to individual data objects and return list of them
        mols, batch_vec = [], data.batch
        for idx in range(batch_vec.max().item() + 1):
            mask = batch_vec == idx
            mols.append(
                Data(
                    pos=data.pos[mask].detach().cpu(),
                    z=data.z[mask].detach().cpu(),
                    h=data.h[mask].detach().cpu(),
                    batch=torch.zeros(data.z[mask].size(0)).long().detach().cpu()
                )
            )

        # TODO: Replace .h with clean one-hot encoding so PaiNNPropertyPredictor works cleanly on it

        return mols
    


from typing import List, Optional
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

class EDMTrajectorySampler(EDMSampler):
    """
    Like EDMSampler, but instead of discarding the intermediate states it
    returns a full trajectory for every molecule.

    >>> sampler = EDMTrajectorySampler(model, noise, cfg,
    ...                                checkpoints=[1000, 900, 800, 0])
    >>> trajs = sampler.sample(n_samples=8)
    >>> len(trajs)                 # 8 trajectories
    8
    >>> len(trajs[0])              # 4 snapshots per trajectory
    4
    """
    def __init__(self,
                 model,
                 noise_schedule,
                 cfg,
                 checkpoints: Optional[List[int]] = None,
                 num_checkpoints: Optional[int] = None,
                 args=None):
        super().__init__(model, noise_schedule, cfg, args)

        if checkpoints is not None and num_checkpoints is not None:
            raise ValueError("Specify either 'checkpoints' or 'num_checkpoints', not both.")

        if checkpoints is None:
            # Evenly spaced, including T and 0
            k = num_checkpoints or 11
            self.checkpoints = [int(round(t))
                                for t in torch.linspace(self.T, 0, k).tolist()]
            
        else:
            # keep the *given* order, just sanity-check the values
            self.checkpoints = []
            for t in checkpoints:
                t_int = int(t)
                if not 0 <= t_int <= self.T:
                    raise ValueError(f"Checkpoint {t_int} outside [0, T={self.T}]")
                if t_int not in self.checkpoints:          # drop duplicates
                    self.checkpoints.append(t_int)

        # We’ll look-up quickly during the loop
        self._checkpoint_set = set(self.checkpoints)

    @torch.no_grad()
    def sample(self, n_samples: int = 16):
        """
        Returns
        -------
        trajectories : List[List[Data]]
            trajectories[i][j] contains the j-th saved snapshot of molecule i.
        """
        device, T, sched, g = self.device, self.T, self.noise, torch.Generator(device=self.device)
        g.manual_seed(self.seed)
        net = self.model
        net.eval()

        # --- identical pre-sampling section from EDMSampler -----------------
        if self.type == 'conditional':
            joint_flat = self.joint_distribution.flatten()
            joint_idx = torch.multinomial(joint_flat, n_samples, replacement=True, generator=g).to(device)
            num_bins, max_Mp1 = self.joint_distribution.shape
            c_bin_idx = joint_idx // max_Mp1
            M_vec = joint_idx % max_Mp1
            c_vals = ((c_bin_idx + 0.5) / num_bins).to(device)
        else:
            M_vec = torch.multinomial(self.categorical_distribution, n_samples, replacement=True, generator=g).to(device)
            c_vals = None

        total_atoms = int(M_vec.sum())
        batch = torch.repeat_interleave(torch.arange(n_samples, device=device), M_vec).long()

        edge_chunks, start = [], 0
        for Mi in M_vec.tolist():
            idx = torch.arange(start, start + Mi, device=device)
            pairs = torch.combinations(idx, r=2, with_replacement=False)
            edges = torch.cat([pairs, pairs.flip(1)], dim=0).T
            edge_chunks.append(edges)
            start += Mi
        edge_index = torch.cat(edge_chunks, dim=1)

        pos = torch.randn(total_atoms, 3, device=device, generator=g)
        pos = net.subtract_CoG(batch, pos)
        h = torch.randn(total_atoms, len(self.data.atom_types), device=device, generator=g)
        data = Data(pos=pos, h=h, edge_index=edge_index, batch=batch, c=c_vals)

        # ---------- helpers --------------------------------------------------
        def _split_to_pyg_list(d: Data) -> List[Data]:
            """Clone tensors *once* to CPU, then slice per-molecule."""
            probs = torch.softmax(d.h / self.atom_scale, dim=1)
            type_idx = probs.argmax(dim=1)
            z = self.data.atom_types_reverse[type_idx]

            h_onehot = F.one_hot(type_idx, num_classes=len(self.data.atom_types)).float()

            mols, bvec = [], d.batch.cpu()
            for idx in range(bvec.max().item() + 1):
                mask = bvec == idx
                mols.append(
                    Data(
                        pos=d.pos[mask].detach().cpu(),
                        z=z[mask].detach().cpu(),
                        h=h_onehot[mask].detach().cpu(),
                        batch=torch.zeros(mask.sum()).long()
                    )
                )
            return mols

        # Prepare empty trajectory holders
        trajectories = [[] for _ in range(n_samples)]

        # ---------- sampling loop with snapshotting --------------------------
        print(f'Sampling {n_samples} molecules with checkpoints {self.checkpoints}')
        for t in range(T, 0, -1):
            t_full = torch.full((total_atoms,), t, device=device)
            s_full = t_full - 1
            t_norm = t_full / T

            eps_pred_x, eps_pred_h = net(data, t_norm)
            alpha_ts = sched.alpha_t_given_s(t_full, s_full)
            sigma_ts = sched.sigma_t_given_s(t_full, s_full)
            sigma_t = sched.sigma(t_full)

            c1 = torch.clip(1 / alpha_ts, max=15)[:, None]
            c2 = torch.clip(-(sigma_ts ** 2) / (alpha_ts * sigma_t), min=-15)[:, None]
            c3 = sched.sigma_t_to_s(t_full, s_full)[:, None]

            eps_x = torch.randn(data.pos.shape, generator=g, device=device)
            eps_x = net.subtract_CoG(batch, eps_x)
            eps_h = torch.randn(data.h.shape, generator=g, device=device)

            data.pos = c1 * data.pos + c2 * eps_pred_x + c3 * eps_x
            data.h = c1 * data.h + c2 * eps_pred_h + c3 * eps_h
            data.pos = net.subtract_CoG(batch, data.pos)

            # ---------- save snapshot if this step is requested ---------------
            if t in self._checkpoint_set:
                for i, mol in enumerate(_split_to_pyg_list(data)):
                    trajectories[i].append(mol)

        # ---------------- grab the final state (t = 0) --------------------
        if 0 in self._checkpoint_set:            # user asked for it
            for i, mol in enumerate(_split_to_pyg_list(data)):
                trajectories[i].append(mol)


        return trajectories
