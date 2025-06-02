import torch
from edm.qm9 import QM9Dataset
from tqdm import tqdm


class EDMSampler:
    def __init__(
            self,
            model,
            noise_schedule,
            cfg, 
            args=None
            ):
        self.model = model
        self.noise = noise_schedule
        self.cfg = cfg
        self.device = cfg['device']
        self.seed = cfg['seed']
        self.T = cfg['T']
        self.atom_scale = cfg['atom_scale']
        self.data = QM9Dataset(p=cfg['p'], device=cfg['device'], atom_scale=cfg['atom_scale'])
        self.categorical_distribution = cfg['categorical_distribution']

        # TODO: Potential overwrite some of the above with arguments from argparse?
        if args.seed:
            self.seed = args.seed


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

        M_vec = torch.multinomial(self.categorical_distribution, n_samples, replacement=True, generator=g).to(device)
        total_atoms = int(M_vec.sum())
        batch = torch.repeat_interleave(torch.arange(n_samples, device=device), M_vec).long()

        # Generate fully connected graphs
        edge_chunks = []
        start = 0
        for Mi in M_vec.tolist():
            idx = torch.arange(start, start + Mi, device=device)
            pairs = torch.combinations(idx, r=2, with_replacement=False)
            edges = torch.cat([pairs, pairs.flip(1)], dim=0).T
            edge_chunks.append(edges)
            start += Mi
        edge_index = torch.cat(edge_chunks, dim=1)  

        # Draw positional and invariant features and construct Data object. 
        pos = torch.randn(total_atoms, 3, device=device, generator=g)
        pos = net.subtract_CoG(batch, pos)
        h = torch.randn(total_atoms, len(self.data.atom_types), device=device, generator=g)
        data = Data(pos=pos, h=h, edge_index=edge_index, batch=batch)

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

        # Convert to individual data objects and return list of them
        mols, batch_vec = [], data.batch
        for idx in range(batch_vec.max().item() + 1):
            mask = batch_vec == idx
            mols.append(
                Data(
                    pos=data.pos[mask].detach().cpu(),
                    z=data.z[mask].detach().cpu(),
                    h=data.h[mask].detach().cpu()
                )
            )

        # TODO: Replace .h with clean one-hot encoding so PaiNNPropertyPredictor works cleanly on it

        return mols