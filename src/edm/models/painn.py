import torch
import torch.nn as nn

from edm.models import ContextEmbedding, PaiNNMessage, PaiNNUpdate, GatedEquivariantBlock, \
                        InvariantReadout, EquivariantReadout, AlphaEquivariantReadout


from pdb import set_trace

class PaiNN(nn.Module):
    """
    Base class for multiple PaiNN variants. This class is not actually used directly. 

    Variants are below and include
    - Property prediction PaiNN (e.g., predict energy, dipole moments, ...)
    - Diffusion backbone PaiNN (predict EDM noise, invariant and equivariant)
    - Conditional diffusion backbone (WIP) (predict, EDM noise, invariant, equivairant, and take in extra context)
    """
    def __init__(
            self,
            num_rounds=9, 
            state_dim=256,
            cutoff=5,
            edge_dim=64,
            geb_layers=2, # for stability reasons should not be higher (empirical)
    ):
        super().__init__()

        # Number of rounds, state dimension, cutoff, and edge dim are universal to PaiNN models
        # Therefore they are in the base class
        self.num_rounds = num_rounds
        self.state_dim = state_dim
        self.cutoff = cutoff
        self.edge_dim = edge_dim # n

        # Since data preprocessing adds .h method to the Data objects, we work on that
        # We would've had to do a one-hot embedding of data.z anyways, so we just use h
        self.node_embedding = torch.nn.Linear(5, state_dim)

        # Message and update layers are taken from PaiNNMessage, PaiNNUpdate without changes
        self.message_layers = nn.ModuleList([
            PaiNNMessage(self.state_dim, self.edge_dim, self.cutoff)
            for _ in range(self.num_rounds)
        ])

        self.update_layers = nn.ModuleList([
            PaiNNUpdate(self.state_dim)
            for _ in range(self.num_rounds)
        ])

        # Gated equvariant blocks are the same
        self.gated_equivariant_blocks = nn.ModuleList([
            GatedEquivariantBlock(state_dim)
            for _ in range(geb_layers)
            ])
        
    def forward(self, x):
        raise NotImplementedError("Don't use this class directly. Instead use PaiNNPropertyPredictor, PaiNNDiffusion, PaiNNConditionalDiffusion")
    

    def get_edge_vectors(self, data):
        row, col = data.edge_index
        r_ij = data.pos[col] - data.pos[row]
        norm_r_ij = r_ij.norm(dim=1, keepdim=True).clamp(min=1e-6) # TODO: Experimental
        return r_ij, norm_r_ij       
    

class PaiNNPropertyPredictor(PaiNN):
    def __init__(
            self, 
            num_rounds=3,
            state_dim=128,
            cutoff=5,
            edge_dim=20,
            property='invariant' # ['invariant' | 'equivariant' | 'alpha_equivariant']
        ):
        super().__init__(num_rounds, state_dim, cutoff, edge_dim, geb_layers=3)

        self.property = property
        if property == 'invariant': 
            self.readout = InvariantReadout(state_dim, 1) # e.g., energy predictions
        elif property == 'equivariant': 
            self.readout = EquivariantReadout(state_dim) # e.g., dipole moment
        elif property == 'alpha_equivariant':
            self.readout = AlphaEquivariantReadout(state_dim)
        else: 
            raise ValueError('property value should be in [invariant | equivariant]')
        

    def forward(self, data):
        num_nodes = data.h.size(0)

        num_nodes = data.h.size(0)
        edge = data.edge_index
        r_ij, norm_r_ij = self.get_edge_vectors(data)

        # Initialise states, time embedding and vector states
        state = self.node_embedding(data.h)


        state_vec = torch.zeros([num_nodes, self.state_dim, 3], device=data.pos.device, dtype=data.pos.dtype)

        # Message passing loop
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            state, state_vec = message_layer(state, state_vec, edge, r_ij, norm_r_ij)
            state, state_vec = update_layer(state, state_vec)

        # gated equivariant blocks
        for gated_block in self.gated_equivariant_blocks: 
            state, state_vec = gated_block(state, state_vec)

        if self.property == 'invariant':
            molwise = torch.zeros((torch.max(data.batch)+1, 1)).to(data.h)
            out = self.readout(state)
            molwise = torch.index_add(molwise, dim=0, index=data.batch, source=out).squeeze(-1)

        elif self.property == 'equivariant':
            # These are special readout layers they define in PaiNN that work on vectors which we then take the norm of

            # Collection variable molwise
            molwise = torch.zeros((torch.max(data.batch)+1, 3)).to(data.pos)

            # Actual readout
            out = self.readout(state, state_vec, data.pos)

            # Adding to molwise and taking the norm for QM9 predictions
            molwise = torch.index_add(molwise, dim=0, index=data.batch, source=out)
            molwise = torch.norm(molwise, dim=1, keepdim=False) # [B, ]

        elif self.property == 'alpha_equivariant':
            molwise = torch.zeros((torch.max(data.batch)+1, 3, 3)).to(data.pos)
            atomwise_alpha = self.readout(state, state_vec, data.pos) # [B, 3, 3]

            molwise = torch.index_add(molwise, dim=0, index=data.batch, source=atomwise_alpha)
            molwise = molwise.diagonal(dim1=1, dim2=2).sum(-1) / 3 # [mols]

        else:
            raise ValueError('What happened to self.property?')
        
        return molwise


class PaiNNDiffusion(PaiNN):
    def __init__(
            self,
            num_rounds=9,
            state_dim=256,
            cutoff=None,
            edge_dim=20,
        ):
        super().__init__(num_rounds, state_dim, cutoff, edge_dim)
        
        self.time_embedding = ContextEmbedding(state_dim)

        self.inv_out = InvariantReadout(state_dim, 5)
        self.equi_out = EquivariantReadout(state_dim)


    def forward(self, data, t):
        num_nodes = data.h.size(0)
        edge = data.edge_index
        r_ij, norm_r_ij = self.get_edge_vectors(data)

        # Initialise states, time embedding and vector states
        state = self.node_embedding(data.h)
        context_embedding = self.time_embedding(t)

        state = state + context_embedding
        state_vec = torch.zeros([num_nodes, self.state_dim, 3], device=data.pos.device, dtype=data.pos.dtype)

        # Message passing loop
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            state, state_vec = message_layer(state, state_vec, edge, r_ij, norm_r_ij)
            state, state_vec = update_layer(state, state_vec)
            state = state + context_embedding

        # gated equivariant blocks
        for gated_block in self.gated_equivariant_blocks: 
            state, state_vec = gated_block(state, state_vec)
            state = state + context_embedding

        inv_out = self.inv_out(state)
        equi_out = self.equi_out(state, state_vec, data.pos)

        return equi_out, inv_out         


class PaiNNConditionalDiffusion(PaiNN):
    def __init__(
            self,
            num_rounds=9,
            state_dim=256,
            cutoff=5,
            edge_dim=64,
            # TODO: What else do we need for conditional?
        ):
        super().__init__(num_rounds, state_dim, cutoff, edge_dim)
        
        self.time_embedding = ContextEmbedding(state_dim)
        self.context_embedding = ContextEmbedding(state_dim)




    







    