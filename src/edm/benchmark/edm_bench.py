
import time

import numpy as np
from tqdm import tqdm
from edm.utils import stable_flags, validity_and_uniqueness


class Benchmarks:

    def stability(self, mols, requested, q=False):

        # Check if any are valid
        total_mols = len(mols)
        if not total_mols:
            if not q:
                print(f'[benchmark | stability ] All samples invalid')
            return 0.0, 0.0
        
        # Count valid ones
        atom_hits, mol_hits = 0, 0
        total_atoms = 0
        for mol in mols:
            atom_mask, mol_ok = stable_flags(mol)
            atom_hits += atom_mask.sum().item()
            total_atoms += len(atom_mask)
            mol_hits += int(mol_ok)

        atom_stab = atom_hits / total_atoms
        mol_stab = mol_hits / requested

        if not q: 
            print(f'[benchmark | stability ] Molecules that contain NaN {(requested - total_mols)/total_mols:.2f}%')
            print(f'[benchmark | stability ] atom stability {atom_stab*100:5.2f}% ± {(np.sqrt(atom_stab*(1 - atom_stab)/total_atoms))*100:.2f}| '
                f'molecule stability {mol_stab*100:5.2f}% ± {(np.sqrt(mol_stab * (1 - mol_stab) / requested)*100):.2f}')
            
        return atom_stab, mol_stab
    

    def validity(self, mols, requested, q=False):
        """
        RDKit-based validity.  Caches everything we need for the downstream
        metrics to avoid recomputation.
        """
        unique_smiles, validity, uniqueness = validity_and_uniqueness(mols)
        self._unique_smiles = unique_smiles         # cache
        self._validity = validity
        self._uniqueness = uniqueness
    
        if not q: 
            print(f'[benchmark | validity  ] molecule validity {validity*100:5.1f}%')

        return validity

    def uniqueness(self, mols, requested, q=False):
        """
        Fraction of the *valid* molecules that are unique.
        """
        if not hasattr(self, '_uniqueness'):
            _, _, self._uniqueness = validity_and_uniqueness(mols)

        if not q:      
            print(f'[benchmark | uniqueness] uniqueness       {self._uniqueness*100:5.1f}%')
    
        return self._uniqueness

    def valid_and_unique(self, requested, q=False):
        """
        EDM’s “valid ∧ unique” – number of *unique valid* molecules divided by
        the requested sample count.  (This equals validity × uniqueness.)
        """
        vu = (len(self._unique_smiles) / requested) if getattr(self, '_unique_smiles', None) else 0.0

        if not q: 
            print(f'[benchmark | V ∧ U     ] valid & unique   {vu*100:5.1f}%')
        
        return vu
    

    def run_all(self, mols, requested, q=False):

        if not q: 
            print(f'[benchmark] Running all benchmarks..')

        start = time.time()

        stability        = self.stability(mols, requested, q)
        validity         = self.validity(mols, requested, q)
        uniqueness       = self.uniqueness(mols, requested, q)
        valid_and_unique = self.valid_and_unique(requested, q)

        benchmarks = {
            'stability'       : stability,
            'validity'        : validity,
            'uniqueness'      : uniqueness,
            'valid_and_unique': valid_and_unique,
        }

        end = time.time()
        if not q: 
            print(f'[benchmark] All benchmarks ran in {(end - start):.1f} seconds.')
        return benchmarks
