import torch
from itertools import combinations

try: 
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    Chem = None

# Ångström versions of Tables 7–9, QM9 subset, order-independent keys
def _k(a, b):             # helper
    return frozenset((a, b))

# Standard distance thresholds for single bonds. 
SINGLE = {
    _k('H','H'):0.74,   _k('H','C'):1.09,   _k('H','O'):0.96,   _k('H','N'):1.01,   _k('H','F'):0.92,
                        _k('C','C'):1.54,   _k('C','O'):1.43,   _k('C','N'):1.47,   _k('C','F'):1.35,
                                            _k('O','O'):1.48,   _k('N','O'):1.40,   _k('O','F'):1.42,
                                                                _k('N','N'):1.45,   _k('N','F'):1.36,
                                                                                    _k('F','F'):1.42
    }

# Standard distance thresholds for double bonds. 
DOUBLE = {
    _k('C','C'):1.34,   _k('C','O'):1.20,   _k('C','N'):1.29,
                        _k('O','O'):1.21,   _k('N','O'):1.21, 
                                            _k('N','N'):1.25
        }

# Standard distance thresholds for triple bonds. 
TRIPLE = {
    _k('C','C'):1.20, _k('C','O'):1.13, _k('C','N'):1.16,
                                        
                                        _k('N','N'):1.10}

# Margins of error found to work well on QM9 by EDM authors
MARGINS = {1: 0.10, 2: 0.05, 3: 0.03}          # Å

# Valence allowed for each atom
ALLOWED_VALENCE = {
    'H': {1},
    'C': {4},
    'N': {3},
    'O': {2},
    'F': {1},
}


def stable_flags(data):
    """
    ## Description of stability benchmark.

    Takes in PyG batch_size 1 Data object and computes whether the atom and 
    molecule are stable. This is done following EDM, checking for each 
    possible combinatorial combination of 2 atoms in the graph, if they are 
    within standard distances for different types of bonds. The table above is 
    taken from EDM (and I removed the GeomDrugs atom types). 

    Whether there is a bond or not is based on 
    1. distance
    2. the combination of atoms

    For 2 atoms, we check first for triple, then double, then single bonds. 
    We compute the euclidean distance between the two atoms, r, and check 
    whether this is smaller than the standard distance for those two atoms
    and a certain bond order + a small margin following EDM as well. If it 
    is, there is determined to be a bond of that order, and this valence 
    is added to the atom. This process continuesfor all atoms.

    After all 2-combinations have been computed, the bonds are in place. We
    compare the number of bonds for each atom with the allowed valence for 
    that atom, which determines whether the atom is stable or not.

    If all atoms in the molecule are stable, the molecule is determined as 
    stable as well. 
    """

    # For reference
    symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
    elem = [symbols[int(z)] for z in data.z]

    n = data.pos.size(0)
    bonds_per_atom = torch.zeros(n, dtype=torch.long)

    for i, j in combinations(range(n), 2):
        r = (data.pos[i] - data.pos[j]).norm().item()
        order = 0 

        # test triple first, then double, then single
        for bond_order, table in ((3, TRIPLE), (2, DOUBLE), (1, SINGLE)):
            d0 = table.get(_k(elem[i], elem[j])) # this is the threshold
            if d0 is not None and r < d0 + MARGINS[bond_order]: # check if there is a bond
                order = bond_order # if there is a bond, save it

                # .. then break out of that loop and move on to next combination of atoms
                break # this way we make sure to not double count

        # If the loop above breaks, that means order != 0, and we add that number of bonds
        # to both atoms (e.g., double bond means +2 bonds for each of the atoms)
        if order:          
            bonds_per_atom[i] += order
            bonds_per_atom[j] += order

    # Check if all atoms are stable by comparing against allowed valence
    atom_ok = torch.tensor([
        bonds_per_atom[k].item() in ALLOWED_VALENCE[elem[k]]
        for k in range(n)])
    
    # Check if mol is stable by seeing if all atoms are stable
    mol_ok = atom_ok.all().item()
    # set_trace()
    return atom_ok, mol_ok


# Validity and uniqueness tests with RDKit
_bondtype = {1: Chem.rdchem.BondType.SINGLE if Chem else None,
             2: Chem.rdchem.BondType.DOUBLE if Chem else None,
             3: Chem.rdchem.BondType.TRIPLE if Chem else None}

_num2sym = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}          # QM9 only

def _bond_order(sym1: str, sym2: str, d: float) -> int:
    """Return 3/2/1 if distance d (Å) satisfies triple/double/single threshold,
    else 0.  Uses the SAME tables & margins as `stable_flags`."""
    key = _k(sym1, sym2)
    # triple → double → single (cheap early exits)
    if TRIPLE.get(key) and d < TRIPLE[key] + MARGINS[3]:
        return 3
    if DOUBLE.get(key) and d < DOUBLE[key] + MARGINS[2]:
        return 2
    if SINGLE.get(key) and d < SINGLE[key] + MARGINS[1]:
        return 1
    return 0



def build_rdkit_mol(data):
    """Convert a PyG `Data` object → sanitised RDKit Mol  → canonical SMILES.
    Returns (mol, smiles) or (None, None) on sanitisation failure."""
    if Chem is None:
        raise ImportError("RDKit is required for validity metrics.  "
                          "Install with `conda install -c conda-forge rdkit` "
                          "or `pip install rdkit-pypi`.")

    rw = Chem.RWMol()
    n = data.pos.size(0)

    # add atoms
    for z in data.z.tolist():
        sym = _num2sym.get(int(z))
        if sym is None:                     # unknown element – invalidate
            return None, None
        rw.AddAtom(Chem.Atom(sym))

    # add bonds
    for i, j in combinations(range(n), 2):
        d = (data.pos[i] - data.pos[j]).norm().item()
        order = _bond_order(_num2sym[int(data.z[i])],
                            _num2sym[int(data.z[j])], d)
        if order:
            rw.AddBond(i, j, _bondtype[order])

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.KekulizeException, ValueError):
        return None, None

    return mol, Chem.MolToSmiles(mol, canonical=True)


def validity_and_uniqueness(molecules):
    """
    Compute validity and uniqueness exactly as in Hoogeboom et al.
    Args
    ----
    molecules : list[torch_geometric.data.Data]
        output of EDMSampler.sample()

    Returns
    -------
    valid_smiles : list[str]
        SMILES strings for the valid subset.
    validity     : float     (#valid / #requested)
    uniqueness   : float     (#unique / #valid)   (0 when no valid mols)
    """
    smiles = []
    for data in molecules:
        _, smi = build_rdkit_mol(data)
        if smi is not None:
            smiles.append(smi)

    validity = len(smiles) / len(molecules) if molecules else 0.0
    unq = set(smiles)
    uniqueness = len(unq) / len(smiles) if smiles else 0.0
    return list(unq), validity, uniqueness


if __name__ == '__main__':
    pass