property_names = {
    0: ('Dipole moment (μ)', 'D'),
    1: ('Isotropic polarizability (α)', 'a₀³'),
    2: ('HOMO energy', 'eV'),
    3: ('LUMO energy', 'eV'),
    4: ('HOMO-LUMO gap', 'eV'),
    5: ('Electronic spatial extent ⟨R²⟩', 'a₀²'),
    6: ('ZPVE', 'eV'),
    7: ('U₀', 'eV'),
    8: ('U', 'eV'),
    9: ('H', 'eV'),
    10: ('G', 'eV'),
    11: ('Heat capacity', 'cal/mol·K')
}

property_names_safe = {
    0: 'dipole moment',
    1: 'isotropic polarizability',
    2: 'HOMO energy',
    3: 'LUMO energy',
    4: 'HUMO LUMO gap',
    5: 'Electronic spatial extent',
    6: 'ZPVE',
    7: 'U0',
    8: 'U',
    9: 'H',
    10: 'H',
    11: 'Heat capacity'
}

atomic_number_to_symbol = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F'
}