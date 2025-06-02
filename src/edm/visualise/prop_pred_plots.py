import matplotlib.pyplot as plt



def plot_predictions(target_idx, metrics, save_path=None):
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

    prop_name, unit = property_names.get(target_idx, (f'Property {target_idx}', ''))
    preds = metrics['preds_original']
    targets = metrics['targets_original']

    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel(f'True {prop_name} ({unit})')
    plt.ylabel(f'Predicted {prop_name} ({unit})')
    plt.title(f'Model Performance on {prop_name}\nTest MAE: {metrics["mae"]:.4f} {unit}')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
