"""
Just FYI, I handed most of this plotting code off to ChatGPT, since I 
didn't really feel like learning PyVista. There might be all kinds of 
annoying things in here like functions that exist elsewhere, bad behaviour,
unused functions, and more. I sanity checked enought to be satisfied that 
it works roughly like it should. Steal at your own risk.
"""

import os
import platform
import imageio
import matplotlib.colors as mcolors
import numpy as np
import pyvista as pv

from itertools import combinations
from pathlib import Path
from torch_geometric.data import Data
from edm.utils.chem_utils import _bond_order, _num2sym


def is_headless():
    return not os.environ.get("DISPLAY") and platform.system() != "Darwin"

if is_headless():
    # Makes pyvista gui free on systems that don't support it
    # ... at least that is intended behaviour. 
    pv.start_xvfb()
    tmp_dir = f"/tmp/xdg_runtime_{os.getuid()}"
    Path(tmp_dir).mkdir(exist_ok=True)
    os.environ["XDG_RUNTIME_DIR"] = tmp_dir


ATOM_COLOUR = {
    'H': "#DAD7D7",   # white
    'C': "#7A7575",   # mid-grey
    'N': "#3352EB",   # blue
    'O': "#DE3333",   # red
    'F': "#D77936",   # green
}

ATOM_RADIUS = {
    'H': 0.46,
    'C': 0.77,
    'N': 0.77,
    'O': 0.77,
    'F': 0.77,
}

DATASET_INFO = {         # minimal dict so we can reuse the paper’s functions
    'name'         : 'qm9',
    'atom_decoder' : ['H', 'C', 'N', 'O', 'F'],
    'atom_encoder' : {s: i for i, s in enumerate(['H', 'C', 'N', 'O', 'F'])},
    'colors_dic'   : [ATOM_COLOUR[s] for s in ['H', 'C', 'N', 'O', 'F']],
    'radius_dic'   : [ATOM_RADIUS[s] for s in ['H', 'C', 'N', 'O', 'F']],
}



def save_molecule_png(
    data: Data,
    outfile: str,
    elev: int = 30,
    azim: int = 135,
    spheres_3d: bool = True,
    dpi: int = 300,
):
    """Render one PyG `Data` molecule to PNG using PyVista/VTK."""
    pos = data.pos.detach().cpu().numpy()
    z   = data.z.detach().cpu().numpy()

    # ---------------- build the scene ----------------
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(int(4 * dpi), int(4 * dpi)),
        lighting=None,                # we’ll add custom lights
        polygon_smoothing=True,
    )
    plotter.set_background("black")

    _plot_molecule_pyvista(plotter, pos, z, spheres_3d)

    # ---------------- camera & lights ----------------
    centre = pos.mean(axis=0)
    max_extent = np.abs(pos - centre).max()
    cam_dist = 6 * max_extent + 1.0
    cam_pos = _cartesian_from_spherical(cam_dist, elev, azim)

    plotter.camera_position = [cam_pos, centre, (0, 0, 1)]

    # key + fill + rim lights for depth cues
    lights = [
        pv.Light(
            position=_cartesian_from_spherical(cam_dist * 1.2, elev, azim),
            focal_point=centre,
            intensity=1.0,
            light_type="scene light",
        ),
        pv.Light(
            position=_cartesian_from_spherical(cam_dist * 1.2, elev + 30, azim + 120),
            focal_point=centre,
            intensity=0.4,
            light_type="scene light",
        ),
        pv.Light(
            position=_cartesian_from_spherical(cam_dist * 1.2, elev - 20, azim - 130),
            focal_point=centre,
            intensity=0.3,
            light_type="scene light",
        ),
    ]
    for L in lights:
        plotter.add_light(L)

    plotter.enable_depth_peeling(number_of_peels=8)
    plotter.render()

    # ---------------- output ----------------
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(outfile), transparent_background=False)
    plotter.close()

    # same optional “brightness pop” you used before
    img = imageio.imread(outfile)
    img = np.clip(img * 1.4, 0, 255).astype("uint8")
    imageio.imwrite(outfile, img)



def _cartesian_from_spherical(r, elev_deg, azim_deg):
    elev, azim = np.deg2rad([elev_deg, azim_deg])
    x = r * np.cos(elev) * np.cos(azim)
    y = r * np.cos(elev) * np.sin(azim)
    z = r * np.sin(elev)
    return x, y, z



def _color_to_rgb_alpha(col):
    """
    Accept anything Matplotlib can parse (“C7”, “royalblue”, 3-tuple, …)
    **or** 6/8-digit hex such as '#AABBCC' or '#AABBCCDD'.  
    Returns (r, g, b)  and  alpha   in 0-1 float range.
    """
    if isinstance(col, tuple):            # already numeric?
        if len(col) == 3:
            return col, 1.0
        if len(col) == 4:
            return col[:3], col[3]
        raise ValueError(f"Bad colour tuple: {col}")

    if col.startswith("#"):
        hexcode = col.lstrip("#")
        if len(hexcode) == 6:
            r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (0, 2, 4))
            return (r, g, b), 1.0
        if len(hexcode) == 8:
            r, g, b, a = (int(hexcode[i : i + 2], 16) / 255 for i in (0, 2, 4, 6))
            return (r, g, b), a
        raise ValueError(f"Bad hex colour {col}")

    # Fallback: let Matplotlib interpret anything else
    r, g, b, a = mcolors.to_rgba(col)
    return (r, g, b), a


def _infer_bonds(pos: np.ndarray, z: np.ndarray):
    """Return list[(i, j, order)] with the same rules as stability()."""
    bonds = []
    n = pos.shape[0]
    for i, j in combinations(range(n), 2):
        order = _bond_order(_num2sym[int(z[i])], _num2sym[int(z[j])],
                            np.linalg.norm(pos[i] - pos[j]))
        if order:
            bonds.append((i, j, order))
    return bonds


def _plot_molecule_pyvista(plotter, pos, z, spheres_3d=True):
    colours = np.array(DATASET_INFO["colors_dic"])
    radii   = np.array(DATASET_INFO["radius_dic"])
    atom_idx = np.array(
        [DATASET_INFO["atom_encoder"][_num2sym[int(zi)]] for zi in z]
    )

    # ---------- bonds (as tubes around a VTK polyline) ----------
    tube_radius = 0.1                      # Å; looks good relative to 0.4*R
    for i, j, order in _infer_bonds(pos, z):
        line = pv.Line(pos[i], pos[j])
        tube = line.tube(radius=tube_radius * np.sqrt(order),
                         n_sides=24, capping=True)
        plotter.add_mesh(
            tube,
            color=(0.8, 0.8, 0.8),
            specular=0.6,
            specular_power=18,
            smooth_shading=True,
        )

    # ---------- atoms (unchanged) ----------
    for p, idx in zip(pos, atom_idx):
        rgb, alpha = _color_to_rgb_alpha(colours[idx])
        sph = pv.Sphere(
            center=p,
            radius=0.5 * radii[idx],
            theta_resolution=64,
            phi_resolution=64,
        )
        plotter.add_mesh(
            sph,
            color=rgb,
            opacity=alpha,
            specular=0.8,
            specular_power=30,
            smooth_shading=True,
        )


