"""
Shared boilerplate for all practice phases.

Builds ONE thing: a point source + a single spherical (plano-convex) lens + a
receiver plane. Every phase imports `build_scene()` so you only focus on the new
concept of that phase, not on re-typing the lens each time.

Geometry (all along +z, units mm):

    point source            spherical lens                receiver
    o=(0,0,-z0)   --->    surf0 (curved)  surf1 (flat)   at z=d_sensor
                          c=1/50, N-BK7, thickness 6.5

Read `diffoptics/optics.py`:
  - Lensgroup.load          (attach surfaces + materials)
  - Aspheric (k=0,ai=None)  == a spherical surface, c = 1/R_curvature
  - render / trace          (used in later phases)
"""
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do  # noqa: E402


# ---- fixed scene parameters (shared across phases) ----
WAVELENGTH = 532.8          # [nm]
Z0 = 40.0                   # point source sits at z = -Z0  [mm]
R_LENS = 12.7               # lens semi-diameter [mm]
D_SENSOR = 40.0             # receiver plane at z = D_SENSOR [mm]
R_RECEIVER = 12.7           # receiver half-size [mm]
FILM = [256, 256]           # receiver pixel grid


def build_scene(c=1 / 50.0, requires_grad=False, device=torch.device("cpu")):
    """Return (lens, wavelength_tensor, o_pt) fully configured.

    c              : curvature of the first (curved) surface = 1/R.
    requires_grad  : if True, make surfaces[0].c a differentiable leaf.
    """
    lens = do.Lensgroup(device=device)

    surfaces = [
        do.Aspheric(R_LENS, 0.0, c=c, device=device),   # curved front face
        do.Aspheric(R_LENS, 6.5, c=0.0, device=device),  # flat back face
    ]
    materials = [
        do.Material("air"),
        do.Material("N-BK7"),
        do.Material("air"),
    ]
    lens.load(surfaces, materials)

    # receiver / sensor config used by lens.render()
    lens.d_sensor = D_SENSOR
    lens.r_last = R_LENS
    lens.film_size = FILM
    lens.pixel_size = 2 * R_RECEIVER / FILM[0]   # [mm/pixel], scalar

    if requires_grad:
        lens.surfaces[0].c = torch.Tensor(np.array(c))
        lens.surfaces[0].c.requires_grad = True

    wavelength = torch.Tensor([WAVELENGTH]).to(device)
    o_pt = torch.Tensor([0.0, 0.0, -Z0]).to(device)   # point source location
    return lens, wavelength, o_pt


def save_dir(name):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    os.makedirs(d, exist_ok=True)
    return d + os.sep
