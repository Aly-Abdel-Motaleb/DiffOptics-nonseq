"""
MODULE 8 (capstone) - Inverse problem: recover a MISALIGNMENT from an image.

This ties everything together, the `misalignment_point.py` idea on CPU:
  point source  +  real lens  +  differentiable irradiance (render)  +  LM
to back-engineer unknown scene/pose parameters from a measured image.

Story: a point source illuminates a Thorlabs singlet onto a sensor. The true rig
has an unknown source position and sensor distance. We have one measured image
`I_mea`. We optimize the pose parameters so the *rendered* image matches `I_mea`.

Differentiable pose parameters live on the Lensgroup as plain tensors
(optics.py:44): `d_sensor`, `theta_x/y/z`, and here a custom `light_o` (source
position) that our `render()` reads. Naming them in `do.LM(...)` differentiates them.
`lens.trace` auto-calls `update()` when a pose angle needs grad (optics.py:807).

Here we recover `d_sensor` (focus distance) and `light_o` (source position).
Note: not every parameter is equally observable -- a centered source makes tilt
nearly invisible (the focus distance absorbs it). We pick a well-conditioned pair.

READ BEFORE (tick when done):
  [ ] diffoptics/optics.py:44-63     Lensgroup.__init__ (pose leaves: d_sensor, theta_*, shift)
  [ ] diffoptics/optics.py:807-830   trace -> update() when a pose needs grad
  [ ] diffoptics/solvers.py:132-...  LM.optimize (residual = I_mea - y)
  [ ] examples/misalignment_point.py the real-data version this mirrors

Run: python m08_misalignment_practice.py
Expect: I_mea / I_init / I_final .png in ./m08b_out/, recovered params printed.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do
from common_lensfile import load_lens
from common_setup import save_dir

out = save_dir("m08b_out")
device = torch.device("cpu")

lens = load_lens("thorlabs_la", device=device)
lens.r_last = torch.Tensor([12.7])
R = lens.surfaces[0].r

N = 128
lens.pixel_size = 3.45e-3 * 8          # [mm]
lens.film_size = [N, N]
wavelength = torch.Tensor([622.5]).to(device)

R_in = 1.2 * R
M = 400
lens.light_o = torch.Tensor([0.0, 0.0, -650.0]).to(device)   # source position hook


def sample_ray(light_o):
    ox, oy = torch.meshgrid(
        torch.linspace(-R_in, R_in, M),
        torch.linspace(-R_in, R_in, M), indexing="ij")
    valid = (ox ** 2 + oy ** 2) < (0.95 * R) ** 2
    o = torch.stack((ox, oy, -torch.ones_like(ox)), axis=-1)
    d = do.normalize(torch.stack((ox, oy, torch.zeros_like(ox)), axis=-1)
                     - light_o[None, None, ...])
    return do.Ray(o[valid], d[valid], wavelength)


def render():
    I = lens.render(sample_ray(lens.light_o))
    return N ** 2 * I / I.sum()        # normalize to constant total energy


if __name__ == "__main__":
    # ---- synthesize the "measurement" at the TRUE (unknown) pose ----
    lens.light_o = torch.Tensor([3.0, 0.0, -650.0])
    lens.d_sensor = torch.Tensor([57.0])
    I_mea = render().detach()
    plt.imsave(out + "I_mea.png", I_mea.numpy(), cmap="gray")

    # ---- reset to a WRONG guess ----
    lens.light_o = torch.Tensor([0.0, 0.0, -650.0])
    lens.d_sensor = torch.Tensor([56.0])
    I_init = render().detach()
    plt.imsave(out + "I_init.png", I_init.numpy(), cmap="gray")
    print(f"loss(init) = {float(((I_init-I_mea)**2).mean()):.4e}")

    # ---- recover pose with LM ----
    # TODO 1: name the pose parameters to recover: sensor distance + source position.
    #   diff_names = ['d_sensor', 'light_o']
    diff_names = ...

    # TODO 2: build LM with damping 1e-2 (keeps the weakly-observed source depth
    #         from wandering), option='diag'.
    #   lm = do.LM(lens, diff_names, 1e-2, option='diag')
    lm = ...

    # TODO 3: optimize render() to match I_mea. Residual = I_mea - y.
    #   res = lm.optimize(render, lambda y: I_mea - y, maxit=40, record=True)
    res = ...

    I_final = render().detach()
    plt.imsave(out + "I_final.png", I_final.numpy(), cmap="gray")
    print(f"loss(final) = {res['ls'][-1]:.4e}")
    print(f"recovered light_o  = {[round(v,3) for v in lens.light_o.tolist()]}  (true [3,0,-650])")
    print(f"recovered d_sensor = {float(lens.d_sensor):.3f}                (true 57.0)")
    print("saved to", out)
