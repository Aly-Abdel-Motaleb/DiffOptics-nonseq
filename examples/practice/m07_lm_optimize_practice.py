"""
MODULE 7 - Classical lens optimization with Levenberg-Marquardt.

Modules 6/8 used autograd + Adam. dO also ships a classical **LM optimizer**
(diffoptics/solvers.py:81) that uses the autodiff Jacobian internally -- the
standard tool in optical design. Here we minimize the RMS spot of a real
Thorlabs asphere by tuning its surface parameters. This is `spherical_aberration.py`.

Key API (LM):
    lm = do.LM(lens, diff_names, damping, option='diag')
    out = lm.optimize(func, func_yref_y, maxit=..., record=True)
  - func():          returns y, the quantity to drive (here sensor hit points ps)
  - func_yref_y(y):  returns (y_ref - y). To drive spots to the axis, y_ref = 0,
                     so this is `lambda y: 0.0 - y`  -> minimizes sum(ps^2) = spot.
  - diff_names:      strings naming the leaves to optimize, e.g. 'surfaces[0].c',
                     'surfaces[0].k', 'surfaces[0].ai'  (curvature, conic, asphere coeffs).
  - out['ls']:       recorded loss per iteration.

READ BEFORE (tick when done):
  [ ] diffoptics/solvers.py:132-...  LM.optimize (func, func_yref_y)
  [ ] diffoptics/solvers.py:93-130   LM.jacobian (used inside optimize)
  [ ] examples/spherical_aberration.py  the workflow this mirrors
  [ ] m07_nikon_GUIDE.md             (after) multi-surface advanced version

Run: python m07_lm_optimize_practice.py
Expect: spot_before/after.png + loss.png in ./m07_out/, loss drops ~100x.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do
from common_lensfile import load_lens, green
from common_setup import save_dir

out = save_dir("m07_out")
device = torch.device("cpu")

lens = load_lens("thorlabs_acl", device=device)
wavelength = green(device)
R = 15.0   # ray bundle radius [mm], < aperture (25) so rays stay valid

RMS = lambda ps: torch.sqrt(torch.mean(torch.sum(ps ** 2, axis=-1)))


def render():
    # collimated on-axis bundle -> sensor hit points (x,y)
    ray = lens.sample_ray(wavelength, M=31, R=R)
    return lens.trace_to_sensor(ray)[..., :2]


if __name__ == "__main__":
    ps0 = render()
    print(f"initial RMS = {RMS(ps0)*1e3:.2f} um")
    lim = 50e-3
    lens.spot_diagram(ps0, xlims=[-lim, lim], ylims=[-lim, lim],
                      savepath=out + "spot_before.png", show=False)

    # TODO 1: name the leaves to optimize: curvature c, conic k, asphere coeffs ai
    #   of the first surface.
    diff_names =   ['surfaces[0].c']

    # TODO 2: build the LM optimizer (damping 1e-4, option='diag').
    #   lm = do.LM(lens, diff_names, 1e-4, option='diag')
    lm =  do.LM(lens,diff_names, 1e-4, option='diag')

    # TODO 3: run it. Drive spots to the axis: y_ref - y with y_ref = 0.
    #   res = lm.optimize(render, lambda y: 0.0 - y, maxit=100, record=True)
    res = lm.optimize(render , lambda y: 0.0 - y, maxit=100, record=True)

    ps1 = render()
    print(f"final   RMS = {RMS(ps1)*1e3:.2f} um")
    lens.spot_diagram(ps1, xlims=[-lim, lim], ylims=[-lim, lim],
                      savepath=out + "spot_after.png", show=False)

    plt.figure()
    plt.semilogy(res["ls"], "-o")
    plt.xlabel("iteration"); plt.ylabel("loss")
    plt.savefig(out + "loss.png", bbox_inches="tight")
    print("saved to", out)
