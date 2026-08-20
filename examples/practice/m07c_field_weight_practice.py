"""
MODULE 7c - Field weighting: where does the spot budget go?

`m07_nikon_GUIDE.md` exercise 3: "Swap the merit to weight the outer field more
heavily. Where does the spot budget go?"  This runs that experiment on the cheap
M07 asphere (seconds on CPU) instead of the 19-surface Nikon (minutes).

THE KEY POINT -- where the weights go:
    LM builds its Jacobian from `func()` alone (solvers.py:161) and only uses
    `func_yref_y` for the loss and the RHS `b = J.T @ r` (solvers.py:188).
    So weighting inside `lambda y: 0.0 - y` gives you `JtJ = J.T J` (unweighted)
    against `b = J.T W r` (weighted) -- mismatched normal equations, wrong step.

    Weight the RESIDUALS inside `func()` instead. Scaling row-block i by sqrt(w_i)
    makes J' = W^(1/2) J and r' = W^(1/2) r, so
        J'.T J' = J.T W J     and     J'.T r' = J.T W r
    which is exactly weighted Gauss-Newton. Free, and correct.

Two more traps handled below:
  - CENTERING. Off-axis spots sit at large y. `0.0 - y` would drive the image
    height to zero, i.e. fight the focal length instead of the aberration.
    Subtract the per-view centroid first (this is what lens.rms does internally,
    optics.py:187).
  - RAY COUNT. `ignore_invalid=True` drops vignetted rays, and LM's loss is a
    plain `torch.mean` over all stacked rows. So a view's effective weight is
    w * n_valid -- the outer field, which vignettes most, is silently
    UNDERweighted before you touch anything. Divide by n to fix.

Run: python m07c_field_weight_practice.py
Expect: on-axis RMS rises, outer-field RMS falls, sum of squares rises.
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

out = save_dir("m07c_out")
device = torch.device("cpu")

VIEWS = [0.0, 5.0, 10.0]          # field angles [deg]
R = 15.0                          # ray bundle radius [mm], < aperture (25)
M = 31                            # rays per side of the grid


def make_render(lens, wavelength, weights):
    """Returns (func, per_view_rms). `func` is what you hand to LM."""
    ws = np.asarray(weights, dtype=float)

    def views_ps():
        """Per-view centroid-subtracted sensor hit points."""
        pss = []
        for v in VIEWS:
            ray = lens.sample_ray(wavelength, view=v, M=M, R=R)
            ps = lens.trace_to_sensor(ray, ignore_invalid=True)[..., :2]
            pss.append(ps - ps.mean(0, keepdim=True))
        return pss

    def func():
        # sqrt(w) on the residuals -> weighted Gauss-Newton (see docstring).
        # /sqrt(n) undoes the accidental weighting by surviving ray count.
        return torch.vstack([
            np.sqrt(w / ps.shape[0]) * ps for ps, w in zip(views_ps(), ws)
        ])

    def per_view_rms():
        """UNWEIGHTED RMS per view [um]. The diagnostic -- never weight this,
        or you hide the very answer the exercise asks for."""
        with torch.no_grad():
            return np.array([
                float(torch.sqrt(torch.mean(torch.sum(ps ** 2, -1)))) * 1e3
                for ps in views_ps()
            ])

    return func, per_view_rms


def run(weights, label):
    lens = load_lens("thorlabs_acl", device=device)   # fresh lens; LM mutates it
    wavelength = green(device)
    func, per_view_rms = make_render(lens, wavelength, weights)

    before = per_view_rms()
    res = do.LM(lens, ["surfaces[0].c", "surfaces[0].k", "surfaces[0].ai"],
                1e-4, option="diag").optimize(func, lambda y: 0.0 - y,
                                              maxit=100, record=True)
    after = per_view_rms()

    print(f"\n=== {label}  weights={list(weights)} ===")
    print("  view[deg]   RMS before [um]   RMS after [um]")
    for v, b, a in zip(VIEWS, before, after):
        print(f"  {v:7.1f}   {b:14.2f}   {a:14.2f}")
    print(f"  sum of squares: {np.sum(before**2):.1f} -> {np.sum(after**2):.1f}")
    return before, after, res["ls"]


if __name__ == "__main__":
    b_uni, a_uni, ls_uni = run([1.0, 1.0, 1.0], "uniform")
    b_out, a_out, ls_out = run([1.0, 1.0, 9.0], "outer-field heavy")

    # NOTE: the two loss curves are NOT comparable to each other -- different
    # weights mean different objectives. Compare the per-view RMS tables.
    plt.figure()
    plt.plot(VIEWS, b_uni, "k-o", label="before")
    plt.plot(VIEWS, a_uni, "b-o", label="after, uniform")
    plt.plot(VIEWS, a_out, "r-o", label="after, outer x9")
    plt.xlabel("field angle [deg]"); plt.ylabel("RMS spot [um]")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(out + "rms_vs_field.png", bbox_inches="tight")

    plt.figure()
    plt.semilogy(ls_uni, "b-", label="uniform")
    plt.semilogy(ls_out, "r-", label="outer x9")
    plt.xlabel("iteration"); plt.ylabel("loss (own merit)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(out + "loss.png", bbox_inches="tight")
    print("\nsaved to", out)
