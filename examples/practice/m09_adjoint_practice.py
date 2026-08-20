"""
MODULE 9 - Adjoint back-propagation (the paper's memory trick).

Plain autograd stores the whole ray-tracing graph for every ray, so memory grows
with ray count -- the bottleneck for Deep Lens designs with millions of rays.
dO's `do.Adjoint` (diffoptics/solvers.py:268) computes the SAME gradients but in
three cheap stages, decoupling the optics graph from the loss/network graph:

  (1) forward render with NO autodiff  -> primal image I
  (2) backprop the loss only to the image -> I.grad   (small graph)
  (3) for each ray batch, VJP from render output using I.grad as the seed
      -> gradients w.r.t. the optical parameters, accumulated batch by batch.

Only one batch's graph is alive at a time -> memory ~ constant in #batches.
This is `backprop_compare.py`, adapted to CPU: we don't time GPU memory, we just
prove the adjoint gradients EQUAL the plain-autograd gradients.

API:
    adj = do.Adjoint(lens, diff_names, network_func, render_batch_func, paras)
    L, Js = adj()          # Js[k] = gradient w.r.t. diff_names[k]

READ BEFORE (tick when done):
  [ ] diffoptics/solvers.py:268-321  Adjoint.__init__ + __call__ (3-stage backprop)
  [ ] diffoptics/solvers.py:323-327  _adjoint_batch (per-batch VJP)
  [ ] examples/backprop_compare.py   baseline vs adjoint (GPU timing/memory version)

Run: python m09_adjoint_practice.py
Expect: baseline vs adjoint loss + gradients match to ~6 digits.
"""
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do
from common_setup import build_scene

DIFF_NAMES = ["surfaces[0].c", "surfaces[1].ai"]
R_src = 8.0
M = 30


def fresh_lens():
    lens, wl, o_pt = build_scene()
    lens.surfaces[0].c = torch.Tensor(np.array(1 / 50.0)); lens.surfaces[0].c.requires_grad = True
    lens.surfaces[1].ai = torch.zeros(2); lens.surfaces[1].ai.requires_grad = True
    return lens, wl, o_pt


def make_rays(o_pt, wl, n_batches=3):
    def one():
        x = torch.linspace(-R_src, R_src, M)
        X, Y = torch.meshgrid(x, x, indexing="ij")
        t = torch.stack((X, Y, torch.zeros_like(X)), axis=-1).reshape(-1, 3)
        return do.Ray(o_pt.expand_as(t).clone(), do.normalize(t - o_pt), wl)
    return [one() for _ in range(n_batches)]


torch.manual_seed(0)
_lens0, _wl0, _o0 = build_scene()
I_ref = torch.rand(*_lens0.film_size)


def network_func(I):
    return ((I - I_ref) ** 2).mean()


if __name__ == "__main__":
    # ---- baseline: plain autograd over all ray batches ----
    lens, wl, o_pt = fresh_lens()
    rays = make_rays(o_pt, wl)
    I = 0.0
    for r in rays:
        I = I + lens.render(r)
    L = network_func(I)
    L.backward()
    g_c = float(lens.surfaces[0].c.grad)
    g_ai = lens.surfaces[1].ai.grad.tolist()
    print(f"baseline  L={L.item():.6f}  d/dc={g_c:.4f}  d/dai={g_ai}")

    # ---- adjoint: same gradients, batch-by-batch ----
    lens2, wl2, o_pt2 = fresh_lens()
    rays2 = make_rays(o_pt2, wl2)
    # TODO 1: build the Adjoint object.
    #   adj = do.Adjoint(lens2, DIFF_NAMES, network_func, lens2.render, rays2)
    adj = ...

    # TODO 2: call it to get (loss, Js).
    #   L2, Js = adj()
    L2, Js = ...

    print(f"adjoint   L={L2:.6f}  d/dc={float(Js[0]):.4f}  d/dai={Js[1].tolist()}")

    assert abs(L2 - L.item()) < 1e-6
    assert abs(float(Js[0]) - g_c) < 1e-2
    print("OK: adjoint gradients match autograd.")
