"""MODULE 9 - SOLUTION. Adjoint back-prop equals autograd (backprop_compare, CPU)."""
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

    lens2, wl2, o_pt2 = fresh_lens()
    rays2 = make_rays(o_pt2, wl2)
    adj = do.Adjoint(lens2, DIFF_NAMES, network_func, lens2.render, rays2)
    L2, Js = adj()
    print(f"adjoint   L={L2:.6f}  d/dc={float(Js[0]):.4f}  d/dai={Js[1].tolist()}")

    assert abs(L2 - L.item()) < 1e-6
    assert abs(float(Js[0]) - g_c) < 1e-2
    print("OK: adjoint gradients match autograd.")
