"""MODULE 4 - SOLUTION. Deterministic irradiance render (shows aliasing)."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_setup import build_scene, save_dir
import diffoptics as do

lens, wavelength, o_pt = build_scene()
out = save_dir("m04_out")

R_src = 8.0


def sample_pointsrc_ray(M):
    x = torch.linspace(-R_src, R_src, M)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    targets = torch.stack((X, Y, torch.zeros_like(X)), axis=-1).reshape(-1, 3)
    o = o_pt.expand_as(targets).clone()
    d = do.normalize(targets - o_pt)
    return do.Ray(o, d, wavelength)


def render(M):
    ray = sample_pointsrc_ray(M)
    return lens.render(ray)


if __name__ == "__main__":
    for M in (20, 60):
        I = render(M).cpu().detach().numpy()
        print(f"M={M:3d}  rays={M*M:5d}  I.sum={I.sum():.1f}  I.max={I.max():.3f}")
        plt.imsave(out + f"I_M{M}.png", I, cmap="inferno")
    print("saved to", out, "-- look for grid/dot aliasing artifacts")
