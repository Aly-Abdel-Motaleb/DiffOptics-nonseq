"""MODULE 1 - SOLUTION. Point source ray generation."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_setup import build_scene
import diffoptics as do

lens, wavelength, o_pt = build_scene()

M = 11
R_src = 8.0


def sample_pointsrc_ray():
    x = torch.linspace(-R_src, R_src, M)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    targets = torch.stack((X, Y, torch.zeros_like(X)), axis=-1).reshape(-1, 3)
    o = o_pt.expand_as(targets).clone()
    d = do.normalize(targets - o_pt)
    return do.Ray(o, d, wavelength)


if __name__ == "__main__":
    ray = sample_pointsrc_ray()
    print("origins shape :", ray.o.shape, "-> all equal?", bool((ray.o == o_pt).all()))
    print("dirs shape    :", ray.d.shape)
    print("dirs unit-norm:", torch.allclose(ray.d.norm(dim=-1), torch.ones(ray.d.shape[0])))

    d = ray.d.detach().numpy()
    plt.scatter(d[:, 0], d[:, 1], s=8)
    plt.gca().set_aspect("equal")
    plt.title("point-source ray directions (dx, dy)")
    plt.xlabel("dx"); plt.ylabel("dy")
    plt.show()
