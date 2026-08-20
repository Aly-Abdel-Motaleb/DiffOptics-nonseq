"""MODULE 5 - SOLUTION. Monte-Carlo jittered sampling + spp averaging."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_setup import build_scene, save_dir
import diffoptics as do

lens, wavelength, o_pt = build_scene()
out = save_dir("m05_out")

M = 30
R_src = 8.0


def sample_pointsrc_ray(random=False):
    x = torch.linspace(-R_src, R_src, M)
    X, Y = torch.meshgrid(x, x, indexing="ij")

    if random:
        p = 2 * R_src / M
        X = X + p * (torch.rand_like(X) - 0.5)
        Y = Y + p * (torch.rand_like(Y) - 0.5)

    targets = torch.stack((X, Y, torch.zeros_like(X)), axis=-1).reshape(-1, 3)
    o = o_pt.expand_as(targets).clone()
    d = do.normalize(targets - o_pt)
    return do.Ray(o, d, wavelength)


# The "proper" MC sampler: uniform points on a DISK via concentric mapping
# (same tool render_psf.py uses). No grid at all -> no square-grid bias, and it
# matches a round emission footprint. See diffoptics/basics.py:116.
sampler = do.Sampler()


def sample_pointsrc_ray_disk(N=M * M):
    u = torch.rand(N)
    v = torch.rand(N)
    px, py = sampler.concentric_sample_disk(u, v)   # unit disk
    X, Y = R_src * px, R_src * py
    targets = torch.stack((X, Y, torch.zeros_like(X)), axis=-1)
    o = o_pt.expand_as(targets).clone()
    d = do.normalize(targets - o_pt)
    return do.Ray(o, d, wavelength)


def render_mc(spp=1, random=True):
    I = torch.zeros(*lens.film_size)
    for _ in range(spp):
        I = I + lens.render(sample_pointsrc_ray(random=random))
    return I / spp


def render_disk(spp=32):
    I = torch.zeros(*lens.film_size)
    for _ in range(spp):
        I = I + lens.render(sample_pointsrc_ray_disk())
    return I / spp


if __name__ == "__main__":
    I_det = render_mc(spp=1, random=False).cpu().detach().numpy()
    I_mc = render_mc(spp=32, random=True).cpu().detach().numpy()
    I_disk = render_disk(spp=32).cpu().detach().numpy()

    plt.imsave(out + "I_deterministic.png", I_det, cmap="inferno")
    plt.imsave(out + "I_montecarlo.png", I_mc, cmap="inferno")
    plt.imsave(out + "I_disk.png", I_disk, cmap="inferno")
    print(f"deterministic     I.max={I_det.max():.3f}")
    print(f"monte-carlo grid  I.max={I_mc.max():.3f}  (spp=32, jittered grid)")
    print(f"monte-carlo disk  I.max={I_disk.max():.3f}  (spp=32, concentric disk)")
    print("saved to", out, "-- both MC images smooth; disk has a round footprint")
