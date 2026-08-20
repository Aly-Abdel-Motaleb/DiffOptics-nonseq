"""
MODULE 1 - Build rays from a POINT SOURCE (not an LED, not collimated).

Goal: understand `do.Ray(o, d, wavelength)`.
  - o : origin of every ray  -> ALL rays start at the same point (the source)
  - d : direction (unit vec)  -> each ray points at a different target
  - a point source = one origin, many directions fanning out.

Contrast with the library's `lens.sample_ray(...)`: that samples origins spread
across the lens plane (collimated / field angle). A point source is the opposite:
one origin, spread of directions.

READ BEFORE (tick when done):
  [ ] diffoptics/basics.py:36-55   Ray.__init__ + __call__(t) = o + t*d
  [ ] diffoptics/basics.py:293     normalize
  [ ] common_setup.py              build_scene (what it returns)

Run: python m01_primitives_rays_practice.py
Expect: printed shapes + a scatter of ray directions.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_setup import build_scene, R_LENS
import diffoptics as do

lens, wavelength, o_pt = build_scene()

M = 11          # rays per axis -> M*M rays total
R_src = 8.0     # half-size of the target grid on the z=0 plane [mm]


def sample_pointsrc_ray():
    """Return a do.Ray whose origins are all o_pt and whose directions fan out
    toward an M x M grid of target points on the z = 0 plane."""
    # TODO 1: make a 1D grid of M coordinates in [-R_src, R_src] with torch.linspace
    x = torch.linspace(-R_src,R_src,M)
    X,Y = torch.meshgrid(x,x,indexing='ij')

    # TODO 2: meshgrid it into X, Y (use indexing="ij"), then stack into
    #         `targets` of shape (M*M, 3) where the z column is 0.
    #         hint: torch.stack((X, Y, zeros), axis=-1).reshape(-1, 3)
    targets = torch.stack((X,Y,torch.zeros_like(x)) , axis=-1).reshape(-1,3)

    # TODO 3: origins `o` = o_pt repeated for every target (same shape as targets)
    #         hint: o_pt.expand_as(targets).clone()
    o = o_pt.expand_as(targets).clone

    # TODO 4: directions `d` = unit vector from source to each target.
    #         hint: do.normalize(targets - o_pt)
    d = do.normalize(targets - o_pt)

    return do.Ray(o, d, wavelength)


if __name__ == "__main__":
    ray = sample_pointsrc_ray()
    print("origins shape :", ray.o.shape, "-> all equal?", bool((ray.o == o_pt).all()))
    print("dirs shape    :", ray.d.shape)
    print("dirs unit-norm:", torch.allclose(ray.d.norm(dim=-1), torch.ones(ray.d.shape[0])))

    # visualize direction spread (dx, dy of each ray)
    d = ray.d.detach().numpy()
    plt.scatter(d[:, 0], d[:, 1], s=8)
    plt.gca().set_aspect("equal")
    plt.title("point-source ray directions (dx, dy)")
    plt.xlabel("dx"); plt.ylabel("dy")
    plt.show()
