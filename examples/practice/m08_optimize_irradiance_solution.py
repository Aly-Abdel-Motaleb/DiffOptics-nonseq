"""MODULE 8 - SOLUTION. Full MC differentiable-illumination optimization."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_setup import build_scene, save_dir, FILM
import diffoptics as do

lens, wavelength, o_pt = build_scene(c=0.028, requires_grad=True)
out = save_dir("m08_out")

# Distant point source: rays arrive near-collimated so the lens forms a spot on
# the receiver whose SIZE varies smoothly with curvature c.
# NOTE on regime: exactly at focus the spot collapses to a few pixels -> the loss
# becomes a razor-thin well on a flat plateau (that IS the aliasing pathology this
# whole exercise is about). So we stay in the DEFOCUSED regime (c below focus),
# where the loss-vs-c bowl is smooth and convex -> reliable convergence.
Z0 = 500.0
o_pt = torch.Tensor([0.0, 0.0, -Z0])

M = 40
R_src = 9.0
SPP = 8


def sample_pointsrc_ray(random=True):
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


def render(spp=SPP):
    I = torch.zeros(*lens.film_size)
    for _ in range(spp):
        I = I + lens.render(sample_pointsrc_ray())
    return I / spp


def make_target(c_target=0.036, spp=64):
    """Reference illumination = irradiance produced by a KNOWN curvature.
    Optimization then tries to recover c_target starting from a different c.
    This guarantees the target is physically reachable, so the loss is
    genuinely sensitive to c and actually decreases."""
    c_save = lens.surfaces[0].c
    with torch.no_grad():
        lens.surfaces[0].c = torch.Tensor(np.array(c_target))
        I = render(spp=spp)
        I = I / I.sum()
    lens.surfaces[0].c = c_save   # restore the differentiable leaf
    return I


if __name__ == "__main__":
    I_ref = make_target()
    plt.imsave(out + "I_target.png", I_ref.numpy(), cmap="inferno")

    I0 = render().detach()
    plt.imsave(out + "I_start.png", I0.numpy(), cmap="inferno")

    optimizer = torch.optim.Adam([lens.surfaces[0].c], lr=2e-3, amsgrad=True)

    for it in range(151):
        I = render()
        I = I / I.sum() * I_ref.sum()
        L = torch.mean((I - I_ref) ** 2)

        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        if it % 25 == 0:
            print(f"it={it:3d}  loss={L.item():.4e}  c={lens.surfaces[0].c.item():.5f}"
                  f"   (target c=0.036)")

    I_final = render().detach()
    plt.imsave(out + "I_final.png", I_final.numpy(), cmap="inferno")
    print("saved to", out)
