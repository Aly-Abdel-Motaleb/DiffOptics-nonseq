"""MODULE 6 - SOLUTION. Differentiable irradiance: dI/dc."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_setup import build_scene, save_dir
import diffoptics as do

lens, wavelength, o_pt = build_scene(requires_grad=True)
out = save_dir("m06_out")

M = 30
R_src = 8.0


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


def render():
    return lens.render(sample_pointsrc_ray(random=True))


if __name__ == "__main__":
    I = render()
    I.sum().backward()
    # ~0.0 expected: total energy conserved, curvature only redistributes it.
    print("d(sum I)/dc =", lens.surfaces[0].c.grad.item())

    lm = do.LM(lens, ["surfaces[0].c"], 1e-2, option="diag")
    JI = lm.jacobian(lambda: lens.render(sample_pointsrc_ray(random=True))).squeeze()

    J = JI.abs().cpu().detach().numpy()
    print("Jacobian shape:", J.shape, " max|dI/dc| =", J.max())
    plt.imsave(out + "J_dIdc.png", J, cmap="inferno")
    print("saved to", out)
