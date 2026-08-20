"""
MODULE 6 - DIFFERENTIABLE irradiance: gradient of the image w.r.t. the lens.

The whole point of dO: irradiance I is differentiable w.r.t. optical parameters.
Here we get dI/dc, the sensitivity of every receiver pixel to the front-surface
curvature c. This is what an optimizer later uses to reshape the illumination.

Two ways to get gradients (both in the repo):
  A) torch autograd directly: I.sum().backward() -> c.grad  (scalar-out shortcut)
  B) do.LM(...).jacobian(func): full per-pixel Jacobian dI/dc  (solvers.py:93)

We make c a differentiable leaf via build_scene(requires_grad=True).

READ BEFORE (tick when done):
  [ ] diffoptics/solvers.py:5-31     Optimization base (diff_parameters_names -> tensors)
  [ ] diffoptics/solvers.py:93-130   LM.jacobian
  [ ] examples/autodiff.py:103-110   LM(...).jacobian on render, working

Run: python m06_jacobian_practice.py
Expect: J_dIdc.png in ./m06_out/, printed grad + Jacobian shape.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_setup import build_scene, save_dir
import diffoptics as do

# c is now a differentiable leaf tensor
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
    # ---- A) scalar shortcut: d(total energy)/dc ----
    I = render()
    # TODO 1: backprop the scalar I.sum() to populate c.grad.
    #         hint: I.sum().backward()
    I.sum().backward()  
    # NOTE: this prints ~0.0 -- expected! Total energy is conserved (splat weights
    # sum to 1 per ray), so d(sum I)/dc = 0. Curvature REDISTRIBUTES energy, it does
    # not create it. That is exactly why the per-pixel Jacobian (below) is needed.
    print("d(sum I)/dc =", lens.surfaces[0].c.grad.item())

    # ---- B) full per-pixel Jacobian dI/dc via LM ----
    # TODO 2: build the LM helper targeting surfaces[0].c.
    #         hint: do.LM(lens, ['surfaces[0].c'], 1e-2, option='diag')
    lm = do.LM(lens, ['surfaces[0].c'], 1e-2, option='diag')

    # TODO 3: evaluate the Jacobian of the render function.
    #         hint: lm.jacobian(lambda: lens.render(sample_pointsrc_ray(random=True))).squeeze()
    JI = lm.jacobian(lambda: lens.render(sample_pointsrc_ray(random=True))).squeeze()

    J = JI.abs().cpu().detach().numpy()
    print("Jacobian shape:", J.shape, " max|dI/dc| =", J.max())
    plt.imsave(out + "J_dIdc.png", J, cmap="inferno")
    print("saved to", out)
