"""MODULE 8 (capstone) - SOLUTION. Recover a misalignment from a measured image."""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do
from common_lensfile import load_lens
from common_setup import save_dir

out = save_dir("m08b_out")
device = torch.device("cpu")

lens = load_lens("thorlabs_la", device=device)
lens.r_last = torch.Tensor([12.7])
R = lens.surfaces[0].r

N = 128
lens.pixel_size = 3.45e-3 * 8
lens.film_size = [N, N]
wavelength = torch.Tensor([622.5]).to(device)

R_in = 1.2 * R
M = 400
lens.light_o = torch.Tensor([0.0, 0.0, -650.0]).to(device)


def sample_ray(light_o):
    ox, oy = torch.meshgrid(
        torch.linspace(-R_in, R_in, M),
        torch.linspace(-R_in, R_in, M), indexing="ij")
    valid = (ox ** 2 + oy ** 2) < (0.95 * R) ** 2
    o = torch.stack((ox, oy, -torch.ones_like(ox)), axis=-1)
    d = do.normalize(torch.stack((ox, oy, torch.zeros_like(ox)), axis=-1)
                     - light_o[None, None, ...])
    return do.Ray(o[valid], d[valid], wavelength)


def render():
    I = lens.render(sample_ray(lens.light_o))
    return N ** 2 * I / I.sum()


if __name__ == "__main__":
    lens.light_o = torch.Tensor([3.0, 0.0, -650.0])
    lens.d_sensor = torch.Tensor([57.0])
    I_mea = render().detach()
    plt.imsave(out + "I_mea.png", I_mea.numpy(), cmap="gray")

    lens.light_o = torch.Tensor([0.0, 0.0, -650.0])
    lens.d_sensor = torch.Tensor([56.0])
    I_init = render().detach()
    plt.imsave(out + "I_init.png", I_init.numpy(), cmap="gray")
    print(f"loss(init) = {float(((I_init-I_mea)**2).mean()):.4e}")

    diff_names = ["d_sensor", "light_o"]
    lm = do.LM(lens, diff_names, 1e-2, option="diag")
    res = lm.optimize(render, lambda y: I_mea - y, maxit=40, record=True)

    I_final = render().detach()
    plt.imsave(out + "I_final.png", I_final.numpy(), cmap="gray")
    print(f"loss(final) = {res['ls'][-1]:.4e}")
    print(f"recovered light_o  = {[round(v,3) for v in lens.light_o.tolist()]}  (true [3,0,-650])")
    print(f"recovered d_sensor = {float(lens.d_sensor):.3f}                (true 57.0)")
    print("saved to", out)
