"""MODULE 7 - SOLUTION. LM spot-size optimization of a real asphere."""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do
from common_lensfile import load_lens, green
from common_setup import save_dir

out = save_dir("m07_out")
device = torch.device("cpu")

lens = load_lens("thorlabs_acl", device=device)
wavelength = green(device)
R = 15.0

RMS = lambda ps: torch.sqrt(torch.mean(torch.sum(ps ** 2, axis=-1)))


def render():
    ray = lens.sample_ray(wavelength, M=31, R=R)
    return lens.trace_to_sensor(ray)[..., :2]


if __name__ == "__main__":
    ps0 = render()
    print(f"initial RMS = {RMS(ps0)*1e3:.2f} um")
    lim = 50e-3
    lens.spot_diagram(ps0, xlims=[-lim, lim], ylims=[-lim, lim],
                      savepath=out + "spot_before.png", show=False)

    diff_names = ["surfaces[0].c", "surfaces[0].k", "surfaces[0].ai"]
    lm = do.LM(lens, diff_names, 1e-4, option="diag")
    res = lm.optimize(render, lambda y: 0.0 - y, maxit=100, record=True)

    ps1 = render()
    print(f"final   RMS = {RMS(ps1)*1e3:.2f} um")
    lens.spot_diagram(ps1, xlims=[-lim, lim], ylims=[-lim, lim],
                      savepath=out + "spot_after.png", show=False)

    plt.figure()
    plt.semilogy(res["ls"], "-o")
    plt.xlabel("iteration"); plt.ylabel("loss")
    plt.savefig(out + "loss.png", bbox_inches="tight")
    print("saved to", out)
