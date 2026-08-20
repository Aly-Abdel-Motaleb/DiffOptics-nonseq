"""MODULE 2 - SOLUTION. Trace + layout + spot diagram."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_setup import build_scene, save_dir
import diffoptics as do

lens, wavelength, o_pt = build_scene()
out = save_dir("m02_out")

M = 15
R_src = 8.0


def sample_pointsrc_ray():
    x = torch.linspace(-R_src, R_src, M)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    targets = torch.stack((X, Y, torch.zeros_like(X)), axis=-1).reshape(-1, 3)
    o = o_pt.expand_as(targets).clone()
    d = do.normalize(targets - o_pt)
    return do.Ray(o, d, wavelength)


RMS = lambda ps: torch.sqrt(torch.mean(torch.sum(torch.square(ps), axis=-1)))

if __name__ == "__main__":
    ray = sample_pointsrc_ray()

    ps, oss = lens.trace_to_sensor_r(ray)
    ax, fig = lens.plot_raytraces(oss, color="b-", show=False)
    ax.set_title("point source -> spherical lens -> receiver")
    fig.savefig(out + "layout_trace.png", bbox_inches="tight")
    plt.close(fig)

    ps2 = lens.trace_to_sensor(ray)[..., :2]
    print(f"RMS spot = {RMS(ps2):.4f} mm")
    lens.spot_diagram(ps2, xlims=[-4, 4], ylims=[-4, 4],
                      savepath=out + "spotdiagram.png", show=False)
    print("saved to", out)
