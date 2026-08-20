"""MODULE 3 - SOLUTION. Load a real lens file; spot diagrams across fields."""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do
from common_lensfile import load_lens, green
from common_setup import save_dir

out = save_dir("m03_out")
device = torch.device("cpu")

lens = load_lens("doublegauss", device=device)
wavelength = green(device)

views = [0.0, 7.0, 14.0, 21.0]


if __name__ == "__main__":
    ax, fig = lens.plot_setup2D_with_trace([0.0, 21.0], wavelength, M=3)
    ax.axis("off"); ax.set_title("Double-Gauss layout")
    fig.savefig(out + "layout.png", bbox_inches="tight")
    plt.close(fig)

    for view in views:
        ray = lens.sample_ray(wavelength, view=view, M=21,
                              sampling="grid", entrance_pupil=True)
        ps = lens.trace_to_sensor(ray, ignore_invalid=True)
        rms = float(lens.rms(ps)[0])
        print(f"view={view:5.1f} deg   rays={ps.shape[0]:4d}   RMS={rms*1e3:.2f} um")

        lim = 30e-3
        lens.spot_diagram(ps[..., :2], xlims=[-lim, lim], ylims=[-lim, lim],
                          savepath=out + f"spot_view_{int(view)}.png", show=False)
    print("saved to", out)
