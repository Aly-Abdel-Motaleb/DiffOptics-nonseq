"""
MODULE 2 - Trace point-source rays THROUGH the spherical lens.

Two views of the same trace:
  1) layout ray-trace (side view of paths)  -> uses trace_to_sensor_r (records paths)
  2) spot diagram (where rays land on receiver) + RMS spot size

New engine calls (read diffoptics/optics.py):
  - lens.trace_to_sensor_r(ray) -> (ps, oss)  # ps=hit points, oss=recorded paths  (L786)
  - lens.plot_raytraces(oss, ...)             # side-view layout                    (L452)
  - lens.trace_to_sensor(ray)   -> ps         # hit points only                     (L768)
  - lens.spot_diagram(ps, ...)                # scatter on receiver                 (L193)

READ BEFORE (tick when done):
  [x] diffoptics/optics.py:807-830   trace (world->local->_trace->world)
  [x] diffoptics/optics.py:768-805   trace_to_sensor + trace_to_sensor_r
  [x] diffoptics/optics.py:452       plot_raytraces
  [x] diffoptics/optics.py:193       spot_diagram
  [x] examples/autodiff.py:40-84     same calls, working example

Run: python m02_build_trace_practice.py
Expect: layout_trace.png + spotdiagram.png in ./m02_out/, printed RMS.
"""
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

    # ---- 1) layout ray-trace ----
    # TODO 1: trace with path recording. Returns (ps, oss).
    #         hint: lens.trace_to_sensor_r(ray)
    ps, oss = lens.trace_to_sensor_r(ray)

    # TODO 2: draw the layout. plot_raytraces returns (ax, fig).
    #         hint: lens.plot_raytraces(oss, color="b-", show=False)
    ax, fig = lens.plot_raytraces(oss)
    ax.set_title("point source -> spherical lens -> receiver")
    fig.savefig(out + "layout_trace.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 2) spot diagram ----
    # TODO 3: get 2D hit points on the receiver (x,y only).
    #         hint: lens.trace_to_sensor(ray)[..., :2]
    ps2 = lens.trace_to_sensor(ray)[...,:2]

    print(f"RMS spot = {RMS(ps2):.4f} mm")

    # TODO 4: save the spot diagram.
    #         hint: lens.spot_diagram(ps2, xlims=[-4,4], ylims=[-4,4],
    #                                 savepath=out+"spotdiagram.png", show=False)
    lens.spot_diagram(ps2, xlims=[-4,4], ylims=[-4,4],savepath=out+"spotdiagram.png", show=False)
    print("saved to", out)
