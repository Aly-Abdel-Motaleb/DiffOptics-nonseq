"""
MODULE 3 - Load a REAL lens from a file; evaluate it across field angles.

So far the lens was hand-built. Production designs live in `.txt` files
(examples/lenses/). `lens.load_file(path)` parses them (diffoptics/optics.py:65,98)
and sets surfaces / materials / d_sensor / r_last for you.

Here: load a Double-Gauss photographic lens, draw its 2D layout with traced rays,
and make a spot diagram + RMS at several FIELD ANGLES (views). This is the
`sanity_check.py` workflow, on CPU.

New engine calls:
  - do.Lensgroup().load_file(Path)                        # load real lens (optics.py:65)
  - lens.sample_ray(wl, view=deg, M, entrance_pupil=True) # field-angle rays (optics.py:539)
  - lens.trace_to_sensor(ray, ignore_invalid=True)        # drop rays that miss
  - lens.rms(ps)[0]                                        # RMS spot (returns (val, ps))
  - lens.plot_setup2D_with_trace(views, wl, M)            # layout (optics.py:486)

`entrance_pupil=True` aims rays through the stop, so off-axis fields are sampled
correctly (only works for lenses that HAVE a stop -- Double-Gauss does).

READ BEFORE (tick when done):
  [ ] diffoptics/optics.py:65-68     load_file
  [ ] diffoptics/optics.py:98-167    read_lensfile (the .txt format)
  [ ] diffoptics/optics.py:539-586   sample_ray (view + entrance_pupil)
  [ ] diffoptics/optics.py:183-191   rms (returns (value, ps))
  [ ] diffoptics/optics.py:486       plot_setup2D_with_trace
  [ ] common_lensfile.py             load_lens / lens list
  [ ] examples/sanity_check.py       the workflow this mirrors

Run: python m03_lensfile_practice.py
Expect: layout.png + spot_view_*.png in ./m03_out/, printed RMS per field.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do
from common_lensfile import load_lens, green
from common_setup import save_dir   # reuse the tiny save_dir helper

out = save_dir("m03_out")
device = torch.device("cpu")

# TODO 1: load the 'doublegauss' lens on CPU (see common_lensfile.load_lens).
lens = load_lens("doublegauss", device=device)
wavelength = green(device)

views = [0.0, 7.0, 14.0, 21.0]




if __name__ == "__main__":
    # ---- layout with traced rays ----
    # TODO 2: draw the 2D layout with traced rays for a couple of views.
    #   hint: lens.plot_setup2D_with_trace([0.0, 21.0], wavelength, M=3)  -> (ax, fig)
    ax, fig = lens.plot_setup2D_with_trace(views,wavelength,M=3)
    ax.axis("off"); ax.set_title("Double-Gauss layout")
    fig.savefig(out + "layout.png", bbox_inches="tight")
    plt.close(fig)

    # ---- spot diagram + RMS per field ----
    for view in views:
        # TODO 3: sample field-angle rays through the entrance pupil.
        #   hint: lens.sample_ray(wavelength, view=view, M=21,
        #                         sampling='grid', entrance_pupil=True)
        ray = lens.sample_ray(wavelength, view=view , M=21 , entrance_pupil=True)

        # TODO 4: trace to sensor, dropping rays that miss (ignore_invalid=True).
        ps = lens.trace_to_sensor(ray=ray , ignore_invalid=True)

        rms = float(lens.rms(ps)[0])
        print(f"view={view:5.1f} deg   rays={ps.shape[0]:4d}   RMS={rms*1e3:.2f} um")

        lim = 30e-3
        lens.spot_diagram(ps[..., :2], xlims=[-lim, lim], ylims=[-lim, lim],
                          savepath=out + f"spot_view_{int(view)}.png", show=False)
    print("saved to", out)
