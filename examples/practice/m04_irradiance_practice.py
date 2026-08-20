"""
MODULE 4 - Differentiable IRRADIANCE at the receiver (deterministic sampling).

This is the illumination goal: an image I[y,x] of energy landing on the receiver.
`lens.render(ray)` does it: trace -> intersect receiver -> bilinear-splat each ray
into the pixel grid with index_put(accumulate=True). (diffoptics/optics.py:712)

It needs (already set by build_scene): d_sensor, film_size, pixel_size.

THE PROBLEM to feel here: we sample the angular domain on a FIXED grid. Increase M
and you see the irradiance is a grid of bright dots / stripes = ALIASING. Module 5
fixes it with Monte-Carlo jitter.

READ BEFORE (tick when done):
  [ ] diffoptics/optics.py:712-759   render (sensor intersect, valid, bilinear splat)
  [ ] diffoptics/optics.py:1014-1044 _refract (sets valid: TIR / aperture miss)

Run: python m04_irradiance_practice.py
Expect: I_M20.png and I_M60.png in ./m04_out/ (both show grid artifacts).
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_setup import build_scene, save_dir
import diffoptics as do

lens, wavelength, o_pt = build_scene()
out = save_dir("m04_out")

R_src = 8.0


def sample_pointsrc_ray(M):
    x = torch.linspace(-R_src, R_src, M)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    targets = torch.stack((X, Y, torch.zeros_like(X)), axis=-1).reshape(-1, 3)
    o = o_pt.expand_as(targets).clone()
    d = do.normalize(targets - o_pt)
    return do.Ray(o, d, wavelength)


def render(M):
    ray = sample_pointsrc_ray(M)
    # TODO 1: compute irradiance image with the engine.
    #         hint: lens.render(ray)
    I = lens.render(ray)
    return I

def spotDiagram(M):
    ray = sample_pointsrc_ray(M)
    # TODO 3: compute the spot diagram with the engine.
    # spot_diagram() writes the figure to savepath and returns the RMS spot size.
    ps = lens.trace_to_sensor(ray)[..., :2]
    rms = lens.spot_diagram(
        ps, xlims=[-4, 4], ylims=[-4, 4],
        savepath=out + f"spot_M{M}.png", show=False
    )
    return rms

if __name__ == "__main__":
    for M in (20, 60):
        I = render(M).cpu().detach().numpy()
        print(f"M={M:3d}  rays={M*M:5d}  I.sum={I.sum():.1f}  I.max={I.max():.3f}")
        # TODO 2: save I as an image (cmap='inferno').
        
        #         hint: plt.imsave(out + f"I_M{M}.png", I, cmap="inferno")
        plt.imsave(out + f"I_M{M}.png", I, cmap="inferno")
        rms = spotDiagram(M)
        print(f"M={M:3d}  spot RMS={rms:.4f} mm")
        
    print("saved to", out, "-- look for grid/dot aliasing artifacts")
