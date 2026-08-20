"""
MODULE 5 - MONTE-CARLO ray sampling (the fix for aliasing).

Deterministic grid -> the same target points every call -> discrete dots on receiver.
Monte-Carlo -> JITTER each target by a random offset inside its cell, and AVERAGE
many independent renders (spp = samples per pixel). Randomness spreads the sampling
so surface features stop aliasing.

Two MC styles here:
  A) jittered grid  -- caustic_pyramid.py: x = x + p*(rand-0.5), p = cell size (L73)
     + render(spp): accumulate spp renders, divide by spp                     (L110)
  B) concentric disk -- render_psf.py: do.Sampler().concentric_sample_disk maps
     uniform [0,1]^2 to a uniform DISK (round footprint, no grid at all).

READ BEFORE (tick when done):
  [ ] examples/caustic_pyramid.py:73-86    sample_ray(random=True) jitter
  [ ] examples/caustic_pyramid.py:110-114  render(spp) accumulation
  [ ] diffoptics/basics.py:116-144         Sampler.concentric_sample_disk
  [ ] examples/render_psf.py:50-70         disk sampler in practice

Run: python m05_montecarlo_practice.py
Expect: I_deterministic / I_montecarlo / I_disk .png in ./m05_out/ -- MC is smooth.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_setup import build_scene, save_dir
import diffoptics as do

lens, wavelength, o_pt = build_scene()
out = save_dir("m05_out")

M = 30
R_src = 8.0


def grid_targets(random=False, M=M):
    """The z=0 aim points of the grid sampler (jittered if random)."""
    x = torch.linspace(-R_src, R_src, M)
    X, Y = torch.meshgrid(x, x, indexing="ij")

    if random:
        # TODO 1: jitter each grid point by a uniform offset within its cell.
        #   cell size p = 2*R_src / M
        #   X = X + p*(torch.rand_like(X) - 0.5)   (same for Y)
        p = 2*R_src / M
        X = X + p*(torch.rand_like(X)-0.5)
        Y = Y + p*(torch.rand_like(Y)-0.5)

    return torch.stack((X, Y, torch.zeros_like(X)), axis=-1).reshape(-1, 3)


def rays_from(targets):
    """Point source: ONE origin o_pt, one direction per target."""
    o = o_pt.expand_as(targets).clone()
    d = do.normalize(targets - o_pt)
    return do.Ray(o, d, wavelength)


def sample_pointsrc_ray(random=False):
    return rays_from(grid_targets(random))


def render_mc(spp=1, random=True):
    """Average `spp` independent (jittered) renders -> Monte-Carlo estimate."""
    I = torch.zeros(*lens.film_size)
    # TODO 2: loop spp times, each time render a freshly-jittered ray batch,
    
    for _ in range(spp):
        I += lens.render(sample_pointsrc_ray(random=random))
    
    #         accumulate into I, then divide by spp.
    #   for _ in range(spp): I = I + lens.render(sample_pointsrc_ray(random=random))
    return I/spp
    ...


# --- style B: concentric-disk sampling ---
sampler = do.Sampler()


def disk_targets(N=M * M):
    # TODO 3: draw N uniform points on a disk of radius R_src.
    u = torch.rand(N); v = torch.rand(N)
    px, py = sampler.concentric_sample_disk(u, v)   # unit disk in [-1,1]
    X, Y = R_src*px, R_src*py
    return torch.stack((X, Y, torch.zeros_like(X)), axis=-1).reshape(-1, 3)


def sample_pointsrc_ray_disk(N=M * M):
    return rays_from(disk_targets(N))


def render_disk(spp=32):
    I = torch.zeros(*lens.film_size)
    for _ in range(spp):
        I = I + lens.render(sample_pointsrc_ray_disk())
    return I / spp


# --------------------------------------------------------------------------
# FLOW VISUALIZATION
#   point source (1 origin)  ->  aim points on z=0  ->  trace  ->  film
# The sampler only chooses DIRECTIONS; the origin o_pt never moves.
# --------------------------------------------------------------------------

def plot_geometry():
    """Stage 1-3: rays leaving the single origin, through the lens, to the sensor."""
    ray = rays_from(grid_targets(random=False, M=9))
    _, oss = lens.trace_to_sensor_r(ray)
    ax, fig = lens.plot_raytraces(oss, color="b-", show=False)
    ax.plot([float(o_pt[2])], [float(o_pt[0])], "r*", markersize=14, zorder=5)
    ax.annotate("point source o_pt\n(ALL rays start here)",
                xy=(float(o_pt[2]), float(o_pt[0])), xytext=(float(o_pt[2]), 9),
                color="r", fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color="r"))
    ax.set_title("stage 1-3: one origin -> many directions -> lens -> film")
    fig.savefig(out + "flow_geometry.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def plot_pipeline(I_det, I_mc, I_disk):
    """Stage 2 (aim points) above stage 4 (irradiance), for all three samplers."""
    cases = [
        ("deterministic grid", grid_targets(random=False), I_det),
        ("jittered grid (MC)", grid_targets(random=True), I_mc),
        ("concentric disk (MC)", disk_targets(), I_disk),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    for j, (name, tg, img) in enumerate(cases):
        t = tg.cpu().detach().numpy()
        a = axes[0, j]
        a.scatter(t[:, 0], t[:, 1], s=3, c="tab:blue")
        a.add_patch(plt.Circle((0, 0), R_src, fill=False, color="r", ls="--", lw=1))
        a.set_aspect("equal")
        a.set_xlim(-R_src * 1.2, R_src * 1.2); a.set_ylim(-R_src * 1.2, R_src * 1.2)
        a.set_title(name, fontsize=10)
        if j == 0:
            a.set_ylabel("stage 2: aim points @ z=0\n[mm]")

        b = axes[1, j]
        b.imshow(img, cmap="inferno")
        b.set_xticks([]); b.set_yticks([])
        b.set_title(f"I.max = {img.max():.3f}", fontsize=9)
        if j == 0:
            b.set_ylabel("stage 4: irradiance on film")

    fig.suptitle("MC flow: sampler pattern (top) -> rendered irradiance (bottom)\n"
                 "same origin every time; only the directions are resampled",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out + "flow_pipeline.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def plot_jitter_zoom(draws=4, ncell=4):
    """Why MC works: each render draws different aim points inside the SAME cells."""
    p = 2 * R_src / M
    lim = ncell * p
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for a, rnd, name in zip(axes, [False, True],
                            ["deterministic: identical every call",
                             f"jittered: {draws} independent draws"]):
        for k in range(draws):
            t = grid_targets(random=rnd).cpu().detach().numpy()
            a.scatter(t[:, 0], t[:, 1], s=16, alpha=0.8, label=f"draw {k}")
        for g in np.arange(-R_src, R_src + p, p):        # cell boundaries
            a.axvline(g, color="0.85", lw=0.6); a.axhline(g, color="0.85", lw=0.6)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
        a.set_title(name, fontsize=10)
        a.set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("cell size p = 2*R_src/M -> jitter fills the cell, "
                 "averaging over spp removes the dots", fontsize=10)
    fig.tight_layout()
    fig.savefig(out + "flow_jitter_zoom.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    I_det = render_mc(spp=1, random=False).cpu().detach().numpy()
    I_mc = render_mc(spp=32, random=True).cpu().detach().numpy()
    I_disk = render_disk(spp=32).cpu().detach().numpy()

    plt.imsave(out + "I_deterministic.png", I_det, cmap="inferno")
    plt.imsave(out + "I_montecarlo.png", I_mc, cmap="inferno")
    plt.imsave(out + "I_disk.png", I_disk, cmap="inferno")
    print(f"deterministic     I.max={I_det.max():.3f}")
    print(f"monte-carlo grid  I.max={I_mc.max():.3f}  (spp=32, jittered grid)")
    print(f"monte-carlo disk  I.max={I_disk.max():.3f}  (spp=32, concentric disk)")
    plot_geometry()
    plot_pipeline(I_det, I_mc, I_disk)
    plot_jitter_zoom()

    print("saved to", out, "-- both MC images smooth; disk has a round footprint")
    print("flow figures: flow_geometry.png, flow_pipeline.png, flow_jitter_zoom.png")
