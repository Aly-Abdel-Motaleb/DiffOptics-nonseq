"""
Visual explainer for `Lensgroup.sample_ray` (diffoptics/optics.py:539).

Renders a 4-panel figure that answers:
  A. What is R, and why is it the half-width of a square?
  B. What does `sag` have to do with it? (side view, x-z plane)
  C. What actually survives the trace, on-axis vs off-axis?
  D. What does `entrance_pupil=True` change?

Run:  python viz_sample_ray.py       (from anywhere)
Out:  viz_sample_ray.png, next to this script
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

HERE = Path(__file__).resolve().parent            # examples/practice/
REPO = HERE.parent.parent                         # repo root
sys.path.insert(0, str(REPO))
import diffoptics as do

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lens = do.Lensgroup(device=device)
lens.load_file(REPO / 'examples/lenses/DoubleGauss/US02532751-1.txt')

wavelength = torch.Tensor([587.5618]).to(device)
s0 = lens.surfaces[0]
r_ap = s0.r                                                    # clear semi-aperture [mm]
sag = s0.surface(torch.tensor(r_ap), torch.tensor(0.0)).item()  # z of surface at edge

M = 31
VIEW = 21.0
angle = np.radians(VIEW)


def auto_R(view_deg):
    """Reproduce the R that sample_ray computes when R is None."""
    return np.tan(np.radians(view_deg)) * sag + r_ap


def grid_and_validity(view_deg, R):
    """Sample the same square sample_ray would, and ask the tracer what survives."""
    x, y = torch.meshgrid(
        torch.linspace(-R, R, M, device=device),
        torch.linspace(-R, R, M, device=device),
        indexing='ij')
    a = np.radians(view_deg)
    ones, zeros = torch.ones_like(x), torch.zeros_like(x)
    o = torch.stack((x, y, zeros), axis=2)
    d = torch.stack((np.sin(a) * ones, zeros, np.cos(a) * ones), axis=-1)
    valid = lens.trace_valid(do.Ray(o, d, wavelength, device=device))
    return x.cpu().numpy(), y.cpu().numpy(), valid.cpu().numpy()


fig, axes = plt.subplots(2, 2, figsize=(13, 12))
fig.suptitle('sample_ray: what R is, and what it costs', fontsize=15, y=0.98)

# --- A: the square vs the circle -------------------------------------------
ax = axes[0, 0]
R0 = auto_R(0.0)
xg, yg, _ = grid_and_validity(0.0, R0)
inside = xg**2 + yg**2 <= r_ap**2

ax.add_patch(Rectangle((-R0, -R0), 2 * R0, 2 * R0, fill=False, ec='C0', lw=2,
                       label=f'sampling square, side 2R = {2*R0:.1f} mm'))
ax.add_patch(Circle((0, 0), r_ap, fill=False, ec='C3', lw=2,
                    label=f'aperture disk, radius r = {r_ap:.1f} mm'))
ax.plot(xg[inside], yg[inside], '.', ms=4, color='C0')
ax.plot(xg[~inside], yg[~inside], 'x', ms=4, color='0.6')
ax.annotate('', xy=(R0, -R0 * 1.18), xytext=(0, -R0 * 1.18),
            arrowprops=dict(arrowstyle='<->', color='C0'))
ax.text(R0 / 2, -R0 * 1.30, 'R = half-width', ha='center', color='C0', fontsize=11)
ax.plot([0, r_ap * np.cos(np.pi/4)], [0, r_ap * np.sin(np.pi/4)], '-', color='C3', lw=1.5)
ax.text(3.0, 3.7, 'r', color='C3', fontsize=12)
ax.plot([0, R0], [0, R0], '--', color='0.4', lw=1)
ax.text(6.0, 8.3, r'corner at $R\sqrt{2}$', color='0.35', fontsize=10)
ax.set_title(f'A. R is the half-width because the code writes linspace(-R, R, M)\n'
             f'grid points inside the disk: {100*inside.mean():.0f}%  '
             r'($\pi/4 = 79\%$)')
ax.legend(loc='upper left', fontsize=9)
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_xlim(-R0 * 1.45, R0 * 1.45); ax.set_ylim(-R0 * 1.45, R0 * 1.45)
ax.grid(alpha=0.2)

# --- B: side view, why sag enters R ----------------------------------------
ax = axes[0, 1]
xs = np.linspace(-r_ap, r_ap, 200)
zs = s0.surface(torch.tensor(xs, dtype=torch.float32, device=device),
                torch.zeros(200, device=device)).cpu().numpy()
ax.plot(zs, xs, color='C3', lw=2.5, label='surface 0 (curved)')
ax.axvline(0, color='0.5', ls='--', lw=1)
ax.text(0.12, -r_ap * 1.28, 'origin plane z = 0', color='0.4', fontsize=10, rotation=90)
ax.plot([sag, sag], [-r_ap * 1.6, r_ap * 1.6], ':', color='0.6', lw=1)
ax.text(sag + 0.12, -r_ap * 1.55, f'z = sag = {sag:.2f}', color='0.35', fontsize=10, rotation=90)

drift = np.tan(angle) * sag
R21 = auto_R(VIEW)
#            launch x0,  colour, label
cases = [
    (-R21, 'C0', f'launch at x = -R = {-R21:.2f}  ->  lands at -r  (edge, BINDING)'),
    (-r_ap, 'C7', f'launch at x = -r = {-r_ap:.2f}  ->  lands short by {drift:.2f}: under-fills'),
    (+R21, 'C1', f'launch at x = +R = {+R21:.2f}  ->  lands at {R21+drift:.2f} > r: WASTED'),
]
for x0, col, lab in cases:
    zz = np.array([0.0, sag])
    xx = x0 + np.tan(angle) * zz
    ax.plot(zz, xx, '-', color=col, lw=2, label=lab)
    ax.plot(0, x0, 'o', color=col, ms=6)
    ax.plot(sag, xx[-1], 's', color=col, ms=6)

for sgn in (+1, -1):
    ax.axhline(sgn * r_ap, color='C3', ls='--', lw=1)
ax.text(-2.9, r_ap + 0.35, 'aperture edge  x = +r', color='C3', fontsize=9)
ax.text(-2.9, -r_ap - 1.5, 'aperture edge  x = -r', color='C3', fontsize=9)
ax.annotate('', xy=(-0.55, -R21), xytext=(-0.55, -r_ap),
            arrowprops=dict(arrowstyle='<->', color='C0'))
ax.text(-1.05, -(r_ap + R21) / 2 - 0.15, r'tan($\theta$)·sag',
        color='C0', fontsize=10, ha='right')
ax.set_title(r'B. Launch at z=0, hit surface at z=sag, drifting $+x$ en route.'
             '\n'
             r'The $-x$ edge sets R = $r+\tan\theta\cdot$sag. $+x$ side is pure padding.')
ax.legend(loc='upper left', fontsize=8.5)
ax.set_xlabel('z [mm]  (optical axis)'); ax.set_ylabel('x [mm]')
ax.set_xlim(-3.0, sag + 2.4); ax.set_ylim(-r_ap * 1.75, r_ap * 1.75)
ax.grid(alpha=0.2)

# --- C: what really survives, on-axis vs off-axis --------------------------
ax = axes[1, 0]
for view, col, mark in [(0.0, 'C0', 'o'), (VIEW, 'C1', 's')]:
    R = auto_R(view)
    xg, yg, valid = grid_and_validity(view, R)
    ax.plot(xg[valid], yg[valid], mark, ms=3.5, color=col, alpha=0.85,
            label=f'view={view:.0f}°  survives: {100*valid.mean():.0f}%  (R={R:.2f})')
    ax.add_patch(Rectangle((-R, -R), 2 * R, 2 * R, fill=False, ec=col, lw=1.2, ls='--'))
ax.add_patch(Circle((0, 0), r_ap, fill=False, ec='C3', lw=1.5, ls=':'))
ax.set_title('C. Rays that actually clear all 11 surfaces + the stop.\n'
             'Off-axis the survivor set shifts in -x and shrinks: vignetting.')
ax.legend(loc='upper left', fontsize=9)
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.grid(alpha=0.2)

# --- D: entrance_pupil=True ------------------------------------------------
ax = axes[1, 1]
R = auto_R(VIEW)
xg, yg, valid = grid_and_validity(VIEW, R)
ax.plot(xg[~valid], yg[~valid], 'x', ms=3.5, color='0.75',
        label=f'entrance_pupil=False: wasted ({100*(~valid).mean():.0f}%)')
ax.plot(xg[valid], yg[valid], '.', ms=3.5, color='0.45',
        label=f'entrance_pupil=False: useful ({100*valid.mean():.0f}%)')

# what sample_ray does when entrance_pupil=True: bounding box of the survivors
_, pxs, pys = lens.calc_entrance_pupil(VIEW, R)
x0, x1 = pxs.min().item(), pxs.max().item()
y0, y1 = pys.min().item(), pys.max().item()
ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='C2', lw=2,
                       label='bounding box from calc_entrance_pupil'))
ex, ey = torch.meshgrid(torch.linspace(x0, x1, M), torch.linspace(y0, y1, M), indexing='ij')
_, _, evalid = None, None, None
a = np.radians(VIEW)
ones, zeros = torch.ones_like(ex).to(device), torch.zeros_like(ex).to(device)
o = torch.stack((ex.to(device), ey.to(device), zeros), axis=2)
d = torch.stack((np.sin(a) * ones, zeros, np.cos(a) * ones), axis=-1)
evalid = lens.trace_valid(do.Ray(o, d, wavelength, device=device)).cpu().numpy()
exn, eyn = ex.numpy(), ey.numpy()
ax.plot(exn[evalid], eyn[evalid], '.', ms=4, color='C2',
        label=f'entrance_pupil=True: useful ({100*evalid.mean():.0f}%)')
ax.add_patch(Rectangle((-R, -R), 2 * R, 2 * R, fill=False, ec='0.7', lw=1, ls='--'))
ax.set_title(f'D. entrance_pupil=True at view={VIEW:.0f}°: pre-trace a 101x101 probe,\n'
             'then sample only inside the survivors\' bounding box.')
ax.legend(loc='upper left', fontsize=8.5)
ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.grid(alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.965])
out = HERE / 'viz_sample_ray.png'
plt.savefig(out, dpi=130)
print(f'wrote {out}')
print(f'r={r_ap} mm, sag={sag:.3f} mm, R(0deg)={auto_R(0):.3f}, R({VIEW}deg)={auto_R(VIEW):.3f}')