"""Generate c05_two_lens.ipynb. Run from examples/nonseq/.

The notebook is written from here rather than by hand so the JSON is valid by
construction and a change to a cell is a normal diff.
"""
import json
import os

MD, CODE = 'markdown', 'code'
cells = []


def add(kind, text):
    src = text.strip('\n').split('\n')
    src = [ln + '\n' for ln in src[:-1]] + [src[-1]]
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == CODE:
        c['execution_count'] = None
        c['outputs'] = []
    cells.append(c)


# ---------------------------------------------------------------- 0 overview
add(MD, r"""
# c05 - two lenses, one reflectivity each, non-sequential MC

Point source, two lenses, two receivers. Lens 1 carries **R = 0.2** on both surfaces, lens 2
carries **R = 0.1**. This notebook runs the scene at **1e6 rays**, sizes each receiver's mesh
from the Monte-Carlo noise it can afford, draws the irradiance maps, and then compares
against a LightTools export you upload.

```
point source            lens 1 (collimator)      lens 2 (mirror image)     forward
(0,0,0), 1 W       ->   S1 asphere  z = 40.0 ->  S3 flat    z = 100.0  ->  z = 200
isotropic               S2 flat     z = 46.5     S4 asphere z = 106.5      +-20 mm
cone 16.2758 deg        N-BK7, sd 12.7           N-BK7, sd 14.0
Phi_cap = 0.020038 W    R1 = 0.2 both            R2 = 0.1 both         <-  backward
                                                 focus at z = 146.5        z = -80
                                                                           +-80 mm
```

**Lens 2 is lens 1 mirrored** about z = 73.25. Lens 1 turns a point at s = 40 mm into a
collimated beam; run the same glass backwards and a collimated beam becomes a point 40 mm
past the aspheric vertex - exactly, by ray reversibility. So there is no second lens design
to justify: its aspheric radius is lens 1's with the sign flipped, its conic constant is
identical. Measured focus spot radius: **2.6e-14 mm**.

The forward receiver sits 53.5 mm *past* the focus, where the beam has reopened to a 15.62 mm
disc. At the focus it would be one hot bin and there would be nothing to compare.

## The bin rule

Per-bin relative noise for `R` rays landing on an `N x N` receiver:

```
eps = 1 / sqrt(R / N^2) = N / sqrt(R)
```

so a 10 % target fixes the mesh from the ray budget:

```
N = 16 * floor(0.1 * sqrt(R) / 16)        snapped DOWN, floored at 16
```

`R` is the count landing on **that** receiver, not the number launched - which is why the two
receivers get different meshes. `c05_two_lens.bins_for` is the one implementation; the
comparison script and the LightTools build spec both use it.

## Runtime

**Runtime > Change runtime type > T4 GPU.** Everything is float64, which a T4 runs at 1/32
of its fp32 rate - correct but not fast. The whole notebook is a few minutes; the 1e6 trace
is the expensive cell.
""")

# ------------------------------------------------------------------- 1 gpu
add(CODE, r"""
!nvidia-smi
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda)
assert torch.cuda.is_available(), 'Runtime > Change runtime type > T4 GPU'
print(torch.cuda.get_device_name(0),
      f'{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB')
print('note: everything below is float64 - a T4 runs fp64 at ~1/32 of fp32')
""")

add(MD, r"""
## 0. Get the code onto the runtime

Run **A or B, not both.**
""")

add(CODE, r"""
# --- OPTION A: clone the branch ---------------------------------------------
# Put a GitHub PAT in Colab Secrets under the name GH_PAT (key icon, left sidebar).
# Never paste a token into a cell - it gets saved inside the notebook.
from google.colab import userdata

REPO   = 'Aly-Abdel-Motaleb/DiffOptics-nonseq'   # private
BRANCH = 'bench'

import os, subprocess
if not os.path.isdir('/content/DiffOptics'):
    tok = userdata.get('GH_PAT')
    url = f'https://{tok}@github.com/{REPO}.git'
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH,
                    url, '/content/DiffOptics'], check=True)
    # The clone URL carries the token, and git writes it in cleartext into
    # .git/config.  Drop the remote immediately - a shallow clone never pushes.
    subprocess.run(['git', '-C', '/content/DiffOptics', 'remote', 'remove', 'origin'])
%cd /content/DiffOptics
!git log --oneline -1
!pip install -q matplotlib
""")

add(CODE, r"""
# --- OPTION B: upload the bundle instead of cloning --------------------------
# Pick examples/nonseq/c05_colab_bundle.zip when the file chooser opens.
import os, zipfile
if not os.path.isdir('/content/DiffOptics'):
    from google.colab import files
    up = files.upload()
    name = next(iter(up))
    with zipfile.ZipFile(name) as z:
        z.extractall('/content')
%cd /content/DiffOptics
!ls diffoptics examples/nonseq
!pip install -q matplotlib
""")

add(CODE, r"""
# --- paths and imports. Everything below depends on this cell. ---------------
import os, sys, subprocess
import numpy as np, torch

sys.path.insert(0, '/content/DiffOptics/examples/nonseq')
sys.path.insert(0, '/content/DiffOptics')
torch.set_default_dtype(torch.float64)      # MUST precede the c05 import

import c05_two_lens as c5
from diffoptics import nonseq

OUT = '/content/c05_out'
LT_DIR = os.path.join(OUT, 'lt')
os.makedirs(LT_DIR, exist_ok=True)
C05_COMPARE = '/content/DiffOptics/examples/nonseq/c05_compare.py'

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device {dev}   R1 = {c5.R1_COAT} (S1,S2)   R2 = {c5.R2_COAT} (S3,S4)')
print(f'Phi_cap = {c5.PHI_CAP:.9f} W over a {np.degrees(c5.THETA_MAX):.6f} deg cone')
print(f'focus at z = {c5.Z_FOCUS:g} mm,  forward receiver z = {c5.Z_RECV:g} +-{c5.R_RECV:g} mm')
print(f'out -> {OUT}')
""")

# ------------------------------------------------------------------- the run
add(MD, r"""
## 1. The 1e6-ray run

One `trace_mc` at 1e6 rays, then the bin table. Two things to read:

* **the ledger** - `|sum(w) + culled + truncated| / Phi_cap - 1` must be under 1e-9. Monte
  Carlo is unbiased but it is not approximate about energy: every ray that leaves the scene,
  gets culled, or hits the depth cap is accounted for. If this does not close, nothing below
  is worth looking at.
* **the bin table** - `rays on` is what sets each mesh. `eps pred` is the rule
  `N/sqrt(R)`; `eps meas` is `sqrt(sum w^2)/sum w` per bin, measured, median over lit bins.

Where the two eps columns disagree, the receiver says so. The rule is a *flat-field*
estimate - it uses the mean rays per bin - and neither receiver is flat:

* **forward** comes in **under** the prediction: only ~62 % of its bins are lit at all (a
  15.6 mm disc inside a 20 mm square), so the bins that exist hold more rays than the average.
* **backward** comes in **over**, and by more. Every bin is lit, but the flux is concentrated:
  mean count per bin is ~214, **median is ~46**. A mean-based rule cannot see that skew.

The `N strict` column is the largest mesh whose *measured* noise is genuinely under 10 %.
""")

add(CODE, r"""
RAYS = int(1e6)

term, tally, cl = c5.run_mc(RAYS, seed_src=11, mc_seed=9007, device=dev)

led = abs(c5.phi_captured_check(term, tally) / c5.PHI_CAP - 1)
print(f'rays {RAYS:.0e}   ledger |sum/Phi_cap - 1| = {led:.2e}   (gate < 1e-9)   '
      f'{"OK" if led < 1e-9 else "FAIL"}')
print(f'truncated at depth {c5.MAX_DEPTH}: '
      f'{float(tally["truncated"]) / c5.PHI_CAP:.2e} of Phi_cap\n')

bins = c5.bin_table(cl, launched=RAYS)

print()
for rc in c5.RECEIVERS:
    b = bins[rc['name']]
    if b['median'] > c5.EPS_TARGET:
        print(f'  NOTE {rc["label"]}: rule picks N = {b["n"]} (predicted '
              f'{b["eps_pred"]:.2%}) but the measured median is {b["median"]:.2%}. '
              f'Flux is concentrated - mean count per bin '
              f'({b["rays_on"] / b["n"] ** 2:.0f}) is well above the typical bin. '
              f'N = {b["n_strict"]} holds the realised noise under target.')
    else:
        print(f'  {rc["label"]}: rule is conservative - measured {b["median"]:.2%} '
              f'vs predicted {b["eps_pred"]:.2%} ({b["lit_frac"]:.0%} of bins lit).')
""")

# ---------------------------------------------------------------------- maps
add(MD, r"""
## 2. Irradiance maps

Left: the map at the mesh the rule chose. Right: the radial profile on a log axis, computed
as power-per-annulus over annulus area so it is a genuine irradiance rather than a raw
histogram that tilts with r.

**Read the plateau, not the hottest pixel.** The peak is one bin's worth of Monte-Carlo
noise; the plateau is the physics.

* **forward**, z = 200, +-20 mm - the `T1T2T3T4` beam, a filled 15.6 mm disc, with ghost haze
  outside it. That haze is why the receiver is +-20 and not c03's +-16.
* **backward**, z = -80, +-80 mm - `R1` off the front asphere plus `T1R2T1`. Note the scale:
  the bright part is a small central patch inside an 80 mm receiver, which is exactly the
  concentration that makes the flat-field rule optimistic here.
""")

add(CODE, r"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(len(c5.RECEIVERS), 2, figsize=(12, 4.6 * len(c5.RECEIVERS)),
                       squeeze=False)
for row, rc in enumerate(c5.RECEIVERS):
    key, half = rc['name'], rc['half']
    n = bins[key]['n']
    pitch = 2 * half / n
    # Divide by bin area -> W/mm^2, so the two receivers are on the same physical
    # scale despite their very different sizes.
    S = nonseq.splat(cl['p_' + key], cl['w_' + key] / pitch ** 2, [n, n], pitch)
    S = S.detach().cpu().numpy()
    lit = S[S > 0]
    vmax = np.log10(lit.max())
    im = ax[row, 0].imshow(np.log10(np.where(S > 0, S, np.nan)), origin='lower',
                           extent=[-half, half, -half, half], vmin=vmax - 5, vmax=vmax)
    ax[row, 0].set(title=f'{rc["label"]}  z = {rc["z"]:g} mm   {n}x{n}, '
                         f'eps = {bins[key]["eps_pred"]:.1%}\nlog10 W/mm^2',
                   xlabel='x [mm]', ylabel='y [mm]')
    fig.colorbar(im, ax=ax[row, 0], fraction=.046)

    r = np.linalg.norm(cl['p_' + key][..., :2].detach().cpu().numpy(), axis=-1)
    ww = cl['w_' + key].detach().cpu().numpy()
    edges = np.linspace(0, half, 33)
    tot, _ = np.histogram(r, bins=edges, weights=ww)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    mid = 0.5 * (edges[1:] + edges[:-1])
    good = tot > 0
    ax[row, 1].semilogy(mid[good], (tot / area)[good], 'o-', ms=3)
    ax[row, 1].set(title=f'{rc["label"]} radial profile', xlabel='r [mm]',
                   ylabel='W/mm^2')
    ax[row, 1].grid(alpha=.3)

    print(f'  {rc["label"]:9s} {float(cl["w_" + key].sum()) / c5.PHI_CAP * 100:6.2f} % '
          f'of Phi_cap,  {int((S > 0).sum())}/{n * n} bins lit')

fig.suptitle(f'two lenses, R1 = {c5.R1_COAT}, R2 = {c5.R2_COAT}, {RAYS:.0e} rays',
             fontsize=12)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'c05_maps_1e6.png'), dpi=120)
plt.show()
""")

# --------------------------------------------------------------------- paths
add(MD, r"""
## 3. Power by path, and what actually has a closed form

`hHrK` = H surface hits, K of them reflections. Three buckets have closed forms, and they are
gated:

| bucket | path | closed form |
|---|---|---|
| `fwd_h4r0` | T1 T2 T3 T4 | `(1-R1)^2 (1-R2)^2 = 0.5184` |
| `back_h1r1` | R1 | `R1 = 0.20` |
| `back_h3r1` | T1 R2 T1 | `(1-R1)^2 R1 = 0.128` |

The rest do **not**, and that is a fact about the scene rather than a gap in the script. With
four partial surfaces the `(hits, reflections)` key stops being unique:

* **`back_h5r1`** measures ~6.66 % where the obvious `(1-R1)^4 R2` gives 4.10 %. The excess is
  a second path in the same bucket - reflect off S4, back out through S3, then **miss** lens
  1's 12.7 mm rim and escape backwards. It is aperture-clipped, so it has no closed form at
  all.
* **`fwd_h6r2`** is the lens-1 ghost and the lens-2 ghost summed. Both real, different
  amplitudes because R1 != R2, and the bucket holds the total.

So the honest gates are the three above plus the energy ledger. The cell also runs
`trace_split` on the *same rays* for a direct MC-vs-deterministic check.
""")

add(CODE, r"""
pp = c5.path_powers(term)
exact = c5.closed_form()

print(f'{"bucket":<14} {"% Phi_cap":>10}  {"closed form":>12}  {"d":>9}')
top = sorted(((k, v) for k, v in pp.items() if k.startswith(('fwd_h', 'back_h'))),
             key=lambda x: -x[1])[:12]
# PATHS maps (side, hits, reflections) -> name, so the tuple is the KEY.
bucket_name = {f'{s}_h{h}r{r}': nm for (s, h, r), nm in c5.PATHS.items()}
for k, v in top:
    frac = 100 * v / c5.PHI_CAP
    nm = bucket_name.get(k)
    if nm:
        want = 100 * exact[nm]
        print(f'{k:<14} {frac:10.4f}  {want:12.4f}  {frac - want:+9.4f}   {nm}')
    else:
        print(f'{k:<14} {frac:10.4f}  {"-":>12}  {"":>9}')

# MC vs the deterministic split tracer, on identical source rays.
o, d, w = c5.sample_point_source(50000, seed=11, device=dev)
els = c5.build_elements(device=dev)
ts, _ = c5.trace_split(o, d, w, els, c5.WAVELENGTH, max_depth=c5.MAX_DEPTH,
                       w_min=c5._w_min(w))
ref_fwd = float(c5.classify(ts)['phi_fwd']) / c5.PHI_CAP
vals = []
for s in (100, 101, 102, 103, 104):
    tm, _ = c5.trace_mc(o, d, w, els, c5.WAVELENGTH, seed=s,
                        max_depth=c5.MAX_DEPTH, w_min=c5._w_min(w))
    vals.append(float(c5.classify(tm)['phi_fwd']) / c5.PHI_CAP)
mean, se = float(np.mean(vals)), float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
print(f'\nforward power, same 5e4 rays:  split {ref_fwd:.5f}   '
      f'MC {mean:.5f} +- {se:.5f}   ({abs(mean - ref_fwd) / se:.1f} s.e.)')
""")

# ------------------------------------------------------------------ lighttools
add(MD, r"""
## 4. LightTools cross-check

Build the scene from **`examples/nonseq/c05_lighttools.md`** and export, then upload here.

**Export with `Analysis > Export Receiver Rays`, not the chart export.** Raw hits carry no
binning convention, so the same `nonseq.splat` call bins both sides identically and there is
no half-pixel question. A chart export has already been binned by LightTools with its own
cell convention, and a half-pixel offset shows up as a spurious radial ring in the difference
map that looks exactly like a physics error.

Upload these (LightTools' own `.1` run suffix is fine, **no renaming needed**):

| file | what |
|---|---|
| `lt_fwd_mc_rays.txt` | forward receiver, Run B (splitting OFF, 1e6 rays) |
| `lt_back_mc_rays.txt` | backward receiver, Run B |
| `lt_paths_mc.txt` | the `ForwardAll` ray-path table - the sharpest gate |

Two things must match the run above or the comparison measures the wrong thing:

1. **ray count** - `--rays` is passed as 1e6 below, so trace 1,000,000 rays in LightTools.
   Different budgets mean different noise, and the map L2 would then be reading the budget.
2. **receiver mesh** - 64x64 over +-20 mm forward, 32x32 over +-80 mm backward, per the bin
   table in section 1. This only matters for the *chart* export; the raw-ray path re-bins
   here and works at any mesh.

The cell is re-runnable: upload a better export and run it again to redraw the figure.
""")

add(CODE, r"""
# Upload the LightTools exports. Skips straight through if they are already there.
from google.colab import files
from IPython.display import Image, display
import glob, shutil

WANT = ['lt_fwd_mc_rays*', 'lt_back_mc_rays*', 'lt_paths_mc*']

have = [p for p in WANT if glob.glob(os.path.join(LT_DIR, p))]
if len(have) < len(WANT):
    print(f'in {LT_DIR}: {len(have)}/{len(WANT)} present - opening the file chooser')
    up = files.upload()
    for name in up:
        shutil.move(name, os.path.join(LT_DIR, name))

print(f'\n{LT_DIR}:')
for p in WANT:
    hits = [os.path.basename(h) for h in glob.glob(os.path.join(LT_DIR, p))]
    print(f'  [{"ok" if hits else "--"}] {p:<20} {hits if hits else "MISSING"}')
print()

r = subprocess.run([sys.executable, C05_COMPARE, '--run', 'b',
                    '--dir', LT_DIR, '--rays', str(RAYS)],
                   capture_output=True, text=True)
print(r.stdout)
if r.stderr:
    print('STDERR:', r.stderr[-3000:])

fig_path = os.path.join(LT_DIR, 'c05_maps_mc.png')
if os.path.exists(fig_path):
    display(Image(fig_path))
else:
    print('no figure written - see the output above')
""")

add(MD, r"""
### Reading the comparison

**Gated:** the three closed-form path fractions, the per-receiver integrated power (+-0.5 %
forward, +-1 % backward), and the energy ledger.

**Reported, not gated: the map L2.** Two independent Monte-Carlo maps at ~9 % per-bin noise
differ in L2 by about `sqrt(2) x 9 % = 13 %` even when both are exactly right, so a fixed L2
gate would be measuring the ray budget rather than the physics. The script prints the noise
floor implied by the meshes actually used and compares against that.

If a number disagrees, work down the forced-agreement checklist in `c05_lighttools.md` before
suspecting physics. The most likely single mistake in this build is **one optical property
applied to all four surfaces** - that gives `T1T2T3T4` = 0.8^4 = 40.96 % instead of 51.84 %.

## 5. What LightTools cannot do

Both coating derivatives come out of a single backward pass, and match central differences to
2e-8 and 6e-13:
""")

add(CODE, r"""
o, d, w = c5.sample_point_source(6000, seed=51, device=dev)

def phi_split(r1, r2, key, depth=6):
    els = c5.build_elements(R1=r1, R2=r2, device=dev)
    t, _ = c5.trace_split(o, d, w, els, c5.WAVELENGTH, max_depth=depth)
    return c5.classify(t)['phi_' + key]

h = 1e-4
for key, which in (('back', 'R1'), ('fwd', 'R2')):
    r1 = torch.tensor(c5.R1_COAT, device=dev, requires_grad=(which == 'R1'))
    r2 = torch.tensor(c5.R2_COAT, device=dev, requires_grad=(which == 'R2'))
    phi_split(r1, r2, key).backward()
    g = float((r1 if which == 'R1' else r2).grad)
    if which == 'R1':
        fd = (float(phi_split(c5.R1_COAT + h, c5.R2_COAT, key))
              - float(phi_split(c5.R1_COAT - h, c5.R2_COAT, key))) / (2 * h)
    else:
        fd = (float(phi_split(c5.R1_COAT, c5.R2_COAT + h, key))
              - float(phi_split(c5.R1_COAT, c5.R2_COAT - h, key))) / (2 * h)
    print(f'dPhi_{key}/d{which}   autograd {g:+.9e}   fd {fd:+.9e}   '
          f'rel {abs(g - fd) / abs(fd):.2e}')

print('\nThe finite differences are run on trace_split, not trace_mc: trace_mc samples its')
print('branch with u < rho where rho = R.detach(), so nudging R also moves rho and some rays')
print('flip branch. Autograd reports the pathwise derivative at fixed decisions; a fixed-seed')
print('difference quotient reports that plus the flips. trace_split takes both branches every')
print('time, so it has no decisions to flip and the comparison is exact.')
""")

nb = {'cells': cells,
      'metadata': {'accelerator': 'GPU',
                   'colab': {'name': 'c05_two_lens.ipynb', 'provenance': []},
                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'nbformat': 4,
      'nbformat_minor': 0}

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, 'c05_two_lens.ipynb')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')
print(f'wrote {path}  ({len(cells)} cells)')
