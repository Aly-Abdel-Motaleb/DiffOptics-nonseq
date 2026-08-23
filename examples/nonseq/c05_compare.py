"""
PHASE 2 - STAGE 5b: the two-lens scene against LightTools.

`c05_two_lens.py` proves the tracer against closed forms and against itself.
This proves it against a commercial non-sequential tracer, on the same scene,
and draws the figure that says so.

    python examples/nonseq/c05_compare.py --rays 1000000
    python examples/nonseq/c05_compare.py --paths-only     # no maps
    python examples/nonseq/c05_compare.py --dir /some/where

Always writes `c05_maps_mc.png` into `--dir`: one row per receiver, four
columns - LightTools | ours | signed difference | radial profile. Receivers
whose LightTools export is missing still get a row, with the LightTools panel
left blank, so a partial upload still produces the figure rather than an
error.

Monte Carlo only - `trace_split` (deterministic ray splitting, LightTools Run
A) is not run here. This compares `trace_mc` against a LightTools MC export
(splitting OFF).

--------------------------------------------------------------------------------
WHAT IT LOOKS FOR IN `--dir`
--------------------------------------------------------------------------------

Per receiver, in this order, first match wins:

    lt_<name>_<tag>_rays*.txt     raw hits - what LightTools actually writes
    lt_<name>_<tag>_rays*.csv     raw hits, renamed
    lt_<name>_<tag>*.csv          chart export (illuminance table)

with `<name>` in {fwd, back} and `<tag>` = mc.
The `*` absorbs the run number LightTools appends - `lt_fwd_mc_rays.1.txt` is
matched as it comes off the export, no renaming needed. If several match, the
newest wins and the chosen name is printed.

Prefer the RAW HITS. A chart export has already been binned by LightTools using
its own cell convention, and a half-pixel disagreement shows up as a spurious
radial ring in the difference map that looks exactly like a physics error. Raw
hits carry no binning convention at all, so `nonseq.splat` - the same call our
own map uses - does the binning for both sides and the two grids are identical
by construction.

--------------------------------------------------------------------------------
THE MESH IS NOT A FREE CHOICE
--------------------------------------------------------------------------------

Per-bin relative noise is `eps = N / sqrt(R)` for `R` rays landing on an `N x N`
receiver, so the mesh follows from the ray budget:

    N = 16 * floor(0.1 * sqrt(R) / 16)

`c05_two_lens.bins_for` is the one implementation and this script calls it, so
the mesh chooses itself from `--rays` unless `--n` overrides it. At 1e6 rays
that is 64x64 forward and 32x32 backward. **The LightTools receiver must be
built with the same mesh** - `read_map` checks the exported cell centres against
`nonseq.splat`'s to 1e-9 mm and refuses a mismatch, because a half-pixel offset
is invisible in a picture and fatal in an L2.

The raw-hit path has no such constraint: it re-bins the hits here, so it is
correct at any mesh. That is the other reason to prefer it.

--------------------------------------------------------------------------------
WHAT IS GATED, AND WHAT IS ONLY REPORTED
--------------------------------------------------------------------------------

Gated: per-receiver integrated power, the three closed-form path fractions, and
the energy ledger.

Reported but NOT gated: the map L2. Two Monte Carlo maps at ~9 % per-bin noise
differ by ~13 % in L2 no matter how right they both are, so a hard L2 gate would
be measuring the ray budget, not the physics. The script prints the noise floor
implied by the meshes actually used and compares the L2 against THAT.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", ".."))
sys.path.append(_HERE)

from diffoptics import nonseq  # noqa: E402
import c05_two_lens as c5  # noqa: E402
# Readers and formatters are c03's, unchanged - they already parse every export
# form LightTools produces, including the raw `.txt`. `c03_compare` guards its
# own `main`, so importing it runs nothing.
from c03_compare import (  # noqa: E402
    _cells, _line, read_map, read_paths, read_rays)

OUT = os.path.join(_HERE, 'c05_out')

TOL = {'closed_form_mc': 0.05,   # % of Phi_cap, on top of the MC sigma
       'recv_rel': 0.5,          # % of Phi_cap, forward
       'recv_rel_back': 1.0,     # backward catches a wider, noisier spray
       'ledger': 0.1}

_OK = {True: 'ok  ', False: 'FAIL'}


# ------------------------------------------------------------------ locating
def find_export(directory, name, tag):
    """First LightTools export matching `name`/`tag`, newest of its class.

    Returns (path, kind) with kind in {'rays', 'map'}, or (None, None).
    Raw-hit exports win over chart exports regardless of date - they are
    strictly better data, not merely different.
    """
    for pattern, kind in ((f'lt_{name}_{tag}_rays*.txt', 'rays'),
                          (f'lt_{name}_{tag}_rays*.csv', 'rays'),
                          (f'lt_{name}_{tag}*.csv', 'map'),
                          (f'lt_{name}_{tag}*.txt', 'map')):
        hits = sorted(glob.glob(os.path.join(directory, pattern)),
                      key=os.path.getmtime, reverse=True)
        # The 'map' patterns also match the '_rays' names; drop those so a
        # raw-hit file is never read with the wrong reader.
        if kind == 'map':
            hits = [h for h in hits if '_rays' not in os.path.basename(h)]
        if hits:
            return hits[0], kind
    return None, None


def find_paths(directory, tag):
    for ext in ('csv', 'txt'):
        hits = sorted(glob.glob(os.path.join(directory, f'lt_paths_{tag}*.{ext}')),
                      key=os.path.getmtime, reverse=True)
        if hits:
            return hits[0]
    return None


# ------------------------------------------------------------------ our side
def run_reference(rays=None, seed_src=11,
                  mc_seeds=(100, 101, 102, 103, 104), n_override=None,
                  strict=False):
    """Rerun the c05 scene with `trace_mc`. Returns (phi_cap, mean, std, maps, bins).

    `maps` is {receiver name: [n,n] W/mm^2}, `bins` the table from
    `c05.bin_table`. Five `trace_mc` seeds are run so every reported number
    carries a standard deviation.

    The mesh comes from the FIRST run's landing counts and is then held fixed
    across the remaining seeds - otherwise each seed would splat onto a slightly
    different grid and the spread would mix two effects.
    """
    N = int(rays or int(1e6))
    o, d, w = c5.sample_point_source(N, seed=seed_src)
    els = c5.build_elements()
    w_min = c5._w_min(w)

    runs = []
    for s in mc_seeds:
        runs.append(c5.trace_mc(o, d, w, els, c5.WAVELENGTH, seed=s,
                                max_depth=c5.MAX_DEPTH, w_min=w_min))

    stats = []
    for term, tally in runs:
        cl = c5.classify(term)
        pp = c5.path_powers(term)
        st = {k: pp.get(k, 0.0) / c5.PHI_CAP for k in c5.closed_form()}
        for rc in c5.RECEIVERS:
            st[rc['name']] = float(cl['phi_' + rc['name']]) / c5.PHI_CAP
        st['off'] = float(cl['phi_off']) / c5.PHI_CAP
        st['leak'] = float(tally['culled'] + tally['truncated']) / c5.PHI_CAP
        stats.append(st)

    term0, _ = runs[0]
    cl0 = c5.classify(term0)
    bins = c5.bin_table(cl0, launched=N, printout=False, strict=strict)
    if n_override:
        # Recompute the noise columns for the mesh actually used, or the table
        # would report the rule's numbers next to an overridden N.
        for k, v in n_override.items():
            if k not in bins:
                raise SystemExit(f'--n {k}=... : no receiver named {k!r}; '
                                 f'have {[r["name"] for r in c5.RECEIVERS]}')
            rc = next(r for r in c5.RECEIVERS if r['name'] == k)
            n = int(v)
            bins[k]['n'] = n
            bins[k]['eps_pred'] = n / np.sqrt(max(bins[k]['rays_on'], 1))
            bins[k].update(c5.measured_eps(cl0['p_' + k], cl0['w_' + k], n,
                                           rc['half']))

    maps = {}
    for rc in c5.RECEIVERS:
        key, n = rc['name'], bins[rc['name']]['n']
        pitch = 2.0 * rc['half'] / n
        maps[key] = nonseq.splat(cl0['p_' + key], cl0['w_' + key] / pitch ** 2,
                                 [n, n], pitch).numpy()

    keys = list(stats[0])
    mean = {k: float(np.mean([s[k] for s in stats])) for k in keys}
    std = {k: float(np.std([s[k] for s in stats], ddof=1)) if len(stats) > 1
           else 0.0 for k in keys}
    return c5.PHI_CAP, mean, std, maps, bins, N


# ---------------------------------------------------------------- comparisons
def compare_paths(lt, phi_cap_lt, mean, std):
    """The three unshared buckets, LightTools vs ours vs the closed form.

    Only three - see `c05_two_lens`'s docstring. With four partial surfaces most
    (hits, reflections) buckets hold more than one path and one of the biggest
    is aperture-clipped, so they have no closed form to compare against.
    """
    print('\nPOWER BY PATH  (% of Phi_cap)')
    if not lt:
        print('  no LightTools path table - this is the sharpest gate you have,'
              '\n  export it (Ray Paths -> ForwardAll) and rerun')
        return True
    exact = {k: 100 * v for k, v in c5.closed_form().items()}
    tol = TOL['closed_form_mc']
    ok = True
    for name, want in exact.items():
        if name not in lt:
            print(f'  [--  ] {name:<12} not in the LightTools table')
            continue
        a = 100 * lt[name] / phi_cap_lt
        b = 100 * mean[name]
        sig = 100 * std[name]
        ok &= _line(f'{name} (exact {want:.3f})', a, b,
                    tol + (3 * sig), '%', sigma=sig)
    return ok


def compare_receivers(tot, mean, phi_cap, phi_cap_lt):
    """Integrated power on each receiver, LightTools vs ours."""
    print('\nRECEIVER TOTALS  (% of Phi_cap)')
    ok = True
    for rc in c5.RECEIVERS:
        key = rc['name']
        if tot.get(key) is None:
            print(f'  [--  ] {rc["label"]:<22} no LightTools map')
            continue
        tol = TOL['recv_rel'] if rc['sign'] > 0 else TOL['recv_rel_back']
        ok &= _line(f'{rc["label"]} receiver', 100 * tot[key] / phi_cap_lt,
                    100 * mean[key], tol, '%')
    print('  the backward receiver is dominated by R1, of which only ~21 % '
          'lands inside\n  +-80 mm. It is the noisy bucket in BOTH codes - do '
          'not read a 1-2 point\n  difference there as a discrepancy.')
    return ok


def compare_maps(I_lt, I_us, half, label, eps_floor):
    """Total power, relative L2 against the noise floor, radial profile.

    The L2 is REPORTED against `eps_floor`, not gated against a fixed number.
    Two independent Monte Carlo maps at per-bin noise `eps` differ in L2 by
    about `sqrt(2) * eps` even when both are exactly right, so the honest
    statement is "the L2 is / is not consistent with the noise these two ray
    budgets buy".
    """
    print(f'\nIRRADIANCE MAP - {label}')
    pitch = 2.0 * half / I_us.shape[0]
    p_lt, p_us = I_lt.sum() * pitch ** 2, I_us.sum() * pitch ** 2
    print(f'  integrated power       LT {p_lt:.6e} W   ours {p_us:.6e} W   '
          f'rel {abs(p_lt / p_us - 1):.3%}')

    l2 = float(np.linalg.norm(I_lt - I_us) / np.linalg.norm(I_us)) * 100
    floor = 100 * np.sqrt(2.0) * eps_floor
    ok = l2 <= max(floor, 2.0)
    print(f'  [{_OK[ok]}] relative L2 = {l2:.3f} %   noise floor for these '
          f'meshes ~{floor:.1f} %')
    if not ok:
        print('         above the floor: either the meshes/ray budgets do not '
              'match, or\n         this is a real disagreement. Read the radial'
              ' profile and the path\n         table before concluding either.')

    x = _cells(half, I_us.shape[0])
    xx, yy = np.meshgrid(x, x, indexing='ij')
    r = np.sqrt(xx ** 2 + yy ** 2)
    edges = np.linspace(0, half, 25)
    print('  radial profile, W/mm^2')
    print(f'    {"r [mm]":>12}  {"LT":>12}  {"ours":>12}  {"rel":>8}')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if not m.any():
            continue
        a, b = I_lt[m].mean(), I_us[m].mean()
        if max(a, b) > 1e-12:
            rel = (a / b - 1) if b > 0 else np.nan
            print(f'    {0.5 * (lo + hi):12.3f}  {a:12.4e}  {b:12.4e}  '
                  f'{rel:+7.1%}')
    return ok


def compare_ledger(mean, phi_cap):
    print('\nENERGY LEDGER  (ours; LightTools reports its own in the Ray Report)')
    keys = [(rc['name'], rc['label'] + ' receiver') for rc in c5.RECEIVERS]
    keys += [('off', 'neither'), ('leak', 'culled + truncated')]
    tot = sum(mean[k] for k, _ in keys)
    for k, lab in keys:
        print(f'    {lab:<24} {100 * mean[k]:9.4f} %')
    rel = abs(tot - 1.0) * 100
    ok = rel < TOL['ledger']
    print(f'  [{_OK[ok]}] closes to {rel:.2e} %')
    print('  the "neither" bucket is a real answer, not a leak - mostly R1, '
          'which sprays\n  far wider than the +-80 mm backward receiver.')
    return ok


# -------------------------------------------------------------------- figure
def plot_maps(rows, tag, out_dir):
    """LightTools | ours | signed difference | radial overlay, one row each.

    Adapted from `c03_compare.plot_maps`, with one change that matters here:
    `I_lt = None` is allowed. A receiver whose export has not been uploaded yet
    still gets its row, with the LightTools and difference panels blanked, so
    the figure is produced from a partial upload instead of failing.

    Colour scale is taken from OUR map and shared across the row. Per-panel
    autoscale would hide exactly the difference being looked for.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n  matplotlib missing, no figure written')
        return None

    fig, ax = plt.subplots(len(rows), 4, figsize=(21, 5 * len(rows)),
                           squeeze=False)

    for r, (label, half, I_lt, I_us) in enumerate(rows):
        a = ax[r]
        ext = [-half, half, -half, half]
        top = max(float(I_us.max()), 1e-30)
        lo, hi = np.log10(top * 1e-6), np.log10(top)

        def logmap(I):
            return np.log10(np.maximum(I, 10.0 ** lo))

        panels = [(I_lt, 'LightTools'), (I_us, 'ours')]
        for k, (I, name) in enumerate(panels):
            if I is None:
                a[k].text(0.5, 0.5, 'no LightTools export', ha='center',
                          va='center', transform=a[k].transAxes, color='0.5')
                a[k].set_title(f'{label} - {name}')
                a[k].set_xticks([])
                a[k].set_yticks([])
                continue
            im = a[k].imshow(logmap(I).T, origin='lower', extent=ext,
                             vmin=lo, vmax=hi)
            a[k].set_title(f'{label} - {name}\nlog10 W/mm^2')
            a[k].set_xlabel('x [mm]')
            a[k].set_ylabel('y [mm]')
            fig.colorbar(im, ax=a[k], fraction=0.046)

        if I_lt is None:
            a[2].set_title(f'{label} - signed difference')
            a[2].set_xticks([])
            a[2].set_yticks([])
        else:
            d = I_lt - I_us
            m = float(np.abs(d).max()) or 1e-30
            im = a[2].imshow(d.T, origin='lower', extent=ext, cmap='RdBu_r',
                             vmin=-m, vmax=m)
            a[2].set_title(f'{label} - signed difference\nLT - ours, W/mm^2')
            fig.colorbar(im, ax=a[2], fraction=0.046)

        x = _cells(half, I_us.shape[0])
        xx, yy = np.meshgrid(x, x, indexing='ij')
        rr = np.sqrt(xx ** 2 + yy ** 2)
        edges = np.linspace(0, half, 65)
        mid, p_lt, p_us = [], [], []
        for a0, a1 in zip(edges[:-1], edges[1:]):
            sel = (rr >= a0) & (rr < a1)
            if sel.any():
                mid.append(0.5 * (a0 + a1))
                p_us.append(I_us[sel].mean())
                p_lt.append(I_lt[sel].mean() if I_lt is not None else np.nan)
        a[3].semilogy(mid, np.maximum(p_us, 1e-30), '-', lw=2, label='ours')
        if I_lt is not None:
            a[3].semilogy(mid, np.maximum(p_lt, 1e-30), '--', lw=1.5,
                          label='LightTools')
        a[3].set_xlabel('r [mm]')
        a[3].set_ylabel('W/mm^2')
        a[3].set_title(f'{label} - radial profile\ncompare the plateau, not '
                       f'the hottest pixel')
        a[3].legend()
        a[3].grid(alpha=0.3)

    fig.suptitle(f'two lenses, R1 = {c5.R1_COAT} (S1,S2), '
                 f'R2 = {c5.R2_COAT} (S3,S4)  -  LightTools vs '
                 f'diffoptics.nonseq', fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f'c05_maps_{tag}.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'\n  wrote {path}')
    return path


# ---------------------------------------------------------------------- main
def _parse_n(items):
    out = {}
    for it in items or []:
        if '=' not in it:
            raise SystemExit(f'--n wants <receiver>=<int>, got {it!r}')
        k, v = it.split('=', 1)
        out[k.strip()] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser(
        description='two-lens scene vs LightTools (Monte Carlo only); '
                    'always writes the map figure')
    ap.add_argument('--dir', default=OUT, help='where the LightTools exports are')
    ap.add_argument('--rays', type=int, default=None,
                    help='OUR ray budget. Set it to the LightTools ray count so '
                         'both maps carry the same MC noise. Default 1e6.')
    ap.add_argument('--n', action='append', metavar='NAME=INT',
                    help='override the bin rule for one receiver, e.g. --n fwd=64')
    ap.add_argument('--strict-bins', action='store_true',
                    help='size the mesh by MEASURED noise instead of the '
                         'N/sqrt(R) rule (coarser where the flux is concentrated)')
    ap.add_argument('--paths-only', action='store_true')
    args = ap.parse_args()

    tag = 'mc'
    os.makedirs(args.dir, exist_ok=True)
    print('LightTools (splitting OFF) vs trace_mc')
    print(f'  two lenses: R1 = {c5.R1_COAT} (S1,S2), R2 = {c5.R2_COAT} (S3,S4)')
    print(f'  reading {args.dir}')

    phi_cap, mean, std, maps, bins, n_rays = run_reference(
        rays=args.rays, n_override=_parse_n(args.n), strict=args.strict_bins)
    print(f'  Phi_cap (ours) = {phi_cap:.9f} W from {n_rays:.0e} rays\n')
    print('  mesh, chosen by N = 0.1 * sqrt(rays landing on the receiver):')
    print(f'  {"receiver":10s} {"rays on":>10s} {"N":>5s} {"eps pred":>9s} '
          f'{"eps meas":>9s} {"N strict":>9s}')
    for rc in c5.RECEIVERS:
        b = bins[rc['name']]
        print(f'  {rc["label"]:10s} {b["rays_on"]:10d} {b["n"]:5d} '
              f'{b["eps_pred"]:8.2%} {b["median"]:8.2%} {b["n_strict"]:9d}')

    ok = True

    p_paths = find_paths(args.dir, tag)
    lt_paths = read_paths(p_paths) if p_paths else {}
    if p_paths:
        print(f'\n  path table: {os.path.basename(p_paths)}')
    phi_cap_lt = lt_paths.get('total', phi_cap)
    ok &= compare_paths(lt_paths, phi_cap_lt, mean, std)

    if not args.paths_only:
        rows, tot = [], {}
        for rc in c5.RECEIVERS:
            key, half = rc['name'], rc['half']
            ours = maps[key]
            n = ours.shape[0]
            path, kind = find_export(args.dir, key, tag)
            if path is None:
                print(f'\n  [--  ] no lt_{key}_{tag}* in {args.dir} - '
                      f'{rc["label"]} row will show ours only')
                rows.append((rc['label'], half, None, ours))
                tot[key] = None
                continue
            print(f'\n  {rc["label"]}: {os.path.basename(path)}  ({kind})')
            if kind == 'rays':
                I_lt = read_rays(path, half, n)
            else:
                I_lt = read_map(path, half, n)
            eps_floor = max(bins[key]['median'], bins[key]['eps_pred'])
            ok &= compare_maps(I_lt, ours, half, rc['label'], eps_floor)
            pitch = 2.0 * half / n
            tot[key] = I_lt.sum() * pitch ** 2
            rows.append((rc['label'], half, I_lt, ours))

        if any(v is not None for v in tot.values()):
            ok &= compare_receivers(tot, mean, phi_cap, phi_cap_lt)
        plot_maps(rows, tag, args.dir)

    ok &= compare_ledger(mean, phi_cap)

    print('\nALL GATES PASSED.' if ok else '\nSOME GATES FAILED - see above.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
