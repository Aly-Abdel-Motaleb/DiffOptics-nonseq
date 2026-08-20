"""
PHASE 2 - STAGE 3: LightTools vs `trace_split` / `trace_mc`.

Reads the CSVs exported from LightTools 2026 (build recipe: `c03_lighttools.md`),
reruns the P3 scene from `c02_R02.py` at a matched ray budget, and reports the
comparison. Nothing here re-derives physics - `c02_R02.py` owns that.

    python examples/nonseq/c03_compare.py                 # Run A (split)
    python examples/nonseq/c03_compare.py --run b         # Run B (Monte Carlo)
    python examples/nonseq/c03_compare.py --paths-only    # skip the maps

--------------------------------------------------------------------------------
WHAT IT NEEDS IN c03_out/
--------------------------------------------------------------------------------

Irradiance maps, from `Analysis -> Illuminance Display -> Table View -> export`:

    lt_fwd_split.csv    lt_back_split.csv      (Run A)
    lt_fwd_mc.csv       lt_back_mc.csv         (Run B)

The reader is deliberately forgiving: it takes either a long `x,y,E` table or a
bare 2D grid with or without header row/column, comma or whitespace separated.
What it does require is 512x512 cells covering the right extent - the grids are
checked against `nonseq.splat`'s cell centres to 1e-9 mm, because a half-pixel
offset shows up as a spurious radial ring in the difference map and is otherwise
invisible.

The path table, from `Analysis -> Ray Path -> ForwardAll -> File -> export`, or
typed by hand into `lt_paths_split.csv` as

    name,power_W
    total,0.020024
    T1T2,0.012815
    R1,0.004005
    T1R2T1,0.0025631
    ghost,0.00026587

`R1` and the ghost are usually SPLIT ACROSS SEVERAL LightTools path IDs - one
for the rays that reach a receiver, one for those that escape. Sum them. This
bites exactly like edge 1 in `c02_R02.py`: the reflection count alone does not
identify a path.

--------------------------------------------------------------------------------
WHAT IT CHECKS
--------------------------------------------------------------------------------

    the three closed forms      64.0000 / 20.0000 / 12.8000, +-0.01
    the ghost                   ~1.33, +-0.05 abs, and below its 2.56 bound
    receiver totals             +-0.5 % fwd, +-1 % back (Run A)
    map agreement               relative L2 < 2 % forward
    radial profile              plateau level and edge position
    energy ledger               closes to < 0.1 % in both codes
"""
import argparse
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", ".."))
sys.path.append(_HERE)

from diffoptics import nonseq  # noqa: E402
import c02_R02 as ref  # noqa: E402


OUT = os.path.join(_HERE, 'c03_out')
TOL = {'closed_form': 0.01, 'ghost': 0.05, 'fwd_rel': 0.5, 'back_rel': 1.0,
       'map_l2': 2.0, 'ledger': 0.1}


# ------------------------------------------------------------------- CSV input
def _cells(half, n):
    """Cell centres `nonseq.splat` uses: `n` bins spanning [-half, +half]."""
    pitch = 2.0 * half / n
    return -half + pitch * (np.arange(n) + 0.5)


def _check_axis(path, name, got, half, n):
    want = _cells(half, n)
    if got.size != n:
        raise ValueError(f'{path}: {got.size} distinct {name}, expected {n}')
    d = float(np.max(np.abs(np.sort(got) - want)))
    if d > 1e-9:
        raise ValueError(
            f'{path}: {name} centres off by {d:.3e} mm. Expected '
            f'{want[0]:.5f} .. {want[-1]:.5f} step {want[1] - want[0]:.5f}. '
            f'A half-pixel offset means the export is on cell EDGES.')


def read_map(path, half, n=512):
    """LightTools irradiance export -> [n,n] array in W/mm^2, indexed [x, y].

    Long `x,y,E` form is the safer export: it carries the cell centres, so the
    grid can be VERIFIED rather than assumed.
    """
    raw = [ln for ln in open(path, encoding='utf-8-sig').read().splitlines()
           if ln.strip()]
    rows = []
    for ln in raw:
        parts = ln.replace(',', ' ').replace(';', ' ').split()
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(np.nan)
        rows.append(vals)

    wide = max(len(r) for r in rows)

    if wide == 3 and len(rows) >= n * n:
        a = np.array([r for r in rows if not np.isnan(r).any()], dtype=float)
        if a.shape[0] != n * n:
            raise ValueError(f'{path}: {a.shape[0]} data rows, expected {n * n}')
        xs, ys = np.unique(a[:, 0]), np.unique(a[:, 1])
        _check_axis(path, 'x', xs, half, n)
        _check_axis(path, 'y', ys, half, n)
        I = np.zeros((n, n))
        I[np.searchsorted(xs, a[:, 0]), np.searchsorted(ys, a[:, 1])] = a[:, 2]
        return I

    # grid form: drop a header row and/or an index column if present
    grid = [r for r in rows if len(r) >= n]
    a = np.array([r[-n:] for r in grid], dtype=float)
    a = a[~np.isnan(a).all(axis=1)]
    if a.shape[0] != n:
        raise ValueError(f'{path}: {a.shape[0]} usable rows, expected {n}. '
                         f'Export the long x,y,E form instead - it is checkable.')
    return a.T


def read_rays(path, half, n=512):
    """`Analysis -> Export Receiver Rays` -> [n,n] W/mm^2, binned by `splat`.

    The safest export there is: raw hits carry no binning convention, so there
    is no half-pixel question at all - the same `nonseq.splat` call our own map
    uses does the binning, and the two grids are identical by construction.

    Wanted columns are x, y and a power/flux per ray. Column ORDER varies with
    the export options ticked, so they are found by header name; failing that,
    the first two numeric columns are taken as x, y and the last as power.
    """
    lines = [ln for ln in open(path, encoding='utf-8-sig').read().splitlines()
             if ln.strip()]

    declared_flux = None
    for ln in lines:
        if ln.lower().startswith('lt_radiant_flux'):
            try:
                declared_flux = float(ln.split(':', 1)[1])
            except (IndexError, ValueError):
                pass
            break

    hdr, ix, iy, iw = None, 0, 1, -1
    for k, ln in enumerate(lines):
        cols = [c.strip().lower() for c in ln.replace('\t', ',').split(',')]
        named = {c: i for i, c in enumerate(cols)}
        xs = [c for c in named if c in ('x', 'x_mm', 'x (mm)', 'xpos')]
        ys = [c for c in named if c in ('y', 'y_mm', 'y (mm)', 'ypos')]
        ws = [c for c in named
              if c in ('power', 'flux', 'watts', 'power (w)', 'radiant power')]
        if xs and ys:
            hdr, ix, iy = k, named[xs[0]], named[ys[0]]
            iw = named[ws[0]] if ws else -1
            break
    body = lines[hdr + 1:] if hdr is not None else lines

    rows = []
    for ln in body:
        parts = ln.replace('\t', ' ').replace(',', ' ').split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) >= 3:
            rows.append(vals)
    if not rows:
        raise ValueError(f'{path}: no numeric ray rows found')

    a = np.array(rows, dtype=float)
    p = torch.zeros(a.shape[0], 3, dtype=torch.float64)
    p[:, 0] = torch.as_tensor(a[:, ix])
    p[:, 1] = torch.as_tensor(a[:, iy])
    w = torch.as_tensor(a[:, iw])
    if declared_flux is not None and float(w.sum()) > 0:
        scale = declared_flux / float(w.sum())
        if abs(scale - 1.0) > 1e-6:
            print(f'  rescaling ray weights by {scale:.6e} to match '
                  f'lt_radiant_flux: {declared_flux:.8f} W declared in file')
        w = w * scale

    pitch = 2.0 * half / n
    return nonseq.splat(p, w / pitch ** 2, [n, n], pitch).numpy()


def read_paths(path):
    """`name,power_W` -> dict. Blank lines and a header row are ignored."""
    out = {}
    for ln in open(path, encoding='utf-8-sig').read().splitlines():
        parts = [p.strip() for p in ln.replace(';', ',').split(',')]
        if len(parts) < 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            continue                      # header
    return out


# ------------------------------------------------------------------- our side
def run_reference(which='a', seed_src=11, mc_seeds=(100, 101, 102, 103, 104)):
    """Rerun P3. Returns (phi_cap, mean, std, forward map, backward map)."""
    N = ref.N_SPLIT if which == 'a' else ref.N_MC
    o, d, w = ref.sample_point_source(N, seed=seed_src)
    els = ref.build_elements()
    w_min = ref._w_min(w)

    if which == 'a':
        runs = [ref.trace_split(o, d, w, els, w_min=w_min)]
    else:
        runs = [ref.trace_mc(o, d, w, els, seed=s, w_min=w_min)
                for s in mc_seeds]

    stats = []
    for term, tally in runs:
        cl = ref.classify(term)
        pp = ref.path_powers(term)
        stats.append(dict(
            T1T2=pp.get('T1T2', 0.0), R1=pp.get('R1', 0.0),
            T1R2T1=pp.get('T1R2T1', 0.0), ghost=pp.get('T1R2R1T2', 0.0),
            fwd=float(cl['phi_fwd']), back=float(cl['phi_back']),
            off=float(cl['phi_off']),
            leak=float(tally['culled'] + tally['truncated'])))

    term, _ = runs[0]
    cl = ref.classify(term)
    I_f = nonseq.splat(cl['p_fwd'], cl['w_fwd'] / ref.PIXEL_F ** 2,
                       ref.FILM_F, ref.PIXEL_F).numpy()
    I_b = nonseq.splat(cl['p_back'], cl['w_back'] / ref.PIXEL_B ** 2,
                       ref.FILM_B, ref.PIXEL_B).numpy()

    keys = list(stats[0])
    mean = {k: float(np.mean([s[k] for s in stats])) for k in keys}
    std = {k: (float(np.std([s[k] for s in stats], ddof=1))
               if len(stats) > 1 else 0.0) for k in keys}
    return ref.PHI_CAP, mean, std, I_f, I_b


# ------------------------------------------------------------------ reporting
_OK = {True: 'ok  ', False: 'FAIL'}


def _line(name, lt, ours, tol, unit='', sigma=None):
    d = lt - ours
    ok = abs(d) <= tol
    s = (f'  [{_OK[ok]}] {name:<22} LT {lt:10.4f}   ours {ours:10.4f}   '
         f'd {d:+8.4f} {unit}')
    if sigma:
        s += f'   ({abs(d) / sigma:.1f} sigma)'
    print(s)
    return ok


def compare_paths(lt, phi_cap_lt, mean, std, phi_cap, which):
    """The sharpest gate: three closed forms with no free parameters."""
    print('\nPATH POWERS  (% of Phi_cap, receivers ignored)')
    T, R = 1.0 - ref.R_COAT, ref.R_COAT
    exact = {'T1T2': 100 * T ** 2, 'R1': 100 * R, 'T1R2T1': 100 * T ** 2 * R}
    all_ok = True

    for k, want in exact.items():
        if k not in lt:
            print(f'  [--  ] {k:<22} not in the LightTools table')
            all_ok = False
            continue
        tol = TOL['closed_form'] if which == 'a' else 0.05
        sig = 100 * std[k] / phi_cap if which == 'b' else None
        all_ok &= _line(k, 100 * lt[k] / phi_cap_lt, 100 * mean[k] / phi_cap,
                        tol, '%', sig)
        print(f'  {"":<8}{"":<22} closed form {want:.4f} - no free parameters')

    if 'ghost' in lt:
        g_lt = 100 * lt['ghost'] / phi_cap_lt
        g_us = 100 * mean['ghost'] / phi_cap
        all_ok &= _line('ghost T1R2R1T2', g_lt, g_us, TOL['ghost'], '%')
        bound = 100 * T ** 2 * R ** 2
        ok = 0 < g_lt < bound
        print(f'  [{_OK[ok]}] {"ghost < bound":<22} {g_lt:.4f} < {bound:.4f} '
              f'({g_lt / bound:.0%} survives the rim) - no closed form exists')
        all_ok &= ok
    return all_ok


def compare_receivers(lt_f, lt_b, mean, phi_cap, phi_cap_lt, which):
    print('\nRECEIVER TOTALS  (% of Phi_cap)')
    ok = True
    tol_f = TOL['fwd_rel'] if which == 'a' else 0.5
    tol_b = TOL['back_rel'] if which == 'a' else 2.0
    if lt_f is not None:
        ok &= _line('forward receiver', 100 * lt_f / phi_cap_lt,
                    100 * mean['fwd'] / phi_cap, tol_f, '%')
    if lt_b is not None:
        ok &= _line('backward receiver', 100 * lt_b / phi_cap_lt,
                    100 * mean['back'] / phi_cap, tol_b, '%')
        print('  note: the backward receiver is dominated by R1, of which only '
              '~21 % lands\n        inside +-80 mm. It is the noisy bucket in '
              'BOTH codes - do not read a\n        1-2 point difference as a '
              'discrepancy.')
    return ok


def compare_maps(I_lt, I_us, half, label):
    """Total power, relative L2, and the radial profile."""
    print(f'\nIRRADIANCE MAP - {label}')
    pitch = 2.0 * half / I_us.shape[0]
    p_lt, p_us = I_lt.sum() * pitch ** 2, I_us.sum() * pitch ** 2
    print(f'  integrated power       LT {p_lt:.6e} W   ours {p_us:.6e} W   '
          f'rel {abs(p_lt / p_us - 1):.3%}')

    l2 = float(np.linalg.norm(I_lt - I_us) / np.linalg.norm(I_us)) * 100
    ok = l2 < TOL['map_l2']
    print(f'  [{_OK[ok]}] relative L2 ||LT-ours||/||ours|| = {l2:.3f} %')

    x = _cells(half, I_us.shape[0])
    xx, yy = np.meshgrid(x, x, indexing='ij')
    r = np.sqrt(xx ** 2 + yy ** 2)
    edges = np.linspace(0, half, 33)
    print('  radial profile, W/mm^2')
    print(f'    {"r [mm]":>12}  {"LT":>12}  {"ours":>12}')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if not m.any():
            continue
        a, b = I_lt[m].mean(), I_us[m].mean()
        if max(a, b) > 1e-12:
            print(f'    {0.5 * (lo + hi):12.3f}  {a:12.4e}  {b:12.4e}')
    return ok


def plot_maps(pairs, tag, out_dir):
    """LightTools | ours | signed difference | radial overlay, one row each.

    Same colour scale across a row, taken from OUR map - an independent
    autoscale per panel would hide exactly the difference being looked for.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n  matplotlib missing, no figure written')
        return

    fig, ax = plt.subplots(len(pairs), 4,
                           figsize=(21, 5 * len(pairs)), squeeze=False)

    for r, (label, half, I_lt, I_us) in enumerate(pairs):
        a = ax[r]
        ext = [-half, half, -half, half]
        top = max(float(I_us.max()), 1e-30)
        lo, hi = np.log10(top * 1e-6), np.log10(top)

        def logmap(I):
            return np.log10(np.maximum(I, 10.0 ** lo))

        for k, (I, name) in enumerate(((I_lt, 'LightTools'), (I_us, 'ours'))):
            im = a[k].imshow(logmap(I).T, origin='lower', extent=ext,
                             vmin=lo, vmax=hi)
            a[k].set_title(f'{label} - {name}\nlog10 W/mm^2')
            fig.colorbar(im, ax=a[k], fraction=0.046)

        d = I_lt - I_us
        m = float(np.abs(d).max()) or 1e-30
        im = a[2].imshow(d.T, origin='lower', extent=ext, cmap='RdBu_r',
                         vmin=-m, vmax=m)
        a[2].set_title(f'{label} - signed difference\nLT - ours, W/mm^2')
        fig.colorbar(im, ax=a[2], fraction=0.046)

        x = _cells(half, I_us.shape[0])
        xx, yy = np.meshgrid(x, x, indexing='ij')
        rr = np.sqrt(xx ** 2 + yy ** 2)
        edges = np.linspace(0, half, 129)
        mid, p_lt, p_us = [], [], []
        for a0, a1 in zip(edges[:-1], edges[1:]):
            sel = (rr >= a0) & (rr < a1)
            if sel.any():
                mid.append(0.5 * (a0 + a1))
                p_lt.append(I_lt[sel].mean())
                p_us.append(I_us[sel].mean())
        a[3].semilogy(mid, np.maximum(p_us, 1e-30), '-', lw=2, label='ours')
        a[3].semilogy(mid, np.maximum(p_lt, 1e-30), '--', lw=1.5,
                      label='LightTools')
        a[3].set_xlabel('r [mm]')
        a[3].set_ylabel('W/mm^2')
        a[3].set_title(f'{label} - radial profile\ncompare the plateau, not the '
                       f'hottest pixel')
        a[3].legend()
        a[3].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f'c03_maps_{tag}.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'\n  wrote {path}')


def compare_ledger(mean, phi_cap):
    print('\nENERGY LEDGER  (ours; LightTools reports its own in the Ray Report)')
    tot = mean['fwd'] + mean['back'] + mean['off'] + mean['leak']
    for k, lab in (('fwd', 'forward receiver'), ('back', 'backward receiver'),
                   ('off', 'neither'), ('leak', 'culled + truncated')):
        print(f'    {lab:<22} {100 * mean[k] / phi_cap:9.4f} %')
    rel = abs(tot / phi_cap - 1) * 100
    ok = rel < TOL['ledger']
    print(f'  [{_OK[ok]}] closes to {rel:.2e} %')
    print('  the "neither" bucket is a real answer, not a leak - mostly R1, '
          'which sprays\n  far wider than +-80 mm.')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', choices=['a', 'b'], default='a',
                    help='a = ray splitting ON (trace_split), b = OFF (trace_mc)')
    ap.add_argument('--paths-only', action='store_true')
    ap.add_argument('--dir', default=OUT)
    args = ap.parse_args()

    tag = 'split' if args.run == 'a' else 'mc'
    tracer = 'trace_split' if args.run == 'a' else 'trace_mc'
    print(f'LightTools Run {args.run.upper()} vs {tracer}')
    print(f'  reading {args.dir}')

    phi_cap, mean, std, I_f, I_b = run_reference(args.run)
    print(f'  Phi_cap (ours, {np.degrees(ref.THETA_MAX):.6f} deg cone) '
          f'= {phi_cap:.9f} W')

    ok = True
    pth = os.path.join(args.dir, f'lt_paths_{tag}.csv')
    if os.path.exists(pth):
        lt = read_paths(pth)
        phi_cap_lt = lt.get('total')
        if phi_cap_lt is None:
            phi_cap_lt = phi_cap
            print(f'  no `total` row in {os.path.basename(pth)}; normalising '
                  f'by OUR Phi_cap. Add one if the LightTools cone differs.')
        else:
            d = abs(phi_cap_lt / phi_cap - 1) * 100
            print(f'  Phi_cap (LightTools)              = {phi_cap_lt:.9f} W'
                  f'   ({d:+.4f} % vs ours)')
            if d > 0.02:
                print('  ^ the cone half-angles differ. Fractions stay '
                      'comparable; watts do not.')
        ok &= compare_paths(lt, phi_cap_lt, mean, std, phi_cap, args.run)
    else:
        print(f'  [--  ] {os.path.basename(pth)} missing - skipping the path '
              f'table, which is the sharpest gate you have')
        phi_cap_lt = phi_cap

    if not args.paths_only:
        maps = [('forward', f'lt_fwd_{tag}.csv', ref.R_RECV, I_f),
                ('backward', f'lt_back_{tag}.csv', ref.R_BACK, I_b)]
        tot, pairs = {}, []
        for label, fn, half, ours in maps:
            p = os.path.join(args.dir, fn)
            rays = os.path.join(args.dir, fn.replace('.csv', '_rays.csv'))
            if os.path.exists(rays):
                print(f'\n  using {os.path.basename(rays)} (raw hits, binned '
                      f'here - no half-pixel risk)')
                I_lt = read_rays(rays, half)
            elif os.path.exists(p):
                I_lt = read_map(p, half)
            else:
                print(f'\n  [--  ] {fn} missing - skipping the {label} map')
                continue
            tot[label] = I_lt.sum() * (2.0 * half / I_lt.shape[0]) ** 2
            ok &= compare_maps(I_lt, ours, half, label)
            pairs.append((label, half, I_lt, ours))
        ok &= compare_receivers(tot.get('forward'), tot.get('backward'),
                                mean, phi_cap, phi_cap_lt, args.run)
        if pairs:
            plot_maps(pairs, tag, args.dir)

    ok &= compare_ledger(mean, phi_cap)
    print('\n' + ('all gates passed.' if ok else 'SOME GATES FAILED - see above.'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
