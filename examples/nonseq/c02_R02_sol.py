"""
PHASE 2 - STAGE 2: R = 0.2 on both lens surfaces.  [SOLUTION]

Reference implementation of `c02_R02.py` - read that file first, all the
reasoning lives there. Same tests, same tolerances.

Measured on this machine (float64, N_split = 50000, N_mc = 200000, depth 10):

    T1  ledger closes                    rel 4.4e-16
        fwd 0.641769  back 0.170727  off 0.187416  culled 5.3e-06  trunc 8.3e-05
    T2  direct forward T1T2              0.640000 x Phi_cap   = (1-R)^2, exact
    T3  backward R1                      0.200000 x Phi_cap   = R, exact
        backward T1R2T1                  0.128000 x Phi_cap   = (1-R)^2 R, exact
        ghost T1R2R1T2                   0.013237 x Phi_cap   = 52 % of its
                                         unvignetted bound (1-R)^2 R^2 = 0.0256
    T4  MC mean vs split, 5 seeds        2.5e-06 apart, se 5.4e-06
        error vs ray count, log-log      slope -0.52
    T5  sequential forward               1.000000 x Phi_cap (no coating loss)
        non-seq forward                  0.6418 of it, and 0.171 x Phi_cap
                                         goes backwards, which sequential
                                         cannot represent at all
    G1  dPhi_back/dR    auto vs fd       2.645045e-02 both
    G2  d<r_fwd>/dc     auto vs fd       -4.984348e+02 both
"""
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do  # noqa: E402
from diffoptics import nonseq  # noqa: E402
from diffoptics.nonseq import (  # noqa: E402
    Element, closest_hit, intersect_one, propagate_to_z, refract)


# ------------------------------------------------------------------ constants
WAVELENGTH = 532.8
S_SRC      = 40.0
R_LENS     = 12.7
THICK      = 6.5
Z_S1       = 40.0
Z_S2       = Z_S1 + THICK
P_TOTAL    = 1.0

Z_RECV = 200.0                      # forward receiver
R_RECV = 16.0
FILM_F = [512, 512]
PIXEL_F = 2 * R_RECV / FILM_F[0]

Z_BACK = -80.0                      # backward receiver
R_BACK = 80.0
FILM_B = [512, 512]
PIXEL_B = 2 * R_BACK / FILM_B[0]

R_COAT = 0.2
MAX_DEPTH = 10
W_MIN_REL = 1e-5

N_AIR   = do.Material('air').ior(WAVELENGTH)
N_GLASS = do.Material('n-bk7').ior(WAVELENGTH)
N_REL   = float(N_GLASS / N_AIR)

C_ASPH = 1.0 / (S_SRC * (N_REL - 1.0))
K_ASPH = -N_REL ** 2


def sag(r):
    r2 = np.asarray(r, dtype=np.float64) ** 2
    return C_ASPH * r2 / (1 + np.sqrt(1 - (1 + K_ASPH) * C_ASPH ** 2 * r2))


THETA_MAX = float(np.arctan(R_LENS / (S_SRC + sag(R_LENS))))
PHI_CAP   = P_TOTAL * (1 - np.cos(THETA_MAX)) / 2.0


# --------------------------------------------------------------------- scenes
def build_lensgroup(device=torch.device('cpu')):
    lens = do.Lensgroup(origin=np.array([0, 0, Z_S1]), device=device)
    lens.load(
        [do.Aspheric(R_LENS, 0.0,   c=C_ASPH, k=K_ASPH, device=device),
         do.Aspheric(R_LENS, THICK, c=0.0,               device=device)],
        [do.Material('air'), do.Material('n-bk7'), do.Material('air')],
    )
    lens.d_sensor = Z_RECV
    lens.r_last = R_LENS
    lens.film_size = FILM_F
    lens.pixel_size = PIXEL_F
    return lens


def build_elements(R=R_COAT, c=C_ASPH, k=K_ASPH, grad_c=False):
    """The two coated surfaces. `R` may be a float or a tensor (G1 needs a
    tensor); `grad_c=True` makes S1's curvature a leaf for G2."""
    eye = torch.eye(3)
    s1_surf = do.Aspheric(R_LENS, 0.0, c=c, k=k)
    if grad_c:
        s1_surf.c = s1_surf.c.clone().requires_grad_(True)
    s1 = Element(s1_surf,
                 do.Transformation(eye, torch.Tensor([0.0, 0.0, Z_S1])),
                 n_in='air', n_out='n-bk7', kind='partial', R_fixed=R, name='S1')
    s2 = Element(do.Aspheric(R_LENS, 0.0, c=0.0),
                 do.Transformation(eye, torch.Tensor([0.0, 0.0, Z_S2])),
                 n_in='n-bk7', n_out='air', kind='partial', R_fixed=R, name='S2')
    return [s1, s2]


def sample_point_source(N, seed=0):
    return nonseq.sample_point_source(N, THETA_MAX, seed=seed, P=P_TOTAL)


# ------------------------------------------------------------------- physics
def reflect(d, n_geom):
    """Mirror `d` about the plane of `n_geom`. The normal appears twice, so its
    orientation cancels - no flip needed."""
    return d - 2.0 * torch.sum(d * n_geom, dim=-1, keepdim=True) * n_geom


def fresnel_R(d, n_geom, eta):
    """Unpolarized Fresnel reflectance, used only when `R_fixed is None`."""
    cosi = torch.abs(torch.sum(d * n_geom, dim=-1))
    sint2 = eta ** 2 * (1.0 - cosi ** 2)
    cost = torch.sqrt(torch.clamp(1.0 - sint2, min=0.0))
    rs = ((eta * cosi - cost) / (eta * cosi + cost)) ** 2
    rp = ((eta * cost - cosi) / (eta * cost + cosi)) ** 2
    return torch.where(sint2 >= 1.0, torch.ones_like(cosi), 0.5 * (rs + rp))


def reflectance(el, d, n_geom, eta):
    """R at this element: the fixed override if it has one, else Fresnel.

    Multiplying by `ones_like` (rather than `expand`) keeps a tensor `R_fixed`
    on the graph, which is what G1 differentiates.
    """
    if el.R_fixed is None:
        return fresnel_R(d, n_geom, eta)
    return torch.ones_like(eta) * el.R_fixed


def _interaction(o, d, elements, eid, wavelength):
    """One hit for a batch: geometry, media, R, and both outgoing directions.

    TIR is not a dead end here (dO's `_refract` kills those rays, optics.py:1014);
    it is R = 1, the one case where a 'partial' surface behaves as a mirror.
    """
    p, n_geom, ok = intersect_one(o, d, elements, eid)

    eta = torch.ones_like(o[..., 0])
    R = torch.zeros_like(o[..., 0])
    for i, el in enumerate(elements):
        m = eid == i
        if bool(m.any()):
            eta[m] = el.eta_at(d[m], n_geom[m], wavelength)
            R[m] = reflectance(el, d[m], n_geom[m], eta[m])

    ok_t, d_refr = refract(d, n_geom, eta)
    R = torch.where(ok_t, R, torch.ones_like(R))
    return p, ok, R, d_refr, reflect(d, n_geom), ok_t


# -------------------------------------------------------------------- tracers
def trace_split(o, d, w, elements, wavelength=WAVELENGTH, max_depth=MAX_DEPTH,
                w_min=0.0):
    """Deterministic ray splitting: every hit spawns BOTH children.

    Generation by generation - the whole depth-k front is one batch, so the
    number of `newtons_method` calls is `max_depth * len(elements)`, not one per
    path. The front does not blow up as 2^k because most reflected children
    leave the scene immediately and retire at the next `closest_hit`.

    Returns (terminal, tally):
        terminal: dict with o, d, w, nrefl, nhit [M] for every ray that left
            the scene - free-flying, so a receiver is just `propagate_to_z`.
            nhit is not decoration: R1 and T1R2T1 both have nrefl == 1 and are
            the two brightest backward paths, so only the hit count tells them
            apart.
        tally: dict with 'culled' (weights dropped under `w_min`) and
            'truncated' (weights still bouncing at `max_depth`). Both are
            needed for the ledger to close.
    """
    o_out, d_out, w_out, nr_out, nh_out = [], [], [], [], []
    culled = torch.zeros((), dtype=w.dtype)

    ignore = torch.full((o.shape[0],), -1, dtype=torch.long)
    nrefl = torch.zeros(o.shape[0], dtype=torch.long)
    nhit = torch.zeros(o.shape[0], dtype=torch.long)

    for _ in range(max_depth):
        if o.shape[0] == 0:
            break

        _, eid = closest_hit(o, d, elements, ignore_id=ignore)

        gone = eid < 0
        if bool(gone.any()):
            o_out.append(o[gone]); d_out.append(d[gone])
            w_out.append(w[gone]); nr_out.append(nrefl[gone])
            nh_out.append(nhit[gone])
        live = ~gone
        if not bool(live.any()):
            o = o[live]
            break
        o, d, w, nrefl, nhit, eid = (o[live], d[live], w[live], nrefl[live],
                                     nhit[live], eid[live])

        p, ok, R, d_refr, d_refl, ok_t = _interaction(o, d, elements, eid,
                                                      wavelength)
        if not bool(ok.all()):
            culled = culled + w[~ok].sum()
            o, d, w, nrefl, nhit, eid, p, R, d_refr, d_refl, ok_t = [
                x[ok] for x in (o, d, w, nrefl, nhit, eid, p, R, d_refr,
                                d_refl, ok_t)]

        w_tr, w_re = w * (1.0 - R), w * R
        keep_tr = ok_t & (w_tr > w_min)
        keep_re = w_re > w_min
        culled = culled + w_tr[~keep_tr].sum() + w_re[~keep_re].sum()

        o = torch.cat((p[keep_tr], p[keep_re]))
        d = torch.cat((d_refr[keep_tr], d_refl[keep_re]))
        w = torch.cat((w_tr[keep_tr], w_re[keep_re]))
        nrefl = torch.cat((nrefl[keep_tr], nrefl[keep_re] + 1))
        nhit = torch.cat((nhit[keep_tr], nhit[keep_re])) + 1
        ignore = torch.cat((eid[keep_tr], eid[keep_re]))

    truncated = w.sum() if o.shape[0] else torch.zeros((), dtype=culled.dtype)

    terminal = {'o': torch.cat(o_out), 'd': torch.cat(d_out),
                'w': torch.cat(w_out), 'nrefl': torch.cat(nr_out),
                'nhit': torch.cat(nh_out)}
    return terminal, {'culled': culled, 'truncated': truncated}


def trace_mc(o, d, w, elements, wavelength=WAVELENGTH, seed=0,
             max_depth=MAX_DEPTH, w_min=0.0):
    """Monte Carlo: ONE child per hit, chosen with probability rho.

    Reflect with probability rho and divide the weight by rho, transmit with
    1 - rho and divide by that - so the expected weight equals the split
    tracer's total and the estimator is unbiased. rho defaults to `R.detach()`,
    which is the zero-variance choice when R is fixed: every path then carries
    its exact weight and only the PATH SET is random.

    Detaching matters: rho is a sampling decision, not physics. Leaving it on
    the graph would put the discrete branch choice into dPhi/dR.

    Same return shape as `trace_split`, so both feed `classify`.
    """
    N = o.shape[0]
    g = torch.Generator().manual_seed(int(seed))
    eps = torch.finfo(w.dtype).eps

    done = torch.zeros(N, dtype=torch.bool)
    nrefl = torch.zeros(N, dtype=torch.long)
    nhit = torch.zeros(N, dtype=torch.long)
    ignore = torch.full((N,), -1, dtype=torch.long)
    culled = torch.zeros((), dtype=w.dtype)

    for _ in range(max_depth):
        act = ~done
        if not bool(act.any()):
            break

        _, eid = closest_hit(o, d, elements, ignore_id=ignore)
        eid = torch.where(act, eid, torch.full_like(eid, -1))
        hit = eid >= 0
        done = done | (act & ~hit)            # left the scene, still flying
        if not bool(hit.any()):
            break

        p, ok, R, d_refr, d_refl, ok_t = _interaction(o, d, elements,
                                                      torch.where(hit, eid,
                                                                  torch.zeros_like(eid)),
                                                      wavelength)
        culled = culled + w[hit & ~ok].sum()
        done = done | (hit & ~ok)
        hit = hit & ok

        rho = R.detach().clone()
        for i, el in enumerate(elements):
            if el.rho_fixed is not None:
                rho = torch.where(eid == i, torch.ones_like(rho) * el.rho_fixed, rho)
        rho = torch.where(ok_t, rho, torch.ones_like(rho)).clamp(eps, 1.0)

        u = torch.rand(N, generator=g, dtype=w.dtype)
        go_re = hit & (u < rho)
        go_tr = hit & ~go_re

        w = torch.where(go_re, w * R / rho, w)
        w = torch.where(go_tr, w * (1.0 - R) / (1.0 - rho).clamp(min=eps), w)
        d = torch.where(go_re[..., None], d_refl,
                        torch.where(go_tr[..., None], d_refr, d))
        o = torch.where(hit[..., None], p, o)
        nrefl = nrefl + go_re.long()
        nhit = nhit + hit.long()
        ignore = torch.where(hit, eid, torch.full_like(eid, -1))

        low = hit & (w < w_min)
        culled = culled + w[low].sum()
        done = done | low

    truncated = w[~done].sum()
    keep = done & (w > 0)
    terminal = {'o': o[keep], 'd': d[keep], 'w': w[keep], 'nrefl': nrefl[keep],
                'nhit': nhit[keep]}
    return terminal, {'culled': culled, 'truncated': truncated}


# ------------------------------------------------------------------ receivers
def classify(term):
    """Sort free-flying terminal rays onto the two receivers.

    Everything that misses both is 'off', which the ledger needs: the backward
    pattern is wide (the asphere's mirror focus is nowhere near the source), so
    a good part of R1 flies past a +-80 mm receiver.
    """
    out = {}
    for key, sgn, z, half in (('fwd', +1, Z_RECV, R_RECV),
                              ('back', -1, Z_BACK, R_BACK)):
        m = term['d'][..., 2] * sgn > 0
        p = propagate_to_z(term['o'][m], term['d'][m], z)
        on = (torch.abs(p[..., 0]) <= half) & (torch.abs(p[..., 1]) <= half)
        out['p_' + key] = p[on]
        out['w_' + key] = term['w'][m][on]
        out['nrefl_' + key] = term['nrefl'][m][on]
        out['nhit_' + key] = term['nhit'][m][on]
        out['phi_' + key] = out['w_' + key].sum()
    out['phi_off'] = term['w'].sum() - out['phi_fwd'] - out['phi_back']
    return out


# (surface hits, reflections) -> path name. The reflection count alone does NOT
# identify a path: R1 and T1R2T1 both reflect once, and they are the two biggest
# things going backwards. The hit count separates them - 1 hit vs 3.
PATHS = {
    ('fwd', 2, 0): 'T1T2',           # direct, (1-R)^2
    ('fwd', 4, 2): 'T1R2R1T2',       # ghost,  (1-R)^2 R^2
    ('back', 1, 1): 'R1',            # off the front face, R
    ('back', 3, 1): 'T1R2T1',        # one round trip in the glass, (1-R)^2 R
}


def path_powers(term):
    """Total weight per (direction, hits, reflections), receivers ignored.

    Keys are 'fwd_h2r0' style, plus the named aliases in PATHS. These are the
    numbers with closed forms, so they are the sharpest gate in the file - no
    tolerance-by-eyeball anywhere.
    """
    fwd = term['d'][..., 2] > 0
    out = {}
    for key, m in (('fwd', fwd), ('back', ~fwd)):
        for h in range(1, 7):
            for k in range(0, 6):
                sel = m & (term['nhit'] == h) & (term['nrefl'] == k)
                if bool(sel.any()):
                    phi = float(term['w'][sel].sum())
                    out[f'{key}_h{h}r{k}'] = phi
                    if (key, h, k) in PATHS:
                        out[PATHS[(key, h, k)]] = phi
    return out


def phi_captured_check(term, tally):
    return float(term['w'].sum() + tally['culled'] + tally['truncated'])


# ============================================================================
#                       TESTS - complete, do not edit
# ============================================================================
N_SPLIT = 50000
N_MC    = 200000


def _check(name, cond, detail=''):
    print(f'  [{"ok " if cond else "FAIL"}] {name}{(" - " + detail) if detail else ""}')
    assert cond, name


def _w_min(w):
    return W_MIN_REL * float(w[0])


def _split_run(N=N_SPLIT, seed=11, R=R_COAT, els=None):
    o, d, w = sample_point_source(N, seed=seed)
    if els is None:
        els = build_elements(R=R)
    return trace_split(o, d, w, els, w_min=_w_min(w))


def test_reduces_to_stage1():
    print('T0  R = 0 reproduces stage 1')
    term, tally = _split_run(N=20000, seed=10, R=0.0)
    cl = classify(term)
    _check('all captured flux goes forward',
           abs(float(cl['phi_fwd']) / PHI_CAP - 1.0) < 1e-12,
           f'{float(cl["phi_fwd"]):.6e} vs {PHI_CAP:.6e} W')
    _check('nothing comes back', float(cl['phi_back']) == 0.0)
    _check('no reflected paths exist', int(term['nrefl'].sum()) == 0)


def test_ledger():
    print('T1  energy ledger, R = 0.2')
    term, tally = _split_run()
    cl = classify(term)

    total = phi_captured_check(term, tally)
    _check('Phi_fwd + Phi_back + Phi_off + culled + truncated = Phi_captured',
           abs(total / PHI_CAP - 1.0) < 1e-3,
           f'rel {abs(total / PHI_CAP - 1.0):.1e}')
    print(f'      fwd  {float(cl["phi_fwd"]) / PHI_CAP:.6f}   '
          f'back {float(cl["phi_back"]) / PHI_CAP:.6f}   '
          f'off  {float(cl["phi_off"]) / PHI_CAP:.6f}   '
          f'culled {float(tally["culled"]) / PHI_CAP:.2e}   '
          f'trunc {float(tally["truncated"]) / PHI_CAP:.2e}')


def test_path_fractions():
    print('T2/T3  path powers and the ghost')
    term, _ = _split_run()
    pp = path_powers(term)
    T, R = 1.0 - R_COAT, R_COAT

    _check('direct T1T2 = (1-R)^2',
           abs(pp['T1T2'] / PHI_CAP - T ** 2) < 1e-9,
           f'{pp["T1T2"] / PHI_CAP:.6f} vs {T ** 2:.6f}')
    _check('backward R1 = R',
           abs(pp['R1'] / PHI_CAP - R) < 1e-9,
           f'{pp["R1"] / PHI_CAP:.6f} vs {R:.6f}')
    _check('backward T1R2T1 = (1-R)^2 R',
           abs(pp['T1R2T1'] / PHI_CAP - T ** 2 * R) < 1e-9,
           f'{pp["T1R2T1"] / PHI_CAP:.6f} vs {T ** 2 * R:.6f}')
    # The ghost is the one path with no closed form: R1 happens on the CURVED
    # face, from inside, so the bounce is a strong concave mirror and a large
    # part of it leaves through the rim instead of reaching S2. (1-R)^2 R^2 is
    # therefore an upper bound, not the answer - which is exactly the kind of
    # number a sequential tracer cannot produce at all.
    ghost = pp['T1R2R1T2'] / PHI_CAP
    _check('ghost is below its unvignetted bound (1-R)^2 R^2',
           0.0 < ghost < T ** 2 * R ** 2,
           f'{ghost:.6f} vs bound {T ** 2 * R ** 2:.6f} '
           f'({ghost / (T ** 2 * R ** 2):.0%} of it survives)')
    _check('ghost survives the weight cull', ghost > 1e-4,
           f'{ghost:.4f} of the captured flux - set any threshold above this '
           f'and the path silently disappears')


def test_mc_matches_split():
    print('T4  Monte Carlo vs the split reference')
    o, d, w = sample_point_source(N_MC, seed=21)
    els = build_elements()
    term_s, _ = trace_split(o, d, w, els, w_min=_w_min(w))
    ref = float(classify(term_s)['phi_fwd'])

    vals = []
    for s in range(5):
        term_m, _ = trace_mc(o, d, w, els, seed=100 + s, w_min=_w_min(w))
        vals.append(float(classify(term_m)['phi_fwd']))
    vals = np.array(vals)
    err = abs(vals.mean() - ref)
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    _check('MC mean = split within its standard error', err < 3 * se + 1e-12,
           f'|{vals.mean():.6e} - {ref:.6e}| = {err:.2e}, se = {se:.2e}')

    # Slope of the RMS error against ray count. The RMS itself is estimated from
    # a finite number of seeds, so it needs enough of them: with 5 the slope
    # wanders by +-0.3 and the gate is meaningless. Depth 6 here (not 10) keeps
    # the sweep cheap - the deep tail is far below this error level anyway.
    Ns, errs, n_seed = [4000, 16000, 64000], [], 24
    for N in Ns:
        o, d, w = sample_point_source(N, seed=31)
        term_s, _ = trace_split(o, d, w, els, max_depth=6, w_min=_w_min(w))
        r = float(classify(term_s)['phi_fwd'])
        e = [abs(float(classify(trace_mc(o, d, w, els, seed=200 + s, max_depth=6,
                                         w_min=_w_min(w))[0])['phi_fwd']) - r)
             for s in range(n_seed)]
        errs.append(float(np.sqrt(np.mean(np.array(e) ** 2))))
    slope = float(np.polyfit(np.log10(Ns), np.log10(errs), 1)[0])
    _check('MC error ~ N^-0.5', abs(slope + 0.5) < 0.2,
           f'slope = {slope:.2f}, RMS ' + ' '.join(f'{e:.2e}' for e in errs))


def test_sequential_cannot_follow():
    """The demo: the same scene through dO's sequential tracer."""
    print('T5  what the sequential tracer sees')
    o, d, w = sample_point_source(N_SPLIT, seed=41)
    lens = build_lensgroup()
    ray = do.Ray(o.clone(), d.clone(), wavelength=torch.Tensor([WAVELENGTH]))
    _, valid = lens.trace(ray)
    phi_seq = float(w[valid].sum())

    term, _ = _split_run(N=N_SPLIT, seed=41)
    cl = classify(term)

    _check('sequential delivers the full captured flux - no coating loss',
           abs(phi_seq / PHI_CAP - 1.0) < 1e-12, f'{phi_seq / PHI_CAP:.6f}')
    # (1-R)^2 = 0.64 of it, plus whatever slice of the ghost lands inside the
    # +-16 mm receiver - the ghost is spread far wider than the direct beam.
    ratio = float(cl['phi_fwd']) / phi_seq
    _check('non-seq forward is ~36 % lower: (1-R)^2 plus a ghost pickup',
           (1 - R_COAT) ** 2 <= ratio < (1 - R_COAT) ** 2 + 0.03,
           f'{ratio:.4f} vs (1-R)^2 = {(1 - R_COAT) ** 2:.4f}')
    _check('and the sequential run has no backward light at all',
           float(cl['phi_back']) / PHI_CAP > 0.1,
           f'non-seq backward {float(cl["phi_back"]) / PHI_CAP:.4f} x Phi_cap, '
           f'sequential 0')


def test_gradients():
    """The bonus no commercial tracer gives: dPhi/dparameter by autodiff."""
    print('G   gradients vs finite differences')
    o, d, w = sample_point_source(4000, seed=51)

    def phi_back_of_R(R):
        term, _ = trace_split(o, d, w, build_elements(R=R), max_depth=4)
        return term['w'][term['d'][..., 2] < 0].sum()

    R_t = torch.tensor(R_COAT, requires_grad=True)
    phi_back_of_R(R_t).backward()
    g_auto = float(R_t.grad)
    h = 1e-4
    g_fd = float((phi_back_of_R(R_COAT + h) - phi_back_of_R(R_COAT - h)) / (2 * h))
    _check('dPhi_back/dR', abs(g_auto - g_fd) / abs(g_fd) < 1e-4,
           f'auto {g_auto:.6e} vs fd {g_fd:.6e}')

    def r_mean_of_c(c, grad=False):
        els = build_elements(c=c, grad_c=grad)
        term, _ = trace_split(o, d, w, els, max_depth=3)
        m = (term['d'][..., 2] > 0) & (term['nrefl'] == 0)
        p = propagate_to_z(term['o'][m], term['d'][m], Z_RECV)
        r = torch.norm(p[..., 0:2], dim=-1)
        return els, (term['w'][m] * r).sum() / term['w'][m].sum()

    els, r_mean = r_mean_of_c(C_ASPH, grad=True)
    r_mean.backward()
    g_auto = float(els[0].surface.c.grad)
    h = 1e-5
    g_fd = float((r_mean_of_c(C_ASPH + h)[1] - r_mean_of_c(C_ASPH - h)[1]) / (2 * h))
    _check('d<r_fwd>/dc', abs(g_auto - g_fd) / abs(g_fd) < 1e-3,
           f'auto {g_auto:.6e} vs fd {g_fd:.6e}')


# ------------------------------------------------------------------- pictures
def _pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print('  matplotlib missing, skipped')
        return None


def _out_dir():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c02_out')
    os.makedirs(out, exist_ok=True)
    return out


def plot_maps():
    """Forward and backward irradiance, the ledger, and the ghost profile."""
    print('P1  irradiance maps')
    plt = _pyplot()
    if plt is None:
        return

    o, d, w = sample_point_source(N_MC, seed=61)
    els = build_elements()
    term, tally = trace_split(o, d, w, els, w_min=_w_min(w))
    cl = classify(term)
    pp = path_powers(term)

    I_f = nonseq.splat(cl['p_fwd'], cl['w_fwd'] / PIXEL_F ** 2, FILM_F, PIXEL_F)
    I_b = nonseq.splat(cl['p_back'], cl['w_back'] / PIXEL_B ** 2, FILM_B, PIXEL_B)

    # The ghost is far wider than the +-16 mm receiver, so its own panel uses a
    # wide virtual screen at the same z - on the real receiver it is a handful
    # of speckles and shows nothing.
    R_WIDE, FILM_W = 60.0, [64, 64]   # coarse bins: the ghost is a thin, wide haze
    pix_w = 2 * R_WIDE / FILM_W[0]
    fw = (term['d'][..., 2] > 0) & (term['nhit'] == 4) & (term['nrefl'] == 2)
    p_g = propagate_to_z(term['o'][fw], term['d'][fw], Z_RECV)
    on = (torch.abs(p_g[..., 0]) <= R_WIDE) & (torch.abs(p_g[..., 1]) <= R_WIDE)
    I_g = nonseq.splat(p_g[on], term['w'][fw][on] / pix_w ** 2, FILM_W, pix_w)

    def logmap(I, floor=1e-6):
        A = I.numpy()
        return np.log10(np.maximum(A, A.max() * floor))

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    ext_f = [-R_RECV, R_RECV, -R_RECV, R_RECV]
    ext_b = [-R_BACK, R_BACK, -R_BACK, R_BACK]

    im = ax[0].imshow(logmap(I_f).T, origin='lower', extent=ext_f)
    ax[0].set_title(f'forward z = {Z_RECV:.0f} mm\nlog10 W/mm^2')
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    im = ax[1].imshow(logmap(I_b).T, origin='lower', extent=ext_b)
    ax[1].set_title(f'backward z = {Z_BACK:.0f} mm\nlog10 W/mm^2')
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    im = ax[2].imshow(logmap(I_g, 1e-3).T, origin='lower',
                      extent=[-R_WIDE, R_WIDE, -R_WIDE, R_WIDE])
    ax[2].plot([-R_RECV, R_RECV, R_RECV, -R_RECV, -R_RECV],
               [-R_RECV, -R_RECV, R_RECV, R_RECV, -R_RECV], 'w--', lw=1)
    ax[2].set_title('forward GHOST only (T1R2R1T2), wide screen\n'
                    'log10 W/mm^2, dashed = real receiver')
    fig.colorbar(im, ax=ax[2], fraction=0.046)

    labels = ['fwd\nT1T2', 'ghost\nT1R2R1T2', 'back\nR1', 'back\nT1R2T1',
              'off\nreceivers', 'culled +\ntruncated']
    vals = [pp.get('T1T2', 0.0), pp.get('T1R2R1T2', 0.0),
            pp.get('R1', 0.0), pp.get('T1R2T1', 0.0),
            float(cl['phi_off']),
            float(tally['culled'] + tally['truncated'])]
    vals = np.array(vals) / PHI_CAP
    ax[3].bar(range(len(vals)), vals, color='steelblue')
    ax[3].set_xticks(range(len(vals)))
    ax[3].set_xticklabels(labels, fontsize=7)
    ax[3].set_yscale('log')
    ax[3].set_ylabel('fraction of Phi_captured')
    ax[3].set_title(f'energy ledger, R = {R_COAT}')
    for i, v in enumerate(vals):
        if v > 0:
            ax[3].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    path = os.path.join(_out_dir(), 'c02_maps.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  wrote {path}')


def _split_segments(els, o, d, w, max_depth=4):
    """[(z0, y0, z1, y1, weight)] for every ray segment, including the split
    branches. `trace_split` only keeps the terminal state, so the loop is
    unrolled here the way `_verts_ns` was in c01."""
    segs = []
    ignore = torch.full((o.shape[0],), -1, dtype=torch.long)

    def fly(o_, d_, w_):
        z = torch.where(d_[..., 2] > 0,
                        torch.full_like(d_[..., 2], Z_RECV),
                        torch.full_like(d_[..., 2], Z_BACK))
        p = o_ + ((z - o_[..., 2]) / d_[..., 2])[..., None] * d_
        for j in range(o_.shape[0]):
            segs.append((float(o_[j, 2]), float(o_[j, 1]),
                         float(p[j, 2]), float(p[j, 1]), float(w_[j])))

    for _ in range(max_depth):
        if o.shape[0] == 0:
            break
        _, eid = closest_hit(o, d, els, ignore_id=ignore)
        gone = eid < 0
        if bool(gone.any()):
            fly(o[gone], d[gone], w[gone])
        live = ~gone
        if not bool(live.any()):
            break
        o, d, w, eid = o[live], d[live], w[live], eid[live]

        p, ok, R, d_refr, d_refl, ok_t = _interaction(o, d, els, eid, WAVELENGTH)
        for j in range(o.shape[0]):
            segs.append((float(o[j, 2]), float(o[j, 1]),
                         float(p[j, 2]), float(p[j, 1]), float(w[j])))

        w_tr, w_re = w * (1.0 - R), w * R
        o = torch.cat((p[ok_t], p))
        d = torch.cat((d_refr[ok_t], d_refl))
        w = torch.cat((w_tr[ok_t], w_re))
        ignore = torch.cat((eid[ok_t], eid))
    return segs


def plot_layout(nfan=9):
    """Side view: one fan, every branch drawn, opacity by weight."""
    print('P2  layout with split branches')
    plt = _pyplot()
    if plt is None:
        return

    t = torch.linspace(-0.9 * THETA_MAX, 0.9 * THETA_MAX, nfan)
    d = torch.stack((torch.zeros_like(t), torch.sin(t), torch.cos(t)), dim=-1)
    o = torch.zeros(nfan, 3)
    w = torch.full((nfan,), 1.0)

    segs = _split_segments(build_elements(), o, d, w)

    y = np.linspace(-R_LENS, R_LENS, 200)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(Z_S1 + sag(y), y, 'k-', lw=1.5)
    ax.plot([Z_S2, Z_S2], [-R_LENS, R_LENS], 'k-', lw=1.5)
    for s in (-1, 1):
        ax.plot([Z_S1 + float(sag(R_LENS)), Z_S2], [s * R_LENS, s * R_LENS],
                'k-', lw=1.5)
    ax.plot([Z_RECV, Z_RECV], [-R_RECV, R_RECV], 'b-', lw=2)
    ax.plot([Z_BACK, Z_BACK], [-R_BACK, R_BACK], 'g-', lw=2)
    ax.plot(0.0, 0.0, 'r*', ms=12)

    for z0, y0, z1, y1, ww in segs:
        a = float(np.clip(0.08 + 0.92 * (np.log10(max(ww, 1e-6)) + 3.0) / 3.0,
                          0.05, 1.0))
        ax.plot([z0, z1], [y0, y1], 'r-', lw=0.8, alpha=a)

    ax.set_xlabel('z [mm]'), ax.set_ylabel('y [mm]')
    ax.set_xlim(Z_BACK - 5, Z_RECV + 5)
    ax.set_ylim(-R_BACK * 0.6, R_BACK * 0.6)
    ax.set_title(f'split tracing, R = {R_COAT} on both surfaces '
                 f'(opacity ~ log weight); blue = forward receiver, '
                 f'green = backward receiver')
    fig.tight_layout()
    path = os.path.join(_out_dir(), 'c02_layout.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  wrote {path}')


if __name__ == '__main__':
    torch.manual_seed(0)
    test_reduces_to_stage1()
    test_ledger()
    test_path_fractions()
    test_mc_matches_split()
    test_sequential_cannot_follow()
    test_gradients()
    plot_maps()
    plot_layout()
    print('\nPhase 2 stage-2 gates passed.')
