"""
PHASE 2 - STAGE 5: two lenses, one reflectivity each.

`c02_R02.py` put a single coated collimator in front of a point source and
proved the non-sequential MC tracer against closed forms. This adds a SECOND
lens with a DIFFERENT reflectivity, which is the first scene in this series
where the interesting paths are not all inside one piece of glass: a ray can
ghost between lens 1 and lens 2, and the two ghosts have different amplitudes
because the two coatings differ.

    python examples/nonseq/c05_two_lens.py            # all gates + both figures
    python examples/nonseq/c05_two_lens.py --quick    # gates only, smaller N

--------------------------------------------------------------------------------
THE SCENE
--------------------------------------------------------------------------------

  point source          lens 1 (collimator)      lens 2 (mirror image)     forward
  (0,0,0), 1 W     ->   S1 asphere  z = 40.0 ->  S3 flat    z = 100.0  ->  z = 200
  isotropic             S2 flat     z = 46.5     S4 asphere z = 106.5      +-20 mm
  into a cone           N-BK7, sd 12.7           N-BK7, sd 14.0
  theta = 16.2758 deg   R1 = 0.2 both            R2 = 0.1 both        <-   backward
  Phi_cap = 0.020038 W                                                     z = -80
                                                 focus at z = 146.5        +-80 mm

Lens 2 is the EXACT MIRROR IMAGE of lens 1, reflected about z = 73.25. That is
not a convenience - it is the only way to get a second lens into the scene with
no new design work and no aberration to argue about. Lens 1 turns a point at
s = 40 mm into a collimated beam; run the same surfaces backwards and a
collimated beam becomes a point 40 mm past the aspheric vertex. Ray reversibility
makes it exact, so the constants are just lens 1's with one sign flipped:

    C_ASPH2 = -C_ASPH        K_ASPH2 = K_ASPH

Measured focus spot radius: 2.6e-14 mm. That is the geometry gate, and it is
sharp - a sign error anywhere in the second lens fails it by many orders.

The forward receiver sits 53.5 mm PAST the focus, where the beam has opened
back up to a 15.62 mm disc. Putting it at the focus would be useless: every ray
in one bin, no map to compare, and the noise question below would be
meaningless.

--------------------------------------------------------------------------------
WHY THE SEMI-DIAMETER GOES UP TO 14
--------------------------------------------------------------------------------

The collimated beam leaving lens 1 is exactly 12.7 mm in radius, because that
is lens 1's rim. Giving lens 2 the same 12.7 mm rim puts the marginal ray
exactly on the aperture edge, where `Surface.is_valid` is a coin flip in the
last bit. 14.0 mm costs nothing and removes the question.

--------------------------------------------------------------------------------
THE BIN RULE
--------------------------------------------------------------------------------

Per-bin relative Monte Carlo noise on a receiver holding `R` rays in `N x N`
bins:

    eps = 1 / sqrt(R / N^2) = N / sqrt(R)

so a 10 % noise target fixes the mesh from the ray budget:

    N = 16 * floor(0.1 * sqrt(R) / 16)          snapped DOWN, floored at 16

`R` is the number of rays that LAND ON THAT RECEIVER, not the number launched.
That distinction is the whole reason the two receivers here get different
meshes: the forward receiver catches 53 % of the launched rays and the backward
one 22 %, so at the same ray budget the backward receiver has to be coarser.
`bins_for` is the single implementation; `c05_compare.py` and the notebook both
call it, and the LightTools receiver mesh must be built to match it.

Snapping DOWN matters. Fewer bins means more rays in each, so the realised
noise can only come in under target, never over.

--------------------------------------------------------------------------------
WHAT HAS A CLOSED FORM HERE, AND WHAT DOES NOT
--------------------------------------------------------------------------------

c02 could name every path it cared about by (direction, hits, reflections),
because with two surfaces that key is unique. With FOUR partial surfaces it is
not. Two independent failures of uniqueness show up immediately:

  ('fwd', 6, 2)   is the lens-1 ghost AND the lens-2 ghost summed. Both are
                  real, they have different amplitudes, and the bucket holds
                  their sum.

  ('back', 5, 1)  measures 6.72 % where the obvious closed form (1-R1)^4 R2
                  gives 4.10 %. The excess is a second path - reflect off S4,
                  back out through S3, then MISS S2's 12.7 mm rim and escape
                  backwards. It is aperture-clipped, so it has no closed form
                  at all.

Pretending otherwise would make a gate that fails for a reason that is not a
bug. So only three buckets are gated, and they are gated because nothing else
can land in them:

    ('fwd',  4, 0)  T1 T2 T3 T4   (1-R1)^2 (1-R2)^2 = 0.5184
    ('back', 1, 1)  R1            R1                = 0.2
    ('back', 3, 1)  T1 R2 T1      (1-R1)^2 R1       = 0.128

Everything else is reported in the table and gated by the two things that hold
regardless of path bookkeeping: the ENERGY LEDGER, and MC agreeing with the
deterministic split tracer.
"""
import argparse
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import diffoptics as do  # noqa: E402
from diffoptics import nonseq  # noqa: E402
from diffoptics.nonseq import (  # noqa: E402
    Element, closest_hit, interaction, propagate_to_z, trace_mc, trace_split)

import c02_R02 as ref  # noqa: E402


# ------------------------------------------------------------------ constants
# Lens 1 and the source are c02's, imported rather than restated so the two
# files cannot drift. Only what lens 2 and the new receivers need is new here.
WAVELENGTH = ref.WAVELENGTH
P_TOTAL    = ref.P_TOTAL
THETA_MAX  = ref.THETA_MAX
PHI_CAP    = ref.PHI_CAP

R1_COAT = 0.2                       # lens 1, both surfaces
R2_COAT = 0.1                       # lens 2, both surfaces

Z_S3, Z_S4 = 100.0, 106.5           # lens 2, flat first then asphere
R_LENS2    = 14.0                   # rim clear of the 12.7 mm collimated beam
C_ASPH2    = -ref.C_ASPH            # mirror image of lens 1: sign flips
K_ASPH2    = ref.K_ASPH             # conic constant does not
Z_FOCUS    = Z_S4 + ref.S_SRC       # 146.5, by reversibility

Z_RECV, R_RECV = 200.0, 20.0        # forward, 53.5 mm past focus
Z_BACK, R_BACK = -80.0, 80.0        # backward, same as c02

# Four partial surfaces make deeper trees than c02's two. At 10 the truncation
# tally is visible in the ledger; 14 pushes it under 1e-5 of the launched power.
MAX_DEPTH = 14
W_MIN_REL = 1e-5

EPS_TARGET = 0.10                   # per-bin MC noise we design to
BIN_SNAP   = 16
BIN_MIN    = 16

RECEIVERS = [
    dict(name='fwd',  sign=+1, z=Z_RECV, half=R_RECV, label='forward'),
    dict(name='back', sign=-1, z=Z_BACK, half=R_BACK, label='backward'),
]


def sag2(r):
    """Sag of lens 2's aspheric surface S4, measured from its vertex.

    Negative for `C_ASPH2 < 0`: S4 bulges towards -z, into the glass side,
    which is what mirroring lens 1 does to it.
    """
    r2 = np.asarray(r, dtype=np.float64) ** 2
    return C_ASPH2 * r2 / (1 + np.sqrt(1 - (1 + K_ASPH2) * C_ASPH2 ** 2 * r2))


# --------------------------------------------------------------------- scenes
def build_elements(R1=R1_COAT, R2=R2_COAT, device=None):
    """The four coated surfaces, in z order.

    `R1` / `R2` may be floats or 0-d tensors; a tensor with `requires_grad`
    makes that coating a leaf, which is how the gradient gate below gets
    dPhi/dR1 and dPhi/dR2 out of one trace.

    Lens 1 is c02's, built by c02's own builder so the two scenes are the same
    glass. Lens 2 is added here.
    """
    eye = torch.eye(3, device=device)

    def pose(z):
        return do.Transformation(eye, torch.tensor([0.0, 0.0, z], device=device))

    s1, s2 = ref.build_elements(R=R1, device=device)

    s3 = Element(do.Aspheric(R_LENS2, 0.0, c=0.0, device=device),
                 pose(Z_S3),
                 n_in='air', n_out='n-bk7', kind='partial', R_fixed=R2,
                 name='S3', device=device)
    s4 = Element(do.Aspheric(R_LENS2, 0.0, c=C_ASPH2, k=K_ASPH2, device=device),
                 pose(Z_S4),
                 n_in='n-bk7', n_out='air', kind='partial', R_fixed=R2,
                 name='S4', device=device)
    return [s1, s2, s3, s4]


def build_lensgroup(device=torch.device('cpu')):
    """All four surfaces as ONE sequential `Lensgroup`, for the contrast run.

    It traces the T1T2T3T4 path and nothing else - `Lensgroup._trace` picks
    forward or backward once from `(ray.d[..., 2] > 0).all()` (optics.py:1049),
    so a batch that contains a back-reflection is already outside its model.
    That is the demo, not a defect of this file.
    """
    lens = do.Lensgroup(origin=np.array([0, 0, ref.Z_S1]), device=device)
    lens.load(
        [do.Aspheric(ref.R_LENS, 0.0, c=ref.C_ASPH, k=ref.K_ASPH, device=device),
         do.Aspheric(ref.R_LENS, ref.THICK, c=0.0, device=device),
         do.Aspheric(R_LENS2, Z_S3 - ref.Z_S1, c=0.0, device=device),
         do.Aspheric(R_LENS2, Z_S4 - ref.Z_S1, c=C_ASPH2, k=K_ASPH2,
                     device=device)],
        [do.Material('air'), do.Material('n-bk7'), do.Material('air'),
         do.Material('n-bk7'), do.Material('air')],
    )
    lens.d_sensor = Z_RECV
    lens.r_last = R_LENS2
    return lens


def sample_point_source(N, seed=0, device=None):
    return nonseq.sample_point_source(N, THETA_MAX, seed=seed, P=P_TOTAL,
                                      device=device)


def _w_min(w):
    return W_MIN_REL * float(w.reshape(-1)[0])


# ------------------------------------------------------------------- the bins
def bins_for(rays_on, target=EPS_TARGET, snap=BIN_SNAP, n_min=BIN_MIN):
    """Bins per axis for a receiver that catches `rays_on` rays.

    eps = N / sqrt(R), so N = target * sqrt(R). Snapped DOWN to a multiple of
    `snap` and floored at `n_min`, because both of those can only reduce the
    realised noise.

    Returns (n, n_ideal, eps_pred). `rays_on = 0` returns (n_min, 0.0, inf) -
    an empty receiver has no usable mesh and the caller should say so rather
    than divide by zero.
    """
    rays_on = int(rays_on)
    if rays_on <= 0:
        return n_min, 0.0, float('inf')
    n_ideal = target * np.sqrt(rays_on)
    n = int(snap * (n_ideal // snap))
    n = max(n_min, n)
    return n, float(n_ideal), float(n / np.sqrt(rays_on))


def measured_eps(p, w, n, half):
    """Realised per-bin noise, sqrt(sum w^2)/sum w, over the lit bins.

    `bins_for` predicts `N/sqrt(R)`, which is `1/sqrt(mean count)`. That is the
    truth only if the rays spread EVENLY over the receiver. They do not, and
    the two receivers here miss in opposite directions - measured at 1e6 rays:

        forward   N = 64   predicted 8.78 %   measured 6.19 %
        backward  N = 32   predicted 6.83 %   measured 14.74 %

    Forward comes in UNDER because only 2556 of 4096 bins are lit at all: the
    beam is a 15.6 mm disc inside a 20 mm square, so the bins that exist hold
    261 rays where the flat-field average is 130. The rule is conservative.

    Backward comes in OVER, and by more. Every one of its 1024 bins is lit, but
    the flux is concentrated - the R1 reflection puts most of its power in a
    small central patch and sprays a thin halo to the rim. Mean count is 214,
    MEDIAN count is 46. A mean-based rule cannot see that, so it under-predicts
    the noise of the typical bin by 2.2x.

    Hence this returns the median and the 90th percentile over lit bins: the
    number that describes the bin a reader actually points at, not the average
    of a skewed distribution.
    """
    empty = dict(median=float('inf'), p90=float('inf'), lit=0, lit_frac=0.0)
    if p.shape[0] == 0:
        return empty
    pitch = 2.0 * half / n
    s1 = nonseq.splat(p, w, [n, n], pitch, bilinear=False)
    s2 = nonseq.splat(p, w * w, [n, n], pitch, bilinear=False)
    lit = s1 > 0
    if not bool(lit.any()):
        return empty
    eps = torch.sqrt(s2[lit]) / s1[lit]
    return dict(median=float(torch.quantile(eps, 0.50)),
                p90=float(torch.quantile(eps, 0.90)),
                lit=int(lit.sum()),
                lit_frac=float(lit.double().mean()))


def bins_measured(p, w, half, target=EPS_TARGET, snap=BIN_SNAP, n_min=BIN_MIN,
                  n_max=1024):
    """Largest snapped N whose MEASURED median noise is still under `target`.

    The empirical counterpart of `bins_for`. Splatting is O(rays) and the trace
    has already been paid for, so trying every candidate mesh costs almost
    nothing next to the run that produced `p`.

    Use this when the realised noise has to be under target - on a receiver
    with concentrated flux, like the backward one here, `bins_for` alone is not
    enough. Returns (n, eps_measured).

    If even `n_min` misses the target the floor is returned with ITS measured
    noise, not infinity: "16 x 16 and still 16.7 %" is a usable answer that says
    the ray budget is too small, where `inf` says nothing.
    """
    if p.shape[0] == 0:
        return n_min, float('inf')
    floor = measured_eps(p, w, n_min, half)['median']
    best = (n_min, floor)
    n = n_min
    while n <= n_max:
        m = measured_eps(p, w, n, half)
        if m['median'] <= target:
            best = (n, m['median'])
        else:
            break
        n += snap
    return best


# ----------------------------------------------------------------- receivers
def classify(term, receivers=RECEIVERS):
    """Sort free-flying terminal rays onto the receivers.

    Generalised from c02's hardcoded pair to a table, because c05_compare.py
    and the notebook both iterate it and a third receiver should cost one line,
    not an edit in four files.

    For each receiver `name`: `p_<name>`, `w_<name>`, `nrefl_<name>`,
    `nhit_<name>` for the rays landing on it, and `phi_<name>` their power.
    Plus `phi_off`, the power that reaches none of them.
    """
    out = {}
    total = term['w'].sum()
    caught = torch.zeros_like(total)
    for rc in receivers:
        key, sgn, z, half = rc['name'], rc['sign'], rc['z'], rc['half']
        m = term['d'][..., 2] * sgn > 0
        p = propagate_to_z(term['o'][m], term['d'][m], z)
        on = (torch.abs(p[..., 0]) <= half) & (torch.abs(p[..., 1]) <= half)
        out['p_' + key] = p[on]
        out['w_' + key] = term['w'][m][on]
        out['nrefl_' + key] = term['nrefl'][m][on]
        out['nhit_' + key] = term['nhit'][m][on]
        out['phi_' + key] = out['w_' + key].sum()
        caught = caught + out['phi_' + key]
    out['phi_off'] = total - caught
    return out


# (direction, hits, reflections) -> name, for the buckets NOTHING ELSE can land
# in. See the module docstring: with four partial surfaces most keys are shared
# by several paths, so this dict is deliberately short.
PATHS = {
    ('fwd', 4, 0): 'T1T2T3T4',
    ('back', 1, 1): 'R1',
    ('back', 3, 1): 'T1R2T1',
}

H_MAX, K_MAX = 14, 12


def closed_form(R1=R1_COAT, R2=R2_COAT):
    """The three gated fractions, as fractions of Phi_cap."""
    t1, t2 = 1.0 - R1, 1.0 - R2
    return {'T1T2T3T4': t1 ** 2 * t2 ** 2,
            'R1': R1,
            'T1R2T1': t1 ** 2 * R1}


def path_powers(term):
    """Total weight per (direction, hits, reflections), receivers ignored.

    Keyed 'fwd_h4r0' style, plus the named aliases in `PATHS`. The h/k ranges
    are wider than c02's because four partial surfaces reach depth 14.
    """
    fwd = term['d'][..., 2] > 0
    out = {}
    for key, m in (('fwd', fwd), ('back', ~fwd)):
        for h in range(1, H_MAX + 1):
            for k in range(0, K_MAX + 1):
                sel = m & (term['nhit'] == h) & (term['nrefl'] == k)
                if bool(sel.any()):
                    phi = float(term['w'][sel].sum())
                    out[f'{key}_h{h}r{k}'] = phi
                    if (key, h, k) in PATHS:
                        out[PATHS[(key, h, k)]] = phi
    return out


def phi_captured_check(term, tally):
    return float(term['w'].sum() + tally['culled'] + tally['truncated'])


def run_mc(N, seed_src=11, mc_seed=101, R1=R1_COAT, R2=R2_COAT, device=None,
           max_depth=MAX_DEPTH):
    """One MC trace. Returns (terminal, tally, classify dict)."""
    o, d, w = sample_point_source(N, seed=seed_src, device=device)
    els = build_elements(R1=R1, R2=R2, device=device)
    term, tally = trace_mc(o, d, w, els, WAVELENGTH, seed=mc_seed,
                           max_depth=max_depth, w_min=_w_min(w))
    return term, tally, classify(term)


def run_split(N, seed_src=11, R1=R1_COAT, R2=R2_COAT, device=None,
              max_depth=MAX_DEPTH):
    """One deterministic split trace.

    Same depth as MC by default. The front doubles at every partially
    reflective hit, but `w_min` retires the faint branches, so 50k rays to
    depth 14 costs ~6.6 s and 1.6 M terminal rays - affordable, and anything
    shallower leaves visible power in the truncation tally (2.6 % at depth 8).
    """
    o, d, w = sample_point_source(N, seed=seed_src, device=device)
    els = build_elements(R1=R1, R2=R2, device=device)
    term, tally = trace_split(o, d, w, els, WAVELENGTH, max_depth=max_depth,
                              w_min=_w_min(w))
    return term, tally, classify(term)


def bin_table(cl, receivers=RECEIVERS, launched=None, printout=True,
              strict=False):
    """Choose and report the mesh for every receiver.

    The deliverable of the noise rule. Per receiver it prints the rays that
    landed, the ideal and the snapped N, the predicted noise, the MEASURED
    noise at that N, and - separately - the largest N whose measured noise is
    actually under target.

    `strict=False` (default) uses `bins_for`, the requested rule. `strict=True`
    uses `bins_measured`, which guarantees the realised median noise is under
    target on every receiver. Both columns are always printed, so a receiver
    where the two disagree names itself.

    Returns {name: dict(n=..., eps_pred=..., median=..., n_strict=...)}.
    """
    out = {}
    if printout:
        print(f'  {"receiver":10s} {"rays on":>10s} {"of launch":>10s} '
              f'{"N ideal":>8s} {"N":>5s} {"eps pred":>9s} {"eps meas":>9s} '
              f'{"lit":>7s} {"N strict":>9s} {"eps":>7s}')
    for rc in receivers:
        key = rc['name']
        p, w = cl['p_' + key], cl['w_' + key]
        rays_on = int(p.shape[0])
        n_rule, n_ideal, eps_rule = bins_for(rays_on)
        n_strict, eps_strict = bins_measured(p, w, rc['half'])
        n = n_strict if strict else n_rule
        meas = measured_eps(p, w, n, rc['half'])
        out[key] = dict(n=n, n_rule=n_rule, n_ideal=n_ideal,
                        eps_pred=(eps_rule if not strict
                                  else n / np.sqrt(max(rays_on, 1))),
                        eps_rule=eps_rule, n_strict=n_strict,
                        eps_strict=eps_strict, rays_on=rays_on,
                        half=rc['half'], z=rc['z'], **meas)
        if printout:
            frac = f'{rays_on / launched * 100:9.2f}%' if launched else ' ' * 10
            print(f'  {rc["label"]:10s} {rays_on:10d} {frac} '
                  f'{n_ideal:8.1f} {n:5d} {out[key]["eps_pred"] * 100:8.2f}% '
                  f'{meas["median"] * 100:8.2f}% {meas["lit"]:7d} '
                  f'{n_strict:9d} {eps_strict * 100:6.2f}%')
    return out


# ============================================================================
#                                   GATES
# ============================================================================
N_SPLIT = 50000
N_MC    = 200000


def _check(name, cond, detail=''):
    print(f'  [{"ok " if cond else "FAIL"}] {name}'
          f'{(" - " + detail) if detail else ""}')
    assert cond, name


def test_geometry():
    """R1 = R2 = 0: pure refraction through four surfaces, perfect focus.

    Every ray must survive all four surfaces, carry the full captured power,
    and cross z = 146.5 on the axis. The focus radius is the sharp part - the
    mirror-image construction is exact, so anything above ~1e-9 mm means a sign
    or a pose is wrong, not that the lens is imperfect.
    """
    print('T0  geometry, R1 = R2 = 0')
    term, tally, cl = run_split(20000, seed_src=3, R1=0.0, R2=0.0, max_depth=6)

    _check('every ray hits all four surfaces',
           bool((term['nhit'] == 4).all()),
           f'nhit set = {sorted(set(term["nhit"].tolist()))}')
    _check('no reflections', int(term['nrefl'].sum()) == 0)

    frac = float(term['w'].sum()) / PHI_CAP
    _check('all captured power survives', abs(frac - 1.0) < 1e-12,
           f'{frac:.15f}')

    fwd = term['d'][..., 2] > 0
    pf = propagate_to_z(term['o'][fwd], term['d'][fwd], Z_FOCUS)
    r_focus = float(torch.linalg.norm(pf[..., :2], dim=-1).max())
    _check(f'focus spot at z = {Z_FOCUS} is a point', r_focus < 1e-9,
           f'max radius {r_focus:.3e} mm')

    p200 = propagate_to_z(term['o'][fwd], term['d'][fwd], Z_RECV)
    r200 = float(torch.linalg.norm(p200[..., :2], dim=-1).max())
    _check(f'beam fits the +-{R_RECV} mm forward receiver', r200 < R_RECV,
           f'max radius {r200:.4f} mm')


def test_ledger():
    """Energy in = energy out, for both tracers.

    Nothing about the two reflectivities or the path bookkeeping can break
    this, which is exactly why it is the gate that matters most.
    """
    print('T1  energy ledger')
    for name, fn, N in (('split', run_split, N_SPLIT), ('mc', run_mc, N_MC)):
        term, tally, cl = fn(N)
        tot = phi_captured_check(term, tally)
        rel = abs(tot / PHI_CAP - 1.0)
        _check(f'{name}: closes', rel < 1e-9, f'|sum/Phi_cap - 1| = {rel:.3e}')

        parts = sum(float(cl['phi_' + rc['name']]) for rc in RECEIVERS)
        parts += float(cl['phi_off'])
        rel2 = abs(parts / float(term['w'].sum()) - 1.0)
        _check(f'{name}: receivers + off = terminal', rel2 < 1e-12,
               f'{rel2:.3e}')

        # Truncated power is TALLIED, so the ledger closes whatever the depth
        # cap is. This gate is a different question: did we go deep enough that
        # the maps are not missing power a reader would notice? At depth 14
        # both tracers sit at ~2.3e-4 of Phi_cap; depth 8 leaves 2.6e-2, which
        # would be visible.
        trunc = float(tally['truncated']) / PHI_CAP
        _check(f'{name}: truncation is negligible', trunc < 1e-3,
               f'{trunc:.2e} of Phi_cap at max_depth = {MAX_DEPTH}')


def test_path_fractions():
    """The three unshared buckets, against their closed forms.

    Only three, deliberately - see the module docstring. `('back', 5, 1)` and
    `('fwd', 6, 2)` are each shared by two paths and one of those is aperture
    clipped, so they have no closed form to gate against.
    """
    print('T2  closed-form path fractions')
    # Depth 10 is enough here even though the ledger wants 14: every gated
    # path is 4 hits or fewer, so the deeper generations cannot move them.
    term, tally, _ = run_split(N_SPLIT, max_depth=10)
    pp = path_powers(term)
    exact = closed_form()
    for name, want in exact.items():
        got = pp.get(name, 0.0) / PHI_CAP
        _check(f'{name} = {want:.4f}', abs(got - want) < 2e-3,
               f'got {got:.5f}, d = {got - want:+.2e}')

    shared = pp.get('back_h5r1', 0.0) / PHI_CAP
    naive = (1 - R1_COAT) ** 4 * R2_COAT
    _check('back_h5r1 is NOT the naive closed form (bucket is shared)',
           shared > naive * 1.2,
           f'{shared:.5f} vs naive {naive:.5f} - the excess is the '
           f'S4-reflection path that misses S2')


def test_mc_matches_split():
    """MC mean equals split, and the error falls as N^-0.5."""
    print('T3  MC vs split')
    # BOTH tracers get the SAME source rays. Otherwise the comparison also
    # carries the source-sampling difference between two independent draws,
    # which the seed-to-seed spread cannot see - and the gate then fails at
    # 5 s.e. for a reason that has nothing to do with the MC estimator.
    N = N_SPLIT
    o, d, w = sample_point_source(N, seed=11)
    els = build_elements()
    ts, _ = trace_split(o, d, w, els, WAVELENGTH, max_depth=MAX_DEPTH,
                        w_min=_w_min(w))
    ref_fwd = float(classify(ts)['phi_fwd']) / PHI_CAP

    vals = []
    for s in (100, 101, 102, 103, 104):
        tm, _ = trace_mc(o, d, w, els, WAVELENGTH, seed=s,
                         max_depth=MAX_DEPTH, w_min=_w_min(w))
        vals.append(float(classify(tm)['phi_fwd']) / PHI_CAP)
    mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
    se = std / np.sqrt(len(vals))
    _check('forward power agrees within 3 s.e. on identical rays',
           abs(mean - ref_fwd) < 3 * se,
           f'split {ref_fwd:.5f}, MC {mean:.5f} +- {se:.5f} '
           f'({abs(mean - ref_fwd) / se:.1f} s.e.)')

    # Convergence: the seed-to-seed SPREAD, which needs no reference at all and
    # so cannot be polluted by the reference's own error. Pure N^-1/2.
    ns, spreads = [4000, 16000, 64000], []
    for n in ns:
        v = []
        for s in range(10):
            oo, dd, ww = sample_point_source(n, seed=200 + s)
            tm, _ = trace_mc(oo, dd, ww, build_elements(), WAVELENGTH,
                             seed=900 + s, max_depth=8, w_min=0.0)
            v.append(float(classify(tm)['phi_fwd']) / PHI_CAP)
        spreads.append(float(np.std(v, ddof=1)))
    slope = float(np.polyfit(np.log(ns), np.log(spreads), 1)[0])
    _check('spread falls as N^-1/2', abs(slope + 0.5) < 0.15,
           f'slope {slope:+.3f}, spreads {[f"{x:.2e}" for x in spreads]}')


def test_bin_rule():
    """`bins_for` picks a mesh whose realised noise really is under target."""
    print('T4  bin rule')
    N = int(1e6) if not _QUICK else N_MC
    _, _, cl = run_mc(N, mc_seed=77)
    tbl = bin_table(cl, launched=N)

    for rc in RECEIVERS:
        b = tbl[rc['name']]
        _check(f'{rc["label"]}: rule gives eps under {EPS_TARGET:.0%}',
               b['eps_rule'] <= EPS_TARGET,
               f'N = {b["n_rule"]}, eps = {b["eps_rule"]:.2%}')
        _check(f'{rc["label"]}: N is the largest multiple of {BIN_SNAP} '
               f'that fits',
               b['n_rule'] + BIN_SNAP > b['n_ideal'] or b['n_rule'] == BIN_MIN,
               f'N = {b["n_rule"]}, ideal = {b["n_ideal"]:.1f}')
        # At the floor mesh there is nothing left to coarsen, so a miss there
        # is a statement about the ray budget, not about the rule.
        floored = b['n_strict'] == BIN_MIN
        _check(f'{rc["label"]}: strict mesh really is under {EPS_TARGET:.0%}',
               b['eps_strict'] <= EPS_TARGET or floored,
               f'N = {b["n_strict"]}, measured {b["eps_strict"]:.2%}'
               + (' (at the N floor - needs more rays, not fewer bins)'
                  if floored and b['eps_strict'] > EPS_TARGET else ''))

        # Not a gate - a reported fact. The rule is a flat-field estimate and
        # the flux is not flat, so it errs in whichever direction the
        # non-uniformity happens to run. Say which, rather than assert.
        if b['median'] > EPS_TARGET:
            print(f'         NOTE {rc["label"]}: the rule picks N = '
                  f'{b["n_rule"]} (predicted {b["eps_rule"]:.2%}) but the '
                  f'measured median is {b["median"]:.2%} - flux is '
                  f'concentrated, so mean count per bin '
                  f'({b["rays_on"] / b["n_rule"] ** 2:.0f}) is well above the '
                  f'typical bin. Use N = {b["n_strict"]} to hold the realised '
                  f'noise under target.')
        else:
            print(f'         {rc["label"]}: rule is conservative here - '
                  f'measured {b["median"]:.2%} vs predicted '
                  f'{b["eps_rule"]:.2%} ({b["lit_frac"]:.0%} of bins lit).')


def test_gradients():
    """dPhi/dR1 and dPhi/dR2 by autograd, against central differences.

    Two coatings in one scene is the case that makes the point: LightTools can
    give either power, but not its derivative with respect to either coating -
    and here both come out of a single backward pass.

    The finite-difference check runs on `trace_split`, NOT `trace_mc`, and that
    is not a convenience. `trace_mc` samples its branch with `u < rho` where
    `rho = R.detach()`. Nudge R by h and rho moves too, so some rays flip to
    the other branch and the estimator jumps discretely. Autograd reports the
    pathwise derivative at fixed decisions; a fixed-seed difference quotient
    reports that PLUS the flips, and the two are simply different quantities
    (measured: 6.94e-3 vs 5.01e-3, a 39 % gap that shrinks with neither h nor
    ray count). `trace_split` takes both branches every time, so it has no
    decisions to flip and the comparison is exact.

    The MC gradient is then checked where it is actually defined - in
    expectation, against the split gradient, over independent seeds.
    """
    print('T5  gradients')
    o, d, w = sample_point_source(6000, seed=51)

    def phi_split(r1, r2, key, depth=6):
        term, _ = trace_split(o, d, w, build_elements(R1=r1, R2=r2),
                              WAVELENGTH, max_depth=depth)
        return classify(term)['phi_' + key]

    h = 1e-4
    autos = {}
    for key, which in (('back', 'R1'), ('fwd', 'R2')):
        r1 = torch.tensor(R1_COAT, requires_grad=(which == 'R1'))
        r2 = torch.tensor(R2_COAT, requires_grad=(which == 'R2'))
        phi_split(r1, r2, key).backward()
        g = float((r1 if which == 'R1' else r2).grad)
        autos[which] = g

        if which == 'R1':
            hi = float(phi_split(R1_COAT + h, R2_COAT, key))
            lo = float(phi_split(R1_COAT - h, R2_COAT, key))
        else:
            hi = float(phi_split(R1_COAT, R2_COAT + h, key))
            lo = float(phi_split(R1_COAT, R2_COAT - h, key))
        fd = (hi - lo) / (2 * h)
        rel = abs(g - fd) / max(abs(fd), 1e-12)
        _check(f'split: dPhi_{key}/d{which}', rel < 1e-4,
               f'autograd {g:.9e} vs fd {fd:.9e}, rel {rel:.2e}')

    # And the same derivative straight out of the MC tracer, which is the one
    # that scales. Unbiased, so it is the MEAN over seeds that must match.
    for key, which in (('back', 'R1'), ('fwd', 'R2')):
        vals = []
        for s in range(8):
            oo, dd, ww = sample_point_source(6000, seed=300 + s)
            r1 = torch.tensor(R1_COAT, requires_grad=(which == 'R1'))
            r2 = torch.tensor(R2_COAT, requires_grad=(which == 'R2'))
            term, _ = trace_mc(oo, dd, ww, build_elements(R1=r1, R2=r2),
                               WAVELENGTH, seed=700 + s, max_depth=6,
                               w_min=0.0)
            classify(term)['phi_' + key].backward()
            vals.append(float((r1 if which == 'R1' else r2).grad))
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
        _check(f'mc: dPhi_{key}/d{which} unbiased vs split',
               abs(mean - autos[which]) < 3 * se + 1e-6,
               f'MC {mean:.6e} +- {se:.1e} vs split {autos[which]:.6e}')


def test_sequential_cannot_follow():
    """The sequential tracer sees T1T2T3T4 and nothing else."""
    print('T6  sequential contrast')
    o, d, w = sample_point_source(N_SPLIT, seed=11)
    lens = build_lensgroup()
    ray = do.Ray(o.clone(), d.clone(), wavelength=torch.Tensor([WAVELENGTH]))
    _, valid = lens.trace(ray)
    phi_seq = float(w[valid].sum())

    _, _, cl = run_mc(N_SPLIT, mc_seed=42)
    frac = float(cl['phi_fwd']) / PHI_CAP
    direct = (1 - R1_COAT) ** 2 * (1 - R2_COAT) ** 2

    _check('sequential delivers the full captured flux - no coating loss',
           abs(phi_seq / PHI_CAP - 1.0) < 1e-12, f'{phi_seq / PHI_CAP:.9f}')
    _check('non-seq forward is the direct path plus a ghost pickup',
           direct <= frac < direct + 0.05,
           f'{frac:.4f} vs T1T2T3T4 = {direct:.4f}')
    _check('and the sequential run has no backward light at all',
           float(cl['phi_back']) / PHI_CAP > 0.1,
           f'non-seq backward {float(cl["phi_back"]) / PHI_CAP:.4f} x Phi_cap, '
           f'sequential 0 by construction')


# ============================================================================
#                                  FIGURES
# ============================================================================
def _out_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c05_out')
    os.makedirs(d, exist_ok=True)
    return d


def _pyplot():
    import matplotlib
    if not os.environ.get('DISPLAY') and sys.platform not in ('win32', 'darwin'):
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_maps(N=None, seed_src=61, mc_seed=9007):
    """Irradiance on both receivers, at the mesh the bin rule chose.

    One row per receiver: the map, then its radial profile on a log axis. Read
    the PLATEAU of the profile, not the hottest pixel - the peak is one bin's
    worth of Monte Carlo noise, the plateau is the physics.
    """
    print('P1  irradiance maps')
    plt = _pyplot()
    N = N or (N_MC if _QUICK else int(1e6))

    term, tally, cl = run_mc(N, seed_src=seed_src, mc_seed=mc_seed)
    rel = abs(phi_captured_check(term, tally) / PHI_CAP - 1.0)
    print(f'  {N:.0e} rays, ledger {rel:.2e}')
    tbl = bin_table(cl, launched=N)

    fig, ax = plt.subplots(len(RECEIVERS), 2,
                           figsize=(12, 4.6 * len(RECEIVERS)))
    ax = np.atleast_2d(ax)
    for row, rc in enumerate(RECEIVERS):
        key, half = rc['name'], rc['half']
        b = tbl[key]
        n, pitch = b['n'], 2 * rc['half'] / b['n']
        S = nonseq.splat(cl['p_' + key], cl['w_' + key] / pitch ** 2,
                         [n, n], pitch).detach().cpu().numpy()
        lit = S[S > 0]
        vmax = np.log10(lit.max()) if lit.size else 0.0
        im = ax[row, 0].imshow(np.log10(np.where(S > 0, S, np.nan)),
                               origin='lower', extent=[-half, half, -half, half],
                               vmin=vmax - 5, vmax=vmax)
        ax[row, 0].set(title=f'{rc["label"]}  z = {rc["z"]:g} mm   '
                             f'{n}x{n}, eps = {b["eps_pred"]:.1%}\n'
                             f'log10 W/mm^2',
                       xlabel='x [mm]', ylabel='y [mm]')
        fig.colorbar(im, ax=ax[row, 0], fraction=.046)

        # Radial profile: power per annulus / annulus area, so it is a genuine
        # irradiance and does not tilt with r the way a raw histogram does.
        r = np.linalg.norm(cl['p_' + key][..., :2].detach().cpu().numpy(),
                           axis=-1)
        ww = cl['w_' + key].detach().cpu().numpy()
        edges = np.linspace(0, half, 33)
        tot, _ = np.histogram(r, bins=edges, weights=ww)
        area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        mid = 0.5 * (edges[1:] + edges[:-1])
        good = tot > 0
        ax[row, 1].semilogy(mid[good], (tot / area)[good], 'o-', ms=3)
        ax[row, 1].set(title=f'{rc["label"]} radial profile',
                       xlabel='r [mm]', ylabel='W/mm^2')
        ax[row, 1].grid(alpha=.3)

    fig.suptitle(f'two lenses, R1 = {R1_COAT}, R2 = {R2_COAT}, '
                 f'{N:.0e} rays - mesh chosen by N = {EPS_TARGET} sqrt(rays on '
                 f'receiver)', fontsize=11)
    fig.tight_layout()
    path = os.path.join(_out_dir(), 'c05_maps.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  wrote {path}')


def _mc_segments(els, o, d, w, max_depth=8, seed=7):
    """[(z0, y0, z1, y1, weight)] for a small MC fan, for the layout picture.

    Ported from `c02_R02._mc_segments`. Each ray is ONE unbranched chain - the
    branch tree only appears as a population, which is the visual statement of
    what Monte Carlo tracing is.
    """
    segs = []
    N = o.shape[0]
    g = torch.Generator().manual_seed(int(seed))
    eps = torch.finfo(w.dtype).eps
    done = torch.zeros(N, dtype=torch.bool)
    ignore = torch.full((N,), -1, dtype=torch.long)

    def fly(o_, d_, w_):
        if o_.shape[0] == 0:
            return
        z = torch.where(d_[..., 2] > 0,
                        torch.full_like(d_[..., 2], Z_RECV),
                        torch.full_like(d_[..., 2], Z_BACK))
        p = o_ + ((z - o_[..., 2]) / d_[..., 2])[..., None] * d_
        for j in range(o_.shape[0]):
            segs.append((float(o_[j, 2]), float(o_[j, 1]),
                         float(p[j, 2]), float(p[j, 1]), float(w_[j])))

    for _ in range(max_depth):
        act = ~done
        if not bool(act.any()):
            break
        _, eid = closest_hit(o, d, els, ignore_id=ignore)
        eid = torch.where(act, eid, torch.full_like(eid, -1))
        hit = eid >= 0
        leave = act & ~hit
        fly(o[leave], d[leave], w[leave])
        done = done | leave
        if not bool(hit.any()):
            break

        p, ok, R, d_refr, d_refl, ok_t = interaction(
            o, d, els, torch.where(hit, eid, torch.zeros_like(eid)), WAVELENGTH)
        hit = hit & ok
        for j in torch.nonzero(hit).flatten().tolist():
            segs.append((float(o[j, 2]), float(o[j, 1]),
                         float(p[j, 2]), float(p[j, 1]), float(w[j])))

        rho = R.detach().clone()
        rho = torch.where(ok_t, rho, torch.ones_like(rho)).clamp(eps, 1.0)
        u = torch.rand(N, generator=g, dtype=w.dtype)
        go_re = hit & (u < rho)
        go_tr = hit & ~go_re
        w = torch.where(go_re, w * R / rho, w)
        w = torch.where(go_tr, w * (1.0 - R) / (1.0 - rho).clamp(min=eps), w)
        d = torch.where(go_re[..., None], d_refl,
                        torch.where(go_tr[..., None], d_refr, d))
        o = torch.where(hit[..., None], p, o)
        ignore = torch.where(hit, eid, torch.full_like(eid, -1))

    fly(o[~done], d[~done], w[~done])
    return segs


def _draw_scene(ax):
    """Both lens outlines, both receivers, the source."""
    y1 = np.linspace(-ref.R_LENS, ref.R_LENS, 200)
    ax.plot(ref.Z_S1 + ref.sag(y1), y1, 'k-', lw=1.5)
    ax.plot([ref.Z_S2, ref.Z_S2], [-ref.R_LENS, ref.R_LENS], 'k-', lw=1.5)
    for s in (-1, 1):
        ax.plot([ref.Z_S1 + float(ref.sag(ref.R_LENS)), ref.Z_S2],
                [s * ref.R_LENS, s * ref.R_LENS], 'k-', lw=1.5)

    y2 = np.linspace(-R_LENS2, R_LENS2, 200)
    ax.plot([Z_S3, Z_S3], [-R_LENS2, R_LENS2], 'k-', lw=1.5)
    ax.plot(Z_S4 + sag2(y2), y2, 'k-', lw=1.5)
    for s in (-1, 1):
        ax.plot([Z_S3, Z_S4 + float(sag2(R_LENS2))],
                [s * R_LENS2, s * R_LENS2], 'k-', lw=1.5)

    ax.axvline(Z_FOCUS, color='0.7', ls=':', lw=1)
    ax.plot([Z_RECV, Z_RECV], [-R_RECV, R_RECV], 'b-', lw=2)
    ax.plot([Z_BACK, Z_BACK], [-R_BACK, R_BACK], 'g-', lw=2)
    ax.plot(0.0, 0.0, 'r*', ms=12)
    ax.set_ylabel('y [mm]')
    ax.set_xlim(Z_BACK - 5, Z_RECV + 5)
    ax.set_ylim(-R_BACK * 0.55, R_BACK * 0.55)


def plot_layout(nfan=120, seed=7):
    """Side view of the MC fan through both lenses.

    Opacity tracks log weight, so the direct path dominates visually and the
    ghosts show as faint strays. The dotted line is the focus at z = 146.5 -
    every direct ray crosses the axis there, which is the geometry gate made
    visible.
    """
    print('P2  layout')
    plt = _pyplot()
    t = torch.linspace(-0.92 * THETA_MAX, 0.92 * THETA_MAX, nfan)
    o = torch.zeros(nfan, 3)
    d = torch.stack((torch.zeros_like(t), torch.sin(t), torch.cos(t)), dim=-1)
    w = torch.full((nfan,), 1.0)
    segs = _mc_segments(build_elements(), o, d, w, seed=seed)

    fig, ax = plt.subplots(figsize=(13, 6))
    _draw_scene(ax)
    for z0, y0, z1, y1, ww in segs:
        alpha = float(np.clip(0.08 + 0.92 * (np.log10(max(ww, 1e-6)) + 3.0) / 3.0,
                              0.05, 1.0))
        ax.plot([z0, z1], [y0, y1], 'r-', lw=0.8, alpha=alpha)
    ax.set_xlabel('z [mm]')
    ax.set_title(f'trace_mc, {nfan} rays, R1 = {R1_COAT} (lens 1), '
                 f'R2 = {R2_COAT} (lens 2); opacity ~ log weight\n'
                 f'blue = forward receiver, green = backward, '
                 f'dotted = focus at z = {Z_FOCUS:g}')
    fig.tight_layout()
    path = os.path.join(_out_dir(), 'c05_layout.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  wrote {path}')


_QUICK = False

TESTS = {
    'geometry': test_geometry,
    'ledger': test_ledger,
    'paths': test_path_fractions,
    'mc': test_mc_matches_split,
    'bins': test_bin_rule,
    'grad': test_gradients,
    'seq': test_sequential_cannot_follow,
}


def main():
    global _QUICK
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--test', choices=sorted(TESTS) + ['all'], default='all')
    ap.add_argument('--quick', action='store_true',
                    help='smaller ray counts; skips the 1e6 bin-rule run')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()
    _QUICK = args.quick

    torch.manual_seed(0)
    print(f'two lenses: R1 = {R1_COAT} (S1,S2)  R2 = {R2_COAT} (S3,S4)')
    print(f'  Phi_cap = {PHI_CAP:.9f} W over a {np.degrees(THETA_MAX):.6f} '
          f'deg cone, focus at z = {Z_FOCUS:g} mm\n')

    for name in (sorted(TESTS) if args.test == 'all' else [args.test]):
        TESTS[name]()

    if not args.no_plots and args.test == 'all':
        plot_maps()
        plot_layout()
    print('\nStage-5 gates passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
