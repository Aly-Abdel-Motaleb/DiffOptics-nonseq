"""
PHASE 2 - STAGE 1: non-sequential vs sequential, reflection OFF.

This is PLAN_A_collimator.md P2. Phase 1 (`n01_geometry.py`, now living in
`diffoptics/nonseq.py`) gave us `Element` / `closest_hit` / `intersect_one`.
That layer only answers "what does this ray hit next". Nothing yet turns a hit
into a new ray, and nothing yet carries power.

The four missing pieces now live in `diffoptics/nonseq.py` next to the geometry
layer, scene-independent, and are imported here:

    sample_point_source   isotropic point source, uniform in SOLID ANGLE
    refract               Snell at a hit, normal flipped against the ray
    trace_nonseq          the bounce loop: closest_hit -> intersect_one -> refract
    propagate_to_z        free flight to the receiver plane
    splat                 weighted bilinear accumulation onto the receiver

What stays in this file is the SCENE: the constants, the two builders, and thin
wrappers that bind the scene constants to those library calls. The job of the
file is to prove the result is not a new physics engine but the SAME physics dO
already has.

With `R_fixed = 0.0` on both surfaces there is no reflection and no Fresnel
loss, so the non-sequential tracer must reproduce dO's sequential tracer
EXACTLY - same survivors, same directions, same irradiance map. Any difference
is a bug in this file, not physics. Stage 2 (`c02_R02.py`) turns reflection on;
that is the point where sequential can no longer follow.

The tests at the bottom are complete and are the gate - do not edit them.
Reference solution: `c01_seq_vs_nonseq_sol.py`.

--------------------------------------------------------------------------------
THE SCENE
--------------------------------------------------------------------------------

    point source          collimating lens              forward receiver
    (0,0,0), 1 W    ->    S1 asphere @ z = 40      ->   z = 200 mm
    isotropic             S2 flat    @ z = 46.5         +-16 mm, 512^2
    into a cone           N-BK7, semi-diameter 12.7

The lens is an EXACT aspheric collimator: a conic refracting surface with the
object at distance `s` collimates perfectly, with no spherical aberration, when

    c = 1 / (s (n_rel - 1)),      k = -n_rel^2,      n_rel = n_glass / n_air

Note `n_rel`, not `n_glass`. dO's "air" is n = 1.000293 (basics.py:188), not 1,
so using the absolute index leaves a residual ~1e-4 rad tilt that the 1e-6
collimation gate below would flag. The rays leave S1 exactly parallel to +z, hit
the flat S2 at normal incidence, and stay parallel. That is what makes this a
sharp test: every output ray must satisfy |d_x|, |d_y| < 1e-6, so a sign error
in the geometry layer fails instantly where an image comparison would hide it.

Numbers that fall out (they are asserted in T0):
    n(N-BK7, 532.8 nm) = 1.5195      c = 0.04816 /mm      k = -2.3076
    sag(12.7) = 3.499 mm             theta_max = 16.28 deg

--------------------------------------------------------------------------------
UNIFORM IN SOLID ANGLE - THE #1 TRAP (PLAN_A gotcha #3)
--------------------------------------------------------------------------------

dO has NO point-source sampler; `sample_ray` and friends (optics.py:539,589,615)
are all collimated, and `pointSrc_spherical.py:55-62` grids a PLANE, which is
uniform in area but NOT in solid angle - it over-weights the centre by 1/cos^3.
For a hit-count image with the same rays in both tracers that bias cancels; for
irradiance in W/mm^2, and for the LightTools comparison in P4, it does not.

An isotropic source of total power P radiates dPhi = P/(4 pi) dOmega. Sampling a
cone of half-angle theta_max uniformly in solid angle:

    cos(theta) ~ U(cos theta_max, 1)      phi ~ U(0, 2 pi)
    Omega  = 2 pi (1 - cos theta_max)
    Phi_captured = P/(4 pi) * Omega = P (1 - cos theta_max) / 2
    per-ray weight w = Phi_captured / N        [W/ray]

Sampling cos(theta) uniformly - not theta - is the whole trick. Here
Phi_captured = 2.0038e-2 W of the 1 W emitted, i.e. the lens catches 2 %.

--------------------------------------------------------------------------------
FOUR MORE SHARP EDGES
--------------------------------------------------------------------------------

1. `Lensgroup.render` (optics.py:712-759) has no per-ray weight: `J = irr` is a
   scalar and invalid rays are dropped by boolean indexing. It is a bilinear
   HIT-COUNT histogram, not an irradiance map. So T3 compares it against
   `splat(..., w = 1)`, and only T4 uses physical weights. `splat` here is
   `render`'s four `index_put(..., accumulate=True)` calls with `J -> w`, and
   must keep render's pixel convention exactly or the maps shift by half a pixel.

2. `Lensgroup.trace` MUTATES the ray it is given (`ray.o = p`, optics.py:1087).
   Build a fresh `do.Ray` per call, or clone `o`/`d`, otherwise the second
   tracer sees rays that already sit on S2.

3. `_refract` needs the normal oriented ALONG propagation: dO calls
   `self._refract(ray.d, -n, eta)` in forward mode (optics.py:1075) because
   `Surface.normal` points to -z. Our `intersect_one` returns the normal toward
   local +z regardless of which side the ray came from, so `refract` here must
   flip it itself using the sign of dot(d, n). `Element.eta_at` already resolves
   n_i/n_t from that same sign, so never hand-pick eta per surface.

4. Float32. dO's Newton solver converges to NEWTONS_TOLERANCE_LOOSE = 300 nm
   (optics.py:1183); in float32 that leaves ~1e-7 rad on the output directions
   and ~5e-5 relative L2 between the two irradiance maps - agreement, but not
   provable agreement at the 1e-6 gates. `torch.set_default_dtype(torch.float64)`
   at the top of the file (before dO is imported) takes both gates to ~1e-13.
   The scene is tiny; the cost is nothing.

Re self-hits: the new ray origin is the hit point itself, so the next
`closest_hit` would find t ~ 0 on the surface just left. `ignore_id` is what
prevents that - no epsilon offset needed (that is why Phase 1 built it).

Run:    python c01_seq_vs_nonseq.py
Gates:  T1 both tracers collimate to < 1e-6;
        T2 identical survivor set, sum of weights = Phi_captured;
        T3 dO `render` vs non-seq `splat` relative L2 < 1e-6;
        T4 radial profiles overlay.
Once these pass, the scene builders / sampler / splat move into
`common_collimator.py` and `c02_R02.py` switches R_fixed to 0.2.
"""
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)   # edge 4 above; set before importing dO

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do  # noqa: E402
from diffoptics.nonseq import (  # noqa: E402
    Element, closest_hit, intersect_one, propagate_to_z, refract)
from diffoptics import nonseq  # noqa: E402


# ------------------------------------------------------------------ constants
WAVELENGTH = 532.8          # [nm]
S_SRC      = 40.0           # [mm] point source -> S1 vertex
R_LENS     = 12.7           # [mm] semi-diameter
THICK      = 6.5            # [mm] centre thickness
Z_S1       = 40.0           # [mm] world z of the S1 vertex (source sits at z = 0)
Z_S2       = Z_S1 + THICK
Z_RECV     = 200.0          # [mm] forward receiver
R_RECV     = 16.0           # [mm] receiver half-size
FILM       = [512, 512]
PIXEL      = 2 * R_RECV / FILM[0]
P_TOTAL    = 1.0            # [W] isotropic point source, total over 4 pi

N_AIR   = do.Material('air').ior(WAVELENGTH)
N_GLASS = do.Material('n-bk7').ior(WAVELENGTH)
N_REL   = float(N_GLASS / N_AIR)

C_ASPH = 1.0 / (S_SRC * (N_REL - 1.0))
K_ASPH = -N_REL ** 2


def sag(r):
    """Aspheric sag z(r) of S1, in mm. Plain numpy - used for THETA_MAX only.

    """
    r2 = np.asarray(r, dtype=np.float64) ** 2
    return C_ASPH * r2 / (1 + np.sqrt(1 - (1 + K_ASPH) * C_ASPH ** 2 * r2))



# S1 is at radius R_LENS but at z = S_SRC + sag(R_LENS), NOT at z = S_SRC -
# forgetting the sag makes the cone too wide and rays start missing the lens.
THETA_MAX = float(np.arctan(R_LENS / (S_SRC + sag(R_LENS))))
PHI_CAP   = P_TOTAL * (1 - np.cos(THETA_MAX)) / 2.0  # [W] captured by the lens


# --------------------------------------------------------------------- scenes
def build_lensgroup(device=torch.device('cpu')):
    """The SEQUENTIAL reference: the same lens as a `do.Lensgroup`.

    Same shape as `examples/practice/common_setup.py:45-54`, except the group
    origin is moved to z = Z_S1 so that world coordinates agree with the
    non-sequential scene (source at the world origin). `Lensgroup.trace` takes
    and returns WORLD rays, and `render` compares against `d_sensor` in world z
    too, so d_sensor = Z_RECV.
    """
    lens = do.Lensgroup(origin=np.array([0, 0, Z_S1]), device=device)
    lens.load(
        [do.Aspheric(R_LENS, 0.0,   c=C_ASPH, k=K_ASPH, device=device),
         do.Aspheric(R_LENS, THICK, c=0.0,               device=device)],
        [do.Material('air'), do.Material('n-bk7'), do.Material('air')],
    )
    lens.d_sensor  = Z_RECV
    lens.r_last    = R_LENS
    lens.film_size = FILM
    lens.pixel_size = PIXEL
    return lens


def build_elements(R=0.0):
    """The NON-SEQUENTIAL scene: the identical lens as two posed `Element`s.

    Same constants as `build_lensgroup` - that is the point of keeping both
    builders in one file. Each element carries its own pose here, so S2's offset
    is in the `Transformation`, not in the surface `d`.

    Media follow the Phase 1 convention: `n_out` is the medium on the element's
    local +z side. Glass is on the +z side of S1 and on the -z side of S2.
    """
    eye = torch.eye(3)
    s1 = Element(do.Aspheric(R_LENS, 0.0, c=C_ASPH, k=K_ASPH),
                     do.Transformation(eye, torch.Tensor([0.0, 0.0, Z_S1])),
                     n_in='air', n_out='n-bk7', kind='refractive', R_fixed=R, name='S1')
    s2 = Element(do.Aspheric(R_LENS, 0.0, c=0.0),
                     do.Transformation(eye, torch.Tensor([0.0, 0.0, Z_S2])),
                     n_in='n-bk7', n_out='air', kind='refractive', R_fixed=R, name='S2')
    return [s1, s2]
    

# -------------------------------------------------- scene-bound library calls
# `refract` and `propagate_to_z` carry no scene state and are imported as they
# are. These three do, so they are bound here once instead of at every call.
def sample_point_source(N, theta_max=THETA_MAX, seed=0, P=P_TOTAL,
                        origin=(0.0, 0.0, 0.0)):
    """`nonseq.sample_point_source` with this scene's cone and source power.

    Uniform in SOLID ANGLE, per the trap above: cos(theta) ~ U(cos theta_max, 1),
    phi ~ U(0, 2 pi), per-ray weight w = Phi_captured / N [W].
    """
    return nonseq.sample_point_source(N, theta_max, seed=seed, P=P, origin=origin)


def splat(p, w, film_size=FILM, pixel_size=PIXEL, device=torch.device('cpu')):
    """`nonseq.splat` on this scene's forward receiver.

    Callers pass w = 1 for a hit-count map (to compare with `render`, edge 1) or
    w = power / pixel_area for irradiance in W/mm^2.
    """
    return nonseq.splat(p, w, film_size, pixel_size, device=device)


def trace_nonseq(o, d, w, elements, wavelength=WAVELENGTH, max_bounces=8):
    """`nonseq.trace_nonseq` at this scene's wavelength.

    Refraction only (R_fixed = 0.0), so no ray splits. Returns
    (o, d, w, alive, nhit); a ray through both lens surfaces has nhit == 2,
    which is what T2 compares against the sequential `valid` mask.
    """
    return nonseq.trace_nonseq(o, d, w, elements, wavelength,
                               max_bounces=max_bounces)



# ============================================================================
#                       TESTS - complete, do not edit
# ============================================================================
N_RAYS   = 200000
TOL_COLL = 1e-6
TOL_L2   = 1e-6


def _check(name, cond, detail=''):
    print(f'  [{"ok " if cond else "FAIL"}] {name}{(" - " + detail) if detail else ""}')
    assert cond, name


def _seq_run(lens, o, d):
    ray = do.Ray(o.clone(), d.clone(), wavelength=torch.Tensor([WAVELENGTH]))
    ray_out, valid = lens.trace(ray)
    return ray_out, valid


def _ns_run(els, o, d, w):
    o1, d1, w1, alive, nhit = trace_nonseq(o.clone(), d.clone(), w.clone(), els)
    through = alive & (nhit == 2)
    return o1, d1, w1, through


def test_scene_constants():
    print('T0  scene constants')
    _check('n(N-BK7) at 532.8 nm', 1.51 < float(N_GLASS) < 1.53, f'n = {float(N_GLASS):.4f}')
    _check('c = 1/(s(n_rel-1))', abs(C_ASPH - 0.04812) < 1e-4, f'c = {C_ASPH:.5f}')
    _check('k = -n_rel^2', abs(K_ASPH + 2.3089) < 2e-3, f'k = {K_ASPH:.4f}')
    _check('sag(12.7) ~ 3.49 mm', abs(float(sag(R_LENS)) - 3.49) < 0.02,
           f'sag = {float(sag(R_LENS)):.4f}')
    _check('theta_max ~ 16.3 deg', abs(np.rad2deg(THETA_MAX) - 16.3) < 0.1,
           f'{np.rad2deg(THETA_MAX):.2f} deg')
    _check('captured flux', abs(PHI_CAP - (1 - np.cos(THETA_MAX)) / 2) < 1e-12,
           f'{PHI_CAP:.6e} W')


def test_collimation():
    print('T1  collimation, both tracers')
    lens, els = build_lensgroup(), build_elements(R=0.0)
    o, d, w = sample_point_source(20000, seed=1)

    ray_out, valid = _seq_run(lens, o, d)
    d_seq = ray_out.d[valid]
    o_ns, d_ns, _, through = _ns_run(els, o, d, w)
    d_nsv = d_ns[through]

    tan_seq = float(torch.max(torch.abs(d_seq[..., 0:2])))
    tan_ns = float(torch.max(torch.abs(d_nsv[..., 0:2])))
    _check('sequential collimated', tan_seq < TOL_COLL, f'max|dxy| = {tan_seq:.2e}')
    _check('non-seq collimated', tan_ns < TOL_COLL, f'max|dxy| = {tan_ns:.2e}')
    _check('both exit inside the clear aperture',
           float(torch.max(torch.sqrt(torch.sum(o_ns[through][..., 0:2] ** 2, dim=-1)))) < R_LENS + 1e-3)


def test_power_and_counts():
    print('T2  ray survival and power')
    lens, els = build_lensgroup(), build_elements(R=0.0)
    o, d, w = sample_point_source(N_RAYS, seed=2)

    _, valid = _seq_run(lens, o, d)
    _, _, w_ns, through = _ns_run(els, o, d, w)

    n_seq, n_ns = int(valid.sum()), int(through.sum())
    agree = int((valid == through).sum())
    _check('same number of survivors', n_seq == n_ns, f'seq {n_seq} vs nonseq {n_ns}')
    _check('same rays survive', agree == N_RAYS, f'{N_RAYS - agree} disagree')
    _check('all sampled rays make it through', n_ns == N_RAYS, f'{N_RAYS - n_ns} lost')

    phi = float(w_ns[through].sum())
    _check('sum w = captured flux', abs(phi / PHI_CAP - 1.0) < 1e-5,
           f'{phi:.6e} vs {PHI_CAP:.6e} W')


def test_irradiance_matches():
    print('T3  irradiance map, dO render vs non-seq splat')
    lens, els = build_lensgroup(), build_elements(R=0.0)
    o, d, w = sample_point_source(N_RAYS, seed=3)

    ray = do.Ray(o.clone(), d.clone(), wavelength=torch.Tensor([WAVELENGTH]))
    I_seq = lens.render(ray, irr=1.0)

    o_ns, d_ns, _, through = _ns_run(els, o, d, w)
    p = propagate_to_z(o_ns[through], d_ns[through], Z_RECV)
    on_film = ((torch.abs(p[..., 0]) <= R_RECV) & (torch.abs(p[..., 1]) <= R_RECV))
    p = p[on_film]
    I_ns = splat(p, torch.ones(p.shape[0], dtype=p.dtype))

    _check('same hit count on film',
           abs(float(I_seq.sum()) - float(I_ns.sum())) < 1e-3,
           f'{float(I_seq.sum()):.3f} vs {float(I_ns.sum()):.3f}')
    rel = float(torch.norm(I_ns - I_seq) / torch.norm(I_seq))
    _check('relative L2 < 1e-6', rel < TOL_L2, f'rel L2 = {rel:.3e}')


def test_radial_profile():
    print('T4  radial profile')
    lens, els = build_lensgroup(), build_elements(R=0.0)
    o, d, w = sample_point_source(N_RAYS, seed=4)

    ray = do.Ray(o.clone(), d.clone(), wavelength=torch.Tensor([WAVELENGTH]))
    I_seq = lens.render(ray, irr=1.0)
    o_ns, d_ns, w_ns, through = _ns_run(els, o, d, w)
    p = propagate_to_z(o_ns[through], d_ns[through], Z_RECV)
    I_ns = splat(p, w_ns[through] / (PIXEL ** 2))

    r_pix = R_RECV / PIXEL
    yy, xx = torch.meshgrid(torch.arange(FILM[0]) - FILM[0] / 2 + 0.5,
                            torch.arange(FILM[1]) - FILM[1] / 2 + 0.5, indexing='ij')
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    nb = 64
    edges = torch.linspace(0, r_pix, nb + 1)

    def profile(I):
        out = np.zeros(nb)
        for i in range(nb):
            m = (rr >= edges[i]) & (rr < edges[i + 1])
            out[i] = float(I[m].mean()) if bool(m.any()) else 0.0
        return out

    pr_seq, pr_ns = profile(I_seq), profile(I_ns)
    scale = pr_seq.sum() / max(pr_ns.sum(), 1e-30)
    band = np.where(pr_seq > 0.05 * pr_seq.max())[0]
    rel = np.abs(pr_ns[band] * scale - pr_seq[band]) / np.maximum(pr_seq[band], 1e-30)
    _check('profiles overlay inside the beam', float(rel.max()) < 1e-6,
           f'max rel = {float(rel.max()):.2e}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c01_out')
        os.makedirs(out, exist_ok=True)
        rc = 0.5 * (edges[:-1] + edges[1:]).numpy() * PIXEL
        ext = [-R_RECV, R_RECV, -R_RECV, R_RECV]
        # I_seq is a bilinear hit-count map (render's J = 1), I_ns is W/mm^2:
        # rescale the non-seq map onto the sequential's units before differencing.
        map_scale = float(I_seq.sum()) / max(float(I_ns.sum()), 1e-30)
        A, B = I_seq.numpy(), I_ns.numpy() * map_scale
        D = np.abs(B - A)

        fig, ax = plt.subplots(1, 4, figsize=(19, 4))
        im0 = ax[0].imshow(A.T, origin='lower', extent=ext)
        ax[0].set_title('sequential [counts]')
        fig.colorbar(im0, ax=ax[0], fraction=0.046)
        im1 = ax[1].imshow(I_ns.numpy().T, origin='lower', extent=ext)
        ax[1].set_title('non-seq irradiance [W/mm^2]')
        fig.colorbar(im1, ax=ax[1], fraction=0.046)
        im2 = ax[2].imshow(D.T, origin='lower', extent=ext)
        ax[2].set_title('|non-seq - seq|')
        ax[2].set_xlabel(f'max {D.max():.3e}   mean {D.mean():.3e}\n'
                         f'rel to peak {D.max() / max(A.max(), 1e-30):.2e}   '
                         f'(peak signal {A.max():.3g})')
        fig.colorbar(im2, ax=ax[2], fraction=0.046)
        ax[3].plot(rc, pr_seq / max(pr_seq.max(), 1e-30), 'k-', label='sequential')
        ax[3].plot(rc, pr_ns * scale / max(pr_seq.max(), 1e-30), 'r--', label='non-seq')
        ax[3].set_xlabel('r [mm]'), ax[3].legend(), ax[3].set_title('radial profile')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'c01_profile.png'), dpi=120)
        plt.close(fig)
        print(f'  wrote {os.path.join(out, "c01_profile.png")}')
    except ImportError:
        pass


def _fan(nfan=13):
    """Meridional fan in the y-z plane, spanning the full collection cone."""
    # 0.99 keeps the outermost pair off the rim, where `is_valid` is a coin flip
    t = torch.linspace(-0.99 * THETA_MAX, 0.99 * THETA_MAX, nfan,
                       dtype=torch.get_default_dtype())
    d = torch.stack((torch.zeros_like(t), torch.sin(t), torch.cos(t)), dim=-1)
    return torch.zeros(nfan, 3, dtype=d.dtype), d


def _verts_seq(lens, o, d):
    """[source, S1 hit, S2 hit, receiver], world frame, from dO's sequential tracer.

    `trace(ray, stop_ind=i)` stops after surface i and returns a WORLD ray whose
    origin is that surface's hit point. `trace_r` would hand back the whole
    record in one call, but its local -> world loop (optics.py:854) rebinds the
    loop variable and is a no-op, so the vertices come back in mixed frames.
    """
    pts = [o.clone()]
    for stop in (0, 1):
        r = do.Ray(o.clone(), d.clone(), wavelength=torch.Tensor([WAVELENGTH]))
        pts.append(lens.trace(r, stop_ind=stop)[0].o.clone())
    r = do.Ray(o.clone(), d.clone(), wavelength=torch.Tensor([WAVELENGTH]))
    r_out, valid = lens.trace(r)
    pts.append(propagate_to_z(r_out.o, r_out.d, Z_RECV))
    return pts, valid


def _verts_ns(els, o, d, max_bounces=4):
    """Same vertex list from the non-sequential tracer, recorded bounce by bounce."""
    oc, dc = o.clone(), d.clone()
    ignore = torch.full((o.shape[0],), -1, dtype=torch.long)
    pts = [oc.clone()]
    for _ in range(max_bounces):
        _, eid = closest_hit(oc, dc, els, ignore_id=ignore)
        hit = eid >= 0
        if not bool(hit.any()):
            break
        p, n_geom, ok = intersect_one(oc, dc, els, eid)
        hit = hit & ok
        eta = torch.ones_like(oc[..., 0])
        for i, el in enumerate(els):
            m = hit & (eid == i)
            if bool(m.any()):
                eta[m] = el.eta_at(dc[m], n_geom[m], WAVELENGTH)
        valid_d, d_new = refract(dc, n_geom, eta)
        oc = torch.where(hit[..., None], p, oc)
        dc = torch.where((hit & valid_d)[..., None], d_new, dc)
        ignore = torch.where(hit, eid, torch.full_like(eid, -1))
        pts.append(oc.clone())
    pts.append(propagate_to_z(oc, dc, Z_RECV))
    return pts


def plot_layout(nfan=13):
    """Side view of the scene with a ray fan, one panel per tracer."""
    print('L   layout view')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib missing, skipped')
        return

    lens, els = build_lensgroup(), build_elements(R=0.0)
    o, d = _fan(nfan)
    pts_seq, valid = _verts_seq(lens, o, d)
    pts_ns = _verts_ns(els, o, d)

    y = np.linspace(-R_LENS, R_LENS, 200)
    s1_z = Z_S1 + sag(y)                       # curved front, sag opens along +z
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c01_out')
    os.makedirs(out, exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    for a, pts, title in ((ax[0], pts_seq, 'sequential (dO Lensgroup.trace)'),
                          (ax[1], pts_ns, 'non-sequential (closest_hit + intersect_one)')):
        a.plot(s1_z, y, 'k-', lw=1.5)                                   # S1
        a.plot([Z_S2, Z_S2], [-R_LENS, R_LENS], 'k-', lw=1.5)           # S2, flat
        for s in (-1, 1):                                               # glass rim
            a.plot([Z_S1 + sag(R_LENS), Z_S2], [s * R_LENS, s * R_LENS], 'k-', lw=1.5)
        a.plot([Z_RECV, Z_RECV], [-R_RECV, R_RECV], 'b-', lw=2)         # receiver
        a.plot(0.0, 0.0, 'r*', ms=12)                                   # point source

        for j in range(nfan):
            zs = [float(p[j, 2]) for p in pts]
            ys = [float(p[j, 1]) for p in pts]
            a.plot(zs, ys, 'r-', lw=0.7, alpha=0.85)

        a.set_ylabel('y [mm]')
        a.set_title(title)
        a.set_ylim(-R_RECV - 2, R_RECV + 2)
    ax[1].set_xlabel('z [mm]')
    ax[0].text(2, R_RECV - 3, f'point source, {np.rad2deg(THETA_MAX):.1f} deg half-cone',
               color='r', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'c01_layout.png'), dpi=120)
    plt.close(fig)
    _check('all fan rays survive the sequential trace', bool(valid.all()))
    print(f'  wrote {os.path.join(out, "c01_layout.png")}')


if __name__ == '__main__':
    torch.manual_seed(0)
    test_scene_constants()
    test_collimation()
    test_power_and_counts()
    test_irradiance_matches()
    test_radial_profile()
    plot_layout()
    print('\nPhase 2 stage-1 gates passed.')
