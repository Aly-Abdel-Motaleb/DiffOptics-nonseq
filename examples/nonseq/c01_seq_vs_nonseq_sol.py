"""
PHASE 2 - STAGE 1: non-sequential vs sequential, reflection OFF.  [SOLUTION]

Reference implementation of `c01_seq_vs_nonseq.py` - read that file first, all
the reasoning lives there. Same tests, same tolerances.

Measured on this machine (float64, N = 2e5):

    T1  max|d_xy| after the lens       3.4e-16 (seq)   3.0e-16 (non-seq)
    T2  survivors                      200000 == 200000, masks identical
        sum w                          2.003822e-02 W == Phi_captured
    T3  relative L2, render vs splat   1.0e-13
    T4  max relative profile error     2.5e-15

In float32 the same code gives 1.8e-07 / 5.5e-05 / 1e-4-ish - correct, but the
1e-6 gates cannot see it. That is the only reason for the float64 line below.
"""
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)   # see the docstring of the exercise file

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do  # noqa: E402
from diffoptics.nonseq import Element, closest_hit, intersect_one  # noqa: E402


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
    r2 = np.asarray(r, dtype=np.float64) ** 2
    return C_ASPH * r2 / (1.0 + np.sqrt(1.0 - (1.0 + K_ASPH) * C_ASPH ** 2 * r2))


THETA_MAX = float(np.arctan2(R_LENS, S_SRC + sag(R_LENS)))
PHI_CAP   = P_TOTAL * (1.0 - np.cos(THETA_MAX)) / 2.0


# --------------------------------------------------------------------- scenes
def build_lensgroup(device=torch.device('cpu')):
    lens = do.Lensgroup(origin=np.array([0.0, 0.0, Z_S1]), device=device)
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
    eye = torch.eye(3)
    s1 = Element(do.Aspheric(R_LENS, 0.0, c=C_ASPH, k=K_ASPH),
                 do.Transformation(eye, torch.Tensor([0.0, 0.0, Z_S1])),
                 n_in='air', n_out='n-bk7', kind='refractive', R_fixed=R, name='S1')
    s2 = Element(do.Aspheric(R_LENS, 0.0, c=0.0),
                 do.Transformation(eye, torch.Tensor([0.0, 0.0, Z_S2])),
                 n_in='n-bk7', n_out='air', kind='refractive', R_fixed=R, name='S2')
    return [s1, s2]


# --------------------------------------------------------------------- source
def sample_point_source(N, theta_max=THETA_MAX, seed=0, P=P_TOTAL,
                        origin=(0.0, 0.0, 0.0)):
    g = torch.Generator().manual_seed(int(seed))
    u1 = torch.rand(N, generator=g, dtype=torch.get_default_dtype())
    u2 = torch.rand(N, generator=g, dtype=torch.get_default_dtype())

    cos_max = float(np.cos(theta_max))
    cost = cos_max + (1.0 - cos_max) * u1
    sint = torch.sqrt(torch.clamp(1.0 - cost ** 2, min=0.0))
    phi = 2.0 * np.pi * u2

    d = torch.stack((sint * torch.cos(phi), sint * torch.sin(phi), cost), dim=-1)
    o = torch.as_tensor(origin, dtype=d.dtype).expand(N, 3).clone()

    phi_cap = P * (1.0 - cos_max) / 2.0
    w = torch.full((N,), phi_cap / N, dtype=d.dtype)
    return o, do.normalize(d), w


# --------------------------------------------------------------------- splat
def splat(p, w, film_size=FILM, pixel_size=PIXEL, device=torch.device('cpu')):
    R_sensor = [film_size[i] * pixel_size / 2 for i in range(2)]
    u = (p[..., 0] + R_sensor[0]) / pixel_size
    v = (p[..., 1] + R_sensor[1]) / pixel_size

    index_l = torch.stack(
        (torch.clamp(torch.floor(u).long(), min=0, max=film_size[0] - 1),
         torch.clamp(torch.floor(v).long(), min=0, max=film_size[1] - 1)), dim=-1)
    index_r = torch.stack(
        (torch.clamp(index_l[..., 0] + 1, min=0, max=film_size[0] - 1),
         torch.clamp(index_l[..., 1] + 1, min=0, max=film_size[1] - 1)), dim=-1)
    w_r = torch.clamp(torch.stack((u, v), dim=-1) - index_l, min=0, max=1)
    w_l = 1.0 - w_r

    I = torch.zeros(*film_size, device=device, dtype=p.dtype)
    I = torch.index_put(I, (index_l[..., 0], index_l[..., 1]), w_l[..., 0] * w_l[..., 1] * w, accumulate=True)
    I = torch.index_put(I, (index_r[..., 0], index_l[..., 1]), w_r[..., 0] * w_l[..., 1] * w, accumulate=True)
    I = torch.index_put(I, (index_l[..., 0], index_r[..., 1]), w_l[..., 0] * w_r[..., 1] * w, accumulate=True)
    I = torch.index_put(I, (index_r[..., 0], index_r[..., 1]), w_r[..., 0] * w_r[..., 1] * w, accumulate=True)
    return I


# ------------------------------------------------------------------- physics
def refract(d, n_geom, eta):
    cosi = torch.sum(d * n_geom, dim=-1)
    n_f = torch.where((cosi < 0)[..., None], -n_geom, n_geom)
    cosi = torch.abs(cosi)

    cost2 = 1.0 - (1.0 - cosi ** 2) * eta ** 2
    valid = cost2 > 0.0
    cost = torch.sqrt(torch.clamp(cost2, min=1e-8))
    d_t = eta[..., None] * d + (cost - eta * cosi)[..., None] * n_f
    return valid, d_t


def trace_nonseq(o, d, w, elements, wavelength=WAVELENGTH, max_bounces=8):
    N = o.shape[0]
    alive = torch.ones(N, dtype=torch.bool)
    nhit = torch.zeros(N, dtype=torch.long)
    ignore = torch.full((N,), -1, dtype=torch.long)

    for _ in range(max_bounces):
        if not bool(alive.any()):
            break

        _, eid = closest_hit(o, d, elements, ignore_id=ignore)
        eid = torch.where(alive, eid, torch.full_like(eid, -1))
        hit = eid >= 0
        if not bool(hit.any()):
            break

        p, n_geom, ok = intersect_one(o, d, elements, eid)
        hit = hit & ok

        eta = torch.ones(N, dtype=d.dtype)
        for i, el in enumerate(elements):
            m = hit & (eid == i)
            if bool(m.any()):
                eta[m] = el.eta_at(d[m], n_geom[m], wavelength)

        valid_d, d_new = refract(d, n_geom, eta)

        o = torch.where(hit[..., None], p, o)
        d = torch.where((hit & valid_d)[..., None], d_new, d)
        nhit = nhit + hit.long()
        ignore = torch.where(hit, eid, torch.full_like(eid, -1))
        alive = alive & (~hit | valid_d)

    return# ============================================================================
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
