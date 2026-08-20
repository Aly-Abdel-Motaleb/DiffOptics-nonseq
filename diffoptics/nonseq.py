"""Non-sequential geometry layer: per-element poses and closest-hit search.

Scene-independent: no Monte Carlo, no Fresnel, no ray splitting.

    Element       a `Surface` + a world pose + an interface description
    closest_hit   pass 1, under no_grad: which element does each ray hit first?
    intersect_one pass 2, with grad: recompute only that one intersection

On top of that geometry, the transport layer of PLAN_A_collimator.md P2 — still
scene-independent, every scene constant is an argument:

    sample_point_source  isotropic point source, uniform in SOLID ANGLE
    refract              Snell at a hit, normal flipped against the ray
    trace_nonseq         the bounce loop: closest_hit -> intersect_one -> refract
    propagate_to_z       free flight to a plane z = const
    splat                weighted bilinear accumulation onto a receiver

And on top of that, the partial-reflection layer of PLAN_A_collimator.md P3 —
the two tracers that follow BOTH children of a hit:

    reflect        mirror direction about a normal
    fresnel_R      unpolarized Fresnel reflectance
    reflectance    R at an element: the `R_fixed` override, else Fresnel
    interaction    one hit for a batch: geometry, media, R, both children
    trace_split    deterministic, every hit spawns both children
    trace_mc       Monte Carlo, one child per hit, weight divided by its prob

Both tracers return the same `(terminal, tally)` pair, so a scene can swap one
for the other without touching its receivers. `terminal` holds every ray that
LEFT the scene, still free-flying — a receiver is a `propagate_to_z` away.
`tally` holds the weight that left the loop without ever reaching a receiver,
which is what closes the energy ledger:

    terminal['w'].sum() + tally['culled'] + tally['truncated'] == input w.sum()

Conventions
-----------
Local frame. A `Surface` is the implicit function f(x,y,z) = g(x,y) + h(z) + d,
so the surface sits at local z = sag(x,y) + d, the sag opens along local +z, and
the finite aperture is the disc/square of radius `surface.r` in local xy.

Media. `n_out` is the medium on the local +z side, `n_in` the medium on the
local -z side. `eta_at` resolves the incident/transmitted pair from the sign of
dot(d, n).

Normal. `intersect_one` always returns the geometric normal pointing toward
local +z (into `n_out`); it is never flipped against the incident ray.
`Surface.normal` points toward -z, so it is negated here.
"""
import math
import os

import torch

from .basics import Material, PrettyPrinter, Transformation, normalize

EPS_DZ = 1e-9      # |d_z| in the local frame below which a ray counts as parallel
EPS_T  = 1e-6      # [mm] minimum travel distance; rejects the self-hit at t ~ 0
T_MAX  = 1e5       # [mm] maximum travel distance, matches Ray.maxt
MAX_DEPTH = 10     # default generations before a splitting tracer gives up

KINDS = ('refractive', 'mirror', 'partial', 'absorber')


class Element(PrettyPrinter):
    """A `Surface` plus a world pose plus an interface description.

    Args:
        surface: any dO Surface (Aspheric / BSpline / XYPolynomial / Mesh).
        to_world: `Transformation` mapping local -> world. None = identity.
        n_in: medium on the local -z side. str or `Material`.
        n_out: medium on the local +z side. str or `Material`.
        kind: one of KINDS.
        R_fixed: constant reflectance overriding Fresnel, or None for Fresnel.
            0.0 turns reflection off entirely.
        rho_fixed: Monte Carlo sampling-probability override, or None to use
            `R.detach()`.
        name: label for printouts.
        device: None leaves everything where it was built (the old behaviour).
            Anything else moves `surface` there — which the surface classes do
            NOT do for themselves: `Aspheric.__init__` (optics.py:1491) builds
            `c` and `k` with a bare `torch.Tensor(...)` and silently ignores its
            own `device=` argument. `Lensgroup` gets away with it because
            `load()` calls `_sync()` (optics.py:75-78); an `Element` has no such
            rescue, so it does the move here.

            `surface.device` is set explicitly afterwards because it is a plain
            `torch.device` attribute, not a tensor, so `PrettyPrinter.to`
            (basics.py:22-33) walks straight past it — and
            `newtons_method_impl` reads it (optics.py:1335).
    """
    def __init__(self, surface, to_world=None, n_in='air', n_out='air',
                 kind='refractive', R_fixed=None, rho_fixed=None, name='',
                 device=None):
        assert kind in KINDS, f'unknown kind {kind!r}, expected one of {KINDS}'
        self.surface = surface
        self.kind = kind
        self.R_fixed = R_fixed
        self.rho_fixed = rho_fixed
        self.name = name

        if device is not None:
            self.surface.to(device)
            self.surface.device = device

        if to_world is None:
            to_world = Transformation(torch.eye(3, device=device),
                                      torch.zeros(3, device=device))
        self.to_world = to_world
        self.to_local = to_world.inverse()

        self.n_in = n_in if isinstance(n_in, Material) else Material(n_in)
        self.n_out = n_out if isinstance(n_out, Material) else Material(n_out)

    # --- frame changes (each one undoes the squeeze in Transformation) -------
    def to_local_ray(self, o, d):
        """World ray (o, d) -> local frame, shapes preserved."""
        o_local = self.to_local.transform_point(o).view_as(o)
        d_local = self.to_local.transform_vector(d).view_as(d)
        return o_local, d_local

    def point_to_world(self, p_local):
        return self.to_world.transform_point(p_local).view_as(p_local)

    def vector_to_world(self, v_local):
        return self.to_world.transform_vector(v_local).view_as(v_local)

    # --- media --------------------------------------------------------------
    def ior(self, wavelength):
        """(n on the -z side, n on the +z side) at `wavelength` [nm]."""
        return self.n_in.ior(wavelength), self.n_out.ior(wavelength)

    def eta_at(self, d_world, n_world, wavelength):
        """Ratio n_incident / n_transmitted per ray, from the side it arrives on.

        `n_world` points toward local +z, so dot(d, n) < 0 means the ray travels
        from the +z side inwards: incident medium n_out, transmitted n_in.
        """
        n_in, n_out = self.ior(wavelength)
        cosi = torch.sum(d_world * n_world, dim=-1)
        from_plus_z = cosi < 0
        n_i = torch.where(from_plus_z,
                          torch.as_tensor(n_out, dtype=d_world.dtype),
                          torch.as_tensor(n_in, dtype=d_world.dtype))
        n_t = torch.where(from_plus_z,
                          torch.as_tensor(n_in, dtype=d_world.dtype),
                          torch.as_tensor(n_out, dtype=d_world.dtype))
        return n_i / n_t

    # --- intersection -------------------------------------------------------
    def intersect_t(self, o, d, maxt=T_MAX):
        """Distance along (o, d) to this element's surface.

        Rays nearly parallel to the local xy-plane are detected before Newton
        runs and given a dummy d_z, so the solver cannot produce inf/NaN.

        Returns:
            t: [N] float, +inf where there is no valid hit.
            valid: [N] bool.
        """
        ol, dl = self.to_local_ray(o, d)

        dz = dl[..., 2]
        parallel = torch.abs(dz) < EPS_DZ
        dl_safe = dl.clone()
        dl_safe[..., 2] = torch.where(parallel, torch.ones_like(dz), dz)

        solved, p_local = self.surface.newtons_method(maxt, ol, dl_safe)
        t = torch.sum((p_local - ol) * dl_safe, dim=-1)

        valid = (solved
                 & torch.isfinite(t)
                 & torch.isfinite(p_local).all(dim=-1)
                 & ~parallel
                 & (t > EPS_T)
                 & (t < maxt)
                 & self.surface.is_valid(p_local[..., 0:2]))

        t = torch.where(valid, t, torch.full_like(t, float('inf')))
        return t, valid


@torch.no_grad()
def closest_hit(o, d, elements, ignore_id=None, maxt=T_MAX):
    """Pass 1: nearest element along each ray.

    Args:
        o, d: [N,3] world-space origins and unit directions.
        elements: list of `Element`.
        ignore_id: [N] long, element index to skip per ray (the element the ray
            just left); -1 ignores nothing.

    Returns:
        t_min: [N] float, +inf where nothing was hit.
        id_min: [N] long, index into `elements`, -1 where nothing was hit.
    """
    N = o.shape[0]
    t_min = torch.full((N,), float('inf'), dtype=o.dtype, device=o.device)
    id_min = torch.full((N,), -1, dtype=torch.long, device=o.device)

    for i, el in enumerate(elements):
        t, valid = el.intersect_t(o, d, maxt)

        if ignore_id is not None:
            valid = valid & (ignore_id != i)

        better = valid & (t < t_min)
        t_min = torch.where(better, t, t_min)
        id_min = torch.where(better, torch.full_like(id_min, i), id_min)

    return t_min, id_min


def intersect_one(o, d, elements, elem_id, maxt=T_MAX):
    """Pass 2: recompute only the chosen intersection, with gradients.

    One `newtons_method` call per distinct element in `elem_id`. Gradients flow
    through the implicit-layer correction to the ray, the pose, and the surface
    coefficients.

    Returns:
        p: [N,3] world-space hit points, zero where `ok` is False.
        n: [N,3] world-space unit normals pointing toward the element's local
           +z (the `n_out` medium), not flipped against the incident ray.
        ok: [N] bool.
    """
    p = torch.zeros_like(o)
    n = torch.zeros_like(o)
    ok = elem_id >= 0

    for i, el in enumerate(elements):
        m = (elem_id == i)
        if not bool(m.any()):
            continue

        ol, dl = el.to_local_ray(o[m], d[m])
        solved, p_local = el.surface.newtons_method(maxt, ol, dl)

        nx, ny, nz = el.surface.surface_derivatives(p_local[..., 0], p_local[..., 1])
        n_local = -normalize(torch.stack((nx, ny, nz), dim=-1))

        p[m] = el.point_to_world(p_local)
        n[m] = normalize(el.vector_to_world(n_local))
        ok[m] = ok[m] & solved

    return p, n, ok


def sample_point_source(N, theta_max, seed=0, P=1.0, origin=(0.0, 0.0, 0.0),
                        device=None, dtype=None):
    """Isotropic point source, sampled uniformly in solid angle inside a cone.

    An isotropic source of total power `P` radiates dPhi = P/(4 pi) dOmega, so a
    cone of half-angle `theta_max` about +z carries

        Omega        = 2 pi (1 - cos theta_max)
        Phi_captured = P (1 - cos theta_max) / 2

    Sampling `cos(theta)` uniformly — not `theta` — is what makes the ray density
    uniform per steradian. Gridding a plane instead over-weights the centre by
    1/cos^3; that bias cancels in a hit-count image but not in W/mm^2.

    Args:
        N: number of rays.
        theta_max: cone half-angle [rad], measured from +z.
        seed: seed of a private `torch.Generator`, so the batch is reproducible
            without touching the global RNG.
        P: total emitted power over the full 4 pi [W].
        origin: world position of the source.
        device: where to build the batch. None = CPU, and then the stream is
            bit-identical to the pre-device version of this function (a bare
            `torch.Generator()` IS `torch.Generator(device='cpu')`).

            NOTE: CUDA's Philox and the CPU's MT19937 are DIFFERENT streams, so
            the same `seed` gives different rays on the two devices. That is
            fine for a statistical comparison and fatal for a ray-by-ray one —
            keep `c01_seq_vs_nonseq.py`'s identity gate on the CPU.

            Building on-device matters at scale: at N = 1e8 a CPU-born batch is
            ~4 GB of host randoms plus a 2.4 GB PCIe copy, which dominates the
            sampling cost and can exhaust host RAM before the GPU is touched.
        dtype: float dtype, defaults to the global default.

    Returns:
        o: [N,3] origins, all at `origin`.
        d: [N,3] unit directions inside the cone.
        w: [N] per-ray power Phi_captured / N [W].
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype
    device = torch.device('cpu') if device is None else device
    g = torch.Generator(device=device).manual_seed(int(seed))
    u1 = torch.rand(N, generator=g, dtype=dtype, device=device)
    u2 = torch.rand(N, generator=g, dtype=dtype, device=device)

    cos_max = math.cos(float(theta_max))
    cost = cos_max + (1.0 - cos_max) * u1
    sint = torch.sqrt(torch.clamp(1.0 - cost ** 2, min=0.0))
    phi = 2.0 * math.pi * u2

    d = torch.stack((sint * torch.cos(phi), sint * torch.sin(phi), cost), dim=-1)
    o = torch.as_tensor(origin, dtype=dtype, device=device).expand(N, 3).clone()

    phi_cap = P * (1.0 - cos_max) / 2.0
    w = torch.full((N,), phi_cap / N, dtype=dtype, device=device)
    return o, normalize(d), w


def refract(d, n_geom, eta):
    """Snell's law at a hit, vectorised, no `Lensgroup` involved.

    `intersect_one` returns the normal toward the element's local +z whichever
    side the ray arrives on, so the normal is flipped here from the sign of
    dot(d, n). `Element.eta_at` resolves n_i/n_t from that same sign — never
    hand-pick `eta` per surface.

    Args:
        d: [N,3] incident unit directions.
        n_geom: [N,3] geometric normals from `intersect_one`, not yet flipped.
        eta: [N] n_incident / n_transmitted.

    Returns:
        valid: [N] bool, False where the ray is totally internally reflected.
        d_t: [N,3] refracted unit directions, garbage where `valid` is False.
    """
    cosi = torch.sum(d * n_geom, dim=-1)
    n_f = torch.where((cosi < 0)[..., None], -n_geom, n_geom)
    cosi = torch.abs(cosi)

    cost2 = 1.0 - (1.0 - cosi ** 2) * eta ** 2
    valid = cost2 > 0.0
    cost = torch.sqrt(torch.clamp(cost2, min=1e-8))
    d_t = eta[..., None] * d + (cost - eta * cosi)[..., None] * n_f
    return valid, d_t


def trace_nonseq(o, d, w, elements, wavelength, max_bounces=8):
    """The bounce loop. Refraction only, so no ray ever splits.

    Everything stays at full [N] width — masked in place, never compacted — so
    the ray order survives the trace and can be compared ray by ray against a
    sequential run. The new origin is the hit point itself, so `ignore_id` (not
    an epsilon offset) is what stops the next `closest_hit` re-finding the
    surface just left.

    Args:
        o, d: [N,3] world origins and unit directions.
        w: [N] per-ray power, carried through untouched here.
        elements: list of `Element`.
        wavelength: [nm], passed to `Element.eta_at`.
        max_bounces: loop cap.

    Returns:
        o, d: [N,3] state after the last interaction — still ON the last
            surface, not on any receiver; `propagate_to_z` does that.
        w: [N] unchanged.
        alive: [N] bool, False once a ray is killed by TIR.
        nhit: [N] long, surfaces actually crossed. Through a two-surface lens
            that is 2; a ray that missed every aperture has 0.
    """
    N = o.shape[0]
    alive = torch.ones(N, dtype=torch.bool, device=o.device)
    nhit = torch.zeros(N, dtype=torch.long, device=o.device)
    ignore = torch.full((N,), -1, dtype=torch.long, device=o.device)

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

        eta = torch.ones(N, dtype=d.dtype, device=d.device)
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

    return o, d, w, alive, nhit


def propagate_to_z(o, d, z):
    """Free flight from (o, d) to the plane z = const. Returns [N,3] points."""
    t = (z - o[..., 2]) / d[..., 2]
    return o + t[..., None] * d


def splat(p, w, film_size, pixel_size, device=None, bilinear=True):
    """Weighted bilinear accumulation of hits `p` [M,3] with weights `w` [M].

    This is `Lensgroup.render`'s tail (optics.py:736-758) with the scalar `J`
    replaced by a per-ray weight — `render` has none, which makes it a hit-count
    histogram rather than an irradiance map. The pixel convention is kept
    identical to `render`'s, or the two maps shift by half a pixel.

    Pass `w = 1` for a hit-count map comparable with `render`, or
    `w = power / pixel_area` for irradiance in W/mm^2.

    Args:
        p: [M,3] hit points on the receiver plane, receiver-centred.
        w: [M] per-ray weight.
        film_size: [nx, ny] pixels.
        pixel_size: pixel pitch [mm].
        device: output device, defaults to `p`'s.
        bilinear: True spreads each hit over the 4 neighbouring bins (the
            `render`-compatible default). False drops it whole into the bin it
            landed in.

            Use `bilinear=False` for SECOND-MOMENT maps. A bilinear bin holds
            `V = sum_i b_i w_i` with `sum_i b_i = 1`, so its variance is
            `sum_i b_i^2 w_i^2` — but `splat(p, w**2)` computes
            `sum_i b_i w_i^2`, and `b_i <= 1` makes that an OVER-estimate by up
            to 2x (a hit dead-centre on a 2x2 corner splits b=1/4 four ways:
            sum b^2 = 1/4 against sum b = 1). Nearest-neighbour is exact, and
            with equal weights it reduces to a plain count, so
            `sqrt(splat(w**2, bilinear=False)) / splat(w, bilinear=False)`
            is exactly `1/sqrt(count)`.

            Keep the VALUE map bilinear, so it stays pixel-registered with
            `Lensgroup.render` and with a LightTools export.

    Returns:
        I: [nx, ny] accumulated map.
    """
    if device is None:
        device = p.device
    R_sensor = [film_size[i] * pixel_size / 2 for i in range(2)]
    u = (p[..., 0] + R_sensor[0]) / pixel_size
    v = (p[..., 1] + R_sensor[1]) / pixel_size

    index_l = torch.stack(
        (torch.clamp(torch.floor(u).long(), min=0, max=film_size[0] - 1),
         torch.clamp(torch.floor(v).long(), min=0, max=film_size[1] - 1)), dim=-1)
    if not bilinear:
        # `index_l` is already the CONTAINING bin, not a corner: a point at the
        # centre of bin i has u = i + 0.5, so floor(u) = i. No rounding needed.
        I = torch.zeros(*film_size, device=device, dtype=p.dtype)
        return torch.index_put(I, (index_l[..., 0], index_l[..., 1]), w,
                               accumulate=True)

    index_r = torch.stack(
        (torch.clamp(index_l[..., 0] + 1, min=0, max=film_size[0] - 1),
         torch.clamp(index_l[..., 1] + 1, min=0, max=film_size[1] - 1)), dim=-1)
    w_r = torch.clamp(torch.stack((u, v), dim=-1) - index_l, min=0, max=1)
    w_l = 1.0 - w_r

    I = torch.zeros(*film_size, device=device, dtype=p.dtype)
    I = torch.index_put(I, (index_l[..., 0], index_l[..., 1]),
                        w_l[..., 0] * w_l[..., 1] * w, accumulate=True)
    I = torch.index_put(I, (index_r[..., 0], index_l[..., 1]),
                        w_r[..., 0] * w_l[..., 1] * w, accumulate=True)
    I = torch.index_put(I, (index_l[..., 0], index_r[..., 1]),
                        w_l[..., 0] * w_r[..., 1] * w, accumulate=True)
    I = torch.index_put(I, (index_r[..., 0], index_r[..., 1]),
                        w_r[..., 0] * w_r[..., 1] * w, accumulate=True)
    return I


# -------------------------------------------------------------------- physics
def reflect(d, n_geom):
    """Mirror `d` about the plane of `n_geom`.

    The normal need not be flipped against the ray: the formula is even in
    `n_geom`, so `intersect_one`'s unflipped normal is used as it comes.

    Args:
        d: [N,3] incident unit directions.
        n_geom: [N,3] normals from `intersect_one`, toward the element's local
            +z and NOT flipped against the ray.

    Returns:
        [N,3] reflected unit directions.
    """
    return d - 2.0 * torch.sum(d * n_geom, dim=-1, keepdim=True) * n_geom


def fresnel_R(d, n_geom, eta):
    """Unpolarized Fresnel reflectance, used only when `R_fixed is None`.

    Args:
        d, n_geom: as in `reflect`.
        eta: [N] n_incident / n_transmitted, from `Element.eta_at`.

    Returns:
        [N] reflectance in [0,1]; exactly 1 where the ray is TIR.
    """
    cosi = torch.abs(torch.sum(d * n_geom, dim=-1))
    sint2 = eta ** 2 * (1.0 - cosi ** 2)
    cost = torch.sqrt(torch.clamp(1.0 - sint2, min=0.0))
    rs = ((eta * cosi - cost) / (eta * cosi + cost)) ** 2
    rp = ((eta * cost - cosi) / (eta * cost + cosi)) ** 2
    return torch.where(sint2 >= 1.0, torch.ones_like(cosi), 0.5 * (rs + rp))


def reflectance(el, d, n_geom, eta):
    """R at this element: the fixed override if it has one, else Fresnel.

    `R_fixed` is a "simple coating" — a constant, and the thing a commercial
    tracer is usually set up with. `R_fixed = None` switches the same code to
    real uncoated Fresnel.
    """
    if el.R_fixed is None:
        return fresnel_R(d, n_geom, eta)
    return torch.ones_like(eta) * el.R_fixed


def interaction(o, d, elements, eid, wavelength):
    """One hit for a batch: geometry, media, R, and both outgoing directions.

    TIR is not a dead end here. `Lensgroup._refract` marks it invalid and drops
    the ray; in a non-sequential tracer TIR is simply R = 1, the one case where
    a partial surface behaves as a perfect mirror. `ok_t` reports it, `R` is
    forced to 1 from it — never use it to kill a ray.

    Args:
        o, d: [N,3] world rays.
        elements: list of `Element`.
        eid: [N] long, the element each ray hit. Must be a valid index for
            every entry — mask non-hitting rays to 0 and discard their results,
            as both tracers below do.
        wavelength: [nm].

    Returns:
        p: [N,3] hit points.
        ok: [N] bool, the intersection solved.
        R: [N] reflectance, forced to 1 where the transmission is TIR.
        d_refr, d_refl: [N,3] the two children's directions.
        ok_t: [N] bool, False where transmission is TIR.
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
def _empty_terminal(o, w):
    """The `terminal` dict of a trace in which no ray ever left the scene."""
    return {'o': o.new_zeros((0, 3)), 'd': o.new_zeros((0, 3)),
            'w': w.new_zeros((0,)),
            'nrefl': torch.zeros(0, dtype=torch.long, device=o.device),
            'nhit': torch.zeros(0, dtype=torch.long, device=o.device)}


def trace_split(o, d, w, elements, wavelength, max_depth=MAX_DEPTH, w_min=0.0):
    """Deterministic ray splitting: every hit spawns BOTH children.

    Work proceeds generation by generation — the whole depth-k front is ONE
    batch, so the cost is `max_depth * len(elements)` Newton solves, not one
    per path. The front is compacted every generation, so retired rays cost
    nothing; in a scene where most reflected children leave immediately it
    grows far slower than the 2^depth worst case.

    This is the reference tracer. Energy is conserved by construction and the
    result carries no noise at all, which is what makes it the thing to check
    `trace_mc` against.

    Args:
        o, d, w: [N,3], [N,3], [N] the source rays and their powers [W].
        elements: list of `Element`.
        wavelength: [nm].
        max_depth: generations before giving up.
        w_min: absolute weight below which a child is dropped. Pass something
            relative to the per-ray weight (`1e-5 * w[0]`, say), never a
            constant — an absolute floor means a different tracer for every
            source power.

    Returns (terminal, tally):
        terminal: dict o, d, w, nrefl, nhit [M] for every ray that LEFT the
            scene. They are free-flying, so a receiver is a `propagate_to_z`.
        nrefl alone does NOT identify a path: one hit with one reflection and
        three hits with one reflection are different paths. Key on the pair.
        tally: dict 'culled' (dropped below `w_min`, or the intersection
            failed) and 'truncated' (still bouncing when `max_depth` ran out).
    """
    o_out, d_out, w_out, nr_out, nh_out = [], [], [], [], []
    culled = torch.zeros((), dtype=w.dtype, device=w.device)

    ignore = torch.full((o.shape[0],), -1, dtype=torch.long, device=o.device)
    nrefl = torch.zeros(o.shape[0], dtype=torch.long, device=o.device)
    nhit = torch.zeros(o.shape[0], dtype=torch.long, device=o.device)

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

        p, ok, R, d_refr, d_refl, ok_t = interaction(o, d, elements, eid,
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

    truncated = (w.sum() if o.shape[0] else
                 torch.zeros((), dtype=culled.dtype, device=culled.device))

    if not o_out:
        return _empty_terminal(o, w), {'culled': culled,
                                       'truncated': truncated}
    terminal = {'o': torch.cat(o_out), 'd': torch.cat(d_out),
                'w': torch.cat(w_out), 'nrefl': torch.cat(nr_out),
                'nhit': torch.cat(nh_out)}
    return terminal, {'culled': culled, 'truncated': truncated}


def trace_mc(o, d, w, elements, wavelength, seed=0, max_depth=MAX_DEPTH,
             w_min=0.0):
    """Monte Carlo: ONE child per hit, chosen with probability rho.

    Reflect with probability rho and divide the weight by rho, transmit with
    1 - rho and divide by that — so the expected weight equals the split
    tracer's total and the estimator is unbiased for ANY rho, with an error
    falling as N^-0.5. rho defaults to `R.detach()`, which is the zero-variance
    choice when R is fixed: both factors are then exactly 1, every surviving
    path carries the weight it was launched with, and only WHICH paths exist is
    random. `Element.rho_fixed` overrides it per element, which is how you
    importance-sample a faint branch — more rays into it, weights correcting
    the bias.

    Detaching matters: rho is a sampling decision, not physics. Leaving it on
    the graph would put the discrete branch choice into the gradient of the
    received power with respect to R.

    Unlike `trace_split` this stays at full [N] width throughout — rays are
    masked in place with a `done` flag, never compacted — so the ray count is
    flat in depth and the ray order survives the trace.

    Same arguments and same return shape as `trace_split`, so a scene can feed
    either to the same receivers.
    """
    N = o.shape[0]
    g = torch.Generator(device=o.device).manual_seed(int(seed))
    eps = torch.finfo(w.dtype).eps

    done = torch.zeros(N, dtype=torch.bool, device=o.device)
    nrefl = torch.zeros(N, dtype=torch.long, device=o.device)
    nhit = torch.zeros(N, dtype=torch.long, device=o.device)
    ignore = torch.full((N,), -1, dtype=torch.long, device=o.device)
    culled = torch.zeros((), dtype=w.dtype, device=w.device)

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

        # non-hitting rays are given element 0 so the index is valid; their
        # results are masked out by `hit` below and never used.
        p, ok, R, d_refr, d_refl, ok_t = interaction(
            o, d, elements, torch.where(hit, eid, torch.zeros_like(eid)),
            wavelength)

        # zero a culled ray's weight, or it lands in `culled` AND in `terminal`
        bad = hit & ~ok
        culled = culled + w[bad].sum()
        w = torch.where(bad, torch.zeros_like(w), w)
        done = done | bad
        hit = hit & ok

        rho = R.detach().clone()
        for i, el in enumerate(elements):
            if el.rho_fixed is not None:
                rho = torch.where(eid == i,
                                  torch.ones_like(rho) * el.rho_fixed, rho)
        rho = torch.where(ok_t, rho, torch.ones_like(rho)).clamp(eps, 1.0)

        u = torch.rand(N, generator=g, dtype=w.dtype, device=o.device)
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
        w = torch.where(low, torch.zeros_like(w), w)
        done = done | low

    truncated = w[~done].sum()
    keep = done & (w > 0)
    terminal = {'o': o[keep], 'd': d[keep], 'w': w[keep], 'nrefl': nrefl[keep],
                'nhit': nhit[keep]}
    return terminal, {'culled': culled, 'truncated': truncated}


