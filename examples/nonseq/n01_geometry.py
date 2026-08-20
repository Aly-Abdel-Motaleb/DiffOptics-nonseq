"""
PHASE 1 - GEOMETRY LAYER: per-element poses + closest-hit search.

This is PLAN.md Phase 1 / PLAN_A_collimator.md P1. It is scene-independent: no
Monte Carlo, no Fresnel, no ray weights. Only three things live here:

    Element       a `Surface` + a world pose + an interface description
    closest_hit   pass 1, under no_grad: which element does each ray hit first?
    intersect_one pass 2, WITH grad: recompute only that one intersection

Once the gates at the bottom pass, this module moves verbatim into
`diffoptics/nonseq.py` and Phase 2 (`trace_split`) builds on top of it.

Fill in every `TODO`. The tests at the bottom are complete and are the gate -
do not edit them. Reference solution: `n01_geometry_sol.py`.

--------------------------------------------------------------------------------
WHY A NEW LAYER IS NEEDED
--------------------------------------------------------------------------------

`Lensgroup` holds ONE pose (`origin`, `theta_x/y/z`, optics.py:44) shared by all
its surfaces; a `Surface` only carries a scalar `d` offset along the group's
local z. A 45 deg extraction mirror is not expressible as a `d` offset, so each
element needs its own `Transformation`.

!so in other words the problem is that each surface should have its own local frame 
! that ensure that the local z axis is normal to the surface, and the local x/y axes are tangent to the surface.


--------------------------------------------------------------------------------
CONVENTIONS (fix these once, everything downstream depends on them)
--------------------------------------------------------------------------------

Local frame of an element. `Surface` is the implicit function

    f(x, y, z) = g(x, y) + h(z) + d = 0,     h(z) = -z  for Aspheric

so the surface sits at local `z = sag(x, y) + d`, the sag opens along local +z,
and the finite aperture is the disc/square of radius `surface.r` in local xy.

Pose. `to_world` maps local -> world; `to_local = to_world.inverse()` maps
world -> local. Rays are transformed into the local frame, intersected there
(that is the only frame `newtons_method` understands), and the hit is pushed
back out to world.

Media. `n_out` is the medium on the local **+z** side, `n_in` the medium on the
local **-z** side. Nothing here consumes them - they are carried for Phase 2 -
but `eta_at()` already resolves the incident/transmitted pair from the sign of
`dot(d, n)` so Phase 2 cannot get the sides backwards.

Normal. `intersect_one` returns the geometric normal pointing toward local +z,
i.e. into the `n_out` medium, ALWAYS - it is never flipped to face the incident
ray. `dO`'s `Surface.normal` (optics.py:1193) returns (gx, gy, -1)/|.| which
points toward -z, so it must be negated here. Phase 2 flips against the ray.

--------------------------------------------------------------------------------
THE TWO SHARP EDGES THIS FILE EXISTS TO BLUNT
--------------------------------------------------------------------------------

1. `newtons_method` seeds itself with `t0 = (self.d - oz) / dz` (optics.py:1298).
   A ray running nearly parallel to a surface's local xy-plane has `dz -> 0` and
   `t0` explodes to inf, then NaN. Sequential tracing never meets this; a light
   guide meets it constantly. Guard is: detect `|dz| < EPS_DZ` BEFORE calling
   Newton, substitute a dummy dz so the solver cannot produce inf/NaN at all,
   and invalidate those rays afterwards. Screening after the fact is not enough
   - NaNs poison the comparison in `closest_hit`.

2. `Transformation.transform_point` ends in `torch.squeeze` (basics.py:75). For
   a single ray, [1,3,1] -> [3], and the batch dimension silently vanishes.
   Every transform in this file must reshape back to the input shape.

Gotcha #7 of PLAN_A also applies: `transform_ray` builds a fresh `Ray` and drops
`mint`/`maxt`, so this file never uses it - transform `o` and `d` directly.

Reuse, do not reimplement:
    Surface.newtons_method            optics.py:1267   (implicit-gradient trick)
    Surface.is_valid / sdf_approx     optics.py:1240   (finite apertures)
    Surface.surface_derivatives       optics.py:1422   (local normal)
    Transformation / .inverse()       basics.py:57
    Material.ior                      basics.py:236
    normalize                         basics.py:293

Run:    python n01_geometry.py
Gate:   closest hit correct from BOTH sides of a two-plane plate; a ray parallel
        to a surface returns "no hit" rather than a NaN; the aperture rejects
        rays outside `surface.r`; autograd through `intersect_one` matches a
        central finite difference on the surface curvature.
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do  


# ------------------------------------------------------------------ constants
EPS_DZ = 1e-9      # |d_z| in the local frame below which a ray counts as parallel
EPS_T  = 1e-6      # [mm] minimum travel distance; rejects the self-hit at t ~ 0
T_MAX  = 1e5       # [mm] maximum travel distance, matches Ray.maxt (basics.py:50)

KINDS = ('refractive', 'mirror', 'partial', 'absorber')


# -------------------------------------------------------------------- element
class Element(do.PrettyPrinter):
    """A `Surface` plus a world pose plus an interface description.

    Args:
        surface:  any `dO` Surface (Aspheric / BSpline / XYPolynomial / Mesh),
                  used unchanged.
        to_world: `Transformation` mapping local -> world. None = identity.
        n_in:     medium on the local -z side. str or `Material`.
        n_out:    medium on the local +z side. str or `Material`.
        kind:     one of KINDS.
        R_fixed:  constant reflectance overriding Fresnel, or None for Fresnel.
                  0.0 turns reflection off entirely (PLAN_A stage 1);
                  0.2 is PLAN_A stage 2. Consumed in Phase 2, stored here.
        rho_fixed: Monte Carlo sampling-probability override, or None to use
                  `R.detach()`. Consumed in Phase 3, stored here.
        name:     label for printouts.
    """
    def __init__(self, surface, to_world=None, n_in='air', n_out='air',
                 kind='refractive', R_fixed=None, rho_fixed=None, name=''):
        assert kind in KINDS, f'unknown kind {kind!r}, expected one of {KINDS}'
        self.surface = surface
        self.kind = kind
        self.R_fixed = R_fixed
        self.rho_fixed = rho_fixed
        self.name = name

        if to_world is None:
            to_world = do.Transformation(torch.eye(3), torch.zeros(3))
        self.to_world = to_world
        self.to_local = to_world.inverse()
        # this is just to either pass a string like 'air'  or a Material object.
        self.n_in = n_in if isinstance(n_in, do.Material) else do.Material(n_in)
        self.n_out = n_out if isinstance(n_out, do.Material) else do.Material(n_out)

    # --- frame changes (each one must undo the squeeze in basics.py:75) ------
    def to_local_ray(self, o, d):
        """World ray (o, d) -> local frame. Shapes must be preserved.
        """
        
        o_local = self.to_local.transform_point(o)
        d_local = self.to_local.transform_vector(d)
        
        # Reshape back to the input shape
        o_local = o_local.view_as(o)
        d_local = d_local.view_as(d)
        
        return o_local, d_local

    def point_to_world(self, p_local):
        return self.to_world.transform_point(p_local).view_as(p_local)

    def vector_to_world(self, v_local):
        return self.to_world.transform_vector(v_local).view_as(v_local)

    # --- media --------------------------------------------------------------
    def ior(self, wavelength):
        """(n on the -z side, n on the +z side) at `wavelength` [nm].

        """
        return self.n_in.ior(wavelength), self.n_out.ior(wavelength)

    def eta_at(self, d_world, n_world, wavelength):
        """
        Ratio n_incident / n_transmitted for each ray, from the side it
        arrives on.

        `n_world` points toward local +z (the `n_out` medium), so
        `dot(d, n) < 0` means the ray travels from the +z side inwards, i.e.
        incident medium = n_out and transmitted medium = n_in. The other sign
        is the other way round.
        """
        n_in , n_out = self.ior(wavelength)
        cosi = torch.sum(d_world * n_world, dim=-1)
        from_plus_z = cosi < 0
        n_i = torch.where(from_plus_z, torch.as_tensor(n_out, dtype=d_world.dtype),
                                torch.as_tensor(n_in, dtype=d_world.dtype))
        n_t = torch.where(from_plus_z, torch.as_tensor(n_in, dtype=d_world.dtype),
                                torch.as_tensor(n_out, dtype=d_world.dtype))
        return n_i / n_t
        
        
    # --- intersection -------------------------------------------------------
    def intersect_t(self, o, d, maxt=T_MAX):
        """
        Distance along (o, d) to this element's surface.
        Returns
        -------
        t : [N] float, +inf where there is no valid hit
        valid : [N] bool
        """
        
        ol, dl = self.to_local_ray(o, d)
        
        dz = dl[..., 2]
        parallel = torch.abs(dz) < EPS_DZ
        dl_safe = dl.clone()
        dl_safe[...,2] = torch.where(parallel, torch.ones_like(dz), dz)
        
        solved , p_local = self.surface.newtons_method(maxt, ol, dl_safe)
        
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


# ------------------------------------------------------------------- pass 1
@torch.no_grad()
def closest_hit(o, d, elements, ignore_id=None, maxt=T_MAX):
    """P
    Args:
        o, d: [N,3] world-space origins and unit directions.
        elements: list of `Element`.
        ignore_id: [N] long, an element index to skip per ray (the element the
            ray just left). Together with the epsilon offset on the new origin
            this is what stops every ray dying at t ~ 0 on bounce 2. Use -1 for
            "ignore nothing".

    Returns:
        t_min: [N] float, +inf where nothing was hit.
        id_min: [N] long, index into `elements`, -1 where nothing was hit.

  
    """

    N = o.shape[0]
    t_min = torch.full((N,), float('inf'), dtype=o.dtype, device=o.device)    
    id_min = torch.full((N,), -1, dtype=torch.long, device=o.device)
    
    for i , el in enumerate(elements):
        t, valid = el.intersect_t(o, d, maxt)
        
        if ignore_id is not None:
            valid = valid & (ignore_id != i)
        
        better = valid & (t < t_min)
        
        t_min = torch.where(better, t, t_min)
        id_min = torch.where(better, torch.full_like(id_min, i), id_min)


    return t_min, id_min


# ------------------------------------------------------------------- pass 2
def intersect_one(o, d, elements, elem_id, maxt=T_MAX):
    """Pass 2: recompute ONLY the chosen intersection, with gradients.

    One `newtons_method` call per distinct element present in `elem_id`, results
    scattered back into [N,...] tensors. Gradients flow through
    `newtons_method`'s implicit-layer correction (optics.py:1312-1313) to the
    ray, the pose, and the surface coefficients.

    Returns:
        p: [N,3] world-space hit points. Zero where `ok` is False.
        n: [N,3] world-space unit normals, pointing toward the element's local
           +z (the `n_out` medium). NOT flipped against the incident ray.
        ok: [N] bool.

    TODO
      - p, n = zeros_like(o); ok = elem_id >= 0;
      - for each element i: m = (elem_id == i); skip if empty;
      - local ray for o[m], d[m], then `newtons_method` again - NO no_grad here,
        this call is the one that carries the gradient;
      - local normal: `surface_derivatives(p_local[...,0], p_local[...,1])`
        gives (gx, gy, -1); stack, `do.normalize`, then NEGATE so it points to
        local +z (see the Normal convention above);
      - write back with `p[m] = ...` / `n[m] = ...` (index_put is
        differentiable, so this keeps the graph intact) and normalize the
        world-space normal after rotating it;
      - ok[m] = ok[m] & solved.
    """
    
    p = torch.zeros_like(o)
    n = torch.zeros_like(o)
    ok = elem_id >= 0
    
    for i, el in enumerate(elements):
        m = (elem_id == i)
        if not bool(m.any()):
            continue
        
        o_m = o[m]
        d_m = d[m]
        
        ol, dl = el.to_local_ray(o_m, d_m)
        solved, p_local = el.surface.newtons_method(maxt, ol, dl)
        
        nx , ny ,nz = el.surface.surface_derivatives(p_local[..., 0], p_local[..., 1])
        n_local = -do.normalize(torch.stack((nx, ny, nz), dim=-1))

                
        p[m] = el.point_to_world(p_local)
        n[m] = do.normalize(el.vector_to_world(n_local))
        
        ok[m] = ok[m] & solved
        
    return p, n, ok
    
    


# ============================================================================
#                       TESTS - complete, do not edit
# ============================================================================
# Scene: the PLAN.md tilted plate, as two independently posed flat elements.
TILT_DEG  = 30.0
THICKNESS = 10.0
Z_FRONT   = 50.0
SEMI_DIA  = 20.0
WAVELENGTH = 532.8


def _rot_y(deg):
    k = torch.Tensor([0.0, 1.0, 0.0])
    return do.rodrigues_rotation_matrix(k, float(np.deg2rad(deg)))


def build_plate(tilt_deg=TILT_DEG, thickness=THICKNESS, z_front=Z_FRONT,
                r=SEMI_DIA):
    """

    Both are flat (`Aspheric` with c = 0, so sag = 0 and the surface is the
    local z = 0 plane), share the same rotation, and are separated along the
    common local +z by `thickness`. Local +z therefore points INTO the glass at
    the front face and OUT of it at the back face, which is exactly why the
    media are swapped between the two.
    """
    R = _rot_y(tilt_deg)
    axis = R @ torch.Tensor([0.0, 0.0, 1.0])
    t_front = torch.Tensor([0.0, 0.0, z_front])
    t_back = t_front + thickness * axis 
    front = Element(do.Aspheric(r, 0.0, c=0.0), do.Transformation(R, t_front),
                    n_in='air', n_out='n-bk7' , kind='refractive', name='front')
    back = Element(do.Aspheric(r, 0.0, c=0.0), do.Transformation(R, t_back),
                   n_in='n-bk7', n_out='air', kind='refractive', name='back')
    return [front, back]


def _plane_ref(o, d, p0, nrm):
    """numpy reference: t of the ray/plane hit, or inf. Independent of dO."""
    denom = float(np.dot(d, nrm))
    if abs(denom) < 1e-12:
        return np.inf
    t = float(np.dot(p0 - o, nrm) / denom)
    return t if t > 1e-9 else np.inf


def _rays(o_list, d_list):
    o = torch.Tensor(np.asarray(o_list, dtype=np.float64))
    d = do.normalize(torch.Tensor(np.asarray(d_list, dtype=np.float64)))
    return o, d


def _check(name, cond, detail=''):
    print(f'  [{"ok " if cond else "FAIL"}] {name}{(" - " + detail) if detail else ""}')
    assert cond, name


# ---------------------------------------------------------------- test cases
def test_single_plane():
    """Chief ray onto the tilted front face: hit point and normal by hand."""
    print('T1  single tilted plane')
    els = build_plate()[:1]
    o, d = _rays([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])

    t, eid = closest_hit(o, d, els)
    p, n, ok = intersect_one(o, d, els, eid)

    th = np.deg2rad(TILT_DEG)
    n_expect = np.array([np.sin(th), 0.0, np.cos(th)])   # local +z, in world

    _check('hit found', bool(ok[0]) and int(eid[0]) == 0)
    _check('t = 50', abs(float(t[0]) - 50.0) < 1e-4, f't = {float(t[0]):.6f}')
    _check('p = (0,0,50)',
           np.allclose(p[0].detach().numpy(), [0.0, 0.0, 50.0], atol=1e-4),
           str(p[0].detach().numpy()))
    _check('normal points to local +z',
           np.allclose(n[0].detach().numpy(), n_expect, atol=1e-6),
           str(n[0].detach().numpy()))


def test_closest_from_both_sides():
    """The Phase 1 gate: the nearer face wins, whichever side the ray is on."""
    print('T2  closest hit from both sides')
    els = build_plate()
    th = np.deg2rad(TILT_DEG)
    axis = np.array([np.sin(th), 0.0, np.cos(th)])
    p_f = np.array([0.0, 0.0, Z_FRONT])
    p_b = p_f + THICKNESS * axis

    cases = [
        ('from -z (source side)',  [0.0, 0.0, 0.0],   [0.0, 0.0,  1.0], 0),
        ('from +z (receiver side)', [0.0, 0.0, 300.0], [0.0, 0.0, -1.0], 1),
    ]
    for label, o_i, d_i, want_id in cases:
        o, d = _rays([o_i], [d_i])
        t, eid = closest_hit(o, d, els)

        on = np.asarray(o_i, dtype=np.float64)
        dn = np.asarray(d_i, dtype=np.float64)
        dn = dn / np.linalg.norm(dn)
        t_ref = min(_plane_ref(on, dn, p_f, axis), _plane_ref(on, dn, p_b, axis))

        _check(f'{label}: element {want_id} wins', int(eid[0]) == want_id,
               f'got {int(eid[0])}')
        _check(f'{label}: t matches numpy', abs(float(t[0]) - t_ref) < 1e-4,
               f'{float(t[0]):.6f} vs {t_ref:.6f}')


def test_ignore_id():
    """`ignore_id` masks the element a ray just left - the self-hit fix."""
    print('T3  ignore_id skips the element just left')
    els = build_plate()
    o, d = _rays([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])

    _, eid_all = closest_hit(o, d, els)
    _, eid_skip = closest_hit(o, d, els, ignore_id=torch.zeros(1, dtype=torch.long))

    _check('without ignore: front', int(eid_all[0]) == 0)
    _check('with ignore=0: back', int(eid_skip[0]) == 1)


def test_parallel_ray():
    """Edge 1. Must return "no hit", must not return NaN."""
    print('T4  ray parallel to a surface')
    els = build_plate()[:1]
    th = np.deg2rad(TILT_DEG)
    in_plane = np.array([np.cos(th), 0.0, -np.sin(th)])   # local +x, in world
    o_i = [0.0, 0.0, 0.0]

    o, d = _rays([o_i], [in_plane.tolist()])
    t, eid = closest_hit(o, d, els)

    _check('no hit reported', int(eid[0]) == -1)
    _check('t is +inf, not NaN', bool(torch.isinf(t[0])) and not bool(torch.isnan(t[0])),
           f't = {float(t[0])}')


def test_aperture():
    """`Surface.is_valid` must cull rays landing outside the semi-diameter."""
    print('T5  finite aperture')
    els = build_plate()[:1]
    o, d = _rays([[0.0,  5.0, 0.0],       # inside  r = 20
                  [0.0, 25.0, 0.0]],      # outside r = 20
                 [[0.0, 0.0, 1.0],
                  [0.0, 0.0, 1.0]])
    _, eid = closest_hit(o, d, els)
    _check('inside kept', int(eid[0]) == 0)
    _check('outside culled', int(eid[1]) == -1)


def test_gradients():
    """Autograd must survive the pose layer: d p_z / d c vs central difference.

    Uses the PLAN_A collimator asphere so the number being differentiated is
    the one Phase 2 actually optimizes.

    On the step size: `dO` runs in float32 and `newtons_method` only converges
    to NEWTONS_TOLERANCE_LOOSE = 300 nm (optics.py:1183). A textbook h = 1e-5
    on c = 0.048 moves the hit by less than that, so the difference is pure
    solver noise and "fails" by a few percent - an autograd bug that is not one.
    h = 1e-3 sits above the noise and below the curvature error. In float64 the
    agreement is 1e-8 at any h.
    """
    print('T6  gradient through intersect_one')
    C, K = 0.04812, -2.3089
    o, d = _rays([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, -1.0, 0.0]],
                 [[0.0, 0.0, 1.0], [0.0, 0.05, 1.0], [-0.02, 0.01, 1.0]])
    t_move = torch.Tensor([0.0, 0.0, 40.0])

    def hit_sum(c_val, grad=False):
        surf = do.Aspheric(12.7, 0.0, c=c_val, k=K)
        if grad:
            surf.c = surf.c.clone().requires_grad_(True)
        el = Element(surf, do.Transformation(torch.eye(3), t_move),
                     n_in='air', n_out='n-bk7')
        _, eid = closest_hit(o, d, [el])
        p, _, ok = intersect_one(o, d, [el], eid)
        assert bool(ok.all()), 'all three rays must hit the asphere'
        return surf, p[..., 2].sum()

    surf, z_sum = hit_sum(C, grad=True)
    z_sum.backward()
    g_auto = float(surf.c.grad)

    h = 1e-3
    g_fd = float((hit_sum(C + h)[1] - hit_sum(C - h)[1]) / (2 * h))

    rel = abs(g_auto - g_fd) / max(abs(g_fd), 1e-12)
    _check('autograd matches finite difference', rel < 1e-3,
           f'auto {g_auto:.6f} vs fd {g_fd:.6f} (rel {rel:.2e})')


def test_eta_sides():
    """Sanity on the media bookkeeping Phase 2 will lean on."""
    print('T7  eta resolved from the side of arrival')
    front = build_plate()[0]
    n_air, n_glass = front.ior(WAVELENGTH)
    th = np.deg2rad(TILT_DEG)
    nrm = torch.Tensor([[np.sin(th), 0.0, np.cos(th)]])

    d_fwd = do.normalize(torch.Tensor([[0.0, 0.0, 1.0]]))    # air -> glass
    d_bwd = do.normalize(torch.Tensor([[0.0, 0.0, -1.0]]))   # glass -> air

    eta_fwd = float(front.eta_at(d_fwd, nrm, WAVELENGTH)[0])
    eta_bwd = float(front.eta_at(d_bwd, nrm, WAVELENGTH)[0])

    _check('n-bk7 index sane', 1.51 < float(n_glass) < 1.53, f'n = {float(n_glass):.4f}')
    _check('air -> glass', abs(eta_fwd - float(n_air / n_glass)) < 1e-6, f'{eta_fwd:.6f}')
    _check('glass -> air', abs(eta_bwd - float(n_glass / n_air)) < 1e-6, f'{eta_bwd:.6f}')


if __name__ == '__main__':
    torch.manual_seed(0)
    test_single_plane()
    test_closest_from_both_sides()
    test_ignore_id()
    test_parallel_ray()
    test_aperture()
    test_gradients()
    test_eta_sides()
    print('\nPhase 1 gates passed.')
