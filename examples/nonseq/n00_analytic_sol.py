"""
PHASE 0 - ANALYTIC GROUND TRUTH for the tilted-plate ghost test.  [SOLUTION]

See n00_analytic.py for the problem statement, scene definition and gate.

Run:  python n00_analytic_sol.py
"""
import os
import sys
import json

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do  # noqa: E402


# ------------------------------------------------------------------ constants
WAVELENGTH   = 532.8          # nm
MATERIAL     = 'N-BK7'
N_AIR        = 1.0

TILT_DEG     = 30.0           # rotation of the plate about the y axis
THICKNESS    = 10.0           # mm, along the plate normal
Z_FRONT      = 50.0           # mm, front face passes through (0, 0, Z_FRONT)

Z_RECV_FWD   = 200.0          # mm, forward receiver plane
Z_RECV_BACK  = 0.0            # mm, backward receiver plane
RECV_SIZE    = 80.0           # mm, square side (aperture is +/- RECV_SIZE/2)
RECV_RES     = 512            # pixels per side

BEAM_RADIUS  = 1.0            # mm
BEAM_POWER   = 1.0            # W

OUT_DIR      = 'n00_out'

PATHS = ('TT', 'R', 'TRRT')

EPS = 1e-12


# ------------------------------------------------------------------- geometry
def plate_planes():
    th = np.deg2rad(TILT_DEG)
    n_front = np.array([-np.sin(th), 0.0, -np.cos(th)])   # points back at source
    p_front = np.array([0.0, 0.0, Z_FRONT])
    p_back = p_front + THICKNESS * (-n_front)             # deeper into +z
    return (p_front, n_front), (p_back, n_front.copy())


def plane_intersect(o, d, p0, n):
    denom = float(np.dot(d, n))
    if abs(denom) < 1e-9:                                 # parallel: no hit, no NaN
        return np.inf
    t = float(np.dot(p0 - o, n) / denom)
    return t if t > 1e-9 else np.inf


def reflect(d, n):
    return d - 2.0 * float(np.dot(d, n)) * n


def refract(d, n, eta):
    # orient the normal against the incident direction
    cosi = -float(np.dot(d, n))
    if cosi < 0.0:
        n, cosi = -n, -cosi
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0.0:
        return False, d                                   # total internal reflection
    dt = eta * d + (eta * cosi - np.sqrt(k)) * n
    return True, dt / np.linalg.norm(dt)


# -------------------------------------------------------------------- physics
def index_of_refraction():
    return float(do.Material(MATERIAL).ior(WAVELENGTH))


def snell_angle(theta_i, n1, n2):
    return float(np.arcsin(np.clip(n1 * np.sin(theta_i) / n2, -1.0, 1.0)))


def fresnel(theta_i, n1, n2):
    theta_t = snell_angle(theta_i, n1, n2)
    ci, ct = np.cos(theta_i), np.cos(theta_t)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)
    R_s, R_p = float(rs ** 2), float(rp ** 2)
    return R_s, R_p, 0.5 * (R_s + R_p)


def path_fractions(R):
    # R is identical at every one of these interfaces: external incidence
    # theta_i and internal incidence theta_t are Snell conjugates, and Fresnel
    # reflectance is reciprocal, so R(air->glass, theta_i) == R(glass->air, theta_t).
    T = 1.0 - R
    return {
        'TT':   T * T,
        'R':    R,
        'TRRT': T * T * R * R,
    }


def ghost_offset(thickness, theta_i, theta_t):
    return float(2.0 * thickness * np.tan(theta_t) * np.cos(theta_i))


# ------------------------------------------------------------- chief-ray trace
def _events(label):
    """Interface sequence for a label: (which face, what happens)."""
    faces = {'TT':   [('front', 'T'), ('back', 'T')],
             'R':    [('front', 'R')],
             'TRRT': [('front', 'T'), ('back', 'R'), ('front', 'R'), ('back', 'T')]}
    return faces[label]


def trace_path(label, n_glass=None):
    if n_glass is None:
        n_glass = index_of_refraction()
    (p_f, n_f), (p_b, n_b) = plate_planes()
    planes = {'front': (p_f, n_f), 'back': (p_b, n_b)}

    o = np.array([0.0, 0.0, 0.0])
    d = np.array([0.0, 0.0, 1.0])
    inside = False                                        # currently in glass?

    for face, event in _events(label):
        p0, nrm = planes[face]
        t = plane_intersect(o, d, p0, nrm)
        assert np.isfinite(t), f'path {label}: missed the {face} face'
        o = o + t * d
        if event == 'R':
            d = reflect(d, nrm)
        else:
            eta = (n_glass / N_AIR) if inside else (N_AIR / n_glass)
            ok, d = refract(d, nrm, eta)
            assert ok, f'path {label}: unexpected TIR at the {face} face'
            inside = not inside

    assert not inside, f'path {label}: ray never left the glass'

    z_recv = Z_RECV_FWD if d[2] > 0 else Z_RECV_BACK
    t = (z_recv - o[2]) / d[2]
    hit = o + t * d
    half = 0.5 * RECV_SIZE
    return {
        'z_recv':   float(z_recv),
        'centroid': [float(hit[0]), float(hit[1])],
        'exit_dir': [float(v) for v in d],
        'on_recv':  bool(abs(hit[0]) <= half and abs(hit[1]) <= half),
    }


# ------------------------------------------------------------------- reporting
def build_report():
    n_glass = index_of_refraction()
    theta_i = np.deg2rad(TILT_DEG)
    theta_t = snell_angle(theta_i, N_AIR, n_glass)
    R_s, R_p, R = fresnel(theta_i, N_AIR, n_glass)
    frac = path_fractions(R)
    spots = {k: trace_path(k, n_glass) for k in PATHS}

    # cross-check the closed-form offset against the traced geometry
    dx_traced = abs(spots['TRRT']['centroid'][0] - spots['TT']['centroid'][0])
    dx_closed = ghost_offset(THICKNESS, theta_i, theta_t)

    return {
        'setup': {
            'wavelength_nm': WAVELENGTH, 'material': MATERIAL,
            'tilt_deg': TILT_DEG, 'thickness_mm': THICKNESS,
            'z_front_mm': Z_FRONT, 'z_recv_fwd_mm': Z_RECV_FWD,
            'z_recv_back_mm': Z_RECV_BACK, 'recv_size_mm': RECV_SIZE,
            'recv_res': RECV_RES, 'beam_radius_mm': BEAM_RADIUS,
            'beam_power_W': BEAM_POWER,
        },
        'n_glass': n_glass,
        'theta_i_deg': float(TILT_DEG),
        'theta_t_deg': float(np.rad2deg(theta_t)),
        'R_s': R_s, 'R_p': R_p, 'R': R,
        'fractions': frac,
        'power_W': {k: BEAM_POWER * v for k, v in frac.items()},
        'ghost_offset_mm': dx_closed,
        'ghost_offset_traced_mm': dx_traced,
        'spots': spots,
    }


def print_report(rep):
    print(f"n({rep['setup']['wavelength_nm']} nm, {rep['setup']['material']}) "
          f"= {rep['n_glass']:.7f}")
    print(f"theta_i = {rep['theta_i_deg']:.4f} deg   "
          f"theta_t = {rep['theta_t_deg']:.4f} deg")
    print(f"R_s = {rep['R_s']:.7f}   R_p = {rep['R_p']:.7f}   "
          f"R = {rep['R']:.7f}\n")

    print(f"{'path':6} {'fraction':>12} {'power [W]':>12} {'z_recv':>8} "
          f"{'x [mm]':>10} {'y [mm]':>8}  on-recv")
    print('-' * 68)
    for k in PATHS:
        s = rep['spots'][k]
        print(f"{k:6} {rep['fractions'][k]:12.7f} {rep['power_W'][k]:12.7f} "
              f"{s['z_recv']:8.1f} {s['centroid'][0]:10.4f} "
              f"{s['centroid'][1]:8.4f}  {'yes' if s['on_recv'] else 'NO'}")

    print(f"\nghost offset  closed form {rep['ghost_offset_mm']:.6f} mm"
          f"   traced {rep['ghost_offset_traced_mm']:.6f} mm"
          f"   |diff| {abs(rep['ghost_offset_mm'] - rep['ghost_offset_traced_mm']):.2e}")
    print(f"sum of fractions = {sum(rep['fractions'].values()):.7f} "
          f"(< 1: higher-order ghosts TR^4T, ... are dropped)")
    if not rep['spots']['R']['on_recv']:
        print("note: the front specular spot falls outside the 80x80 mm back "
              "receiver -- widen it or move it if Phase 2 must catch that path.")


def save_report(rep, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'analytic.json')
    with open(path, 'w') as f:
        json.dump(rep, f, indent=2, sort_keys=True)
    return path


def _self_check(rep):
    """Gate for Phase 0."""
    assert abs(rep['ghost_offset_mm'] - rep['ghost_offset_traced_mm']) < 1e-9
    assert abs(rep['theta_t_deg'] - 19.211319) < 1e-5
    assert abs(rep['R'] - 0.0440790) < 1e-6
    assert abs(rep['fractions']['TT'] - 0.9137849) < 1e-6
    assert abs(rep['fractions']['TRRT'] - 0.0017754) < 1e-6
    assert abs(rep['ghost_offset_mm'] - 6.035476) < 1e-5
    # main beam exits a parallel plate undeviated
    assert np.allclose(rep['spots']['TT']['exit_dir'], [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(rep['spots']['TRRT']['exit_dir'], [0.0, 0.0, 1.0], atol=1e-12)
    print('self-check OK')


if __name__ == '__main__':
    rep = build_report()
    print_report(rep)
    _self_check(rep)
    print('\nwrote', save_report(rep))
