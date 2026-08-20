"""
PHASE 0 - ANALYTIC GROUND TRUTH for the tilted-plate ghost test.

Pure numpy + closed-form optics. No ray tracer, sequential or otherwise. Every
later phase (n01..n06) asserts its results against the JSON this file writes, and
this file is the arbiter whenever `dO`-nonseq and LightTools disagree.

--------------------------------------------------------------------------------
SCENE (all lengths in mm, angles in degrees unless stated)
--------------------------------------------------------------------------------

    collimated beam                tilted plate                    receivers
    r = 1 mm, 1 W, 532.8 nm  ->  N-BK7 slab, t = 10 mm    ->   FWD: z = 200
    along +z, centred on the      tilted 30 deg about y         BACK: z = 0
    z axis (chief ray starts      front face through            both 80 x 80 mm,
    at the origin)                (0, 0, 50)                    512 x 512

Plate orientation. The front face is the plane through P_FRONT = (0, 0, Z_FRONT)
with outward normal (pointing back at the source)

    n_front = (-sin(tilt), 0, -cos(tilt))

The back face is the parallel plane through P_FRONT + THICKNESS * (-n_front),
with the same normal. Both faces are infinite planes for this phase; apertures
are Phase 1's problem.

Refractive index comes from the repo so Phase 0 and the tracer cannot drift:

    n = float(do.Material('N-BK7').ior(532.8))        # Cauchy A + B/lambda^2

Media: air (n = 1) outside, glass inside. No absorption, no coatings, no
polarization tracking - unpolarized Fresnel only, R = (R_s + R_p) / 2.

--------------------------------------------------------------------------------
THE THREE PATHS
--------------------------------------------------------------------------------

Only three paths carry meaningful flux. Label them by the sequence of events at
the interfaces (T = transmit, R = reflect), in the order the ray meets them:

    'TT'    front T, back T            main beam,  forward, exits parallel to +z
    'R'     front R                    specular,   backward, ~60 deg off axis
    'TRRT'  front T, back R, front R, back T
                                       ghost,      forward, parallel to the main
                                                   beam, laterally offset

Each path's power fraction is the product of its per-event Fresnel coefficients.
A sequential tracer produces 'TT' only - that contrast is the whole demo.

--------------------------------------------------------------------------------
WHAT THIS FILE MUST PRODUCE
--------------------------------------------------------------------------------

For the chief ray (origin (0,0,0), direction (0,0,1)):

  1. theta_t                       refraction angle inside the glass
  2. R_s, R_p, R                   Fresnel reflectance at the first interface
  3. frac['TT'], frac['R'], frac['TRRT']    power fractions of the three paths
  4. ghost_offset                  in-plane separation of the ghost from the main
                                   beam on the forward receiver
  5. spot centroid (x, y) of each path, on whichever receiver it lands on, plus
     an in/out-of-bounds flag against the 80 x 80 mm receiver aperture

Print a readable table, then write everything to `n00_out/analytic.json`.

Run:    python n00_analytic.py
Gate:   the printed offset and fractions are stable to ~1e-6 and are reproduced
        by the deterministic split tracer in Phase 2.
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


# ------------------------------------------------------------------- geometry
def plate_planes():
    """Front and back faces of the slab.

    Returns
    -------
    (p_front, n_front), (p_back, n_back)
        Each a point on the plane and its unit normal, both shape (3,), with the
        normals oriented as described in the module docstring.
    """
    th = np.deg2rad(TILT_DEG)
    n_front = np.array([-np.sin(th), 0.0, -np.cos(th)])
    n_back = -n_front

    p_front = np.array([0.0, 0.0, Z_FRONT])
    p_back = p_front + THICKNESS * (-n_front)
    return (p_front, n_front), (p_back, n_back)



def plane_intersect(o, d, p0, n):
    """Intersect the ray o + t*d with the plane through p0 with normal n.

    Returns
    -------
    t : float
        Ray parameter of the hit, or np.inf if the ray is parallel to the plane
        or the hit is behind the origin.
    """
    denom = float(np.dot(d,n))
    
    if(abs(denom) < 1e-9):
        return np.inf
    
    t = float(np.dot(p0 - o, n) / denom)
    return t if t > 1e-9 else np.inf
    


def reflect(d, n):
    """Specular reflection of unit direction d off a surface with unit normal n.

    Returns the reflected unit direction, shape (3,).
    """
    
    return d - 2.0 * float(np.dot(d, n)) * n
    
    
    


def refract(d, n, eta):
    """Refraction of unit direction d at a surface with unit normal n.

    Parameters
    ----------
    d, n : (3,) arrays, unit length. n may point either way relative to d.
    eta : float
        Ratio of the incident-side index to the transmitted-side index.

    Returns
    -------
    ok : bool
        False on total internal reflection, in which case `dt` is meaningless.
    dt : (3,) array
        Transmitted unit direction.
    """
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
    """Refractive index of the plate at WAVELENGTH, as a float."""
    return float(do.Material(MATERIAL).ior(WAVELENGTH))


def snell_angle(theta_i, n1, n2):
    """Refraction angle in radians for incidence angle theta_i (radians)."""
    return float(np.arcsin(np.clip(n1 * np.sin(theta_i) / n2, -1.0, 1.0))) 



def fresnel(theta_i, n1, n2):
    """Unpolarized Fresnel reflectance at a single interface.

    Returns
    -------
    (R_s, R_p, R) : floats
        s- and p-polarized power reflectances, and their unpolarized average.
    """
    theta_t = snell_angle(theta_i, n1, n2)
    ci, ct = np.cos(theta_i), np.cos(theta_t)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)
    R_s, R_p = float(rs ** 2), float(rp ** 2)
    return R_s, R_p, 0.5 * (R_s + R_p)



def path_fractions(R):
    """Power fraction of each path in PATHS, given the single-interface R.

    Returns a dict keyed by the labels in PATHS.
    """
    T = 1.0 - R
    return {
        'TT':   T * T,
        'R':    R,
        'TRRT': T * T * R * R,
    }
    

def ghost_offset(thickness, theta_i, theta_t):
    """Lateral separation between the ghost and the main beam, in mm."""
    return float(2.0 * thickness * np.tan(theta_t) * np.cos(theta_i))

def _events(label):
    """Interface sequence for a label: (which face, what happens)."""
    faces = {'TT':   [('front', 'T'), ('back', 'T')],
             'R':    [('front', 'R')],
             'TRRT': [('front', 'T'), ('back', 'R'), ('front', 'R'), ('back', 'T')]}
    return faces[label]

# ------------------------------------------------------------- chief-ray trace
def trace_path(label):
    """Follow the chief ray along one labelled path, by hand.

    Applies the events of `label` in order at the front/back planes (no search,
    no closest-hit logic - the sequence is known), then propagates the exit ray
    to whichever receiver plane it reaches.

    Returns
    -------
    dict with keys:
        'z_recv'   : float, receiver plane the path lands on
        'centroid' : (x, y) tuple, mm, hit point on that plane
        'exit_dir' : (3,) list, unit exit direction
        'on_recv'  : bool, whether the centroid is inside the 80 x 80 aperture
    """

    if n_glass is None:
        n_glass = index_of_refraction()
    (p_f, n_f) , (p_b, n_b) = plate_planes()
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
        
        

# -------------------------------------------------------------------- reporting
def build_report():
    """Assemble every quantity listed in the module docstring into one dict."""
    raise NotImplementedError


def print_report(rep):
    """Print the report as a readable table."""
    raise NotImplementedError


def save_report(rep, out_dir=OUT_DIR):
    """Write the report to <out_dir>/analytic.json and return the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'analytic.json')
    with open(path, 'w') as f:
        json.dump(rep, f, indent=2, sort_keys=True)
    return path


if __name__ == '__main__':
    rep = build_report()
    print_report(rep)
    print('\nwrote', save_report(rep))
