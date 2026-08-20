# Differentiable non-sequential Monte Carlo ray tracing in `dO`

## Context

**Internship** (`IMLEX2526_Internship_1_YouriMeuret.pdf`, Youri Meuret, Light & Lighting Lab, KU Leuven), three stages:

1. Understand differentiable **sequential** ray tracing via `diffoptics` — study and reproduce examples. *(largely done: `examples/practice/` M1–M11 curriculum exists and runs)*
2. Implement differentiable **non-sequential Monte Carlo** ray tracing, following `arXiv:2601.04370`.
3. Use it to design an **illumination component** that maps a source distribution to a prescribed target.

**Why it's needed.** `dO` is sequential by construction and says so — Wang 2022 §III-B: *"we focus on the sequential mode"*; §VI lists non-sequential tracing as future work. Concretely, `Lensgroup._forward_tracing` (`diffoptics/optics.py:1057`) is a `for i in range(stop_ind+1)` over an ordered surface list, one intersection per surface, no closest-hit search. `_refract` (`optics.py:1014`) marks TIR as `valid=False` — the ray is **killed, never reflected**. There is no reflection branch, no ray weight, no path loop. Every physical effect the internship targets (TIR collimators, multi-bounce light guides, ghosts, concentrators) is exactly what that code discards.

**Method to follow.** Yang et al. 2026 (KAUST + Meta), *End-to-end differentiable design of geometric waveguide displays*. Their Methods section §"Differentiable Monte Carlo non-sequential ray tracing" is the recipe: at each partially-reflective interaction, **sample one branch instead of splitting**, and rescale by the sampling probability (their Eq. 6). Ray count stays constant, memory is bounded, gradients survive because the *decision* is detached while the *physical coefficient* stays in the autodiff graph.

**Decisions taken** (from clarifying questions): tracer ships as a real library module `diffoptics/nonseq.py`; LightTools is available and driven directly; polarization + thin-film TMM are **out of scope** (scalar unpolarized Fresnel only) but the coefficient hook is designed so they could drop in later.

**Outcome.** A differentiable non-sequential MC tracer in the repo, validated three ways (analytic / deterministic split / LightTools), then used to optimize an illumination component against a prescribed target.

---

## The physics, settled on paper first

Do this before any code. Everything downstream is checked against these numbers.

### Validation setup — tilted glass plate ghost test

Chosen because it is genuinely non-sequential (the same two surfaces are revisited, hit order is not known in advance), yet has a closed-form answer.

```
collimated beam              tilted plate                    receiver
r = 1 mm, λ = 532.8 nm  →  N-BK7, t = 10 mm, tilt 30°  →   z = 200 mm, 80×80 mm, 512²
1 W, along +z              front face at z = 50 mm
                           tilt about y axis                back-receiver at z = 0
                                                            (catches the front reflection)
```

`n(532.8 nm) ≈ 1.5195` from `Material('N-BK7').ior(532.8)` (Cauchy `A + B/λ²`, `basics.py:236`).
Snell: `θ_i = 30°` → `θ_t = 19.2°`. Unpolarized `R = ½(R_s + R_p) ≈ 0.0455`.

Three separated spots:

| Path | Fraction of input | Position |
|---|---|---|
| `T·T` main beam | `(1−R)² ≈ 0.909` | forward, laterally shifted |
| `R` front specular | `R ≈ 0.0455` | backward, ~60° off axis |
| `T·R·R·T` ghost | `(1−R)²R² ≈ 0.00188` | forward, parallel to main, offset **≈ 6.04 mm** |

Ghost offset `Δ = 2·t·tan(θ_t)·cos(θ_i) = 2·10·0.3486·0.866 ≈ 6.04 mm`.
Beam radius 1 mm ⇒ spots cleanly separated. The geometry numbers were picked for exactly that; do not shrink `t` or the ghost merges into the main beam.

**A sequential tracer produces spot 1 only.** That contrast is the demo.

### The Monte Carlo estimator

At each partially reflective hit, with sampling probability `ρ`:

```
u ~ U(0,1)
u < ρ   →  reflect,   w ← w · R/ρ
u ≥ ρ   →  transmit,  w ← w · (1−R)/(1−ρ)
```

- **Unbiased**: expectation reproduces the deterministic split. Error falls as `1/√N`.
- **`ρ` must be detached** (`rho = R.detach()`). The paper goes further and uses a *learned* `ρ` from a cheap pre-optimization pass. Keeping `ρ` out of the graph is what makes the gradient a clean pathwise estimator with no score-function term.
- **`ρ` must be clamped** off 0 and 1 (`clamp(1e-3, 1-1e-3)`), else rare branches give infinite weights.
- **TIR reflects with probability 1**, weight ×1. This is the single most important behavioural difference from current `dO`.

### Two-pass intersection (paper Fig. 3b)

Pass 1 finds *which* surface each ray hits, under `no_grad`, recording surface IDs. Pass 2 recomputes only that intersection with autodiff on. Same spirit as `dO`'s implicit-Newton trick (`optics.py:1305`) lifted one level up: from "which `t`" to "which surface".

---

## Phase 0 — Analytic ground truth

**File:** `examples/nonseq/n00_analytic.py`

Pure numpy. No tracer. Computes, for the plate setup: `θ_t`, `R_s`, `R_p`, `R`, the three path fractions, the ghost offset, and the three spot centroids on both receivers. Prints a table and saves `n00_out/analytic.json`.

Every later phase asserts against this JSON. It is the arbiter when the tracer and LightTools disagree.

**Done when:** table matches the numbers above to ~1e-6.

---

## Phase 1 — Geometry layer: poses and closest-hit

This is the real architectural work, and the part that has nothing to do with Monte Carlo. Do it first and alone.

**File:** `diffoptics/nonseq.py` (new)

### Why a new layer is needed

`Lensgroup` holds **one** pose (`origin`, `theta_x/y/z`, `optics.py:44`) shared by all its surfaces; individual `Surface` objects only carry a scalar `d` offset along the group's local z. Non-sequential scenes need each surface independently placed and oriented — a 45° extraction mirror is not expressible as a `d` offset.

There is also a sharp edge in `Surface.newtons_method` (`optics.py:1298`): the initial guess is `t0 = (self.d - oz) / dz`. When a ray runs nearly parallel to a surface's local xy-plane, `dz → 0` and `t0` explodes. Sequential tracing never hits this; a light guide hits it constantly. Per-surface local frames fix it *if* every surface is oriented so its local +z faces the traffic — that is a documented constraint of this design, plus a `|dz| > eps` guard.

### What to add

```python
class Element:
    """A Surface plus a world pose plus an interface description."""
    surface:  Surface          # reuse Aspheric / BSpline / XYPolynomial / Mesh unchanged
    to_world: Transformation   # reuse basics.py:57
    n_in, n_out: Material      # which side is which, by local +z
    kind: 'refractive' | 'mirror' | 'partial' | 'absorber'
    rho_fixed: float | None    # optional override of the sampling probability

def closest_hit(o, d, elements):
    """Pass 1. Under no_grad. Returns (t_min, elem_id) over all elements.
    Per element: transform ray to local frame, call surface.newtons_method,
    reject t <= eps, reject via surface.is_valid (optics.py:1240) for aperture,
    then argmin over elements."""

def intersect_one(o, d, elements, elem_id):
    """Pass 2. WITH grad. Re-runs the chosen intersection only, returns (p, n)
    in world coordinates."""
```

**Reuse, do not reimplement:** `Surface.newtons_method` / `newtons_method_impl` (`optics.py:1267`, `1321`) — the implicit-gradient trick is already correct and is the Wang 2022 contribution; `Surface.is_valid` / `sdf_approx` (`optics.py:1240`, `1221`) for finite apertures; `Transformation.transform_ray` (`basics.py:80`); `Material.ior` (`basics.py:236`); `normalize` (`basics.py:293`).

**Test:** `examples/nonseq/n01_geometry_test.py` — single tilted plane, known ray, assert hit point against hand-computed value. Then two planes, assert the closest one wins from both directions.

**Done when:** closest-hit is correct for a ray fired at the plate from either side, and a ray parallel to a surface returns "no hit" rather than a NaN.

---

## Phase 2 — Deterministic split tracer (the reference)

**File:** `diffoptics/nonseq.py`, function `trace_split(rays, elements, max_depth=4)`

Explicit recursion. At each partially-reflective hit, spawn **both** children with weights `w·R` and `w·(1−R)`. Depth-capped. This is deliberately the slow, exponential, obviously-correct implementation — it exists to be the reference the MC tracer is checked against, and to demonstrate in the write-up why MC is needed.

Also add here, since both tracers need them:

```python
def fresnel_unpolarized(d, n, eta):  # returns R, differentiable
def reflect(d, n):                   # d - 2(d·n)n
def refract(d, n, eta):              # port of Lensgroup._refract (optics.py:1014),
                                     # but returns ok=False for TIR WITHOUT killing the ray
```

**Test:** `examples/nonseq/n02_split_plate.py` — the plate setup, three spots, fluxes asserted against `analytic.json` to ~1e-6.

**Done when:** all three path fractions and the 6.04 mm ghost offset match analytic.

---

## Phase 3 — Monte Carlo tracer (the core deliverable)

**File:** `diffoptics/nonseq.py`, function `trace_mc(rays, elements, max_bounces=100, rr=True)`

Iterative, not recursive. Fixed ray count. State per ray: `o [N,3]`, `d [N,3]`, `w [N]`, `alive [N]`.

```python
for bounce in range(max_bounces):
    with torch.no_grad():                                  # pass 1 (paper Fig. 3b)
        t_hit, elem_id = closest_hit(o, d, elements)
    alive &= (elem_id >= 0)
    if not alive.any(): break

    p, n = intersect_one(o, d, elements, elem_id)          # pass 2, WITH grad

    R = fresnel_unpolarized(d, n, eta)                     # differentiable
    rho = R.detach().clamp(1e-3, 1 - 1e-3)                 # sampling prob, DETACHED

    do_reflect = torch.rand_like(rho) < rho                # decision: no gradient
    d_r = reflect(d, n)
    ok_t, d_t = refract(d, n, eta)
    do_reflect = do_reflect | ~ok_t                        # TIR: reflect, probability 1

    d = torch.where(do_reflect[:, None], d_r, d_t)
    w = w * torch.where(do_reflect, R / rho, (1 - R) / (1 - rho))
    o = p + 1e-6 * d                                       # epsilon offset
    alive &= (w > 1e-6)                                    # or russian roulette
```

### The four things that will actually cost time

1. **Self-intersection.** Offset the new origin along `d`, *and* exclude `elem_id` from the next `closest_hit`. Without both, every ray dies at `t ≈ 0` on bounce 2. Expect to lose a day here if skipped.
2. **TIR must reflect, not die.** `ok_t == False` ⇒ forced reflection. This is the line that `dO` does not have.
3. **`rho` detached.** If it stays in the graph the gradient is biased and blows up — and it will look like a physics bug, not an autograd bug.
4. **Clamp `rho`.** At near-normal incidence `R ≈ 0.045`; the ghost branch is rare and a `rho` of exactly 0 gives `inf`.

### Splatting to a receiver

Reuse the differentiable bilinear splat from `Lensgroup.render` (`optics.py:712`, the four `index_put(..., accumulate=True)` calls) — it already carries gradients correctly and conserves energy. Wrap it as a `Receiver` in `nonseq.py` that takes world-space hits plus per-ray weight `w`.

**Tests:** `examples/nonseq/n03_mc_plate.py`
- MC vs split tracer: three fluxes agree within MC standard error, over 5 seeds (the paper reports over 5 seeds — copy that).
- Error vs ray count, log-log, slope ≈ −0.5.
- Peak memory flat in bounce count, unlike `trace_split`.

**Done when:** all three hold.

---

## Phase 4 — LightTools side-by-side

You have LightTools and can run it directly, so this is a full three-way comparison, not a spot check.

**Build:** collimated source (1 mm radius, 1 W, 532.8 nm) → rectangular solid 40×40×10 mm, N-BK7 from the SCHOTT catalog, rotated 30° about y → two receiver planes at z = 200 and z = 0.

- Surface optical property: **uncoated Fresnel** on both faces. No coating file.
- Lower the minimum-flux threshold (`1e-6`) so the 0.19% ghost is not culled. Default settings will silently drop it.
- Receiver mesh 512×512. Export illuminance to CSV.

**Run twice:**
1. **Ray splitting ON** (LightTools default) — deterministic branch tree, the commercial reference.
2. **Ray splitting OFF**, 10⁶ rays — LightTools then Monte-Carlo-samples one branch per hit, which *is* the paper's estimator. This is the direct apples-to-apples check on your `trace_mc`.

**File:** `examples/nonseq/n04_compare_lighttools.py` — reads the CSVs, produces the deliverable figure.

### The deliverable figure

Three columns — **analytic | dO-nonseq | LightTools** — with:
1. receiver illuminance map (512²),
2. horizontal cross-section through the three spots,
3. table of the three fluxes with MC standard error over 5 seeds,
4. MC error vs ray count, log-log, expected slope −0.5.

**Then the payoff no commercial tracer gives you:** `dΦ_ghost/dθ_tilt` and `dΦ_ghost/dt_plate` by autograd, verified against finite differences from a LightTools tilt sweep (5 tilt angles, central difference). This is the single strongest slide in the final report — same physics as LightTools, plus gradients.

**Done when:** fluxes agree across all three columns within MC error, and the autograd gradient matches the LightTools finite difference to a few percent.

---

## Phase 5 — Slab light guide (many bounces)

**File:** `examples/nonseq/n05_lightguide.py`

Launch into the edge of a glass slab; rays TIR along it; a 45° partial mirror (`ρ = 0.3`) extracts. Now bounce counts are high (10–50) and hit order genuinely varies per ray — the regime where deterministic splitting dies and MC does not.

This is the GWG of the paper in miniature, and the structural rehearsal for the collimator.

**Checks:** energy conservation (in = out + absorbed) to <0.1%; extraction efficiency vs mirror `ρ` matches a hand-computed geometric series; the split tracer becomes infeasible past depth ~12 while MC memory stays flat — measure and plot this, it justifies the whole method.

---

## Phase 6 — Illumination design (internship stage 3)

**File:** `examples/nonseq/n06_collimator_design.py`

The LED collimator from the internship brief: aspheric TIR surface + aspheric refractive surface + flat exit. Point or small-area source, prescribed target irradiance on a far plane.

- Surfaces: reuse `Aspheric` (`optics.py:1471`) to start, then `BSpline` (`optics.py:1554`) for freeform degrees of freedom.
- Optimizer: `torch.optim.Adam` on the surface coefficients — same loop shape as `examples/caustic_pyramid.py:129-148`, which is the closest existing reference in the repo (MC jitter + `spp` accumulation + Adam on freeform coefficients + differentiable irradiance).
- Loss: normalized MSE against the target, plus a smoothness regularizer on the freeform coefficients.

**Carry over the gotchas already documented in `examples/practice/STUDY_GUIDE.md` §6** — they all still apply and were learned the hard way:
- energy conservation ⇒ optimize the *distribution*, never the sum;
- stay defocused, the in-focus regime is a razor-thin loss well on a flat plateau;
- keep rays valid, an all-zero image has no gradient and `.backward()` fails.

**New MC-specific one:** gradient noise scales with `1/√(spp)`. If Adam stalls, raise `spp` before touching the learning rate.

**Validate the final design in LightTools** — export the optimized surface (sag table or coefficients), rebuild, confirm the target irradiance. Closes the loop and is what makes the result credible to the lab.

---

## File manifest

**New:**
```
diffoptics/nonseq.py            Element, closest_hit, intersect_one,
                                trace_split, trace_mc, Receiver,
                                fresnel_unpolarized, reflect, refract
examples/nonseq/n00_analytic.py            ground truth → analytic.json
examples/nonseq/n01_geometry_test.py       poses + closest-hit
examples/nonseq/n02_split_plate.py         deterministic reference
examples/nonseq/n03_mc_plate.py            MC tracer + convergence
examples/nonseq/n04_compare_lighttools.py  three-way figure + gradient check
examples/nonseq/n05_lightguide.py          many-bounce TIR slab
examples/nonseq/n06_collimator_design.py   illumination optimization
examples/nonseq/README.md                  what each phase shows
```

**Modified:** `diffoptics/__init__.py` — export the `nonseq` symbols.

**Untouched:** everything else. `optics.py`, `basics.py`, `shapes.py`, `solvers.py`, all existing examples and the whole `examples/practice/` curriculum keep working unchanged. `nonseq.py` imports from them; nothing imports back.

---

## Verification

Per phase, in order — each gate must pass before the next phase starts:

| Phase | Gate |
|---|---|
| 0 | Analytic table matches the three fractions and 6.04 mm offset |
| 1 | Closest-hit correct from both sides; parallel ray → no hit, no NaN |
| 2 | Split tracer matches analytic to ~1e-6 |
| 3 | MC matches split within standard error over 5 seeds; error slope ≈ −0.5; memory flat in bounces |
| 4 | Three-way flux agreement; autograd gradient matches LightTools finite difference |
| 5 | Energy conserved <0.1%; extraction vs `ρ` matches geometric series |
| 6 | Optimized target irradiance reproduced in LightTools |

Run everything on CPU first — the plate setup is small and CPU keeps the debug loop fast. Move to GPU only at Phase 5.

## Out of scope (documented extension points)

- **Polarization + thin-film TMM** (paper Methods Eq. 3–5). Design `fresnel_unpolarized` behind a coefficient hook so a TMM solver returning `(r_s, r_p, t_s, t_p)` can replace it later without touching the tracer loop.
- **Learned sampling probabilities.** The paper pre-optimizes `ρ` per mirror to stop rays starving the target. Only needed once a scene has many partial mirrors in series — likely relevant at Phase 5 if extraction efficiency is low, not before.
- **Multi-GPU FoV partitioning** (paper Fig. 3a). Irrelevant at this scale.
