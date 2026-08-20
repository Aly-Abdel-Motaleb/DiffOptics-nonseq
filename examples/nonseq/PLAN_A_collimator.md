# Sub-plan A — point source + collimating lens

## Goal (Jeroen's email)

Simplest possible first test of the non-sequential tracer:

1. **R = 0** — point source + refractive collimating lens. Non-seq must match dO **sequential**: irradiance map, total power, ray directions.
2. **R = 0.2** on the lens — same geometry, now reflected + transmitted paths. Sequential cannot do this.
3. **LightTools** — compare case 2: irradiance, transmitted/reflected power, energy balance.

The sequential tracer is the reference in step 1; LightTools is the reference in step 3. No analytic solution is used anywhere.

Replaces PLAN.md's tilted-plate ghost test as the entry path. PLAN.md **Phase 1** (geometry layer, `diffoptics/nonseq.py`) still comes first — it is scene-independent. The plate test (`n00_analytic*.py`) is demoted to a later regression, not deleted.

---

## The scene

```
point source        collimating lens              forward receiver
(0,0,0), 1 W    →   S1 asphere, S2 flat      →    z = 200 mm, ±16 mm, 512²
isotropic           N-BK7, t = 6.5 mm             backward receiver (stage 2+)
into cone           semi-diameter 12.7 mm         z = −80 mm, ±80 mm, 512²
```

Lens = exact aspheric collimator, so a source at distance `s = 40 mm` comes out collimated with no spherical aberration. Closed form, no optimization needed:

| param | value |
|---|---|
| `s` (source → S1 vertex) | 40 mm |
| `n` at 532.8 nm (`Material('N-BK7').ior`) | 1.5195 |
| `c = 1/(s(n−1))` | 0.04812 mm⁻¹ |
| `k = −n²` | −2.3089 |
| semi-diameter | 12.7 mm |
| thickness (S2 `d`) | 6.5 mm, flat |
| sag at r = 12.7 | 3.49 mm |
| collection half-angle `θ_max = atan2(12.7, s+sag)` | 16.3° |

Exact only at 532.8 nm (`k` depends on `n(λ)`) — fine, everything is monochromatic.

Why this lens: perfect collimation makes the stage-1 gate razor sharp — every output ray must have `|d_x|, |d_y| < 1e-6`. A sign error in the new geometry layer fails that instantly, where an image comparison would hide it.

---

## Blocking gotchas in the existing code (read first)

| # | Fact | Where | Consequence |
|---|---|---|---|
| 1 | `Lensgroup.render` has no per-ray weight — `J = irr` is scalar, invalid rays dropped by boolean indexing | `optics.py:712,732` | It is a bilinear hit-count histogram. Non-seq needs its own weighted splat: copy the four `index_put(..., accumulate=True)` calls at `optics.py:754-758`, multiply by `w`. |
| 2 | `Ray` has no weight field | `basics.py:43` | Carry `w [N]` as a separate tensor beside `o, d`. |
| 3 | **No point-source sampler exists**; `optics.py:539,589,615` are all collimated. `pointSrc_spherical.py:55-62` samples a grid on a *plane* → non-uniform in solid angle | — | Must write one. This is the #1 trap for irradiance and for the LightTools comparison. |
| 4 | `_trace` picks forward/backward via `(ray.d[...,2] > 0).all()` | `optics.py:1049` | Mixed-direction batches misroute. Sequential structurally **cannot** trace the R = 0.2 back-reflection — that is the point of stage 2. |
| 5 | `_refract` marks TIR `valid=False` | `optics.py:1014` | Ray killed, never reflected. |
| 6 | `sdf_approx` returns **squared** units for round apertures | `optics.py:1221` | Not a distance. |
| 7 | `Transformation.transform_ray` builds a fresh `Ray`, dropping `mint/maxt` | `basics.py:80` | Transform any extra per-ray state manually. |

---

## Phases

### P1 — geometry layer (= PLAN.md Phase 1, unchanged)
**File:** `diffoptics/nonseq.py` — `Element`, `closest_hit`, `intersect_one`. Test: `examples/nonseq/n01_geometry_test.py`.

Reuse `Surface.newtons_method` (`optics.py:1267`), `Surface.is_valid` (`optics.py:1240`), `Transformation.transform_ray` (`basics.py:80`), `Material.ior` (`basics.py:236`), `normalize` (`basics.py:293`).

One addition for this sub-plan — `Element` gets a fixed-reflectivity override:

```python
kind: 'refractive' | 'mirror' | 'partial' | 'absorber'
R_fixed:   float | None   # constant reflectance, overrides Fresnel. 0.0 = reflection off
rho_fixed: float | None   # MC sampling-probability override
```

`R_fixed = 0.0` is the stage-1 mode; `R_fixed = 0.2` is stage 2.

**Gate:** closest hit correct from both sides; ray parallel to a surface → no hit, no NaN.

---

### P2 — stage 1: non-seq vs sequential, reflection OFF
**Files:** `examples/nonseq/common_collimator.py`, `examples/nonseq/c01_seq_vs_nonseq.py`

`common_collimator.py` builds **one** parameter set two ways so the scenes cannot drift:

- `build_lensgroup()` → `do.Lensgroup` with `[Aspheric(12.7, 0.0, c=C, k=K), Aspheric(12.7, 6.5, c=0.0)]`, materials `air / N-BK7 / air` (same shape as `examples/practice/common_setup.py:45-54`);
- `build_elements(R=0.0)` → the `nonseq` `Element` list from the identical constants;
- `sample_point_source(N, theta_max, rng)` → **uniform in solid angle**: `cosθ ~ U(cosθ_max, 1)`, `φ ~ U(0, 2π)`; per-ray power `w = Φ_captured / N` with `Φ_captured = P(1 − cosθ_max)/2`;
- `splat(p_world, w, film_size, pixel_size)` → the weighted `index_put` splat.

`c01_seq_vs_nonseq.py` runs both tracers on the **same ray batch**, `R_fixed = 0.0` on both surfaces (no Fresnel at all — matches dO's physics exactly):

1. **Collimation** — `max(|d_x|,|d_y|) < 1e-6` after the lens, in both tracers.
2. **Total power** — identical valid-ray count; `Σw` equals `Φ_captured`.
3. **Irradiance** — `lens.render(rays)` vs non-seq splat with all `w = 1`, forward receiver. Relative L2 < 1e-6 (same rays, no MC branching yet, so this is deterministic).
4. **Radial profile** — plot both; should overlay within sampler shot noise.

**Gate:** all four. Isolates geometry bugs before any Monte Carlo enters.

---

### P3 — stage 2: R = 0.2 on the lens
**File:** `examples/nonseq/c02_R02.py`

`R_fixed = 0.2` on **both** surfaces. Two receivers now (forward `z = 200`, backward `z = −80`) plus an "escaped" tally for rays hitting neither.

Run the deterministic split tracer (`trace_split`) first, then the MC tracer (`trace_mc`) — split is the local reference for MC.

Paths that appear, in decreasing size: forward `T₁T₂`, backward `R₁` off the asphere (neither collimated nor point-focused — the hyperboloid's mirror focus is not the source), forward ghost `T₁R₂R₁T₂`, then higher orders.

**Gates:**
- energy ledger `Φ_fwd + Φ_back + Φ_escaped = Φ_captured` to < 0.1 %;
- forward fraction ≈ `0.8 × 0.8 = 0.64` before ghost pickup — a coarse ledger check only, not a formal reference;
- MC ≈ split within MC standard error over 5 seeds; MC error vs ray count log-log slope ≈ −0.5;
- ghost **visible** — set any weight-culling threshold below 1e-4 or the ~0.6 % path is silently dropped;
- sequential run on this scene captures the forward path only (gotcha #4) — screenshot it, that contrast is the demo.

Optional variant, only if useful for the write-up: `R_fixed = None` → real unpolarized Fresnel on both surfaces. Same script, one flag.

Free bonus from autodiff, which no commercial tracer gives: `dΦ_fwd/dc`, `dΦ_back/dR`, `dΦ_ghost/dt`, checked against finite differences in the same script.

---

### P4 — stage 3: LightTools comparison
**File:** `examples/nonseq/c03_lighttools.md` (build spec) → run, then compare

Build spec, written while the geometry is fresh:

- source: isotropic point source, 1 W, 532.8 nm, cone half-angle 16.3° (state explicitly whether the cone or full 4π is used — normalization differs);
- lens: N-BK7, aspheric S1 `c = 0.04812`, `k = −2.3089`, semi-diameter 12.7, thickness 6.5, flat S2;
- surface properties: both surfaces fixed reflectance 0.2 (a "simple coating", **not** uncoated Fresnel);
- receivers: forward `z = 200`, ±16 mm, 512²; backward `z = −80`, ±80 mm, 512²;
- minimum-flux threshold down to 1e-8 so the ghost survives;
- run twice — ray splitting **ON** (deterministic reference) and **OFF** at 10⁶ rays (that is the MC estimator, the apples-to-apples check on `trace_mc`);
- export illuminance CSV, contract `x_mm, y_mm, E` per row, header skipped.

**Compare:** irradiance maps, transmitted/reflected power, energy balance. Report to Jeroen.

---

## Files

**New:**
```
examples/nonseq/common_collimator.py   one scene two builders, solid-angle sampler, weighted splat
examples/nonseq/c01_seq_vs_nonseq.py   P2  R = 0, must match dO sequential
examples/nonseq/c02_R02.py             P3  R = 0.2 both surfaces, ledger, MC, gradients
examples/nonseq/c03_lighttools.md      P4  build spec + comparison
```
**Modified:** `diffoptics/nonseq.py` (`R_fixed` on `Element`); `examples/nonseq/PLAN.md` (note the plate test now follows the collimator).

**Untouched:** `optics.py`, `basics.py`, `shapes.py`, `solvers.py`, `examples/practice/`, `examples/pointSrc_spherical.py`, `n00_analytic*.py`.

CPU throughout — the scene is tiny, keep the debug loop fast.