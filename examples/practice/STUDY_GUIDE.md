# DiffOptics (dO) — Study Guide

One place that explains **everything**: what the library is, how it fits together,
and — for each practice module — **what to read before**, what you build, and what you
should see. Work top to bottom.

Driving goal: a **point source + spherical lens illumination component** with
**Monte-Carlo** sampling and **differentiable irradiance** at a receiver. The
curriculum starts there and expands to every major dO capability.

Everything runs on **CPU**. Run a module, then diff your `_practice.py` against its
`_solution.py`. Heavy/GPU examples are covered by `*_GUIDE.md` (read-only).

```bash
cd examples/practice
python m01_primitives_rays_practice.py   # edit TODOs, then compare to _solution
```

---

## 1. What dO is

A differentiable ray tracer in PyTorch. You build an optical system, trace rays through
it, and read out an image (irradiance) or spot positions. Because every step is a
PyTorch op, you can take **gradients** of the output w.r.t. any optical parameter
(curvature, thickness, pose, source position) and **optimize** the system — for lens
design, illumination shaping, calibration, or end-to-end "Deep Lens" learning.

Key trick (the paper): the ray-surface intersection solver runs **without autograd** to
find the root, then re-attaches gradients — cheap memory, scales to millions of rays.

## 2. The library map (`diffoptics/`)

- **`basics.py`** — data primitives
  - `Ray(o, d, wavelength)` and `ray(t) = o + t*d`
  - `Material` + `ior(wavelength)` (refractive index)
  - `normalize`, `Sampler.concentric_sample_disk`, `Transformation`
- **`shapes.py`**
  - `Endpoint` base; `Screen` — a textured plane (the scene, for imaging)
- **`optics.py`** — the engine
  - `Lensgroup`: `load` / `load_file`, `sample_ray*`, `trace*`, **`render`**,
    `prepare_mts`, `sample_ray_sensor`
  - `Surface` → `Aspheric` (spherical = conic k=0), `BSpline`, `XYPolynomial`, `Mesh`
- **`solvers.py`**
  - `Optimization` → `Adam`, `LM` (`jacobian` / `optimize`), `Adjoint`

Class tree:

```
Endpoint      ── Lensgroup, Screen
Surface       ── Aspheric, BSpline, XYPolynomial, Mesh
PrettyPrinter ── Ray, Material, Sampler, Transformation, Spectrum
Optimization  ── Adam, LM, Adjoint
```

Forward data flow:

```
source rays → trace (per-surface _refract, sets `valid`) → intersect sensor/screen
            → render splat (bilinear index_put) → image I
gradients flow back through all of it (autograd, or the adjoint shortcut in M9)
```

## 3. Suggested order

`1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10`, then the M11 guides.
Illumination spine = `1 → 4 → 5 → 6 → 8`. The rest broaden into the whole library.

## 4. Read the core library once (before/around M1–M4)

1. `basics.py` — `Ray`, `Material`, `normalize`, `Sampler`
2. `optics.py` — `load`/`load_file`, `Aspheric`, `sample_ray*`,
   `trace`/`_trace`/`_refract`, **`render`**, then `prepare_mts`/`sample_ray_sensor`
3. `solvers.py` — `LM.jacobian`/`optimize`, `Adam`, `Adjoint`
4. `examples/caustic_pyramid.py` — the reference MC differentiable-illumination design

---

## 5. The modules

Line numbers point into the repo; tick as you read.

### M1 — Primitives & rays  (`m01_primitives_rays_*`)
A point source = **one origin, many directions**. Build `do.Ray(o, d, wl)`.
**Read before**
- [ ] `diffoptics/basics.py:36-55` — `Ray.__init__` + `__call__(t)`
- [ ] `diffoptics/basics.py:293` — `normalize`
- [ ] `common_setup.py` — `build_scene`
**Build:** rays fanning from one point to a target grid. **See:** 121 unit-norm rays,
shared origin.

### M2 — Build & trace a lens  (`m02_build_trace_*`)
Push rays through the spherical lens; layout + spot diagram.
**Read before**
- [ ] `diffoptics/optics.py:807-830` — `trace`
- [ ] `diffoptics/optics.py:768-805` — `trace_to_sensor` / `trace_to_sensor_r`
- [ ] `diffoptics/optics.py:452` — `plot_raytraces`
- [ ] `diffoptics/optics.py:193` — `spot_diagram`
- [ ] `examples/autodiff.py:40-84` — same calls, working
**Build:** layout figure + spot diagram + RMS. **See:** rays converging past the lens.

### M3 — Load real lens files  (`m03_lensfile_*`)
Load a Double-Gauss from `.txt`; evaluate across field angles.
**Read before**
- [ ] `diffoptics/optics.py:65-68` — `load_file`
- [ ] `diffoptics/optics.py:98-167` — `read_lensfile` (the `.txt` format)
- [ ] `diffoptics/optics.py:539-586` — `sample_ray` (`view`, `entrance_pupil`)
- [ ] `diffoptics/optics.py:183-191` — `rms` (returns `(value, ps)`)
- [ ] `diffoptics/optics.py:486` — `plot_setup2D_with_trace`
- [ ] `common_lensfile.py` — `load_lens`
- [ ] `examples/sanity_check.py` — the workflow this mirrors
**Build:** layout + per-field spot diagrams. **See:** RMS grows ~4→7 µm off-axis.

### M4 — Irradiance rendering  (`m04_irradiance_*`)
The illumination readout: `render` splats rays into a pixel grid.
**Read before**
- [ ] `diffoptics/optics.py:712-759` — `render` (intersect, `valid`, bilinear splat)
- [ ] `diffoptics/optics.py:1014-1044` — `_refract` (sets `valid`: TIR / miss)
**Build:** irradiance image at M=20 and M=60. **See:** grid-dot **aliasing** (the problem).

### M5 — Monte-Carlo sampling  (`m05_montecarlo_*`)
The fix: jitter each ray + average `spp`; plus the proper disk sampler.
**Read before**
- [ ] `examples/caustic_pyramid.py:73-86` — `sample_ray(random=True)` jitter
- [ ] `examples/caustic_pyramid.py:110-114` — `render(spp)` accumulation
- [ ] `diffoptics/basics.py:116-144` — `Sampler.concentric_sample_disk`
- [ ] `examples/render_psf.py:50-70` — disk sampler in practice
**Build:** deterministic vs jittered-grid vs disk images. **See:** MC images smooth,
no grid dots; disk = round footprint.

### M6 — Differentiability & Jacobians  (`m06_jacobian_*`)
Irradiance is differentiable: `dI/dc`.
**Read before**
- [ ] `diffoptics/solvers.py:5-31` — `Optimization` base (`diff_parameters_names` → tensors)
- [ ] `diffoptics/solvers.py:93-130` — `LM.jacobian`
- [ ] `examples/autodiff.py:103-110` — `LM(...).jacobian` on `render`
**Build:** scalar `d(sumI)/dc` + per-pixel Jacobian. **See:** scalar ≈ 0 (energy
conserved) but per-pixel Jacobian nonzero → optimize the *distribution*, not the sum.

### M7 — Optimize: spot minimization (LM)  (`m07_lm_optimize_*`, `m07_nikon_GUIDE.md`)
Classical Levenberg-Marquardt to shrink the spot of a real asphere.
**Read before**
- [ ] `diffoptics/solvers.py:132-…` — `LM.optimize(func, func_yref_y)`
- [ ] `diffoptics/solvers.py:93-130` — `LM.jacobian` (used inside)
- [ ] `examples/spherical_aberration.py` — the workflow this mirrors
- [ ] `m07_nikon_GUIDE.md` — (after) multi-surface advanced version
**Build:** LM over `c, k, ai`. **See:** RMS ~48 → ~3 µm.

### M8 — Optimize: differentiable illumination  (`m08_optimize_irradiance_*` + capstone `m08_misalignment_*`)
(a) Adam recovers a target curvature from irradiance. (b) **Capstone:** recover an
unknown source position + sensor distance from a measured image (inverse problem).
**Read before**
- [ ] `examples/caustic_pyramid.py:116-156` — the optimization loop
- [ ] `examples/caustic_pyramid.py:126-143` — the core 5 lines
- [ ] `diffoptics/optics.py:44-63` — `Lensgroup.__init__` (pose leaves)
- [ ] `diffoptics/optics.py:807-830` — `trace` calls `update()` when a pose needs grad
- [ ] `examples/misalignment_point.py` — the real-data version
**Build:** Adam c-recovery; LM pose recovery. **See:** c 0.028→0.036; source/sensor
recovered exactly. **Gotchas:** focus regime is rugged (stay defocused); a centered
source makes tilt unobservable (identifiability).

### M9 — Adjoint back-prop  (`m09_adjoint_*`)
Same gradients as autograd, memory ~constant in #ray-batches.
**Read before**
- [ ] `diffoptics/solvers.py:268-321` — `Adjoint.__init__` + `__call__` (3-stage)
- [ ] `diffoptics/solvers.py:323-327` — `_adjoint_batch` (per-batch VJP)
- [ ] `examples/backprop_compare.py` — baseline vs adjoint (GPU version)
**Build:** baseline autograd vs `do.Adjoint`. **See:** losses + gradients match to ~6 digits.

### M10 — Image rendering (MTS backward tracing)  (`m10_image_render_*`)
Trace backward from sensor pixels through the lens to a textured screen → image a scene.
**Read before**
- [ ] `diffoptics/optics.py:863-902` — `prepare_mts` (reverse lens into a camera)
- [ ] `diffoptics/optics.py:953-1010` — `sample_ray_sensor` + `_sample_ray_render`
- [ ] `diffoptics/shapes.py:29-111` — `Screen` (`intersect` + `shading`)
- [ ] `examples/render_image.py` — the workflow this mirrors
**Build:** RGB render of the squirrel through a Double-Gauss. **See:** a real image (~8 s).

### M11 — Advanced (read-only guides)
- [ ] `m11_psf_GUIDE.md` — `render_psf.py`: PSF maps over field & depth
- [ ] `m11_end2end_GUIDE.md` — `end2end_edof_backward_tracing.py`: joint optics + neural
  network (Deep Lens / EDoF), needs GPU + GAN deps

---

## 6. Concept cheat-sheet (recurring gotchas)

- **Energy conservation** — splat weights sum to 1 per ray ⇒ `sum(I)` is independent of
  curvature. Optimize the distribution, not the total.
- **Focus is rugged** — near exact focus the spot is a few pixels ⇒ a razor-thin loss
  well on a flat plateau (aliasing). Stay defocused for smooth optimization.
- **Identifiability** — not every parameter is observable from a given measurement
  (centered source ⇒ tilt ≈ invisible). Condition the inverse problem.
- **Keep rays valid** — bundles must stay within the aperture, else `render` returns an
  all-zero image with no gradient (and `.backward()` fails).
- **Spherical lens = `Aspheric` with `k=0, ai=None`**, curvature `c = 1/R`.
- **Deterministic vs Monte-Carlo** — jitter + average (`spp`) is the whole point of the
  sampling switch; it removes aliasing on the optical surfaces during optimization.

## 7. File index

- `common_setup.py` — from-scratch point-source + spherical lens (`build_scene`)
- `common_lensfile.py` — load shipped `.txt` lenses (`load_lens`)
- `m01`–`m10` `_practice.py` / `_solution.py` — the runnable modules
- `m07_nikon_GUIDE.md`, `m11_psf_GUIDE.md`, `m11_end2end_GUIDE.md` — read-only guides
- `README.md` — short version of this guide
