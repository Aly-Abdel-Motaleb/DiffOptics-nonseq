# DiffOptics (dO) — hands-on learning curriculum

A phased, do-it-yourself path through the **whole `dO` library**, built around one
driving goal — a **point source + spherical lens illumination component** with
**Monte-Carlo** sampling and **differentiable irradiance** — then branching out to
every major capability the library ships (real lenses, LM design, adjoint back-prop,
image rendering, Deep Lens).

Everything here runs on **CPU**. It is additive: nothing in `diffoptics/` or the
original `examples/*.py` is changed.

## How to use

Each runnable module has two files:

- `mNN_<topic>_practice.py` — has `# TODO` blanks (`...`). **You fill these in.**
- `mNN_<topic>_solution.py` — reference answer. `diff` them to check yourself.

Heavy/GPU examples that don't reduce cleanly to CPU are covered by a
`mNN_<topic>_GUIDE.md` — an annotated walkthrough + exercises (no runnable file).

Run from this folder:

```bash
cd examples/practice
python m01_primitives_rays_practice.py     # then diff against the _solution
```

Outputs land in `mNN_out/`. Shared helpers:
- `common_setup.py` — `build_scene()`: the from-scratch point-source + spherical lens.
- `common_lensfile.py` — `load_lens(name)`: load a shipped `.txt` lens on CPU.

## Modules

| M | Topic | Files | Teaches | Source example |
|---|-------|-------|---------|----------------|
| 0 | Library map | (this README, below) | class tree + data flow | — |
| 1 | Primitives & rays | `m01_primitives_rays_*` | `Ray`, `normalize`, point source = one origin/many dirs | (new) |
| 2 | Build & trace a lens | `m02_build_trace_*` | `Lensgroup.load`, `Aspheric`, `trace*`, `spot_diagram`, `plot_raytraces` | `autodiff.py` |
| 3 | Load real lens files | `m03_lensfile_*` | `load_file`, field angles, `entrance_pupil`, `rms` | `sanity_check.py` |
| 4 | Irradiance rendering | `m04_irradiance_*` | `render` (splat) + **aliasing** | (new) |
| 5 | Monte-Carlo sampling | `m05_montecarlo_*` | jittered grid + `concentric_sample_disk` | `render_psf.py`, `caustic_pyramid.py` |
| 6 | Differentiability | `m06_jacobian_*` | autograd on `ps`, `LM.jacobian`, energy conservation | `autodiff.py` |
| 7 | Optimize: spot (LM) | `m07_lm_optimize_*` + `m07_nikon_GUIDE.md` | `LM.optimize`, multi-param `diff_names` | `spherical_aberration.py`, `nikon.py` |
| 8 | Optimize: illumination | `m08_optimize_irradiance_*` (Adam) + `m08_misalignment_*` (capstone) | Adam loop; inverse pose recovery | `caustic_pyramid.py`, `misalignment_point.py` |
| 9 | Adjoint back-prop | `m09_adjoint_*` | `do.Adjoint`, memory scaling; grads == autograd | `backprop_compare.py` |
| 10 | Image rendering (MTS) | `m10_image_render_*` | `prepare_mts`, `sample_ray_sensor`, `Screen` | `render_image.py` |
| 11 | Advanced (read only) | `m11_psf_GUIDE.md`, `m11_end2end_GUIDE.md` | PSF field/depth maps; end-to-end Deep Lens | `render_psf.py`, `end2end_*` |

**Suggested order:** 1→2→3→4→5→6→7→8→9→10, then the M11 guides. The illumination goal
is the spine M1→M4→M5→M6→M8; M3/M7/M9/M10 broaden into the rest of the library.

## M0 — Library map

Core files in `diffoptics/`:

- **`basics.py`** — data primitives: `Ray(o,d,wavelength)`, `Material`+`ior`,
  `normalize`, `Sampler.concentric_sample_disk`, `Transformation`.
- **`shapes.py`** — `Endpoint` base; `Screen` (textured plane for imaging).
- **`optics.py`** — the engine:
  - `Lensgroup` (build/`load`/`load_file`, `sample_ray*`, `trace*`, **`render`**,
    `prepare_mts`, `sample_ray_sensor`).
  - `Surface` → `Aspheric` (spherical = k=0), `BSpline`, `XYPolynomial`, `Mesh`.
- **`solvers.py`** — `Optimization` → `Adam`, `LM` (`jacobian`/`optimize`), `Adjoint`.

Class tree:

```
Endpoint ── Lensgroup, Screen
Surface  ── Aspheric, BSpline, XYPolynomial, Mesh
PrettyPrinter ── Ray, Material, Sampler, Transformation, Spectrum
Optimization ── Adam, LM, Adjoint
```

Forward data flow: **source rays → `trace` (per-surface `_refract`, sets `valid`) →
intersect sensor/screen → `render` splat → image**. Gradients flow back through all of
it (autograd, or the adjoint shortcut in M9).

## The core idea (M4 → M5)

**Deterministic** angular sampling (a fixed target grid) sends rays through the same
points every call → the irradiance shows a grid of dots and *aliases* onto surfaces
during optimization. **Monte-Carlo** jitters each ray and averages many draws (`spp`),
so features stop aliasing. That is exactly `caustic_pyramid.py`'s
`sample_ray(random=True)` + `spp` accumulation.

## Gotchas baked into the modules

- **Energy conservation** (M6): splat weights sum to 1 per ray, so `sum(I)` is
  independent of curvature — optimize the *distribution*, not the sum.
- **Focus is rugged** (M8): at exact focus the spot is a few pixels → a razor-thin loss
  well on a flat plateau (the aliasing pathology). Stay defocused for smooth convergence.
- **Identifiability** (M8 capstone): a centered source makes tilt nearly invisible (the
  focus distance absorbs it). Pick observable parameters / condition the problem.
- **Keep rays valid**: keep bundles within the aperture, else `render` returns an
  all-zero image with no gradient.

## Reading order into the core library

1. `basics.py` — `Ray`, `Material`, `normalize`, `Sampler`.
2. `optics.py` — `load`/`load_file`, `Aspheric`, `sample_ray*`, `trace`/`_trace`/`_refract`,
   **`render`**, then `prepare_mts`/`sample_ray_sensor`.
3. `solvers.py` — `LM.jacobian`/`optimize`, `Adam`, `Adjoint`.
4. `examples/caustic_pyramid.py` — the reference MC differentiable-illumination design.
