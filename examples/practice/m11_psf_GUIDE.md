# M11 (reading guide): `render_psf.py` — PSF maps over field & depth

`examples/render_psf.py` renders the **point-spread function** (image of a point
source) across a grid of field positions and several object depths — the standard
way to characterize a camera lens. It reuses everything from M5 (MC disk sampling)
and M4 (bilinear splat). Runs on CPU with smaller settings.

## The pieces you already know

- **Disk sampling** (render_psf.py:50-54): `do.Sampler().concentric_sample_disk`
  draws uniform aperture points — exactly M5's "style B".
- **Point-source rays** (render_psf.py:56-70): one object point `o_obj`, directions to
  aperture samples — M1's "one origin, many directions".
- **Splat** (render_psf.py:28-48): the same bilinear `index_put(accumulate=True)` as
  `lens.render` (M4), written out inline so you can see it.

## The new structure

- `render(o_obj, M, rep_count)` (L72): MC-average `rep_count` passes of `M*M`
  aperture rays for one object point.
- `render_at_depth(z)` (L90): loop a grid of field points `(x,y)` at depth `z`,
  summing each PSF into one image — a field map of PSFs.
- outer loop over depths `zs` (L103) → one PSF map per depth.

## Run on CPU (tweaks)

1. `device = torch.device('cuda')` → `'cpu'`.
2. Shrink `film_size` (e.g. `[300,400]`), `M` (e.g. 201), keep `rep_count=1`.
3. Reduce the field grid `Nx,Ny` (e.g. 5,4) and use 2-3 depths.
4. Uses `do.Sampler` and `lens.trace_to_sensor(..., ignore_invalid=True)` — no other deps.

## Exercises

1. Render the on-axis PSF at best focus vs a defocused depth — see it blur.
2. Overlay PSFs across the field: where is the lens sharp/soft?
3. Swap in the Thorlabs/Nikon lens and compare PSF quality.

## Link back

This is the *characterization* counterpart to the *optimization* modules: M7 minimized
the spot (a single-point PSF proxy); this maps the PSF everywhere. Same rays, same splat.
