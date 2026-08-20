# M07 advanced (reading guide): `nikon.py` — full multi-surface design

`examples/nikon.py` scales M07's idea to a real camera lens: a **19-surface Nikon
Z 35 mm f/1.8** design, optimized over **many parameters at once**, across
**3 wavelengths** and **4 field angles**. Read it after doing `m07_lm_optimize`.
It runs on CPU (`device = torch.device('cpu')` at the top) but is slow.

## What is new vs M07

| M07 (asphere) | nikon.py |
|---|---|
| 1 surface, on-axis | 19 surfaces, 4 field views |
| 1 wavelength | 3 wavelengths (RGB) |
| 3 params (`c,k,ai`) | dozens: `c` of all non-stop surfaces + `k,ai` of 2 aspheres |
| `render()` = ps | `render()` = per-view, per-λ RMS accumulation |

## The parameter list (nikon.py:101-110)

```python
id_range = list(range(0, 19)); id_range.pop(lens.aperture_ind)  # skip the stop
id_asphere = [16, 17]
for i in id_asphere: lens.surfaces[i].ai = torch.Tensor([0.0])  # seed asphere coeffs
diff_names  = ['surfaces[{}].c'.format(i)  for i in id_range]
diff_names += ['surfaces[{}].k'.format(i)  for i in id_asphere]
diff_names += ['surfaces[{}].ai'.format(i) for i in id_asphere]
```

`do.LM` accepts any number of named leaves; the Jacobian is assembled column-by-column
(`solvers.py:110-124`). Same call as M07, just a longer `diff_names`.

## The merit function (nikon.py:55-88)

`render()` loops views x wavelengths, traces `sample_ray(..., entrance_pupil=True)`,
and sums squared RMS (`lens.rms(ps, squared=True)`). `func()` stacks all hit points;
`loss_func()` returns the scalar. LM drives them toward the axis with the same
`lambda y: 0.0 - y`.

## Run it yourself (CPU tweaks)

1. It uses `torch.cuda.Event` for timing (nikon.py:116-124) — on CPU replace with
   `import time; t0 = time.perf_counter()` … `print(time.perf_counter()-t0)`.
2. Lower `maxit` (e.g. 20) and `M` (e.g. 15) first — the full run is minutes on CPU.
3. Expect RMS to drop across all fields/wavelengths (see the `rms_*` plots it saves).

## Exercises

1. Optimize **only** the curvatures (`c`), then add `k`/`ai`. How much do the
   asphere coefficients buy you?
2. Freeze the stop-adjacent surfaces (remove them from `diff_names`). Does the
   design still converge?
3. Swap the merit to weight the outer field more heavily. Where does the spot budget go?

## Note on refractive index (nikon.py:14-31)

The paper figure assumed `air = 1.0`; the shipped table uses `1.000293`. The docstring
explains how to switch it in `diffoptics/basics.py` `MATERIAL_TABLE` to reproduce the paper.
