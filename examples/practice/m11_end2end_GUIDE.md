# M11 (reading guide): `end2end_edof_backward_tracing.py` — end-to-end Deep Lens

`examples/end2end_edof_backward_tracing.py` is the library's most advanced demo:
**jointly** train an optical design AND a neural post-processing network so the whole
imaging system (optics + software) is optimized together — "Deep Lens" / end-to-end.
The application is **extended depth of field (EDoF)**: encode depth-invariant blur
optically, then deblur with a network.

**Reading-only.** It needs heavy external deps (the bundled DeblurGANv2 network,
`utils_end2end.py`, a training image folder) and is written for CUDA. Study it; do not
expect to run it on CPU without significant effort.

## How it builds on the modules you did

| Concept | Where you learned it |
|---|---|
| Load lens + `prepare_mts` camera | M10 |
| `Screen` texture + backward `sample_ray_sensor` | M10 |
| Multi-pass MC averaging over aperture | M5 / M10 |
| Differentiable render + `.backward()` | M6 / M8 |
| Adjoint-style decoupling of optics vs network graph | M9 |

## The new idea

- The **screen sits at several depths** (EDoF): render each training image at random
  depths → depth-varying blur (the optical "code").
- A **network** (DeblurGANv2, loaded via `load_deblurganv2`) deblurs the rendered image.
- The **loss** compares the deblurred output to the sharp original; gradients flow back
  through the network AND through the differentiable optics to the lens parameters.
- Training couples optics params and network weights — the paper's
  adjoint back-prop (M9) is what makes tracing millions of rays affordable here.

## Key spots to read

- `render_single` / `render` (L42-): backward MTS render of a *batch* of images per
  wavelength — the M10 loop, batched.
- lens parameters marked differentiable + optimizer setup (search for `requires_grad`
  and `torch.optim`).
- the depth sampling and the network forward/deblur + loss.

## If you want to attempt a CPU cut-down

1. `device='cuda'` → `'cpu'`; `downsample_factor` up (smaller film).
2. Replace the DeblurGANv2 network with a tiny CNN (or skip the network and just
   optimize the optics to a fixed target) to drop the GAN dependency.
3. Use a couple of small images and 1-2 depths.

This is the capstone of the *whole library*: illumination/imaging + MC + differentiable
render + adjoint scaling + a learned back-end, all in one training loop.
