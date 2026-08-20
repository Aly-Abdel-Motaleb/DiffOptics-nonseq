"""
MODULE 8 - Full differentiable-illumination OPTIMIZATION (the payoff).

Everything combines: point source + spherical lens + Monte-Carlo irradiance +
autograd -> optimize the lens curvature c so the receiver irradiance matches a
TARGET distribution. This is the caustic_pyramid.py loop, minimal form.

Loop (caustic_pyramid.py:116-143):
  1. I = render(spp)                         # MC differentiable irradiance
  2. normalize I to the target's total energy
  3. L = mean((I - I_ref)**2)                # loss
  4. optimizer.zero_grad(); L.backward()     # gradients dL/dc
  5. optimizer.step()                        # update c

Target here: the irradiance produced by a KNOWN curvature c=0.036 -- an inverse
problem "recover the lens that made this illumination". Using a physically
reachable target guarantees the loss actually decreases.

Regime note: we use a DISTANT source (near-collimated) and stay DEFOCUSED (c below
focus). Exactly at focus the spot collapses to a few pixels and the loss becomes a
razor-thin well on a flat plateau -- the aliasing pathology this whole exercise is
about. Defocused = smooth convex bowl = reliable convergence.

READ BEFORE (tick when done):
  [ ] examples/caustic_pyramid.py:116-156  the optimization loop
  [ ] examples/caustic_pyramid.py:126-143  the core 5 lines you replicate
  [ ] (recap) diffoptics/optics.py:712     render, the differentiable irradiance

SOLVER COMPARISON (the extra experiment): the same problem is solved three ways --
Adam, do.LM on the raw pixels, and do.LM on a blurred residual. See the writeup at
the bottom of this file for what the numbers show.

Run: python m08_optimize_irradiance_practice.py
Expect: falling loss printout + I_start / I_final_* / I_target .png in ./m08_out/,
plus compare_solvers.png.
"""
import time

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common_setup import build_scene, save_dir, FILM
import diffoptics as do

out = save_dir("m08_out")

Z0 = 500.0                          # distant source -> near-collimated rays
o_pt = torch.Tensor([0.0, 0.0, -Z0])

C_START = 0.028
C_TARGET = 0.036

M = 40
R_src = 9.0
SPP = 8

_, wavelength, _ = build_scene(c=C_START)


def sample_pointsrc_ray(random=True):
    x = torch.linspace(-R_src, R_src, M)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    if random:
        p = 2 * R_src / M
        X = X + p * (torch.rand_like(X) - 0.5)
        Y = Y + p * (torch.rand_like(Y) - 0.5)
    targets = torch.stack((X, Y, torch.zeros_like(X)), axis=-1).reshape(-1, 3)
    o = o_pt.expand_as(targets).clone()
    d = do.normalize(targets - o_pt)
    return do.Ray(o, d, wavelength)


def render(lens, spp=SPP, seed=None):
    """Monte-Carlo irradiance: average spp jittered renders.

    seed=None -> fresh jitter every call (true MC, what Adam wants).
    seed=int  -> identical jitter every call. LM needs this: its damping loop
    re-evaluates the objective and compares `L_current >= L` (solvers.py:193),
    so MC noise between evaluations would look like a real change and make the
    damping thrash.
    """
    if seed is not None:
        torch.manual_seed(seed)
    I = torch.zeros(*FILM, device=lens.device)
    for _ in range(spp):
        I += lens.render(sample_pointsrc_ray())
    return I / spp


def gauss_blur(I, sigma):
    """Separable Gaussian blur, differentiable. Used to smooth the LM residual."""
    r = int(3 * sigma)
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=I.device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    J = I[None, None]
    J = F.conv2d(F.pad(J, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
    J = F.conv2d(F.pad(J, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1))
    return J[0, 0]


def make_target(c_target=C_TARGET, spp=64):
    """Reference illumination = irradiance from a KNOWN curvature. Optimization
    recovers c_target from a different start c, so the target is reachable and the
    loss genuinely drops.

    Scale: renormalize to MEAN pixel = 1, not to total energy = 1. LM's stopping
    tests are absolute (`|dloss| < 1e-8`, `|x_delta| < 1e-8`, solvers.py:246,250);
    with sum=1 over 65536 pixels the loss starts at ~1e-9 and LM quits on the
    second iteration having done nothing. Adam is unaffected either way since its
    step is gradient-normalized."""
    lens, _, _ = build_scene(c=c_target)
    with torch.no_grad():
        I = render(lens, spp=spp, seed=1234)
    return I / I.sum() * (FILM[0] * FILM[1])


def eval_loss(lens, I_ref):
    """Common yardstick: same fixed-jitter, high-spp render for both methods."""
    with torch.no_grad():
        I = render(lens, spp=64, seed=1234)
        I = I / I.sum() * I_ref.sum()
        return torch.mean((I - I_ref) ** 2).item(), I


# ---------------------------------------------------------------- Adam
def run_adam(I_ref, iters=500, lr=2e-3):
    lens, _, _ = build_scene(c=C_START, requires_grad=True)
    optimizer = torch.optim.Adam([lens.surfaces[0].c], lr=lr, amsgrad=True)

    hist_c, hist_L = [], []
    t0 = time.time()
    for it in range(iters):
        I = render(lens)                        # fresh jitter -> stochastic grad
        I = I / I.sum() * I_ref.sum()           # match total energy (like caustic)
        L = torch.mean((I - I_ref) ** 2)

        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        hist_c.append(lens.surfaces[0].c.item())
        hist_L.append(L.item())
        if it % 50 == 0:
            print(f"  [adam] it={it:3d}  loss={L.item():.4e}  c={hist_c[-1]:.5f}")
    return lens, hist_c, hist_L, time.time() - t0


# ---------------------------------------------------------------- LM
def run_lm(I_ref, maxit=50, lamb=1e-4, sigma=None):
    """do.LM is NOT a torch optimizer -- it owns the whole loop.

    Three API differences from Adam:
      1. `func()` returns the per-pixel vector y, not a scalar loss; LM builds the
         Jacobian from it and squares/means internally (solvers.py:155).
      2. `func_yref_y(y)` returns the residual y_ref - y.
      3. The parameter is passed BY NAME, not as a raw tensor: LM writes updates
         back via exec('self.lens.{name} = ...') in _change_parameters
         (solvers.py:257). Passing a bare tensor leaves the name-loop index `i`
         undefined and crashes at solvers.py:263.

    sigma: if set, both sides of the residual are Gaussian-blurred. This is what
    makes LM work here -- see the writeup at the bottom of the file.
    """
    lens, _, _ = build_scene(c=C_START, requires_grad=True)
    optimizer = do.LM(lens, ['surfaces[0].c'], lamb=lamb, option='diag')

    I_ref_fit = I_ref if sigma is None else gauss_blur(I_ref, sigma)
    hist_c = []

    def func():
        hist_c.append(lens.surfaces[0].c.item())
        I = render(lens, seed=1234)             # deterministic, see render()
        I = I / I.sum() * I_ref.sum()
        return I if sigma is None else gauss_blur(I, sigma)

    def func_yref_y(y):
        return I_ref_fit - y

    t0 = time.time()
    res = optimizer.optimize(func, func_yref_y, maxit=maxit)
    return lens, hist_c, list(res['ls']), time.time() - t0


if __name__ == "__main__":
    I_ref = make_target()
    plt.imsave(out + "I_target.png", I_ref.numpy(), cmap="inferno")

    lens0, _, _ = build_scene(c=C_START)
    L0, I0 = eval_loss(lens0, I_ref)
    plt.imsave(out + "I_start.png", I0.numpy(), cmap="inferno")
    print(f"start: c={C_START:.5f}  loss={L0:.4e}   (target c={C_TARGET})\n")

    runs = []   # (label, hist_c, hist_L, time, final eval loss)

    print("--- Adam ---")
    lens_a, c_a, L_a, t_a = run_adam(I_ref)
    La, Ia = eval_loss(lens_a, I_ref)
    plt.imsave(out + "I_final_adam.png", Ia.numpy(), cmap="inferno")
    runs.append(("Adam", c_a, L_a, t_a, La))

    print("\n--- LM, raw pixel residual ---")
    lens_r, c_r, L_r, t_r = run_lm(I_ref, maxit=50)
    Lr, Ir = eval_loss(lens_r, I_ref)
    plt.imsave(out + "I_final_lm_raw.png", Ir.numpy(), cmap="inferno")
    runs.append(("LM raw", c_r, L_r, t_r, Lr))

    print("\n--- LM, blurred residual (sigma=12 px) ---")
    lens_b, c_b, L_b, t_b = run_lm(I_ref, maxit=50, sigma=12.0)
    Lb, Ib = eval_loss(lens_b, I_ref)
    plt.imsave(out + "I_final_lm_blur.png", Ib.numpy(), cmap="inferno")
    runs.append(("LM blur", c_b, L_b, t_b, Lb))

    # ---- comparison ----
    # `iters` counts render() calls, not outer iterations: LM re-evaluates inside
    # its damping loop, so outer-iteration counts are not comparable to Adam's.
    print("\n" + "=" * 66)
    print(f"{'method':<9}{'renders':>9}{'time[s]':>10}{'final c':>11}"
          f"{'|c-c*|':>11}{'eval loss':>13}")
    print("-" * 66)
    print(f"{'(start)':<9}{0:>9}{0.0:>10.1f}{C_START:>11.5f}"
          f"{abs(C_START - C_TARGET):>11.2e}{L0:>13.4e}")
    for name, c_h, _, t, Lf in runs:
        print(f"{name:<9}{len(c_h):>9}{t:>10.1f}{c_h[-1]:>11.5f}"
              f"{abs(c_h[-1] - C_TARGET):>11.2e}{Lf:>13.4e}")
    print("=" * 66)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, c_h, L_h, t, _ in runs:
        style = '-' if name == "Adam" else 'o-'
        ax[0].semilogy(L_h, style, ms=4, label=f"{name} ({t:.0f}s)")
        ax[1].plot(c_h, style, ms=4, label=name)
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("training loss")
    ax[0].set_title("loss vs iteration (own objective)"); ax[0].legend()
    ax[1].axhline(C_TARGET, ls='--', c='k', label="target c")
    ax[1].set_xlabel("render call"); ax[1].set_ylabel("c")
    ax[1].set_title("curvature trajectory"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out + "compare_solvers.png", dpi=120)
    print("saved to", out)


# ============================================================================
# WRITEUP: Adam vs do.LM on this problem
#
# Measured (c: 0.02800 -> target 0.03600; eval loss = fixed-jitter spp=64 MSE):
#
#   method     renders   time[s]    final c     |c-c*|    eval loss
#   (start)          0       0.0    0.02800   8.00e-03   9.5368e+00
#   Adam           500      52.9    0.03609   9.05e-05   5.4312e-02
#   LM raw         150      17.1    0.02885   7.15e-03   9.0648e+00
#   LM blur         21       1.9    0.03600   9.16e-08   5.8505e-08
#
# 1. LM on RAW PIXELS FAILS. It moves c by only ~1.8e-5 per iteration and would
#    need ~450 iterations to cross an 8e-3 basin. Not a damping problem: every
#    step decreases the loss, so lamb keeps shrinking (solvers.py:222) and the
#    step converges to pure Gauss-Newton -- the tiny step IS the GN step.
#
#    Why GN under-steps: the step is (J^T r)/(J^T J), and here J^T J ~ 1.1e12
#    over ~9500 lit pixels, i.e. |dI/dc| ~ 1e4 per pixel while I itself is O(10).
#    Those derivatives come from rays sliding across pixel-BIN EDGES, not from
#    the smooth way the caustic actually grows. The linearization is only honest
#    over dc ~ 1e-5, so GN correctly refuses to step further -- based on a
#    Jacobian that describes aliasing rather than optics.
#
#    Adam is immune because its step is gradient-NORMALIZED: it only uses the
#    sign/direction, so an inflated Jacobian magnitude costs it nothing.
#
# 2. BLURRING THE RESIDUAL FIXES LM. With a sigma=12px Gaussian on both I and
#    I_ref, the edge-derivative spikes average away and J reflects the real
#    response. LM then takes ONE step of 8.7e-3 -- straight onto the target --
#    and converges in 4 outer iterations / 1.9 s. That is 28x faster than Adam
#    and ~1e6x lower final loss. This is why real caustic work (and
#    caustic_pyramid.py) optimizes coarse-to-fine.
#
# 3. Adam plateaus at eval loss 5.4e-2 and never does better: its stochastic MC
#    gradient makes c rattle around +-3e-4 of the optimum forever (see the right
#    panel of compare_solvers.png). LM's deterministic seed lets it stop exactly.
#
# 4. Two API traps hit along the way, both from LM's absolute stopping tests
#    (solvers.py:246,250) and its by-name parameter writeback (solvers.py:257):
#      - target normalized to sum=1 puts the loss at ~1e-9, already below the
#        |dloss| < 1e-8 threshold, so LM exits on iteration 2 having done
#        nothing. Fixed by normalizing to mean pixel = 1 (see make_target).
#      - do.LM(lens.surfaces[0].c, ...) does not work; the parameter must be the
#        STRING 'surfaces[0].c' (see run_lm).
#
# Takeaway: Adam is the robust default on raw aliased pixels. LM is dramatically
# better -- but only once the objective is smoothed enough that its Jacobian
# means something. Second-order methods buy speed by trusting the derivative
# magnitude, so they need a derivative worth trusting.
# ============================================================================
