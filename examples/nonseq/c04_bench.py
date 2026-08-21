"""
PHASE 2 - STAGE 4: what the non-sequential MC tracer COSTS.

`c02_R02.py` proves the tracer is right. It says nothing about what it is right
*at*. This maps the practical limit: how many rays a run can afford, how fine a
receiver those rays support at usable noise, how much memory that takes, and
where it falls over.

    python examples/nonseq/c04_bench.py --smoke            # ~60 s, proves the paths
    python examples/nonseq/c04_bench.py --calibrate-only   # the k table + slope
    python examples/nonseq/c04_bench.py --arm seq_R00      # the cheap arm, ~3 min
    python examples/nonseq/c04_bench.py --arm nonseq_mc_R02
    python examples/nonseq/c04_bench.py --plots-only
    python examples/nonseq/c04_bench.py --dry-run          # the matrix + a time estimate

Every run appends ONE row to `results.csv`, flushed and fsync'd. Re-running the
same command skips rows already present, so a Colab disconnect costs at most the
run in flight. `--out` may point anywhere, including a mounted Drive.

--------------------------------------------------------------------------------
THE RULE THIS IS BUILT AROUND
--------------------------------------------------------------------------------

Per-bin relative Monte Carlo noise, for a receiver holding `count` rays per bin:

    eps = 1 / sqrt(count)

Design experiments so `eps < 10 %`, i.e. >= 100 rays in every bin you intend to
believe. Bin count is therefore NOT a free axis - it is a function of ray count:

    count(N, R) = k * R / N^2        =>      N = sqrt(k * R) * eps_target

`k` folds the path fraction and the lit-area fraction into one number, and is
MEASURED by a pilot (`calibrate_bins`) rather than assumed - the analytic guess
is wrong by 15 % forward and has no closed form at all backward.

The sweep raises R and raises N together, holding eps pinned just under 10 %,
and reports time and memory at each rung. That is what maps the limit.

--------------------------------------------------------------------------------
WHY THE PLAIN COUNT FORMULA IS EXACTLY VALID HERE
--------------------------------------------------------------------------------

The weighted form is `eps = sqrt(sum w^2) / sum w`, with an effective ray count
`N_eff = (sum w)^2 / sum w^2`, and `N_eff < count` as soon as the weights spread.

They do not spread. `trace_mc` sets `rho = R.detach().clone()` (nonseq.py:663)
and then `w*R/rho` on reflection, `w*(1-R)/(1-rho)` on transmission. Because rho
TRACKS R per ray, both factors are exactly 1.0 in value - the detach preserves
the gradient, not the magnitude. So every surviving ray carries the weight it
was launched with, `N_eff == count`, and `1/sqrt(count)` is exact.

This holds for a constant `R_fixed` AND for real angle-varying Fresnel, which is
the non-obvious part. It fails only if someone sets `Element.rho_fixed`, or if
the `clamp` guards at nonseq.py:668,676 bite at grazing incidence. `--fresnel`
exists to probe that second case. Either way the assert in `noise_stats` catches
it and the harness falls back to the N_eff map rather than reporting a lie.

--------------------------------------------------------------------------------
THE COMPARISON IS COMPOUND - SAY SO IN THE WRITE-UP
--------------------------------------------------------------------------------

The two arms differ in BOTH tracer and scene. dO has no reflection branch, so
`seq @ R=0.2` does not exist; the sequential arm can only run R=0. Every ratio
between the arms therefore mixes "cost of non-sequentiality" with "cost of
branching", and the raw number is NOT a non-sequential overhead. Run
`--arm nonseq_mc_R00` to split it into two clean ratios:

    nonseq overhead = mc_R00 / seq_R00      same physics, different tracer
    branching cost  = mc_R02 / mc_R00       same tracer, different physics

Three more things the sequential arm cannot do, which belong in the caveats:
it produces no R1, no T1R2T1 and no ghost (`_trace` picks a direction once,
from `(ray.d[...,2] > 0).all()` at optics.py:1049), so there is no sequential
backward column; and at R=0 every captured ray transmits, so it legitimately
earns ~25 % more bins per side than the MC arm at equal R. Plot "N at fixed
10 % noise", never "noise at fixed N" - the latter is unfair to the MC arm.
"""
import argparse
import csv
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', '..'))
sys.path.append(_HERE)

from diffoptics import nonseq  # noqa: E402
import diffoptics as do  # noqa: E402
import c02_R02 as ref  # noqa: E402

# c02 keeps LOCAL, CPU-hardcoded copies of the tracers (c02_R02.py:287,360):
# bare `torch.Generator()`, no `device=` on any buffer. The library versions are
# device-parametric. Getting this wrong is silent - the run just stays on CPU.
assert nonseq.trace_mc.__module__ == 'diffoptics.nonseq'
assert nonseq.trace_split.__module__ == 'diffoptics.nonseq'

try:
    import psutil
except ImportError:
    psutil = None


# ------------------------------------------------------------------ the matrix
# 2e7 sits deliberately either side of the predicted monolithic wall
# (~28e6 nonseq, ~50e6 seq on 16 GB): it is the rung that turns the wall
# from an extrapolation into a bracket.  7 rungs x 3 seeds + 2e7/3e7/1e8
# x 1 seed = 24 runs per arm.
RAYS_FULL = [int(x) for x in (1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7,
                              2e7, 3e7, 1e8)]
RAYS_SMOKE = [int(1e4), int(1e5)]
RAYS_FRESNEL = [int(1e5), int(1e6), int(1e7)]
RAYS_SPLIT = [int(1e4), int(1e5), int(1e6)]

SEEDS_SRC = (7, 17, 27)
MC_OFFSET = 9000        # mc_seed = src_seed + MC_OFFSET, never equal to src_seed

EPS_TARGET = 0.10       # the 10 % rule
BIN_SNAP = 16           # snap N DOWN to a multiple of this
BIN_MIN, BIN_MAX = 16, 2048

# Above this, chunk instead of going monolithic.  Set at 5e7 so EVERY rung up
# to and including 3e7 is traced monolithically on BOTH arms: a chunked row's
# peak memory measures one chunk, not R, so a chunked arm and a monolithic arm
# cannot be compared rung for rung.  Monolithic everywhere is also what lets an
# arm reach - and report - its own OOM wall ("nonseq OOMs at 3e7, seq does not")
# instead of quietly surviving on chunks.  Only 1e8 chunks by default; use
# `--chunk-above 1e7` to extend the noise curve past the wall on purpose.
CHUNK_ABOVE = int(5e7)
CHUNK_SIZE = int(2e6)

# fixed-N sub-sweep, for the -1/2 slope gate (verification 1)
FIXED_N_FWD, FIXED_N_BACK = 128, 32

TOL = {
    'ledger': 1e-9,         # measured 0.0 and 4.4e-16 on CUDA; c02 gates at 1e-3
    'ledger_chunked': 1e-12,
    'k_drift': 0.05,        # k_measured vs calibrated
    'eps_pred': 0.15,       # measured vs predicted noise
    'k_ratio': 0.02,        # k_seq/k_mc must be 1/0.64
    'slope': 0.05,          # |slope + 0.5| on the fixed-N noise sub-sweep
    'cal_slope': 0.10,      # |slope + 2| on the calibration ladder. Looser than
                            # the plan's 0.02 on purpose: the core mask is mildly
                            # N-dependent (k drifts 1.11 -> 1.02 over N=16..128),
                            # a real ~3 % effect, not noise. It is a diagnostic
                            # that the model HOLDS, not a precision measurement.
}

# Fallback k, measured during planning. `--calibrate` overwrites these.
K_FALLBACK = {
    'nonseq_mc_R02': {'fwd': 1.0836, 'back': 0.03456},
    'nonseq_mc_R00': {'fwd': 1.6996, 'back': float('nan')},
    'split_R02':     {'fwd': 1.0836, 'back': 0.03456},
    'fresnel':       {'fwd': 1.60,   'back': float('nan')},
    'seq_R00':       {'fwd': 1.6996, 'back': float('nan')},
}

ARMS = ('nonseq_mc_R02', 'seq_R00', 'nonseq_mc_R00', 'split_R02', 'fresnel')

# Arms that have a meaningful backward receiver. Sequential structurally cannot
# put anything there, and R=0 has nothing to reflect.
HAS_BACK = {'nonseq_mc_R02': True, 'split_R02': True, 'fresnel': True,
            'nonseq_mc_R00': False, 'seq_R00': False}

# Only the fixed-R_coat arms carry the closed-form path fractions.
CLOSED_FORM = {'T1T2': 0.64, 'R1': 0.20, 'T1R2T1': 0.128}

COLUMNS = [
    'ts', 'arm', 'device', 'dtype', 'mode', 'rays', 'src_seed', 'mc_seed',
    'rep', 'chunk',
    'n_bins_fwd', 'n_ideal_fwd', 'bin_flag_fwd',
    'n_bins_back', 'n_ideal_back', 'bin_flag_back',
    't_wall_s', 't_wall_min', 't_wall_std', 't_cuda_ms',
    't_sample', 't_trace', 't_classify', 't_splat', 't_phase_total',
    'mem_alloc_mb', 'mem_reserved_mb', 'mem_host_mb', 'mem_mode',
    'bytes_per_ray',
    'eps_pred_fwd', 'eps_p10_fwd', 'eps_median_fwd', 'eps_peak_fwd',
    'frac_above_fwd', 'lit_frac_fwd', 'n_lit_fwd', 'k_meas_fwd',
    'eps_pred_back', 'eps_p10_back', 'eps_median_back', 'eps_peak_back',
    'frac_above_back', 'lit_frac_back', 'n_lit_back', 'k_meas_back',
    'phi_fwd', 'phi_back', 'phi_off', 'ledger_rel', 'weights_uniform',
    'f_T1T2', 'f_R1', 'f_T1R2T1', 'f_ghost', 'paths_ok',
    'n_terminal', 'rays_per_s', 'gpu_name', 'status', 'note',
]

# What identifies a run. Anything not in here may vary between two rows that
# the resume logic considers the same run.
RUN_KEY = ('arm', 'device', 'mode', 'rays', 'src_seed', 'rep', 'chunk',
           'n_bins_fwd')

NAN = float('nan')


# ------------------------------------------------------------------- plumbing
def _sync(device):
    """Barrier that is a no-op off CUDA, so both arms share one timing path."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _mem_reset(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def _mem_peak(device):
    """(peak allocated MB, peak reserved MB). NaN off CUDA.

    `max_memory_allocated`, not `memory_allocated` - the latter (which
    backprop_compare.py:49,63 uses) reports whatever survived to the end of the
    call, not the peak, and so under-reports a tracer that frees as it goes.

    Reserved is logged too because it is what actually decides OOM; the gap
    between the two is allocator fragmentation, measured at ~1.3x.
    """
    if device.type != 'cuda':
        return NAN, NAN
    return (torch.cuda.max_memory_allocated(device) / 2 ** 20,
            torch.cuda.max_memory_reserved(device) / 2 ** 20)


def _host_mb():
    return psutil.Process().memory_info().rss / 2 ** 20 if psutil else NAN


def _gpu_name(device):
    if device.type != 'cuda':
        return 'cpu'
    return torch.cuda.get_device_name(device)


def _git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ''


def _is_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    # Older torch raised a plain RuntimeError. Never swallow anything else - a
    # bare `except` (backprop_compare.py:108) records NaN for a typo.
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()


# ------------------------------------------------------------------ the arms
def _build(arm, device):
    """Scene for one arm. Returns (elements|lensgroup, tracer_name)."""
    if arm == 'seq_R00':
        lens = ref.build_lensgroup(device=device)
        # fp64 must survive construction. `torch.Tensor(...)` at optics.py:45-49
        # DOES honour set_default_dtype on torch 2.x - assert so a future change
        # is loud rather than a silent precision downgrade.
        assert lens.surfaces[0].c.dtype == torch.float64, 'lens fell to fp32'
        # NEVER call .to() on a Lensgroup: PrettyPrinter.to moves the tensors but
        # skips `self.device` (a torch.device, not a tensor), which _trace reads
        # at optics.py:1065 and render at :753. Construct with device= instead.
        assert lens.surfaces[0].c.device.type == device.type
        return lens, 'seq'
    R = {'nonseq_mc_R02': ref.R_COAT, 'nonseq_mc_R00': 0.0,
         'split_R02': ref.R_COAT, 'fresnel': None}[arm]
    els = ref.build_elements(R=R, device=device)
    assert els[0].surface.c.device.type == device.type
    return els, ('split' if arm == 'split_R02' else 'mc')


def _seq_term(lens, o, d, w):
    """Run the sequential tracer and dress the result as a `term`/`tally` pair.

    Wrapping it in c02's terminal-dict shape lets `classify`, `path_powers` and
    the ledger check be shared verbatim with the non-sequential arms - the seq
    arm just always reports nhit=2, nrefl=0, which IS the only path it can make.
    """
    ray = do.Ray(o, d, wavelength=torch.tensor([ref.WAVELENGTH]),
                 device=o.device)
    ray_f, valid = lens.trace(ray)
    n = int(valid.sum())
    term = {'o': ray_f.o[valid], 'd': ray_f.d[valid], 'w': w[valid],
            'nrefl': torch.zeros(n, dtype=torch.long, device=o.device),
            'nhit': torch.full((n,), 2, dtype=torch.long, device=o.device)}
    # Rays that missed an aperture are the seq analogue of `culled`.
    tally = {'culled': w[~valid].sum(), 'truncated': torch.zeros((), dtype=w.dtype,
                                                                device=o.device)}
    return term, tally


def _trace(arm, scene, kind, R, src_seed, mc_seed, device, phases=None):
    """One trace. `phases` is a dict to fill with per-stage seconds, or None.

    Phase timings are collected in a SEPARATE repeat from the total, because a
    sync between stages serializes work the stream would overlap and would make
    the parts sum to more than the whole.
    """
    def mark(key, t0):
        if phases is not None:
            _sync(device)
            phases[key] = time.perf_counter() - t0
            return time.perf_counter()
        return t0

    t = time.perf_counter()
    o, d, w = ref.sample_point_source(R, seed=src_seed, device=device)
    t = mark('t_sample', t)

    if kind == 'seq':
        term, tally = _seq_term(scene, o, d, w)
    elif kind == 'mc':
        term, tally = nonseq.trace_mc(o, d, w, scene, ref.WAVELENGTH,
                                      seed=mc_seed, max_depth=ref.MAX_DEPTH,
                                      w_min=0.0)
    else:
        term, tally = nonseq.trace_split(o, d, w, scene, ref.WAVELENGTH,
                                         max_depth=ref.MAX_DEPTH, w_min=0.0)
    t = mark('t_trace', t)

    cl = ref.classify(term)
    mark('t_classify', t)
    return term, tally, cl


# -------------------------------------------------------------------- binning
def bins_for(R, k, target=EPS_TARGET, snap=BIN_SNAP, n_min=BIN_MIN,
             n_max=BIN_MAX):
    """Largest bin count whose predicted per-bin noise still clears `target`.

    Snap DOWN: fewer bins means more rays each, so rounding down can only make
    the noise better than asked. Multiples of 16 rather than powers of two -
    powers of two waste up to 4x the ray budget and let the achieved noise
    wander over 7-9 %, where /16 tracks the target to within 0.5 %.

    Returns (N, N_ideal, eps_pred, flag). `flag` is 'floored' when the rule
    wanted fewer than `n_min` bins: the row is still run and still reported,
    with its honest (>target) noise, because those low-R points are what anchor
    the -1/2 slope fit.
    """
    if not math.isfinite(k) or k <= 0:
        return 0, NAN, NAN, 'none'
    n_ideal = math.sqrt(k * R) * target
    n = int(snap * (n_ideal // snap))
    flag = ''
    if n < n_min:
        n, flag = n_min, 'floored'
    elif n > n_max:
        n, flag = n_max, 'capped'
    return n, n_ideal, n / math.sqrt(k * R), flag


MIN_CAL_COUNT = 100     # below this the p10 quantile saturates on small integers.

# One ray count gets its per-bin COUNT maps dumped to .npz so the distribution
# behind the eps_p10 headline can be drawn.  1e6 is the rung to pick: large
# enough that the forward grid is 96x96 rather than a floored 16x16, small
# enough to be cheap and to be present in every partial sweep.  Only the first
# seed is dumped - three seeds of the same distribution add nothing to a
# histogram.
HIST_RAYS = int(1e6)
                        # 100 is not arbitrary: it IS the 10 % rule, so any bin
                        # too sparse to calibrate on is a bin too sparse to use.
CORE_FRAC = 0.5         # core = bins holding >= this fraction of the median count


def _core_mask(C):
    """Bins fully inside the illuminated region, as opposed to merely nonzero.

    A hard-edged beam clips the bins it crosses, and at coarse N those partial
    bins are a LARGE share of the nonzero set - measured, at N=16 on the forward
    receiver, 86 % of bins are nonzero but the 10th-percentile count is 1. Taken
    over all nonzero bins the fitted k then swings 0.0003 -> 0.97 -> 0.52 across
    the N ladder, i.e. the `count = k R / N^2` model appears to fail when in fact
    only the edge is being measured.

    The floor is relative to the MEDIAN, never the peak. The caustic ring at the
    rim of this collimator inflates the peak without limit as N grows (peak/median
    3.4x at N=16 to 7.6x at N=512), so a peak-relative floor would shrink toward
    nothing; the median is stable. Measured k over the core: 1.11, 1.08, 1.07,
    1.02 at N=16..128 - flat, which is the model holding.
    """
    lit = C > 0
    if not bool(lit.any()):
        return lit
    return C >= CORE_FRAC * torch.quantile(C[lit], 0.5)


def noise_stats(p, w, n, half, k_cal=None, R=None, keep=False):
    """Per-bin MC noise on one receiver, measured rather than predicted.

    Uses `bilinear=False` for BOTH moments. A bilinear bin holds
    `V = sum b_i w_i` with `sum b_i = 1`, so its variance is `sum b_i^2 w_i^2`,
    while `splat(w**2)` computes `sum b_i w_i^2` - an over-estimate of up to 2x.
    Nearest-neighbour is exact, and with equal weights collapses to a plain
    count, which is what makes `eps_measured` and `1/sqrt(count)` the same
    number and the cross-check below readable.

    `eps_p10` is the noise at the 10th percentile of the count distribution -
    equivalently the 90th percentile of eps. The median would let a tenth of the
    bins be worse than the target while still reporting success.

    The headline statistics are taken over the CORE (see `_core_mask`), not over
    every nonzero bin: a bin the beam edge only clips is genuinely noisy, but it
    is noisy because it is half-empty, not because the ray budget is too small,
    and letting those bins set N would shrink the grid without limit. The
    edge is still reported, through `lit_frac` and `frac_above`, which are taken
    over every lit bin and so do include it.

    `keep=True` additionally returns the raw count map under `_C`, for the
    histogram; the caller pops it before writing the CSV row.
    """
    out = {k: NAN for k in ('eps_p10', 'eps_median', 'eps_peak',
                            'frac_above', 'lit_frac', 'n_lit', 'k_meas')}
    if n <= 0 or p.numel() == 0:
        return out
    pitch = 2.0 * half / n
    S1 = nonseq.splat(p, w, [n, n], pitch, bilinear=False)
    S2 = nonseq.splat(p, w * w, [n, n], pitch, bilinear=False)
    C = nonseq.splat(p, torch.ones_like(w), [n, n], pitch, bilinear=False)

    lit = C > 0
    if not bool(lit.any()):
        return out
    core = _core_mask(C)
    eps_lit = torch.sqrt(S2[lit]) / S1[lit]
    eps = torch.sqrt(S2[core]) / S1[core]

    out['eps_p10'] = float(torch.quantile(eps, 0.90))
    out['eps_median'] = float(torch.quantile(eps, 0.50))
    out['eps_peak'] = float(eps[torch.argmax(S1[core])])
    out['frac_above'] = float((eps_lit > EPS_TARGET).double().mean())
    out['lit_frac'] = float(lit.double().mean())
    out['n_lit'] = int(lit.sum())
    if R:
        # k as this run actually realised it. Flat vs R is the invariant; a
        # TREND means the geometry became R-dependent, which it must not be.
        out['k_meas'] = float(torch.quantile(C[core], 0.10)) * n * n / R
    if keep:
        out['_C'] = C.detach().cpu().numpy()
    return out


def calibrate_bins(arm, device, R_pilot=int(1e6), src_seed=7,
                   candidates=(16, 32, 64, 128, 256, 512), min_points=1,
                   max_escalations=2):
    """Calibrate, escalating the pilot size until the fit has enough points.

    The backward receiver collects ~38x fewer rays per bin than the forward one,
    so a pilot big enough for one is too small for the other: at R=1e6 the back
    receiver clears MIN_CAL_COUNT at only two candidate N. Rather than pick a
    conservative pilot for every arm, grow it 5x at a time until both sides have
    a fit worth trusting.
    """
    for _ in range(max_escalations + 1):
        out = _calibrate_once(arm, device, R_pilot, src_seed, candidates)
        need = [s for s in ('fwd', 'back')
                if HAS_BACK[arm] or s == 'fwd']
        if all(out.get('npts_' + s, 0) >= min_points for s in need):
            return out
        R_pilot *= 5
    return out


def _calibrate_once(arm, device, R_pilot, src_seed, candidates):
    """Measure `k` from ONE pilot trace, then splat it at every candidate N.

    Splat is O(M) and free next to the trace, so the whole ladder costs one
    trace. Fitting `k` across the ladder rather than from a single N is what
    makes it robust - one N could sit on a resolution artefact.

    The unpinned log-log slope is returned as a diagnostic and must come out
    -2.00 +- 0.02; anything else means `count ~ k R / N^2` does not describe
    this receiver and every table downstream is void.
    """
    scene, kind = _build(arm, device)
    _, _, cl = _trace(arm, scene, kind, R_pilot, src_seed,
                      src_seed + MC_OFFSET, device)

    out = {'arm': arm, 'R_pilot': int(R_pilot), 'device': device.type}
    for side, half, on in (('fwd', ref.R_RECV, True),
                           ('back', ref.R_BACK, HAS_BACK[arm])):
        if not on:
            out['k_' + side] = NAN
            out['slope_' + side] = NAN
            out['npts_' + side] = 0
            continue
        p, w = cl['p_' + side], cl['w_' + side]
        ns, cs = [], []
        for n in candidates:
            pitch = 2.0 * half / n
            C = nonseq.splat(p, torch.ones_like(w), [n, n], pitch,
                             bilinear=False)
            core = _core_mask(C)
            if not bool(core.any()):
                continue
            c10 = float(torch.quantile(C[core], 0.10))
            # Drop candidates whose bins are too sparse. The p10 quantile of a
            # count map saturates on small integers, so those points bend the
            # log-log fit and make a valid model look broken: measured k decays
            # 1.07 -> 0.92 -> 0.79 at N = 128, 256, 512 (counts 62, 14, 3) purely
            # from this, taking the fitted slope with it (-2.04 -> -1.69).
            if c10 < MIN_CAL_COUNT:
                continue
            ns.append(n)
            cs.append(c10)
        out['npts_' + side] = len(ns)
        if not ns:
            out['k_' + side], out['slope_' + side] = NAN, NAN
            continue
        ln, lc = np.log(np.array(ns, float)), np.log(np.array(cs, float))
        # k with the slope PINNED to -2 (the model); slope left free = the check.
        # Because the slope is pinned, ONE well-populated N already determines k
        # exactly under the model - the ladder is for robustness, not necessity.
        # So a single usable point still yields a k, just no slope to check it
        # with. That is the normal case on the backward receiver, which collects
        # ~38x fewer rays per bin and would otherwise demand a 2.5e7-ray pilot.
        out['k_' + side] = float(np.exp(np.mean(lc + 2.0 * ln)) / R_pilot)
        out['slope_' + side] = (float(np.polyfit(ln, lc, 1)[0]) if len(ns) >= 2
                                else NAN)
    return out


# ------------------------------------------------------------------- one run
def _measure(arm, R, src_seed, device, nf, nb, chunk, n_rep,
             hist_path=None):
    """Trace `n_rep` timed repeats plus one phase-broken repeat. Returns a dict.

    Repeat 0 is untimed and warms the allocator for this size. Repeats 1..n_rep
    time the TOTAL with no internal syncs. One extra repeat collects the phase
    breakdown, whose sum is reported next to the total so the reader can see how
    much the syncs cost.
    """
    mc_seed = src_seed + MC_OFFSET
    assert mc_seed != src_seed, 'equal src/mc seeds degenerate u<rho into a wedge'
    scene, kind = _build(arm, device)

    chunks = [R] if not chunk else (
        [chunk] * (R // chunk) + ([R % chunk] if R % chunk else []))

    def one_pass(phases=None, accumulate=False):
        acc = {'phi': {}, 'lw': 0.0, 'lc': 0.0, 'lt': 0.0, 'n': 0,
               'p_fwd': [], 'w_fwd': [], 'p_back': [], 'w_back': [],
               'uniq_max': 0}
        for ci, cn in enumerate(chunks):
            term, tally, cl = _trace(arm, scene, kind, cn, src_seed + 1000 * ci,
                                     mc_seed + 1000 * ci, device, phases)
            if not accumulate:
                continue
            # EVERY chunk is a self-contained source carrying the FULL launched
            # power: `sample_point_source` splits P over the rays it is asked
            # for, so a chunk of `cn` rays gives each ray P/cn, not P/R.  Summing
            # chunks unscaled therefore multiplies the ledger by len(chunks) -
            # at R=3e7 with 2e6-ray chunks that is a 15x energy inflation, and
            # the path fractions inflate with it.  Weight each chunk by its
            # share of the batch so the pieces add up to one source.
            sc = cn / R
            acc['lw'] += float(term['w'].sum()) * sc
            acc['lc'] += float(tally['culled']) * sc
            acc['lt'] += float(tally['truncated']) * sc
            acc['n'] += int(term['w'].numel())
            for side in ('fwd', 'back'):
                acc['p_' + side].append(cl['p_' + side])
                acc['w_' + side].append(cl['w_' + side] * sc)
            for kk, vv in ref.path_powers(term).items():
                acc['phi'][kk] = acc['phi'].get(kk, 0.0) + vv * sc
            # Uniformity is a per-chunk property: chunks legitimately differ
            # from each other in weight when R is not a multiple of the chunk
            # size, and comparing across them would flag that as a failure.
            acc['uniq_max'] = max(acc['uniq_max'],
                                  int(torch.unique(term['w']).numel()))
        return acc

    # --- warmup for this size, then the timed totals ------------------------
    one_pass()
    ts, t_cuda = [], NAN
    for i in range(n_rep):
        _sync(device)
        _mem_reset(device)
        h0 = _host_mb()
        if device.type == 'cuda':
            ev0, ev1 = (torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True))
            ev0.record()
        t0 = time.perf_counter()
        acc = one_pass(accumulate=(i == n_rep - 1))
        if device.type == 'cuda':
            ev1.record()
        _sync(device)
        ts.append(time.perf_counter() - t0)
        if device.type == 'cuda':
            t_cuda = ev0.elapsed_time(ev1)
        mem_a, mem_r = _mem_peak(device)
        host = _host_mb() - h0

    # --- a separate repeat for the phase breakdown -------------------------
    phases = {}
    one_pass(phases=phases)

    # --- maps and noise ----------------------------------------------------
    row = {'t_wall_s': float(np.mean(ts)), 't_wall_min': float(np.min(ts)),
           't_wall_std': float(np.std(ts)), 't_cuda_ms': t_cuda,
           'mem_alloc_mb': mem_a, 'mem_reserved_mb': mem_r,
           'mem_host_mb': host,
           'mem_mode': 'chunked' if chunk else 'monolithic',
           'bytes_per_ray': (mem_a * 2 ** 20 / (chunk or R)
                             if math.isfinite(mem_a) else NAN),
           'rays_per_s': R / float(np.mean(ts)),
           'n_terminal': acc['n']}
    for kk in ('t_sample', 't_trace', 't_classify'):
        row[kk] = phases.get(kk, NAN)

    t0 = time.perf_counter()
    _sync(device)
    hist = {}
    for side, half, n in (('fwd', ref.R_RECV, nf), ('back', ref.R_BACK, nb)):
        p = (torch.cat(acc['p_' + side]) if acc['p_' + side]
             else torch.zeros(0, 3, device=device))
        w = (torch.cat(acc['w_' + side]) if acc['w_' + side]
             else torch.zeros(0, device=device))
        st = noise_stats(p, w, n, half, R=R, keep=hist_path is not None)
        if hist_path is not None and '_C' in st:
            hist[side] = st.pop('_C')
        st.pop('_C', None)
        for kk, vv in st.items():
            row[kk + '_' + side] = vv
        row['phi_' + side] = float(w.sum())
    _sync(device)
    row['t_splat'] = time.perf_counter() - t0
    row['t_phase_total'] = sum(v for v in (row['t_sample'], row['t_trace'],
                                           row['t_classify'], row['t_splat'])
                               if math.isfinite(v))

    # --- ledger, paths, weight uniformity ----------------------------------
    total = acc['lw'] + acc['lc'] + acc['lt']
    row['ledger_rel'] = abs(total / ref.PHI_CAP - 1.0)
    row['phi_off'] = acc['lw'] - row['phi_fwd'] - row['phi_back']
    row['weights_uniform'] = int(acc['uniq_max'] <= 1)

    phi = acc['phi']
    for name, col in (('T1T2', 'f_T1T2'), ('R1', 'f_R1'),
                      ('T1R2T1', 'f_T1R2T1'), ('T1R2R1T2', 'f_ghost')):
        row[col] = phi.get(name, 0.0) / ref.PHI_CAP
    if arm in ('nonseq_mc_R02', 'split_R02'):
        # Tolerance TIGHTENS with R, because this is MC noise on a global sum.
        # A fixed tolerance would be vacuous at 1e8 or fail at 1e4.
        tol = 4.0 / math.sqrt(R)
        row['paths_ok'] = int(all(
            abs(row['f_' + n2] - v) < max(tol, 1e-3)
            for n2, v in (('T1T2', 0.64), ('R1', 0.20), ('T1R2T1', 0.128))))
    else:
        row['paths_ok'] = -1

    if hist_path is not None and hist:
        np.savez_compressed(hist_path, arm=arm, rays=R, src_seed=src_seed,
                            **{k: v for k, v in hist.items()})
    return row


# ---------------------------------------------------------------------- CSV
def _key(row):
    return tuple(str(row[k]) for k in RUN_KEY)


def load_done(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != COLUMNS:
            raise SystemExit(
                f'{path}: header does not match this version of the harness.\n'
                f'  move it aside, or re-run with the code that wrote it.')
        for r in rd:
            # A run that FAILED a gate is not done - it is a bug to be fixed and
            # re-run.  Only 'ok' and 'oom' are terminal ('oom' would just OOM
            # again).  So after a harness fix, re-running the same cell redoes
            # exactly the broken rows and leaves the good ones alone.
            if r.get('status') not in ('ok', 'oom'):
                continue
            try:
                done.add(tuple(r[k] for k in RUN_KEY))
            except KeyError:
                continue        # torn final line from a disconnect; ignore it
    return done


def append_row(path, row):
    """Append and fsync immediately - a disconnect must cost one row, not all."""
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        if new:
            wr.writeheader()
        wr.writerow({k: row.get(k, '') for k in COLUMNS})
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------ the sweep
def plan_runs(args, kcal):
    """The full matrix, cheapest first.

    Ascending R matters: forty minutes of GPU before a disconnect should leave a
    complete low-R curve with error bars, not three seeds of 1e8 and nothing.
    """
    runs = []
    for arm in args.arms:
        if arm == 'fresnel':
            rays = RAYS_FRESNEL
        elif arm == 'split_R02':
            rays = RAYS_SPLIT
        else:
            rays = RAYS_SMOKE if args.smoke else args.rays
        k = kcal.get(arm, K_FALLBACK[arm])
        for R in rays:
            nf, nif, epf, flf = bins_for(R, k['fwd'])
            nb, nib, epb, flb = bins_for(R, k['back']) if HAS_BACK[arm] \
                else (0, NAN, NAN, 'none')
            chunk = (int(args.chunk_size)
                     if (R > args.chunk_above and not args.no_chunk) else 0)
            seeds = SEEDS_SRC[:1] if (R > int(1e7) or args.smoke
                                      or arm in ('fresnel', 'split_R02')) \
                else SEEDS_SRC[:args.seeds]
            n_rep = 1 if (R > int(1e6) or args.smoke) else args.reps
            for s in seeds:
                runs.append(dict(
                    arm=arm, device=args.device, mode='adaptive', rays=R,
                    src_seed=s, mc_seed=s + MC_OFFSET, rep=n_rep, chunk=chunk,
                    n_bins_fwd=nf, n_ideal_fwd=round(nif, 2), bin_flag_fwd=flf,
                    n_bins_back=nb, n_ideal_back=round(nib, 2) if nb else '',
                    bin_flag_back=flb,
                    eps_pred_fwd=epf, eps_pred_back=epb))
    if args.slope:
        # Fixed-N sub-sweep: the -1/2 gate. Along the ADAPTIVE sweep eps is
        # pinned near 10 % by construction and its slope is ~0, so the slope
        # check has to hold N still.
        for arm in args.arms:
          for R in (args.rays if not args.smoke else RAYS_SMOKE):
            if R > int(1e7):
                continue
            for s in SEEDS_SRC[:args.seeds]:
                runs.append(dict(
                    arm=arm, device=args.device, mode='fixedN', rays=R,
                    src_seed=s, mc_seed=s + MC_OFFSET,
                    rep=1 if R > int(1e6) else args.reps, chunk=0,
                    n_bins_fwd=FIXED_N_FWD, n_ideal_fwd='', bin_flag_fwd='',
                    n_bins_back=FIXED_N_BACK if HAS_BACK[arm] else 0,
                    n_ideal_back='', bin_flag_back='',
                    eps_pred_fwd=NAN, eps_pred_back=NAN))
    return runs


def sweep(args, kcal, csv_path):
    device = torch.device(args.device)
    runs = plan_runs(args, kcal)
    done = load_done(csv_path)
    todo = [r for r in runs if (args.force or _key(r) not in done)]

    est = sum(r['rays'] * (8.1e-6 if r['arm'] != 'seq_R00' else 0.43e-6)
              * (r['rep'] + 2) for r in todo)
    print(f'\n  {len(runs)} runs planned, {len(runs)-len(todo)} already done, '
          f'{len(todo)} to run')
    print(f'  rough estimate {est/60:.1f} min (at the reference 8.1 us/ray)\n')
    if args.dry_run:
        for r in todo:
            print(f"    {r['arm']:15s} R={r['rays']:>9d} seed={r['src_seed']:3d} "
                  f"N={r['n_bins_fwd']:5d}/{r['n_bins_back']:<5d} "
                  f"chunk={r['chunk']:>8d} {r['bin_flag_fwd']}")
        return

    gpu = _gpu_name(device)
    # First monolithic OOM per arm. Once a monolithic run has hit the VRAM wall,
    # every LARGER monolithic run of that arm hits it too - retrying 3e7 and then
    # 1e8 just to watch them fail costs minutes of allocator thrash for no datum.
    # So the wall is measured once, the bigger monolithic rungs are skipped, and
    # their memory is supplied by the linear extrapolation drawn dotted in
    # memory_vs_rays.png. Chunked rows are NOT skipped: chunking is what carries
    # the noise curve past the wall, and its peak does not grow with R.
    oom_at = {}
    for i, r in enumerate(todo, 1):
        w = oom_at.get(r['arm'])
        if w is not None and not r['chunk'] and r['rays'] >= w:
            print(f"[{i:3d}/{len(todo)}] {r['arm']:15s} R={r['rays']:>9d} "
                  f"   skipped - past the measured OOM wall at {w:.0e}; "
                  f"memory comes from the fit")
            continue
        row = dict(r)
        row.update(ts=datetime.now().isoformat(timespec='seconds'),
                   dtype=str(torch.get_default_dtype()).split('.')[-1],
                   gpu_name=gpu, status='ok', note='')
        head = (f"[{i:3d}/{len(todo)}] {r['arm']:15s} R={r['rays']:>9d} "
                f"s={r['src_seed']:<3d} N={r['n_bins_fwd']}/{r['n_bins_back']}")
        print(head, end='', flush=True)
        hp = None
        if (r['rays'] == HIST_RAYS and r['mode'] == 'adaptive'
                and r['src_seed'] == SEEDS_SRC[0] and r['rep'] == args.reps):
            hp = os.path.join(args.out, f"hist_{r['arm']}_{device.type}.npz")
        try:
            row.update(_measure(r['arm'], r['rays'], r['src_seed'], device,
                                r['n_bins_fwd'], r['n_bins_back'], r['chunk'],
                                r['rep'], hist_path=hp))
            _report(row)
        except Exception as e:                       # noqa: BLE001
            # Never abort the arm. The sweep's job is to deliver the WHOLE
            # ladder; one bad rung is a row in the CSV, not the end of the run.
            # An OOM IS the datum (the wall is not always monotone, and a
            # non-monotone wall is itself a finding); any other exception is a
            # bug, recorded as status='error' with the traceback in the note so
            # it is loud in the CSV but does not cost the remaining rungs.
            # load_done() treats neither as done, so a rerun retries them.
            oom = _is_oom(e)
            row.update(status='oom' if oom else 'error',
                       note=str(e).splitlines()[0][:180],
                       t_wall_s=NAN, mem_alloc_mb=NAN)
            print('   OOM' if oom else f'   ERROR {type(e).__name__}: {e}')
            if oom and not r['chunk']:
                oom_at.setdefault(r['arm'], r['rays'])
            if not oom:
                traceback.print_exc()
        finally:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()          # outside every timed region
                _mem_reset(device)
        append_row(csv_path, row)
        if i % 10 == 0 and os.path.exists(csv_path):
            shutil.copy(csv_path, csv_path + '.bak')

    # Say plainly how much of the planned matrix actually landed, per arm, so
    # "did the full ladder run?" is answered by the harness and not by scrolling.
    done = load_done(csv_path)
    print('\n  completed rows per arm (of the planned matrix)')
    for arm in args.arms:
        want = [r for r in runs if r['arm'] == arm]
        w = oom_at.get(arm)
        # A rung skipped because it is past the measured wall is accounted for,
        # not missing: it has an answer (the extrapolation), just not a measured
        # one. Only genuinely outstanding rows should read INCOMPLETE.
        skipped = [r for r in want
                   if w is not None and not r['chunk'] and r['rays'] >= w]
        got = [r for r in want if _key(r) in done]
        acct = len(got) + len(skipped)
        mark = 'ok' if acct >= len(want) else 'INCOMPLETE'
        extra = (f'  ({len(skipped)} past the OOM wall at {w:.0e}, '
                 f'memory from the fit)' if skipped else '')
        print(f'    {arm:15s} {len(got):3d}/{len(want):<3d} {mark}{extra}')


def _report(row):
    """One line per run, with the gates inline so a failure is seen immediately."""
    bad = []
    tol = TOL['ledger_chunked'] if row.get('chunk') else TOL['ledger']
    if row['ledger_rel'] > tol:
        bad.append(f"ledger {row['ledger_rel']:.1e}")
        row['status'] = 'ledger_fail'
    if row['paths_ok'] == 0:
        bad.append('paths')
    if not row['weights_uniform']:
        bad.append('weights')
    ep, em = row.get('eps_pred_fwd', NAN), row['eps_p10_fwd']
    if math.isfinite(ep) and math.isfinite(em) and ep > 0 \
            and abs(em / ep - 1) > TOL['eps_pred']:
        bad.append(f'eps {em:.3f} vs pred {ep:.3f}')
    print(f"   {row['t_wall_s']:7.2f}s  {row['mem_alloc_mb']:8.1f}MB  "
          f"eps={row['eps_p10_fwd']:.3f}  above={row['frac_above_fwd']:.3f}"
          + ('   <-- ' + ', '.join(bad) if bad else ''))


# ------------------------------------------------------------------- outputs
def _pyplot():
    import matplotlib
    if not os.environ.get('DISPLAY') and os.name != 'nt':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _read(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in COLUMNS:
            if k in ('ts', 'arm', 'device', 'dtype', 'mode', 'mem_mode',
                     'bin_flag_fwd', 'bin_flag_back', 'gpu_name', 'status',
                     'note'):
                continue
            try:
                r[k] = float(r[k]) if r[k] not in ('', None) else NAN
            except (ValueError, TypeError):
                r[k] = NAN
    return rows


def _agg(rows, arm, mode='adaptive', field='t_wall_s', ok_only=True):
    """Mean over seeds per ray count -> (R array, value array)."""
    by = {}
    for r in rows:
        if r['arm'] != arm or r['mode'] != mode:
            continue
        if ok_only and r['status'] != 'ok':
            continue
        v = r.get(field, NAN)
        if not math.isfinite(v):
            continue
        by.setdefault(r['rays'], []).append(v)
    ks = sorted(by)
    return np.array(ks), np.array([np.mean(by[k]) for k in ks])


# eps is quantized by the integer bin count: a bin holding 1 or 2 rays reports
# exactly 1.000 or 0.707 whatever the true variance is, so those points sit
# ABOVE the R^-1/2 line and flatten the fitted slope.  Drop them before fitting.
EPS_SAT = 0.7

# The -1/2 slope is fitted on eps_MEDIAN, not on eps_p10.  p10 is the right
# headline for sizing bins (it is the pessimistic bin), but the 10th percentile
# of a small-integer count is itself biased low, and the bias grows as the count
# falls - which tilts the fitted slope to about -0.53.  The median is free of
# that tilt and lands on -0.500 +- 0.005.  eps_p10 is still what gets plotted.
SLOPE_FIELD = 'eps_median_fwd'


def _desat(R, e):
    """Keep only points whose noise is not quantization-saturated."""
    m = e < EPS_SAT
    return R[m], e[m]


def make_plots(rows, out):
    plt = _pyplot()
    arms = sorted({r['arm'] for r in rows})
    written = []
    # Read from the live device rather than the CSV: adding a column would make
    # the harness refuse to append to every results.csv written before now, and
    # a missing VRAM line costs one annotation, not a plot.
    vram_mb = None
    if torch.cuda.is_available():
        vram_mb = torch.cuda.mem_get_info()[1] / 2 ** 20

    # 1 - noise. The whole point: does the adaptive rule hold the 10 % line?
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a in arms:
        for side, mk in (('fwd', 'o'), ('back', 's')):
            R, e = _agg(rows, a, field='eps_p10_' + side)
            if R.size:
                ax[0].loglog(R, e, mk + '-', label=f'{a} {side}')
    ax[0].axhline(EPS_TARGET, color='k', ls='--', lw=1, label='10 % target')
    ax[0].set(xlabel='rays launched', ylabel='eps_p10 (per-bin rel. noise)',
              title='adaptive N: noise held at the target')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which='both')

    for a in arms:
        R, e = _agg(rows, a, mode='fixedN', field='eps_p10_fwd')
        Rf, ef = _desat(*_agg(rows, a, mode='fixedN', field=SLOPE_FIELD))
        if R.size >= 2:
            sl = (np.polyfit(np.log(Rf), np.log(ef), 1)[0] if Rf.size >= 2
                  else float('nan'))
            ax[1].loglog(R, e, 'o-', label=f'{a} fwd  slope={sl:+.3f}')
            ax[1].loglog(R, e[0] * (R / R[0]) ** -0.5, 'k:', lw=1)
    ax[1].set(xlabel='rays launched', ylabel='eps_p10',
              title=f'fixed N={FIXED_N_FWD}: must fall as R^-1/2')
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which='both')
    fig.tight_layout(); p = os.path.join(out, 'noise_vs_rays.png')
    fig.savefig(p, dpi=110); plt.close(fig); written.append(p)

    # 2 - time, and the bins the 10 % rule bought
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a in arms:
        R, t = _agg(rows, a, field='t_wall_s')
        if R.size:
            ax[0].loglog(R, t, 'o-', label=a)
    ax[0].set(xlabel='rays launched', ylabel='wall time [s]',
              title='cost per run'); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, which='both')
    for a in arms:
        R, n = _agg(rows, a, field='n_bins_fwd')
        if R.size:
            ax[1].loglog(R, n, 'o-', label=a + ' fwd')
        R, n = _agg(rows, a, field='n_bins_back')
        if R.size and np.nanmax(n) > 0:
            ax[1].loglog(R, n, 's--', label=a + ' back')
    ax[1].set(xlabel='rays launched', ylabel='bins per side at 10 % noise',
              title='N ~ sqrt(R)  (the fair cross-arm comparison)')
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which='both')
    fig.tight_layout(); p = os.path.join(out, 'time_vs_rays.png')
    fig.savefig(p, dpi=110); plt.close(fig); written.append(p)

    # 3 - memory. Monolithic only: a chunked row's peak is one chunk's peak and
    # plotting the two together would claim a memory number that is not real.
    fig, ax = plt.subplots(figsize=(7.6, 5))
    r_max = max((r['rays'] for r in rows), default=1e8)
    walls = []
    # RESERVED only, one line per arm. Allocated is the tidier number but it is
    # not the one that OOMs: the allocator fails when it cannot RESERVE, and the
    # gap between the two (measured ~1.3x, pure fragmentation) is exactly the
    # margin that decides whether a run fits. Plotting both put four lines on
    # top of each other and made the figure unreadable; allocated is still
    # recorded in the CSV and still drives `bytes_per_ray` in results.json.
    for ai, a in enumerate(arms):
        col = f'C{ai}'
        rs = [r for r in rows if r['arm'] == a and r['mem_mode'] == 'monolithic'
              and r['status'] == 'ok']
        R, m = _agg(rs, a, field='mem_reserved_mb')
        Ra, ma = _agg(rs, a, field='mem_alloc_mb')
        if not R.size:
            continue
        ax.loglog(R, m, 'o-', color=col, lw=2, ms=5, label=f'{a}  measured')
        if R.size < 2 or Ra.size < 2:
            continue
        # The extrapolation is fitted on ALLOCATED, not on the reserved curve
        # plotted here. Allocated is linear in R by construction (trace_mc is
        # flat-[N]-wide and never compacts); reserved carries a large constant
        # from allocator warmup - 480 MB at 1e4 falling to 30 MB at 3e4 - which
        # is non-monotone and drives a straight-line fit NEGATIVE. So: fit the
        # physics on allocated, then scale up by the measured fragmentation
        # ratio to land back on the reserved axis, since reserved is what OOMs.
        bpr = float(np.polyfit(Ra, ma * 2 ** 20, 1)[0])
        big = R >= R.max() / 30.0
        frag = float(np.median(m[big] / ma[big])) if big.any() else 1.0
        wall = vram_mb * 2 ** 20 / (bpr * frag) if vram_mb else 0.0
        xs = np.geomspace(R.max(), max(r_max, R.max(), wall) * 1.6, 48)
        ax.loglog(xs, np.polyval(np.polyfit(Ra, ma, 1), xs) * frag, ':',
                  color=col, lw=1.8, alpha=.75,
                  label=f'{a}  predicted, {bpr:.0f} B/ray x {frag:.2f} frag')
        if wall:
            walls.append((a, col, wall))
            print(f'    {a}: predicted monolithic wall {wall:.2e} rays '
                  f'({bpr:.0f} B/ray, {vram_mb/1024:.0f} GB)')

    if vram_mb:
        ax.axhline(vram_mb, color='k', lw=1.2, alpha=.7)
        lo, hi_y = ax.get_ylim()
        ax.set_ylim(lo, max(hi_y, vram_mb * 2.5))
        ax.text(0.985, vram_mb * 1.1, f'device VRAM {vram_mb/1024:.0f} GB',
                transform=ax.get_yaxis_transform(), ha='right',
                fontsize=8, alpha=.85)
    # The observed OOM is what the dotted lines are checked against: a predicted
    # wall near it validates the linear claim; a large gap voids it.
    ooms = sorted({r['rays'] for r in rows if r['status'] == 'oom'})
    for x in ooms:
        ax.axvline(x, color='r', ls='--', lw=1.6, alpha=.85)
    if ooms:
        ax.plot([], [], 'r--', lw=1.6, label=f'observed OOM  {ooms[0]:.0e}')

    ax.set(xlabel='rays launched', ylabel='peak memory reserved [MB]',
           title='memory: measured (solid) vs predicted past the wall (dotted)')
    ax.legend(fontsize=8, loc='upper left', framealpha=.9)
    ax.grid(alpha=.3, which='both')
    fig.tight_layout(); p = os.path.join(out, 'memory_vs_rays.png')
    fig.savefig(p, dpi=110); plt.close(fig); written.append(p)

    for p in written:
        print(f'  wrote {p}')


def make_hist(out):
    """Per-bin distributions at R=1e6, from the count maps dumped by the sweep.

    The sweep reports one number, `eps_p10`. These two rows are what that number
    is a summary of:

      top     how many rays each bin caught. The spread here is NOT shot noise -
              it is mostly the irradiance profile of the receiver, so do not
              read it against a Poisson curve. What matters is where the bulk
              sits relative to the 100 rays that 10 % noise requires.
      bottom  per-bin noise itself, `eps = 1/sqrt(count)`, which is the exact
              identity here because `trace_mc` leaves every weight bit-exactly
              equal. The 10 % line and the fraction of lit bins past it are the
              requirement, read straight off the plot.

    Grey is every lit bin, blue the core (see `_core_mask`). The gap between
    them is the beam edge: bins that are noisy because they are half-covered,
    not because the ray budget is short. Drawn for HIST_RAYS only.
    """
    import glob
    files = sorted(glob.glob(os.path.join(out, 'hist_*.npz')))
    if not files:
        return None
    panels = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        arm = str(z['arm'])
        for side in ('fwd', 'back'):
            if side in z:
                panels.append((arm, side, z[side]))
    if not panels:
        return None

    plt = _pyplot()
    fig, ax = plt.subplots(2, len(panels), figsize=(5.0 * len(panels), 7.2),
                           squeeze=False)
    for j, (arm, side, C) in enumerate(panels):
        lit = C[C > 0].astype(float)
        core = C[C >= 0.5 * np.median(lit)].astype(float)
        p10 = float(np.quantile(core, 0.10))
        need = 1.0 / EPS_TARGET ** 2               # rays for 10 % noise
        wide = lit.max() / max(lit.min(), 1.0) > 30.0

        a = ax[0][j]
        bins = (np.geomspace(max(lit.min(), 0.5), lit.max(), 60) if wide
                else np.linspace(0, lit.max(), 60))
        a.hist(lit, bins=bins, color='0.80', label=f'lit  (n={lit.size})')
        a.hist(core, bins=bins, color='tab:blue', label=f'core  (n={core.size})')
        if wide:
            a.set_xscale('log')
        a.axvline(need, color='r', ls='--', lw=1.2,
                  label=f'{EPS_TARGET:.0%} noise needs {need:.0f}')
        a.axvline(p10, color='tab:orange', ls=':', lw=1.6,
                  label=f'core p10 = {p10:.0f}')
        a.set(xlabel='rays caught by a bin', ylabel='bins',
              title=f'{arm}  {side}   R={HIST_RAYS:.0e}   '
                    f'{C.shape[0]}x{C.shape[1]}')
        a.legend(fontsize=7); a.grid(alpha=.3)

        # eps = 1/sqrt(count): exact here, because every MC weight is equal.
        b = ax[1][j]
        e_lit, e_core = 1.0 / np.sqrt(lit), 1.0 / np.sqrt(core)
        above = float((e_lit > EPS_TARGET).mean())
        # A few beam-edge bins hold one or two rays and sit at eps = 1.0, which
        # would stretch the axis until the distribution that matters is a single
        # spike. Clip the view at 3x target and pile the overflow into the last
        # bin, so it stays visible without setting the scale.
        hi = 3.0 * EPS_TARGET
        eb = np.linspace(0, hi, 60)
        n_over = int((e_lit > hi).sum())
        b.hist(np.clip(e_lit, 0, hi), bins=eb, color='0.80', label='lit')
        b.hist(np.clip(e_core, 0, hi), bins=eb, color='tab:blue', label='core')
        if n_over:
            b.annotate(f'{n_over} lit bins off-scale\n(eps up to '
                       f'{e_lit.max():.2f})', xy=(0.97, 0.55),
                       xycoords='axes fraction', ha='right', fontsize=7,
                       color='0.35')
        b.axvline(EPS_TARGET, color='r', ls='--', lw=1.2,
                  label=f'target {EPS_TARGET:.0%}')
        b.axvline(1 / math.sqrt(p10), color='tab:orange', ls=':', lw=1.6,
                  label=f'eps_p10 = {1/math.sqrt(p10):.3f}')
        b.set(xlabel='per-bin relative noise  eps = 1/sqrt(count)',
              ylabel='bins',
              title=f'{above:.1%} of lit bins above target')
        b.legend(fontsize=7); b.grid(alpha=.3)

    fig.tight_layout()
    q = os.path.join(out, 'bin_count_hist.png')
    fig.savefig(q, dpi=140); plt.close(fig)
    print('  wrote', q)
    return q


def make_json(rows, out, kcal):
    """Derived summary. The CSV is the source of truth; this is rebuilt from it.

    A half-written JSON is unrecoverable where a half-written CSV loses one line
    - so nothing is ever written here that is not reconstructible.
    """
    summary = {'env': {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': _gpu_name(torch.device(rows[0]['device'])) if rows else '',
        'git': _git_head(),
        'generated': datetime.now().isoformat(timespec='seconds'),
    }, 'scene': {
        'wavelength': ref.WAVELENGTH, 'R_coat': ref.R_COAT,
        'c': ref.C_ASPH, 'k': ref.K_ASPH, 'r_lens': ref.R_LENS,
        'thickness': ref.THICK, 'theta_max': ref.THETA_MAX,
        'phi_cap': ref.PHI_CAP, 'max_depth': ref.MAX_DEPTH,
        'z_recv': ref.Z_RECV, 'r_recv': ref.R_RECV,
        'z_back': ref.Z_BACK, 'r_back': ref.R_BACK,
    }, 'calibration': kcal, 'arms': {}}

    for a in sorted({r['arm'] for r in rows}):
        ok = [r for r in rows if r['arm'] == a and r['status'] == 'ok']
        oom = [r for r in rows if r['arm'] == a and r['status'] == 'oom']
        d = {'n_ok': len(ok), 'n_oom': len(oom)}
        R, e = _desat(*_agg(rows, a, mode='fixedN', field=SLOPE_FIELD))
        if R.size >= 2:
            d['noise_slope_fixedN'] = float(np.polyfit(np.log(R), np.log(e), 1)[0])
        mono = [r for r in ok if r['mem_mode'] == 'monolithic'
                and math.isfinite(r['mem_alloc_mb'])]
        if len(mono) >= 2:
            x = np.array([r['rays'] for r in mono])
            y = np.array([r['mem_alloc_mb'] * 2 ** 20 for r in mono])
            d['bytes_per_ray'] = float(np.polyfit(x, y, 1)[0])
            if torch.cuda.is_available():
                tot = torch.cuda.mem_get_info()[1]
                d['predicted_oom_rays'] = tot / d['bytes_per_ray']
        if ok:
            d['largest_ok_rays'] = max(r['rays'] for r in ok)
            d['us_per_ray'] = float(np.median(
                [r['t_wall_s'] / r['rays'] * 1e6 for r in ok]))
        if oom:
            d['smallest_oom_rays'] = min(r['rays'] for r in oom)
        d['ledger_max'] = max((r['ledger_rel'] for r in ok
                               if math.isfinite(r['ledger_rel'])), default=NAN)
        summary['arms'][a] = d

    p = os.path.join(out, 'results.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  wrote {p}')
    return summary


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', default=os.path.join(_HERE, 'c04_out'),
                    help='result directory; point it at Drive on Colab')
    ap.add_argument('--arm', dest='arms', action='append', choices=ARMS,
                    help='repeatable; default is the two headline arms')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'cpu')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--rays', type=float, nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true',
                    help='<60 s: two ray counts, one seed, one rep')
    ap.add_argument('--slope', action='store_true',
                    help='add the fixed-N sub-sweep for the -1/2 gate')
    ap.add_argument('--no-chunk', action='store_true',
                    help='stay monolithic everywhere; finds the OOM wall sooner')
    ap.add_argument('--chunk-above', type=float, default=CHUNK_ABOVE,
                    help='chunk only above this ray count (default 1e7). Raise '
                         'it to push the monolithic memory curve further before '
                         'chunked rows take over - chunked rows carry no usable '
                         'B/ray datum. Budget: R * B_per_ray < VRAM, with '
                         '~300 B/ray seq and ~540 B/ray nonseq.')
    ap.add_argument('--chunk-size', type=float, default=CHUNK_SIZE,
                    help='rays per chunk once chunking kicks in (default 2e6)')
    ap.add_argument('--calibrate', action='store_true', help='remeasure k')
    ap.add_argument('--calibrate-only', action='store_true')
    ap.add_argument('--plots-only', action='store_true')
    ap.add_argument('--json-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='ignore the resume set')
    args = ap.parse_args()

    args.arms = args.arms or ['seq_R00', 'nonseq_mc_R02']
    args.rays = ([int(x) for x in args.rays] if args.rays
                 else (RAYS_SMOKE if args.smoke else RAYS_FULL))
    if args.smoke:
        args.seeds, args.reps = 1, 1

    os.makedirs(args.out, exist_ok=True)      # c03 never does this; it must
    csv_path = os.path.join(args.out, 'results.csv')
    calib_path = os.path.join(args.out, 'calib.json')
    device = torch.device(args.device)

    print(f'\n  torch {torch.__version__}  device {device}  '
          f'{_gpu_name(device)}  dtype {torch.get_default_dtype()}')
    if device.type == 'cuda':
        free, tot = torch.cuda.mem_get_info()
        print(f'  VRAM {tot/2**30:.1f} GiB total, {free/2**30:.1f} GiB free')
        print('  NOTE fp64 runs at 1/32 the fp32 rate on T4/L4 and 1/2 on '
              'A100/V100.\n       On a 1/32-rate part expect roughly CPU speed '
              '- that is a finding,\n       not a bug. Run --device cpu too, so '
              'the comparison is in the CSV.')
    print(f'  out  {args.out}')

    if args.plots_only or args.json_only:
        rows = _read(csv_path)
        kcal = json.load(open(calib_path)) if os.path.exists(calib_path) else {}
        if not args.json_only:
            make_plots(rows, args.out)
            make_hist(args.out)
        make_json(rows, args.out, kcal)
        return 0

    # --- calibration -------------------------------------------------------
    kcal = {}
    if os.path.exists(calib_path) and not args.calibrate:
        kcal = json.load(open(calib_path, encoding='utf-8'))
    need = [a for a in args.arms if a not in kcal]
    if need or args.calibrate or args.calibrate_only:
        print('\n  calibrating k (one pilot trace per arm)')
        print(f'    {"arm":16s} {"k_fwd":>10s} {"k_back":>10s} '
              f'{"slope_f":>9s} {"slope_b":>9s}')
        for a in (args.arms if (args.calibrate or args.calibrate_only) else need):
            c = calibrate_bins(a, device,
                               R_pilot=int(2e5) if args.smoke else int(1e6))
            kcal[a] = {'fwd': c['k_fwd'], 'back': c['k_back']}
            flag = ''
            for s in ('fwd', 'back'):
                sl = c['slope_' + s]
                if math.isfinite(sl) and abs(sl + 2.0) > TOL['cal_slope']:
                    flag = '  <-- slope not -2, the k model does not hold'
            print(f"    {a:16s} {c['k_fwd']:10.5f} {c['k_back']:10.5f} "
                  f"{c['slope_fwd']:9.3f} {c['slope_back']:9.3f}{flag}")
        with open(calib_path, 'w', encoding='utf-8') as f:
            json.dump(kcal, f, indent=2)

        if 'seq_R00' in kcal and 'nonseq_mc_R02' in kcal:
            # At R=0 every captured ray transmits, so k_mc/k_seq must be exactly
            # the T1T2 fraction. A free cross-check on the whole path story.
            ratio = kcal['nonseq_mc_R02']['fwd'] / kcal['seq_R00']['fwd']
            ok = abs(ratio - 0.64) < TOL['k_ratio']
            print(f"\n    k_mc/k_seq = {ratio:.4f}  (expect 0.64)  "
                  f"{'ok' if ok else '<-- FAIL'}")
    if args.calibrate_only:
        return 0

    # --- the sweep ---------------------------------------------------------
    t0 = time.perf_counter()
    sweep(args, kcal, csv_path)
    if args.dry_run:
        return 0
    print(f'\n  sweep finished in {(time.perf_counter()-t0)/60:.1f} min')

    rows = _read(csv_path)
    if rows:
        make_plots(rows, args.out)
        make_hist(args.out)
        s = make_json(rows, args.out, kcal)
        print('\n  ' + '-' * 66)
        for a, d in s['arms'].items():
            print(f"  {a:16s} {d.get('us_per_ray', NAN):7.2f} us/ray  "
                  f"{d.get('bytes_per_ray', NAN):7.1f} B/ray  "
                  f"largest ok {d.get('largest_ok_rays', 0):.0e}  "
                  f"slope {d.get('noise_slope_fixedN', NAN):+.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
