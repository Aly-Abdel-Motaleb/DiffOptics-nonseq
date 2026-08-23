# P5 — LightTools build + comparison, two lenses

Build the `c05_two_lens.py` scene in LightTools, run it with probabilistic ray
splitting (Monte Carlo), and compare against `trace_mc` with `c05_compare.py`.

This is a **complete build from `File → New`**. It does not assume the c03
one-lens model is open. Where a step is identical to c03 it is repeated here in
full rather than cross-referenced, so this file can be followed start to finish
at the workstation.

> **Version:** written for **LightTools 2026**. Confidence tags carried over
> from the c03 build, which was checked against this install:
>
> - **[2026-verified]** — confirmed against the actual 2026 UI.
> - **[confirm]** — stable across 2021–2026 in principle, not re-checked here.
>
> The *values* are exact regardless of where the field lives.
>
> **GPU trace: leave it off.** It is a forward-only accelerator and its handling
> of splitting, TIR and the depth cap is not guaranteed to match the CPU tracer.
> Every reference number below comes from a float64 CPU tracer.

Build order: source → lens 1 → lens 2 → coatings → receivers. Placing a receiver
before the lenses exist makes it easy to attach it to the wrong surface.

---

## What is different from the c03 one-lens build

If you have done c03, these are the only things that change — but read Step 2b
and Step 3 carefully, they are where the mistakes will be.

| | c03 (one lens) | **c05 (two lenses)** |
|---|---|---|
| lenses | 1 | **2** |
| coatings | one property, 20 % on both surfaces | **two properties** — 20 % on S1/S2, **10 % on S3/S4** |
| forward receiver | z = 200, ±16 mm, 512² | z = 200, **±20 mm**, **64²** |
| backward receiver | z = −80, ±80 mm, 512² | z = −80, ±80 mm, **32²** |
| rays | 50 k / 200 k | **1,000,000** |
| direct path | `T1T2` = 64.0000 % | **`T1T2T3T4` = 51.8400 %** |

The receiver meshes are much coarser than c03's 512², and that is deliberate —
see **Step 4a**. 512² at these ray counts is not a finer picture, it is a noisier
one.

---

## The scene

```
point source            lens 1 (collimator)      lens 2 (mirror image)     forward
(0,0,0), 1 W       ->   S1 asphere  z = 40.0 ->  S3 flat    z = 100.0  ->  z = 200
isotropic               S2 flat     z = 46.5     S4 asphere z = 106.5      +-20 mm
cone half-angle         N-BK7, sd 12.7           N-BK7, sd 14.0            64 x 64
16.2758 deg             R = 20 % both            R = 10 % both
                                                                      <-  backward
                                                 focus at z = 146.5       z = -80
                                                                          +-80 mm
                                                                          32 x 32
```

Lens 2 is lens 1 mirrored about z = 73.25. Lens 1 turns the point source into a
collimated beam; run the same glass backwards and a collimated beam becomes a
point 40 mm past the aspheric vertex — exactly, by ray reversibility. So lens 2
is *not* a new design: it is lens 1 flipped, which is why its aspheric radius is
lens 1's with the sign changed and its conic constant is identical.

The forward receiver sits **53.5 mm past the focus**, where the beam has opened
back out to a 15.62 mm disc. At the focus it would be a single hot bin and there
would be nothing to compare.

---

## Step 0 — new model, units and axes

1. `File → New`.
2. `Model → System Data` **[confirm]**:
   - length units **mm**
   - photometry **Radiometric** (watts). This greys out the *Photometric Flux*
     box in the source dialog. If lumens are live and watts are not, this is set
     wrong.
   - **Ambient Refractive Index**: **1.000293**
3. Global axes: **+z is the optical axis**, source at the origin, light toward +z.

> **If the ambient field rounds to 1.0003** (4 decimals) that is fine, but fix
> the *ratio* — nothing in the comparison depends on either index alone, only on
> `n_rel = n_glass / n_air = 1.519066026`.
>
> | ambient | glass to enter in Step 2 | n_rel |
> |---|---|---|
> | 1.000293 | 1.519511112 | 1.519066026 |
> | 1.0003 (rounded) | **1.5195217454** | 1.519066026 |
>
> Leaving 1.0003 with 1.519511112 shifts n_rel by −1.06e-5. Power, path
> fractions and the ledger are untouched; only the focus quality moves.

Everything is on-axis and centred at (0, 0). Leave all x/y offsets at 0.

---

## Step 1 — the point source

`Insert → Light Source → Point Source` **[2026-verified]**, then double-click it.
Tabs: **Coordinates | Emittance | Aim Sphere | Immersion | Spectral Region |
Spectral Region Chart | Display**. The cone is *not* on the same tab as the flux.

### Coordinates tab

Position **(0, 0, 0)**, all rotations 0 → emission axis along **+z**.

### Emittance tab **[2026-verified]**

| control | set to |
|---|---|
| Enabled | checked |
| Weight Factor | **1.0000** — relative ray share between sources, not a flux scale |
| Total Flux/Power | **Radiometric Power** (watts) |
| Radiometric Power | **1.0 W** |
| Measured Over | **Whole Sphere** |
| Starting Point Classification | **Semi-automatic** |
| Aim Region | **Aim Sphere** |
| **Angular Distribution** | **Uniform** |
| Polarization | **Unpolarized** |

**Uniform, not Cosine.** Uniform is constant radiant intensity per solid angle —
an isotropic point lamp, which is what `nonseq.sample_point_source` draws
(cos θ uniform). *Cosine* is Lambertian and would bias the beam toward the axis;
every irradiance number below would be wrong while the total flux still looked
right.

**`Measured Over` decides what the number in the box means:**

| Radiometric Power | Measured Over | flux traced into the cone | verdict |
|---|---|---|---|
| **1.0 W** | **Whole Sphere** | 0.020023941 W | **use this** |
| 0.020023941 W | Aim Region | 0.020023941 W | equivalent, entered directly |
| 0.020023941 W | Whole Sphere | 4.01e-4 W | **wrong, ~50× low** |

### Aim Sphere tab **[2026-verified]** — the cone lives here

| group | field | set to |
|---|---|---|
| Angle | **Upper** | **0.00** degrees |
| Angle | **Lower** | **16.27** degrees |
| Orientation | Alpha | 0.00 |
| Orientation | Beta | 0.00 |
| | Draw Aim Region | **checked** |

`Upper`/`Lower` are polar angles from the aim axis; `Upper = 0, Lower = θ` is a
forward cone of half-angle θ. Keep **Draw Aim Region** checked and confirm in the
3D view that the cone opens toward the lenses at +z. If it points at the backward
receiver, set `Beta = 180`.

#### The field takes 2 decimals — round the cone DOWN

Exact fill angle is `atan(12.7 / 43.498695) = 16.275839356°`. The two roundings
are **not** symmetric:

| entered | ray at Lower hits S1 at r = | consequence |
|---|---|---|
| **16.27** | 12.694365 mm | cone sits **inside** the rim, nothing vignettes → the closed forms stay exact |
| 16.28 | 12.704016 mm | cone **overfills** the rim by 4 µm, ~0.05 % of rays miss the lens → `T1T2T3T4` drops off 51.8400 and fails Step 9 with the physics perfectly correct |

So **Lower = 16.27**. The closed forms are per-ray ratios, so shrinking the cone
costs nothing while growing it past the rim breaks them.

At 16.27° the collimated beam leaving lens 1 is 12.694365 mm in radius, well
inside lens 2's 14 mm rim, so lens 2 does not vignette either.

#### Φ_cap for the 16.27° cone

| θ | Φ_cap = 1 W × (1 − cos θ)/2 |
|---|---|
| 16.275839356° (ideal) | 0.020038221 W |
| **16.27° (what you enter)** | **0.020023941 W** (−0.0713 %) |

Every percentage in §7 is a fraction of Φ_cap, so all fractions are unchanged —
**normalise by what LightTools reports, not by 0.020038221**. Absolute watts run
0.0713 % low across the board. Compare *fractions*, not watts.

### Immersion tab **[confirm]**

Source **not immersed** — starting medium is ambient air.

### Spectral Region tab **[confirm]**

Single wavelength **532.8 nm**. Verify on the Spectral Region Chart tab: one
line, nothing else.

### Ray count is not here

Rays per run live in the Simulation Manager (Step 5).

---

## Step 2a — lens 1 (the collimator)

`Insert → Lens` **[2026-verified]** opens a **Place Singlet** palette.

**Pick the parametric icon — the one labelled `R1` / `R2` with the diameter arrow
`D`.** Avoid every icon with **numbered points (1,2,3,4,5,6)**: those are
interactive placement modes that infer geometry from where you click, and nothing
here can be clicked to 6 decimals. Folder icons are library elements.

It runs **`CreateQuickLens`**, which prompts for **curvature**, not radius
**[2026-verified]**. Curvature is `c = 1/R`:

| prompt | value |
|---|---|
| front surface curvature | **0.0481634296** (= 1/20.762641) |
| back surface curvature | **0** (flat) |
| thickness | **6.5** |
| diameter / aperture | **25.4** (semi-diameter 12.7) |
| position | front vertex at z = **40.0** |

Full precision matters — 0.04816 instead of 0.0481634296 moves the collimation
point by ~0.3 mm.

`CreateQuickLens` also prompts for **material**. Type **`BK7`** to get past it;
the index is overridden in Step 2c. The catalogue dispersion is not what this
comparison uses.

**Conic is not in the Place Singlet dialog** — it creates spherical surfaces.
After placing, open S1's surface properties, change type to **Aspheric/Conic**,
set `Conic Constant = −2.307561590` with all polynomial terms 0, and re-check the
radius survived the type change.

### S1 — front, aspheric

| field | value |
|---|---|
| surface type | Aspheric (or Conic) |
| vertex position z | **40.0** |
| Radius | **+20.762641** mm |
| Conic Constant | **−2.307561590** |
| A4 … A12 | **0** — pure conic |
| aperture | circular, semi-diameter **12.7** |

### S2 — back, flat

| field | value |
|---|---|
| surface type | Plane |
| vertex position z | **46.5** |
| Radius | infinite / curvature 0 |
| aperture | circular, semi-diameter **12.7** |

### Sag sign — check before continuing

`sag(12.7) = +3.498695`, so:

- S1 **vertex** at z = 40.000000 — the closest point of the glass to the source
- S1 **rim** at z = 43.498695
- **edge thickness = 6.5 − 3.498695 = 3.001305 mm**

If the edge measures 9.999 instead of 3.001, the curvature sign is flipped — the
surface bulges away from the source and collimation is dead.

---

## Step 2b — lens 2 (the mirror image)

Same procedure, **flat side first**. This is the step where a sign error is
easiest to make and hardest to see, because a wrong-way lens 2 still produces a
plausible-looking spot on the receiver.

| prompt | value |
|---|---|
| front surface curvature | **0** (flat) |
| back surface curvature | **−0.0481634296** (= 1/−20.762641) |
| thickness | **6.5** |
| diameter / aperture | **28.0** (semi-diameter 14.0) |
| position | front vertex at z = **100.0** |

Then change S4's type to Aspheric/Conic and set the conic constant — **the same
−2.307561590 as lens 1**. Mirroring a conic flips the radius sign; it does not
touch the conic constant. If you find yourself typing +2.307 or −1/2.307, stop.

### S3 — front, flat

| field | value |
|---|---|
| surface type | Plane |
| vertex position z | **100.0** |
| Radius | infinite / curvature 0 |
| aperture | circular, semi-diameter **14.0** |

### S4 — back, aspheric

| field | value |
|---|---|
| surface type | Aspheric (or Conic) |
| vertex position z | **106.5** |
| Radius | **−20.762641** mm |
| Conic Constant | **−2.307561590** |
| A4 … A12 | **0** |
| aperture | circular, semi-diameter **14.0** |

### Sag sign for lens 2 — the check that matters

`sag(14.0) = −4.171955`, so S4 bulges toward **−z**, into the glass:

- S4 **vertex** at z = 106.500000 — the *furthest* point of lens 2 from the source
- S4 **rim** at z = 102.328045
- **edge thickness = 102.328045 − 100 = 2.328045 mm**

So lens 2 is thickest on axis, like lens 1, but its curved face points **away**
from the source where lens 1's points **toward** it. In the 3D view the two
lenses should look like mirror images across the gap, not like two copies of the
same part.

If the edge measures 10.67 mm instead of 2.328, the curvature sign is flipped.

### Geometry check — the focus

Before adding coatings, set both optical properties to plain transmitting (no
reflection) and trace a few rays. **Every ray must cross the axis at
z = 146.500 mm.** In `diffoptics` this focus is a point to 2.6e-14 mm; in
LightTools it will be limited by your index precision, but it should be tight and
symmetric. A focus at the wrong z, or a blur instead of a point, means a
curvature, a conic or a vertex position is wrong — find it now, not after the
coatings are on.

---

## Step 2c — glass, both lenses

Double-click each solid → `Material` tab **[confirm]** → **User Defined**,
constant index:

- ambient entered as 1.000293 → glass **1.519511112**
- ambient rounded to 1.0003 → glass **1.5195217454**

Not the Schott catalogue: LightTools' N-BK7 fit gives ≈1.51947 at 532.8 nm, ours
1.519511. A 4e-5 difference is irrelevant to power but turns the focus check from
razor-sharp into a judgement call.

Both lenses get the **same** index. No mount, no barrel, no baffles — the rims
are hard circular edges and rays past them simply miss.

---

## Step 3 — the coatings: TWO properties, not one

This is the substantive difference from c03. In the one-lens build a single
property named `Transmitting` covered both surfaces. **Here that is wrong** — the
two lenses have different reflectances, and applying one property to all four
surfaces is the single most likely way to get a run that looks fine and is
silently the wrong experiment.

Make two Smooth Optical properties **[2026-verified]**:

| property name | Reflectance | Transmittance | applied to |
|---|---|---|---|
| **`Transmitting_20`** | **20.00 %** | **80.00 %** | **S1, S2** (lens 1) |
| **`Transmitting_10`** | **10.00 %** | **90.00 %** | **S3, S4** (lens 2) |

For each one:

1. Right-click the surface → `Optical Properties` (or the Optical Property
   Manager).
2. **Optical Properties** tab → type **Smooth Optical** (the default). Give it
   the Description above so the two can be told apart at a glance.
3. **Smooth Optical** tab:

   | field | value |
   |---|---|
   | **Ray Trace Mode** | **Split Rays (Reflected and Transmitted)** |
   | **Reflectance** | 20.00 % / 10.00 % |
   | **Transmittance** | 80.00 % / 90.00 % |
   | Absorption | 0.00 % (greyed, = 100 − R − T) |
   | Preferred Direction | **checked** — required for probabilistic splitting, Step 5 |
   | **Advanced Properties** | **None** |
   | Include Contamination Scatter | unchecked |

**`Ray Trace Mode` is the ray-splitting control** — not the Simulation Manager.
The default **Transmitted/TIR Rays** suppresses reflection entirely: both lenses
would transmit perfectly, the backward receiver would stay empty, and the forward
receiver would read ~100 % of Φ_cap instead of 51.84 %.

**`Advanced Properties = None` is "Fresnel OFF".** Selecting **Fresnel Loss**
replaces the flat percentages with the angle-dependent physical reflectance and
every closed form below stops being exact. `Coating` and `Polarizing Element` are
equally wrong here. A constant R at all angles is what `Element(R_fixed=...)`
implements.

**Verify the assignment before running.** In the System Navigator, click each of
the four surfaces in turn and read back which property it carries. Expected:
S1 → 20, S2 → 20, S3 → 10, S4 → 10. A run with all four at 20 % gives
`T1T2T3T4` = 0.8⁴ = 40.96 % instead of 51.84 %, which is a large enough gap to
catch — but only if you look.

### Edge surfaces

Each lens also has an **EdgeSurface** — the cylindrical rim (3.001305 mm on lens
1, 2.328045 mm on lens 2). Leave both on **Optical Absorber**. `diffoptics` has
no edge surface at all; its surfaces are simply cut at the semi-diameter and rays
past them fly away. Either way such a ray never completes a ghost path. Only the
ledger label differs — dO calls it `neither`, LightTools calls it absorbed.

---

## Step 4 — the receivers

LightTools attaches receivers to geometry, so each is two objects: a carrier
dummy surface, then a receiver on it.

### 4a — why these meshes, and why they are so coarse

Per-bin relative Monte Carlo noise for `R` rays landing on an `N × N` receiver:

```
eps = 1 / sqrt(R / N^2) = N / sqrt(R)
```

Designing for **eps < 10 %** fixes the mesh from the ray budget:

```
N = 16 * floor(0.1 * sqrt(R) / 16)          snapped DOWN, floored at 16
```

`R` is the number of rays landing on **that** receiver, not the number launched —
which is why the two receivers get different meshes. At **1,000,000 launched
rays**:

| receiver | rays landing | share | N ideal | **mesh to build** | eps |
|---|---|---|---|---|---|
| forward | ~532,000 | 53.2 % | 72.9 | **64 × 64** | **8.8 %** |
| backward | ~219,000 | 21.9 % | 46.8 | **32 × 32** | **6.8 %** |

`c05_two_lens.bins_for` is the implementation, and `c05_compare.py` calls the
same function, so the two sides agree by construction — **as long as the
LightTools receiver is built with the mesh in that table**. If you export the raw
rays rather than the chart (Step 8), the mesh does not have to match at all,
because the binning then happens once, here, for both codes. That is the main
reason to prefer the raw export.

> **Do not build 512 × 512 "for detail".** At 1e6 rays a 512² forward mesh holds
> ~2 rays per bin: `eps = 512/sqrt(532000) = 70 %`. The map would be noise, and
> the L2 in Step 9 would be comparing two noise fields rather than two
> irradiances.

> **Honest caveat on the backward receiver.** The rule above is a flat-field
> estimate — it uses the *mean* rays per bin. The backward receiver's flux is
> concentrated: a bright central patch plus a thin halo out to the rim. Mean
> count per bin is 214, **median is 46**, so the typical bin's realised noise at
> 32 × 32 is ~14.7 %, not 6.8 %. A **16 × 16** backward mesh brings the measured
> median to 7.3 %. `c05_compare.py` prints both numbers in its `N strict` column
> and `c05_two_lens.py --test bins` explains it. Build 32 × 32 to follow the
> stated rule; build 16 × 16 if you want the realised noise under target.

### 4b — forward receiver, z = +200

1. `Insert → Dummy Surface` → rectangular, size **40 × 40 mm**, centred at
   **(0, 0, 200)**, normal along z.
2. `Insert → Receiver` → **Surface Receiver**, attached to that dummy surface.
3. Mesh **[confirm]**:
   - **64 × 64** bins
   - extents **±20 mm** in x and y (pitch **0.625 mm**)
   - quantity **irradiance, W/mm²** (reads W because Step 0 set Radiometric)
4. **Enable ray path sorting on this receiver.**

Why ±20 mm and not c03's ±16: the beam at z = 200 is a 15.62 mm disc, so ±16
would clip it with 0.4 mm to spare and cut off every ghost. ±20 leaves the
direct beam clear and keeps the ghost haze on the map.

### 4c — backward receiver, z = −80

- dummy surface centred at **(0, 0, −80)**, size **160 × 160 mm**
- mesh **32 × 32**, extents **±80 mm** (pitch **5.0 mm**)
- irradiance, W/mm²
- ray path sorting enabled

---

## Step 5 — simulation settings

`Ray Trace → Simulation Manager` → **Simulation Input**. Tabs:
`Forward | Backward | Hybrid | Sequences | Data Collection | Ray Paths | Update |
Spectral | Random Numbers`. **[2026-verified]**

### Forward tab

| field | value | why |
|---|---|---|
| Enable Forward Simulation | checked | |
| **Total Rays to Trace** | **1,000,000** | what the mesh table in Step 4a is sized for |
| **Relative Ray Power Threshold** | **1e-8** | default is 0.01 — far too tight, it eats the ghosts. Ours is 1e-5 relative to the per-ray weight |
| Show Preview Rays | off | |
| Show Ray Report | on | gives the flux summary |

### Ray Paths tab — required, or there is no path gate

Default reads *"No Ray Paths will be collected."* Tick **Collect** on:

| row | gives |
|---|---|
| **ForwardAll** | the `path_powers` equivalent — receivers ignored, every ray that flew that path. **This is where 51.84 / 20.00 / 12.80 live** |
| **Forward.\<forward receiver\>** | forward receiver landing table |
| **Forward.\<backward receiver\>** | backward receiver landing table |

**Raise the max-paths cap.** Four partial surfaces generate far more distinct
paths than c03's two — the reference table in §7 already shows 14 buckets above
0.1 %, and the true path count is larger still because several paths share a
bucket. With a low cap the tail is merged into an "other" row and the three gated
fractions can pick up contamination.

Leave **"Limit Rays per Path" unchecked** — that is a display cap, not a depth
limit.

### Other tabs

| tab | set |
|---|---|
| Spectral | single wavelength **532.8 nm** |
| Random Numbers | note the seed; run 5 different ones |
| Data Collection / Update | check here for a max-intersection cap |

### Depth cap **[open]**

Not found on Forward or Ray Paths in 2026. The power threshold does most of the
work: with R = 0.2 and 0.1, weight after k partial reflections falls fast and a
1e-8 cull terminates well before `MAX_DEPTH = 14`. Our own truncation at depth 14
is 2.3e-4 of Φ_cap, so the two should be comparable — but record it as a known
asymmetry rather than assume parity.

### Splitting is NOT set here **[2026-verified]**

It lives on the **optical property**, `Ray Trace Mode` (Step 3). Both properties
must be set the same way:

| Ray Trace Mode | Preferred Direction | equivalent |
|---|---|---|
| **Split Rays (Reflected and Transmitted)** | **checked** + Probabilistic Ray Split, threshold **1.0** | `trace_mc` |

---

## Step 6 — run

Splitting probabilistic (one child per hit, branch chosen with probability ρ,
weight divided by ρ) at 1e6 rays → compares against `trace_mc`. LightTools' MC
and `trace_mc` are the same estimator class, so this is an apples-to-apples
check. Run 5 seeds at 1e6 so the comparison has an error bar. Repeat at
1e5 / 1e6 / 1e7 to confirm the error falls with a log-log slope of ≈ −0.5.

---

## Step 7 — reference numbers

Closed-form / exact values, depth 14. **Percentages are of Φ_cap** — normalise
LightTools by the flux it reports launching into the cone.

### Power by path

`hHrK` = H surface hits, K of them reflections.

| bucket | % of Φ_cap | W (at Φ_cap = 0.020038221) | path | closed form |
|---|---|---|---|---|
| `fwd_h4r0` | **51.84000** | 1.039e-02 | T1 T2 T3 T4 | `(1−R1)²(1−R2)² = 0.5184` |
| `back_h1r1` | **20.00000** | 4.008e-03 | R1 | `R1 = 0.20` |
| `back_h3r1` | **12.80000** | 2.565e-03 | T1 R2 T1 | `(1−R1)²R1 = 0.128` |
| `back_h5r1` | 6.66115 | 1.335e-03 | *two paths* | **none — see below** |
| `fwd_h6r2` | 2.26718 | 4.543e-04 | *two ghosts* | sum of both |
| `fwd_h8r4` | 1.76930 | 3.545e-04 | | |
| `fwd_h4r2` | 1.25568 | 2.516e-04 | | |
| `back_h7r5` | 0.83923 | 1.682e-04 | | |
| `back_h7r3` | 0.80544 | 1.614e-04 | | |
| `back_h5r3` | 0.51314 | 1.028e-04 | | |
| `fwd_h6r4` | 0.31066 | 6.225e-05 | | |
| `back_h9r5` | 0.28484 | 5.708e-05 | | |
| `fwd_h10r6` | 0.18746 | 3.756e-05 | | |
| `back_h11r5` | 0.10671 | 2.138e-05 | | |

**Only the first three are gates**, and that is a physical fact about this scene,
not a limitation of the script. With four partial surfaces the (hits,
reflections) key stops being unique:

- **`back_h5r1` = 6.66115 %**, where the obvious closed form `(1−R1)⁴R2` gives
  4.096 %. The excess is a *second* path in the same bucket — reflect off S4,
  back out through S3, then **miss** lens 1's 12.7 mm rim and escape backwards.
  It is aperture-clipped, so it has no closed form at all.
- **`fwd_h6r2`** is the lens-1 ghost and the lens-2 ghost summed. Both are real,
  they have different amplitudes because R1 ≠ R2, and the bucket holds the total.

Compare the whole table for shape; gate on the three exact rows.

### Where the power lands

| bucket | % of Φ_cap | W |
|---|---|---|
| forward receiver (z = 200, ±20 mm) | **53.20343** | 1.066102e-02 |
| backward receiver (z = −80, ±80 mm) | **21.79931** | 4.368195e-03 |
| neither | 24.96727 | |
| culled + truncated | 0.02999 | |
| **total** | **100.00000** | |

Note forward receiver 53.203 % vs direct path 51.840 %: the 1.36 point difference
is ghost light landing inside ±20 mm. That is why the receiver is ±20 and not
±16.

The backward receiver catches 21.80 % while `R1` alone is 20.00 % — but only part
of `R1` lands inside ±80 mm, and the rest of the 21.80 is `T1R2T1` and friends.
The "neither" bucket is a real answer, not a leak: `R1` sprays far wider than the
backward receiver.

### Geometry

| quantity | value |
|---|---|
| focus (all direct rays cross the axis) | z = **146.500** mm |
| direct beam radius at z = 200 | **15.62** mm |
| lens 1 edge thickness | **3.001305** mm |
| lens 2 edge thickness | **2.328045** mm |
| collimated beam radius between the lenses | **12.694365** mm (at Lower = 16.27) |

---

## Step 8 — export

**Use `Analysis → Export Receiver Rays`, not the chart export.** Raw hits carry
no binning convention, so `nonseq.splat` — the same call our own map uses — bins
both sides identically and there is no half-pixel question. A chart export has
already been binned by LightTools using its own cell convention, and a half-pixel
disagreement shows up as a spurious radial ring in the difference map that looks
exactly like a physics error.

Export one file per receiver into `examples/nonseq/c05_out/`:

```
lt_fwd_mc_rays.txt      forward receiver
lt_back_mc_rays.txt     backward receiver
```

LightTools appends its own run number — `lt_fwd_mc_rays.1.txt` is fine and is
matched as-is. **No renaming is needed**; `c05_compare.py` globs
`lt_<name>_<tag>_rays*.txt`.

The file it writes looks like this, and the reader parses it directly:

```
# LightTools 2026
# Ray Data Export File
lt_rdf_version: 2.0
lt_datatype: radiant_power
lt_length_units: millimeters
lt_radiant_flux: 0.01284768
lt_startofdata
      4.206182      0.5588877            200   6.140793e-11   8.159451e-12              1     0.02002394
     -9.088956      -3.348481            200  -1.020052e-10  -3.757995e-11              1     0.02002394
```

Columns are `x y z dx dy dz power`, whitespace separated. The reader takes the
first two as x/y and the last as the per-ray weight, then **rescales the weights
so they sum to `lt_radiant_flux`** — so the declared flux in the header is the
number that has to be right, and per-ray equal weights are fine.

Also export the **ray path table** as `lt_paths_mc.txt` (or `.csv`), one
`name,power_W` per line, from `ForwardAll`. Include a `total` row if the exporter
offers one — it is used to normalise. This is the sharpest gate available; a run
without it is missing its best check.

If you must use the chart export instead, the receiver mesh **has to** match Step
4a exactly (64² over ±20 mm, 32² over ±80 mm) and the rows must be
`x_mm,y_mm,E` at cell centres in W/mm². The reader verifies the cell centres
against `nonseq.splat`'s to 1e-9 mm and refuses a mismatch.

### Step 8a — chart export walkthrough, both receivers **[confirm]**

Menu path for the CSV, per receiver:

1. In the **System Navigator**, click the receiver (not the dummy surface it
   sits on) — **Forward** for z = 200, **Backward** for z = −80.
2. `Analysis → Illuminance/Irradiance Receiver Data` (or double-click the
   receiver, which opens the same chart) → confirm **Irradiance, W/mm²** is
   the plotted quantity, not illuminance/lux — Step 0 set Radiometric, but
   this dialog has its own unit dropdown and it does not inherit that setting.
3. On the chart window: `File → Export → Data (CSV)` (or the export/save icon
   on the chart toolbar — labelling has moved between 2021–2026 builds).
4. Save as:

   | receiver | file |
   |---|---|
   | forward, z = 200 | `lt_fwd_mc.csv` |
   | backward, z = −80 | `lt_back_mc.csv` |

   into `examples/nonseq/c05_out/`. `c05_compare.py` globs
   `lt_<name>_mc*.csv` (`_rays` in the name routes to the raw-hit reader
   instead, so do not add it to a chart export's filename).
5. Repeat for the other receiver before closing the Simulation Manager —
   closing it can discard the in-memory chart data on some builds.

Open the CSV once before trusting it: header row `x_mm,y_mm,E`, one row per
cell, cell count **exactly** 64×64 (forward) or 32×32 (backward). A row count
of 4096 / 1024 confirms the mesh; anything else means Step 4a's mesh was not
what actually got built, and the reader will refuse it rather than silently
rebinning.

---

## Step 9 — compare

```
python examples/nonseq/c05_compare.py --rays 1000000
```

`--rays` must equal the LightTools ray count, or the two maps carry different MC
noise and the L2 measures the difference in ray budget. Add `--dir` if the
exports are elsewhere.

It always writes **`c05_out/c05_maps_mc.png`** — one row per receiver, four
columns: LightTools | ours | signed difference | radial profile. A receiver whose
export is missing still gets its row with the LightTools panel blank, so a
partial upload still produces the figure.

### Tolerances

| quantity | MC, 1e6 | gated? |
|---|---|---|
| `T1T2T3T4`, `R1`, `T1R2T1` | ±0.05 % + 3σ | **yes** |
| forward receiver total | ±0.5 % | **yes** |
| backward receiver total | ±1.0 % | **yes** |
| energy ledger | 0.1 % | **yes** |
| map relative L2 | vs noise floor | reported only |

The L2 is deliberately **not** gated against a fixed number. Two independent
Monte Carlo maps at ~9 % per-bin noise differ in L2 by about `sqrt(2) × 9 % ≈
13 %` even when both are exactly right. The script prints the floor implied by
the meshes actually used and compares against that.

---

## Forced-agreement checklist

Deliberate modelling choices that make the two codes comparable. If a number
disagrees, check this list before suspecting physics.

| # | choice | both sides |
|---|---|---|
| 1 | lens 1 reflectance | constant **20 %** at all angles, both surfaces |
| 2 | lens 2 reflectance | constant **10 %** at all angles, both surfaces |
| 3 | Fresnel | **off** (`Advanced Properties = None` / `R_fixed` set) |
| 4 | absorption | **0 %** everywhere |
| 5 | glass index | constant **1.519511112**, no dispersion, both lenses |
| 6 | ambient index | **1.000293** |
| 7 | wavelength | **532.8 nm**, monochromatic |
| 8 | source | isotropic point, **Uniform** angular distribution |
| 9 | cone half-angle | **16.27°** (rounded down, nothing vignettes) |
| 10 | polarization | unpolarized, scalar |
| 11 | edge/rim | absorbing; ours simply cuts at the semi-diameter |
| 12 | coatings applied per lens | **not one property on all four surfaces** |
| 13 | power threshold | 1e-8 (LT) vs 1e-5 relative (ours) |
| 14 | depth cap | threshold-driven (LT) vs `MAX_DEPTH = 14` (ours) |
| 15 | receiver mesh | 64² / 32², or export raw rays and the mesh stops mattering |
| 16 | normalisation | fractions of Φ_cap, never absolute watts |

---

## Report order

1. The scene and why lens 2 is lens 1 mirrored (exact, no new design).
2. The focus check at z = 146.5 — geometry before physics.
3. The three closed forms vs `trace_mc` vs LightTools.
4. `trace_mc` repeatability, with the 5-seed error bar.
5. The bin rule: why 64² and 32² and not 512², with the noise arithmetic.
6. `c05_maps_mc.png` — the deliverable figure.
7. The honest bit: `back_h5r1` has no closed form, and the backward receiver's
   realised noise is worse than the flat-field rule predicts. Both are stated,
   neither is hidden.
8. What LightTools cannot do: `dΦ_back/dR1` and `dΦ_fwd/dR2` out of one backward
   pass through `trace_mc`, matching a fixed-seed finite difference to a few
   percent.

---

## Files

**New:**
```
examples/nonseq/c05_two_lens.py     the scene, the bin rule, the gates
examples/nonseq/c05_compare.py      this comparison, writes c05_maps_mc.png
examples/nonseq/c05_lighttools.md   this file
examples/nonseq/c05_two_lens.ipynb  the 1e6 Colab run + the comparison cell
```

**Unchanged:** `c02_R02.py`, `c03_compare.py`, `c03_lighttools.md`, `c04_*`, and
`diffoptics/` — c05 imports the library tracers and c02's optical constants, and
writes nothing back.

---

## Still unverified in 2026

1. Receiver mesh dialog location **[confirm]**.
2. Material tab path for User Defined constant index **[confirm]**.
3. Whether an explicit max-intersection cap exists **[open]**.
4. `Probabilistic Ray Split` + threshold 1.0 as the `trace_mc` equivalent
   **[confirm]** — carried over from the c03 build, not independently
   re-checked for two properties in one model.
