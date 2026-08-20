# P4 — LightTools build + comparison

Rebuild the P3 scene (`c02_R02.py`) in LightTools, run it twice — splitting ON
and OFF — and compare against `trace_split` and `trace_mc` respectively.

> **Version:** written for **LightTools 2026**. Every path below carries a
> confidence tag:
>
> - **[2026-verified]** — confirmed against the actual 2026 UI (screenshot).
> - **[confirm]** — stable across 2021–2026 in principle, but *not* yet checked
>   in this install. If the dialog is not where stated, search the item name in
>   Help; item names outlive the tabs they sit on.
>
> The *values* are exact regardless of where the field lives.
>
> **GPU trace:** 2026 adds GPU-accelerated forward ray tracing. Leave it **off**
> for both runs — it is a forward-only accelerator and its handling of splitting,
> TIR and the depth cap is not guaranteed to match the CPU tracer. Reference
> numbers here come from a float64 CPU tracer. Re-run on GPU afterwards as a
> separate experiment if you want the speed.

Build order below is deliberate: source first, then geometry, then receivers.
Placing a receiver before the lens exists makes it easy to attach it to the
wrong surface.

---

## Step 0 — new model, units and axes

1. `File → New`.
2. Units and ambient index — `Model → System Data` **[confirm]**:
   - length units **mm**
   - photometry **Radiometric** (watts). This is what greys out the *Photometric
     Flux* box in the source dialog; if lumens are live and watts are not, you
     set this wrong.
   - **Ambient Refractive Index**: enter **1.000293**.
3. Global axes: **+z is the optical axis**, source at the origin, light toward +z.

> **If the ambient field rounds to 1.0003** (4 decimals) — that is fine, but fix
> the *ratio*, because nothing in `c02_R02.py` depends on either index alone;
> everything comes from `N_REL = n_glass/n_air` ([c02_R02.py:166-171](../../examples/nonseq/c02_R02.py#L166-L171)).
>
> | ambient | glass to enter in Step 2 | n_rel |
> |---|---|---|
> | 1.000293 | 1.519511112 | 1.519066026 |
> | 1.0003 (rounded) | **1.5195217454** | 1.519066026 |
>
> Leaving 1.0003 with 1.519511112 shifts n_rel by −1.06e-5. Power, path
> fractions, ghost and ledger are untouched; only the collimation gate moves —
> 0.82 µm of equivalent defocus, ~1e-5 rad rim divergence, so sanity check 4
> would need relaxing to `< 1e-5`. Compensating the glass index is free; do that.

Everything below is on-axis and centred at (0, 0), so leave all x/y offsets at 0.

---

## Step 1 — the point source

`Insert → Light Source → Point Source` **[2026-verified]**, then double-click it. The 2026
Properties dialog has tabs **Coordinates | Emittance | Aim Sphere | Immersion |
Spectral Region | Spectral Region Chart | Display** — the cone is *not* on the
same tab as the flux. **[2026-verified]**

### Coordinates tab

Position **(0, 0, 0)**, all rotations 0 → emission axis along **+z**.

### Emittance tab **[2026-verified]**

| control | set to |
|---|---|
| Enabled | checked |
| Weight Factor | **1.0000** — relative ray share between sources, not a flux scale |
| Total Flux/Power | **Radiometric Power** (watts) |
| Radiometric Power | **1.0 W** — see the box below |
| Measured Over | **Whole Sphere** |
| Starting Point Classification | **Semi-automatic** (default; source sits in ambient) |
| Aim Region | **Aim Sphere** |
| Angular Distribution | **Uniform** |
| Polarization | **Unpolarized** |

**Angular Distribution = Uniform, not Cosine.** Uniform means constant radiant
intensity per solid angle — an isotropic point lamp, which is what
`nonseq.sample_point_source` samples (cos θ uniform). *Cosine* is a Lambertian
emitter and would bias the beam toward the axis; every irradiance number below
would then be wrong while the total flux still looked right.

**`Measured Over` decides what the number in the box means.** Two correct ways,
one trap:

| Radiometric Power | Measured Over | traced flux in the cone | verdict |
|---|---|---|---|
| **1.0 W** | **Whole Sphere** | 0.020038221 W | **use this** — mirrors `P_TOTAL = 1 W` isotropic in the repo |
| **0.020038221 W** | **Aim Region** | 0.020038221 W | equivalent, entered directly |
| 0.020038221 W | Whole Sphere | 4.015e-4 W | **wrong, ~50× low** — the screenshot state |

Your screenshot has 0.020038221 W *with* Whole Sphere. Change one of the two.

### Aim Sphere tab **[2026-verified]** — this is where the cone lives

| group | field | set to |
|---|---|---|
| Angle | **Upper** | **0.00** degrees |
| Angle | **Lower** | **16.27** degrees |
| Orientation | Alpha | 0.00 |
| Orientation | Beta | 0.00 |
| | Draw Aim Region | **checked** |

`Upper`/`Lower` are polar angles measured from the aim axis, and the dialog
enforces `0 ≤ Upper ≤ Lower ≤ 180`. The default 0/180 is the whole sphere. A
forward cone of half-angle θ is therefore `Upper = 0`, `Lower = θ` — the source
emits into the polar band between them, and with Upper = 0 that band is the cone.

Alpha/Beta = 0/0 aims the cone along the source's local +Z, which is global +z
once Coordinates has zero rotations. Keep **Draw Aim Region** checked and look at
the 3D view: the cone must open toward the lens at z = +40. If it points at the
backward receiver instead, set `Beta = 180`.

#### The field takes 2 decimals — round the cone DOWN, never up

Exact fill angle is `atan(12.7 / 43.498695) = 16.275839356°` (43.498695 = 40 +
sag at r = 12.7). You cannot type that. The two roundings are **not** symmetric:

| entered | ray at Lower hits S1 at r = | consequence |
|---|---|---|
| **16.27** | 12.694365 mm | cone sits **inside** the rim. Nothing vignettes → `T1T2`, `R1`, `T1R2T1` stay **exactly** 64.0000 / 20.0000 / 12.8000 |
| 16.28 | 12.704016 mm | cone **overfills** the rim by 4 µm. ~0.05 % of rays miss the lens entirely → `T1T2` drops to ≈ 63.967 %, which **fails** the ±0.01 tolerance in Step 9 with every bit of physics still correct |

So: **Lower = 16.27**. The three closed forms hold for *any* cone contained in
the aperture — they are per-ray ratios — so shrinking the cone costs nothing,
while growing it past the rim breaks them.

#### Φ_cap for the 16.27° cone

| θ | Φ_cap = 1 W × (1 − cos θ)/2 | vs exact |
|---|---|---|
| 16.275839356° (ideal) | 0.020038221 W | — |
| **16.27° (what you enter)** | **0.020023941 W** | −0.0713 % |

Every percentage in §7/§8 is a fraction of Φ_cap, so with `1.0 W` + `Whole
Sphere` (Step 1) LightTools launches 0.020023941 W into the cone and **all
fractions are unchanged** — normalise by what it reports, not by 0.020038221. If
instead you enter the power directly with `Measured Over = Aim Region`, type
**0.020023941 W**.

Absolute watts in §7 are then 0.0713 % low across the board — 0.0128245 →
0.0128154 W for `T1T2`, and so on. Compare *fractions*, not watts.

### Immersion tab **[confirm]**

Source **not immersed** — starting medium is ambient air. If an immersing element
is picked up here, the first refraction is computed against glass and the cone no
longer fills S1.

### Spectral Region tab **[confirm]**

Single wavelength **532.8 nm** (monochromatic). Verify on the **Spectral Region
Chart** tab: one line, nothing else. The greyed *Photometric Flux* readout
(679.55 lm in the screenshot) is just the photopic conversion of the radiometric
value — a useful "did the wavelength take" indicator, nothing to set.

### Ray count is not here

Rays per run live in the Simulation Manager (Steps 5–6). Weight Factor only
splits rays *between* sources.

**Check before moving on:** cone edge at 16.28°, and the flux actually launched
into the cone reads 0.0200382 W in the simulation output — not just in this box.

---

## Step 2 — the lens

`Insert → Lens` **[2026-verified]** opens a **Place Singlet** palette
of icons. **[2026-verified, partial]**

**Pick the parametric one — the icon labelled `R1` / `R2` with the diameter
arrow `D`** (the biconvex profile with dimension callouts, no numbered points).
It opens a dialog where R1, R2, thickness and diameter are typed as numbers.

Avoid every icon with **numbered points (1,2,3,4,5,6)** — those are interactive
placement modes: LightTools asks you to click points in the 3D view and infers
the geometry from where you clicked. Nothing here can be clicked to 6 decimals.
The folder icons are library/saved elements, not what you want.

Picking it runs **`CreateQuickLens`**, which prompts in the command bar for
**curvature**, not radius. **[2026-verified]** Curvature is `c = 1/R`:

| prompt | value |
|---|---|
| front surface curvature | **0.0481634296** (= 1/20.762641) |
| back surface curvature | **0** (flat) |
| thickness | **6.5** |
| diameter / aperture | **25.4** (semi-diameter 12.7) |
| position | front vertex at z = **40.0** |

Full precision matters — 0.04816 instead of 0.0481634296 moves the collimation
point by ~0.3 mm and check 4 fails.

`CreateQuickLens` also prompts for **material**. It wants a catalogue name, not
an index — type **`BK7`** (or `N-BK7`) to get past it, then override the index in
the Glass step below. The catalogue dispersion is *not* what this comparison
uses; a placeholder here is fine because the next step replaces it with a
constant index. If the prompt does accept a numeric index or a "user defined"
keyword, enter the constant from the Glass step directly.

Sign: **positive** curvature must bulge the front surface *toward the source*
(vertex at z = 40, rim at z = 43.498695). If the solid comes out bulging the
other way, negate it and re-check edge thickness = 3.001305 mm.

**Conic is not in the Place Singlet dialog.** It creates spherical surfaces.
After placing, open S1's surface properties and change type to
**Aspheric/Conic**, then set `Conic Constant = −2.307561590` with all polynomial
terms 0. Re-check the radius survived the type change.

If your workflow builds surfaces individually instead, make S1 and S2 and join
them into one solid in the System Navigator — the result must be **one solid** so
LightTools knows the glass is between the surfaces.

### S1 — front, aspheric

| field | value |
|---|---|
| surface type | Aspheric (or Conic) |
| vertex position z | **40.0** |
| Radius | **+20.762641** mm |
| Conic Constant | **−2.307561590** |
| A4 … A12 polynomial terms | **0** — pure conic |
| aperture | circular, semi-diameter **12.7** |

### S2 — back, flat

| field | value |
|---|---|
| surface type | Plane |
| vertex position z | **46.5** (centre thickness 6.5) |
| Radius | infinite / curvature 0 |
| aperture | circular, semi-diameter **12.7** |

### Glass

Double-click the solid → `Material` tab **[confirm]** → **User Defined**,
constant index:

- ambient entered as 1.000293 → glass **1.519511112**
- ambient rounded to 1.0003 → glass **1.5195217454** (Step 0)

Not the Schott catalogue: LightTools' N-BK7 fit gives ≈1.51947 at 532.8 nm, ours
1.519511. A 4e-5 difference is irrelevant to power but turns the collimation gate
from razor-sharp into a judgement call.

Aperture, if not set inline: right-click each surface → `Aperture` → circular,
12.7 semi-diameter. No mount, no barrel, no baffles, no edge treatment — the rim
is a hard circular edge and rays past it simply miss.

### Sag sign — check this before continuing

`z = 40 + sag(r)`, and `sag(12.7) = 3.498695`, so:

- S1 **vertex** at z = 40.000000 — the *closest* point of the lens to the source
- S1 **rim** at z = 43.498695
- **edge thickness = 6.5 − 3.498695 = 3.001305 mm**

Measure the edge thickness in LightTools. If it reads 9.999 instead of 3.001,
the radius sign is flipped: the surface bulges away from the source instead of
toward it, and collimation is dead. This is the single easiest thing to get
wrong in the whole build.

Why this prescription collimates: `c = 1/(s(n−1))` with `k = −n²` is the exact
aspheric collimator for a point at s = 40 mm — zero spherical aberration at any
aperture. Direct rays leave S2 parallel to +z to machine precision.

---

## Step 3 — the coating, both surfaces

Flat **20 % reflectance, 80 % transmittance, 0 % absorptance**, at every angle,
unpolarized.

There is no control literally named "Simple Coating" in 2026. The equivalent is
the **Smooth Optical** property type with the percentages typed in directly.
**[2026-verified]**

1. Right-click S1 → `Optical Properties` (or the Optical Property Manager).
2. **Optical Properties** tab → type **Smooth Optical** (the default), give it a
   Description so you can find it again.
3. **Smooth Optical** tab:

   | field | value |
   |---|---|
   | **Ray Trace Mode** | **Split Rays (Reflected and Transmitted)** |
   | **Reflectance** | **20.00 %** |
   | **Transmittance** | **80.00 %** |
   | Absorption | 0.00 % (greyed, = 100 − R − T) |
   | Preferred Direction | **unchecked** |
   | **Advanced Properties** | **None** |
   | Include Contamination Scatter | unchecked |

4. **`Ray Trace Mode` is the ray-splitting control** — not the Simulation
   Manager, not `Preferred Direction`. The default **Transmitted/TIR Rays**
   suppresses reflection entirely: the lens collimates perfectly, the backward
   receiver stays empty, and the forward receiver reads ~100 % of Phi_cap
   instead of 64 %. The dropdown offers `Transmitted/TIR Rays`, **`Split Rays
   (Reflected and Transmitted)`**, `Transmitted Rays Only`, `Reflected Rays
   Only`, `TIR Rays Only`. **Split Rays is `trace_split`.**
5. **`Advanced Properties = None` is the "Fresnel OFF" of the old wording.**
   Selecting **Fresnel Loss** replaces your flat 20 % with the angle-dependent
   physical reflectance and the closed forms stop being exact. `Coating` and
   `Polarizing Element` are equally wrong here.
6. Apply the same property to **S2**. In practice one property named
   `Transmitting` already covers both lens surfaces — edit it once rather than
   making a second.

The lens also has a third surface, **EdgeSurface** — the 3.001305 mm cylindrical
rim. Leave it on **Optical Absorber**. `diffoptics` has no edge surface at all;
its two surfaces are simply cut at r = 12.7 and rays going past fly away. Either
way such a ray never completes `T1R2R1T2`, which is exactly why the ghost is
0.0132 rather than its 0.0256 bound. Only the ledger label differs — dO calls it
`off`, LightTools calls it absorbed.

`Advanced Properties` and ray splitting are **independent knobs**. Turning
Fresnel off does not disable Monte Carlo sampling — it only decides *what R is*.
`trace_mc` takes ρ from R (`rho = R.detach().clone()`,
[c02_R02.py:398](../../examples/nonseq/c02_R02.py#L398)), so a fixed R = 0.2
gives ρ = 0.2, `w × R/ρ = w`, and the MC estimator becomes zero-variance in
weight — only *which* path exists stays random. Fixed R makes MC cleaner, not
absent.

Both surfaces carry it on both sides. A constant 0.2 at all angles is what
`Element(R_fixed=0.2)` implements, and it is what makes the closed forms in the
reference table exact. An angle-dependent reflectance drifts `T1T2` off
64.0000 % and the comparison loses its anchor.

---

## Step 4 — the receivers

LightTools attaches receivers to geometry, so each one is two objects: a carrier
surface, then a receiver on it.

### 4a — forward receiver, z = +200

The menu is **`Insert`**, not `Create`. **[2026-verified]**

1. **`Insert → Dummy Surface`** → rectangular, size **32 × 32 mm**, centred at
   **(0, 0, 200)**, normal along z.

   A dummy surface is non-blocking *by definition* — rays cross it with no
   refraction, reflection or loss. That is why it is the right carrier: no
   optical property to set, nothing to get wrong. Do **not** build a thin Block
   and try to make it transparent instead.

2. **`Insert → Receiver`** → **Surface Receiver**, attached to that dummy
   surface.
3. Receiver mesh **[confirm]**:
   - Illuminance mesh, **512 × 512** bins
   - extents ±16 mm in x and y (pitch 0.0625 mm)
   - quantity **irradiance, W/mm²** (reads W because Step 0 set Radiometric)
4. **Enable ray path sorting on this receiver** — the per-path power table in §7
   only exists if paths are recorded during the run, and it cannot be recovered
   afterwards. Raise the max-paths cap so the four named paths are not merged
   into an "other" bucket.

### 4b — backward receiver, z = −80

Same procedure — `Insert → Dummy Surface`, then `Insert → Receiver` — with:

- dummy surface centred at **(0, 0, −80)**, size **160 × 160 mm**
- mesh **512 × 512**, extents ±80 mm (pitch 0.3125 mm)
- irradiance, W/mm²
- ray path sorting enabled

---

## Step 5 — simulation settings

`Ray Trace → Simulation Manager` → **Simulation Input** dialog. Tabs:
`Forward | Backward | Hybrid | Sequences | Data Collection | Ray Paths | Update
| Spectral | Random Numbers`. **[2026-verified]**

### Forward tab

| field | value | why |
|---|---|---|
| Enable Forward Simulation | checked | |
| **Total Rays to Trace** | **50,000** (Run A) / **200,000** (Run B) | matches `N_SPLIT` / `N_MC` ([c02_R02.py:492-493](../../examples/nonseq/c02_R02.py#L492-L493)) |
| **Relative Ray Power Threshold** | **1e-8** | default is **0.01** — far too tight, eats the ghost tail and the TIR-guided path. Ours is 1e-5 |
| Show Preview Rays | off | slow at 50k |
| Show Ray Report | on | gives the flux summary |

### Ray Paths tab — required, or there is no comparison

Default reads *"No Ray Paths will be collected."* Tick **Collect** on all three:

| row | gives |
|---|---|
| **ForwardAll** | `path_powers` equivalent — receivers ignored, every ray that flew that path. **This is where 0.640 / 0.200 / 0.128 / 0.0132 live** |
| **Forward.Receiver_5** | forward receiver landing table |
| **Forward.Receiver_7** | backward receiver landing table |

`ForwardAll` is the one that matters: the three closed forms are
aperture-independent. `R1` is 20.0000 % of flux but only 4.1988 % lands inside
±80 mm — different numbers, both correct.

Leave **"Limit Rays per Path" unchecked** — that is a display cap, not a depth
limit.

### Other tabs

| tab | set |
|---|---|
| Spectral | single wavelength **532.8 nm** |
| Random Numbers | note the seed; Run B needs 5 different ones |
| Data Collection / Update | check here for a max-intersection cap |

### Max intersections — not found in 2026 **[open]**

Not on Forward or Ray Paths. The **threshold does most of the work**: with
R = 0.2, weight after k partial reflections ≈ 0.2^k, so a 1e-8 cull terminates
near k ≈ 11 — close to `MAX_DEPTH = 10`.

**But TIR paths do not decay.** R = 1 at TIR, so weight never drops and no
threshold stops them. That is exactly the `h8r6` path in §7 which *outranks*
shorter partial-reflection paths ([c02_R02.py:296-300](../../examples/nonseq/c02_R02.py#L296-L300)).
If no explicit cap exists, record it as a known asymmetry rather than assume
parity.

### Splitting is NOT set here **[2026-verified]**

It lives on the **optical property**, `Ray Trace Mode` dropdown (Step 3):

| run | Ray Trace Mode | Preferred Direction | equivalent |
|---|---|---|---|
| **Run A** | **Split Rays (Reflected and Transmitted)** | unchecked | `trace_split` |
| **Run B** | Split Rays | **checked** + Probabilistic Ray Split, threshold **1.0** | `trace_mc` |

Leaving Ray Trace Mode on its **Transmitted/TIR Rays** default is the single
most likely way to get a clean-looking run with no reflection anywhere: perfect
collimation, empty backward receiver, forward receiver at ~100 % instead of
64 %.

---

## Step 6 — run twice

### Run A — ray splitting **ON** → compares against `trace_split`

`Ray Splitting = ON`, 10⁵–10⁶ rays. Splitting is not noise-limited on the direct
path, so more rays buy little.

Expect the exact fractions in §7 — 64.0000 / 20.0000 / 12.8000 — to the digit.
Pure geometry plus a constant coating, no sampling anywhere in either code.

### Run B — ray splitting **OFF**, 10⁶ rays → compares against `trace_mc`

`Ray Splitting = OFF`. This is the apples-to-apples check: LightTools' MC and
`trace_mc` are the same estimator class — one child per hit, branch chosen with
probability ρ, weight divided by ρ.

Repeat at 10⁵ / 10⁶ / 10⁷ rays to confirm the error falls with a log-log slope
of ≈ −0.5. Run 5 different random seeds at 10⁶ so you have an error bar; §8
explains why you need one.

---

## Step 7 — sanity checks before trusting anything

Run these on Run A, in this order. Each isolates one class of setup error.

| # | check | expect | fails if |
|---|---|---|---|
| 1 | edge thickness of the lens | 3.001305 mm | S1 radius sign flipped |
| 2 | flux launched into the cone | 0.0200239 W (16.27° cone) | `Measured Over` / power pair wrong (Step 1), or cone angle wrong |
| 3 | direct beam diameter at z = 200 | 25.4 mm, hard-edged | not collimated — index or conic wrong |
| 4 | direct beam divergence | \|d_x\|,\|d_y\| < 1e-6 (< 1e-5 if you left ambient 1.0003 uncompensated) | index or conic wrong |
| 5 | forward receiver total | ≈ 64.2 % of Φ_cap | coating is Fresnel, not 0.2 |
| 6 | energy balance closes | to < 0.1 % | flux threshold or depth too tight |

Check 3 is the fastest visual: the direct beam is a flat disc of radius 12.7 mm
at **2.531e-5 W/mm²**, plus a faint ghost halo out to 22.6 mm.

---

## Reference numbers

### §7 — `trace_split`, deterministic

2×10⁶ rays, seed 0, `w_min = 1e-5 × w_ray`, `max_depth = 10`, float64.
Reproduce with `python examples/nonseq/c02_R02.py`.
`Φ_cap = 0.020038221 W`.

**Ledger**

| bucket | W | % of Φ_cap |
|---|---|---|
| forward receiver | 0.0128600 | 64.1772 |
| backward receiver | 0.0034307 | 17.1208 |
| neither | 0.0037458 | 18.6933 |
| culled | 1.06e-07 | 0.0005 |
| in flight at depth 10 | 1.66e-06 | 0.0083 |
| **total** | **0.020038221** | **100.0000** |

The 18.7 % is a real answer, not a leak — mostly R1, which sprays far wider than
±80 mm.

**Power by path** — the sharpest comparison. Aperture-independent: every ray that
flew that way. In LightTools, get these from the ray path table filtered on hit
sequence.

| path | hits | refl | W | % | closed form |
|---|---|---|---|---|---|
| `T1T2` direct | 2 | 0 | 0.0128245 | **64.0000** | T² = 0.640 exact |
| `R1` front face | 1 | 1 | 0.0040076 | **20.0000** | R = 0.200 exact |
| `T1R2T1` round trip | 3 | 1 | 0.0025649 | **12.8000** | T²R = 0.128 exact |
| `T1R2R1T2` ghost | 4 | 2 | 0.0002658 | **1.3265** | < T²R² = 2.56, vignetted |

First three are exact — the cone fills S1 exactly, T1T2 clears S2, R1 leaves
immediately. Nothing vignettes.

The ghost is the point of the exercise. Its R1 bounce is on the *curved* face
from inside — a strong concave mirror — and about half leaves through the rim.
No thin-lens argument gives 1.33 %. **If LightTools also lands near 1.33 %, the
geometry layer is right.**

**What lands on each receiver**

| recv | hits | refl | W | % | reading |
|---|---|---|---|---|---|
| fwd | 2 | 0 | 0.0128245 | 64.0000 | `T1T2` |
| fwd | 4 | 2 | 0.0000026 | 0.0132 | ghost, only 2 % of it lands |
| fwd | 8 | 6 | 0.0000327 | 0.1631 | TIR-guided |
| back | 1 | 1 | 0.0008414 | 4.1988 | `R1`, only 21 % lands |
| back | 3 | 1 | 0.0025649 | 12.8000 | `T1R2T1`, all lands |
| back | 5 | 3 | 0.0000163 | 0.0812 | |
| back | 9 | 7 | 0.0000066 | 0.0331 | |

`h8r6` beats `h4r2` and `h6r4` because those reflections are **TIR** — R = 1,
free. Partial reflections cost 0.2 each; TIR costs nothing, so a long guided
path can outrank a short partial one. LightTools reproduces this if TIR is on
and depth allows. A clean monotonic decay with order means your depth cap is too
low.

**Beam geometry**

| | |
|---|---|
| forward 95th-pct radius | 14.58 mm |
| forward max radius | 22.61 mm (ghost tail, off receiver) |
| `T1T2` divergence | \|d_x\|, \|d_y\| < 1e-6 |
| backward 95th-pct radius | 80.57 mm |

Compare the plateau level and the radial profile, not the hottest pixel — the
disc edge lands between pixels.

### §8 — `trace_mc`, for Run B

10⁶ rays, mean ± sample std over seeds 0–4, % of `Φ_cap`:

| quantity | MC, 10⁶ rays | split (exact) | agreement |
|---|---|---|---|
| forward receiver | 64.1522 ± 0.0852 | 64.1772 | 0.3 σ |
| backward receiver | 16.2986 ± 1.8681 | 17.1208 | 0.4 σ |
| neither | 19.5416 ± 1.9460 | 18.6933 | 0.4 σ |
| `T1T2` | 63.9911 ± 0.0540 | 64.0000 | 0.2 σ |
| `R1` | 20.0025 ± 0.0590 | 20.0000 | 0.0 σ |
| `T1R2T1` | 12.8230 ± 0.0283 | 12.8000 | 0.8 σ |
| ghost `T1R2R1T2` | 1.3879 ± 0.1415 | 1.3265 | 0.4 σ |
| ledger closure | 1.0000 ± 0.0000 | 1.0000 | exact |

Every bucket is unbiased — that is the T4 gate in `c02_R02.py`.

Match your LightTools tolerance to the spread. The **backward receiver is ~20×
noisier in relative terms than the forward one** (±1.87 vs ±0.085, on a smaller
number). Reason: it is dominated by `R1`, of which only 21 % lands inside
±80 mm, so at 10⁶ rays only ~2×10⁵ rays take that branch and a small noisy
subset survives the aperture. LightTools' MC behaves the same way. **Do not read
a 1–2 point backward difference as a discrepancy** — run 5 seeds and compare
distributions, not single numbers.

The path fractions are the tight ones: ±0.05 on `T1T2` and `R1`. Compare those.

---

## Step 8 — export

Four files into `examples/nonseq/c03_out/`:

```
lt_fwd_split.csv    lt_back_split.csv
lt_fwd_mc.csv       lt_back_mc.csv
```

**Where:** right-click the receiver in the System Navigator → `Charts →
Illuminance` → in the chart window, `File → Export` or right-click →
`Export Data` → CSV.

Row format, header skipped on read:

```
x_mm,y_mm,E
-15.96875,-15.96875,0.0
```

- `x_mm`, `y_mm` = **cell centres**, receiver-centred. Forward runs
  −15.96875 … +15.96875 step 0.0625; backward −79.84375 … +79.84375 step
  0.3125. `nonseq.splat` uses cell centres — a half-pixel offset shows up as a
  spurious radial ring in the difference map.
- `E` in **W/mm²**. Convert on export if LightTools gives lm/mm² or W/cm².
- 262144 rows per file.

Also export as plain text: total flux per receiver, total emitted, lost/escaped,
and the ray path table filtered to the four named paths. That path table is more
diagnostic than either irradiance map.

---

## Step 9 — compare

`examples/nonseq/c03_compare.py` — reruns P3 at matched ray budget, dumps the
same maps via `nonseq.splat`, reads the CSVs, asserts the grids match to
< 1e-9 mm, then reports per receiver per run: total power, relative L2
`||A−B||₂/||A||₂`, radial profile overlay, signed difference map, path table vs
closed forms, both ledgers.

| comparison | tolerance |
|---|---|
| `T1T2` | 64.0000 % ± 0.01 |
| `R1` | 20.0000 % ± 0.01 |
| `T1R2T1` | 12.8000 % ± 0.01 |
| ghost | 1.33 % ± 0.05 abs |
| forward total, run A | ± 0.5 % rel |
| backward total, run A | ± 1 % rel |
| ledger closure, either code | < 0.1 % |
| forward map rel. L2, run A | < 2 % |
| run B `T1T2`, `R1` vs `trace_mc` | ± 0.05 abs |
| run B backward total vs `trace_mc` | ± 2 abs — genuinely that noisy |
| run B error vs ray count | log-log slope ≈ −0.5 |

The first three have no free parameters — a mismatch there is a real
disagreement about physics or geometry, not a setup difference.

---

## Forced-agreement checklist

Deliberate modelling choices, not physics. Get one wrong and you measure the
setup instead of the tracer.

| # | item | this repo | force in LightTools | set in |
|---|---|---|---|---|
| 1 | reflectance | constant 0.2 all angles | simple coating, Fresnel OFF | Step 3 |
| 2 | glass index | 1.519511112 | fixed, not catalogue | Step 2 |
| 3 | ambient index | 1.000293 | not 1.0 | Step 0 |
| 4 | source angular law | uniform in **solid angle** | isotropic point, cone-limited | Step 1 |
| 5 | cone half-angle | 16.275839356° | **16.27** — round down; 2-dp field | Step 1 |
| 6 | source flux | 0.020038221 W in cone | not 1 W in cone | Step 1 |
| 7 | absorption | none | off | Step 5 |
| 8 | TIR | R = 1, ray continues | on — carries real flux here | Step 5 |
| 9 | ray depth | 10 | ≥ 10 | Step 5 |
| 10 | flux threshold | 1e-5 rel | 1e-8 | Step 5 |
| 11 | apertures | hard circle r = 12.7 | no mount, baffle, or edge | Step 2 |
| 12 | receiver pixels | cell-centred | cell-centred export | Step 8 |
| 13 | trace engine | CPU, float64 | CPU — GPU trace OFF (2026) | Step 5 |
| 14 | flux normalisation | 1 W isotropic, cone carries Φ_cap | Radiometric Power + `Measured Over` consistent | Step 1 |
| 15 | source immersion | source in air | not immersed | Step 1 |

Item 4 is the *Uniform vs Cosine* radio button in the Emittance tab, and it
quietly ruins irradiance comparisons. Gridding a plane and shooting one
ray per cell — what `examples/pointSrc_spherical.py` does — over-weights the
axis by 1/cos³θ. It cancels in a hit-count image and does not cancel in W/mm².
`nonseq.sample_point_source` samples `cos θ` uniformly for exactly this reason.

Item 5 matters more than it looks: the cone fills S1 to the rim, so even
+0.005° adds rays that miss the lens entirely and shifts every fraction by
hundredths of a percent with all the physics still correct. Rounding *down* is
free, rounding *up* is not — see the Aim Sphere table in Step 1.

---

## Report order

1. **The three exact fractions** — 64.0000 / 20.0000 / 12.8000 in both codes.
   Headline: the non-sequential tracer is right.
2. **The ghost** — 1.33 % in both vs a thin-lens 2.56 %. Neither an analytic
   model nor a sequential tracer gives this. The argument for the whole layer.
3. **MC vs MC** — Run B against `trace_mc`, same estimator class, plus the
   N^−0.5 slope.
4. **Irradiance maps** side by side, difference map, radial profile overlay.
5. **Energy ledger**, both codes, showing the 18.7 % "neither" bucket accounted
   for rather than lost.
6. **Sequential contrast** — `Lensgroup._trace` picks a direction once from
   `(ray.d[...,2] > 0).all()` (`optics.py:1049`), so it returns the forward path
   only. `c02_R02.py` already produces the screenshot.
7. **Gradients** — `dΦ_back/dR`, `dΦ_fwd/dc` vs finite differences, in
   `c02_R02.py`. No commercial tracer has an equivalent.

---

## Files

**New:**
```
examples/nonseq/c03_compare.py    reads the CSVs, produces the comparison
examples/nonseq/c03_out/          four CSVs + plots
```

**Existing, unchanged:** `diffoptics/nonseq.py` (`trace_split`, `trace_mc`,
`interaction`, `splat`, …), `examples/nonseq/c02_R02.py` (produces every
reference number above).

---

## Still unverified in 2026

Everything tagged **[confirm]** above. Screenshots of these dialogs would pin
them the way the source Emittance tab is now pinned:

1. `Model → System Data` — units, photometry, ambient index (does it take 6 dp?)
2. Receiver mesh dialog + ray path sorting controls
3. Simulation Manager ray-trace options — depth, threshold, TIR, splitting, GPU
4. Receiver chart export dialog — CSV column order and cell-centre convention

**Settled during the 2026 build:** the top-level menu is **`Insert`**, not
`Create` — `Insert → Light Source`, `Insert → Lens`, `Insert → Dummy Surface`,
`Insert → Receiver`. The other top-level menus seen alongside it are `Modify`,
`Ray Trace`, `Analysis`.
