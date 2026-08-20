"""
Shared helper for modules that use a REAL lens loaded from a `.txt` file
(as shipped in examples/lenses/). Used by M03 (spot diagrams), M07 (LM spot
optimization), M10 (image rendering).

`lens.load_file(path)` (diffoptics/optics.py:65) parses the ASCII lens table via
`read_lensfile` (optics.py:98): each row = surface_type, thickness, ROC, diameter,
material. It sets lens.surfaces, lens.materials, lens.r_last, lens.d_sensor.

The `.txt` format (first two lines are comments), one surface per line:
    type   thickness   ROC(0=flat)   diameter   material   [conic  a4 a6 ...]
types: O object, S aspheric, A aperture(stop), I sensor/image, X xy-poly, B b-spline.
"""
import os
from pathlib import Path

import torch

import diffoptics as do  # assumes sys.path already set by the caller module

# absolute path to examples/lenses/
LENSES = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lenses"))

# a few shipped lenses, keyed by short name
LENS_FILES = {
    "doublegauss": LENSES / "DoubleGauss" / "US02532751-1.txt",
    "thorlabs_acl": LENSES / "ThorLabs" / "ACL5040U.txt",
    "thorlabs_la": LENSES / "ThorLabs" / "LA1131.txt",
    "nikon": LENSES / "Zemax_samples" / "Nikon-z35-f1.8-JPA2019090949-example2.txt",
}

# standard photographic wavelengths [nm]: red / green(d) / blue
WAVELENGTHS_RGB = [656.2725, 587.5618, 486.1327]


def load_lens(name="doublegauss", device=torch.device("cpu"),
              film_size=None, pixel_size=None):
    """Load a shipped lens on CPU. Returns the Lensgroup.

    film_size / pixel_size are only needed if you will call lens.render().
    """
    if name not in LENS_FILES:
        raise KeyError(f"unknown lens '{name}', choose from {list(LENS_FILES)}")
    lens = do.Lensgroup(device=device)
    lens.load_file(LENS_FILES[name])
    if film_size is not None:
        lens.film_size = film_size
    if pixel_size is not None:
        lens.pixel_size = pixel_size
    return lens


def green(device=torch.device("cpu")):
    return torch.Tensor([WAVELENGTHS_RGB[1]]).to(device)
