"""
MODULE 10 - Render a real IMAGE through the lens (backward/MTS tracing).

Modules 4-8 rendered irradiance from a source. To image a SCENE we trace
BACKWARD: rays start at sensor pixels, go out through the lens, and hit a textured
screen; the screen colour they land on becomes the pixel value. This is
`render_image.py`, shrunk for CPU.

Pipeline:
  1. lens.prepare_mts(pixel_size, film_size)   # reverse the lens into a camera (optics.py:863)
  2. do.Screen(transform, size, texture)       # the scene, a textured plane (shapes.py:29)
  3. per pixel: valid, ray = lens.sample_ray_sensor(wl)  # backward rays (optics.py:953)
  4. uv, valid_s = screen.intersect(ray)[1:]   # where each ray hits the screen
  5. colour = screen.shading(uv, mask)         # bilinear texture lookup (shapes.py:68)
  6. average many aperture samples per pixel (multi-pass) -> smooth image.

We render R, G, B separately (chromatic tracing) and stack.

READ BEFORE (tick when done):
  [ ] diffoptics/optics.py:863-902   prepare_mts (reverse lens into a camera)
  [ ] diffoptics/optics.py:953-1010  sample_ray_sensor + _sample_ray_render
  [ ] diffoptics/shapes.py:29-111    Screen (intersect + shading)
  [ ] examples/render_image.py       the workflow this mirrors

Run: python m10_image_render_practice.py
Expect: I_rendered.png in ./m10_out/ (a squirrel, upside-down flips handled).
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.image import imread

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import diffoptics as do
from common_lensfile import load_lens
from common_setup import save_dir

out = save_dir("m10_out")
device = torch.device("cpu")

lens = load_lens("doublegauss", device=device)

pixel_size = 6.45e-3 * 4        # [mm] (4x coarser than sensor for CPU speed)
film_size = [128, 96]

# TODO 1: turn the lens into a backward-tracing camera.
#   hint: lens.prepare_mts(pixel_size, film_size)
...
lens.prepare_mts(pixel_size, film_size)

# scene: a textured screen far in front of the lens
z0 = 10e3                       # [mm]
tex_pixel = 1.1                 # [mm] per texture pixel
tex = imread(r"C:\Users\Twins\Desktop\Imlex\Internships\DiffOptics\examples\images\squirrel.jpg").astype(np.float32)
tex = np.flip(tex, axis=(0, 1)).copy()      # pre-flip so final image is upright
tex_size = np.array(tex.shape[0:2])
screen = do.Screen(
    do.Transformation(np.eye(3), np.array([0, 0, z0])),
    tex_size * tex_pixel, torch.Tensor(tex).to(device), device=device,
)

wavelengths = [656.2725, 587.5618, 486.1327]    # R, G, B [nm]
rays_per_pixel = 100


def render_single(wavelength):
    # TODO 2: sample backward rays from the sensor for this wavelength.
    #   hint: valid, ray = lens.sample_ray_sensor(wavelength)
    valid, ray = lens.sample_ray_sensor(wavelength)

    # TODO 3: intersect the screen -> uv + validity, combine masks, shade.
    #   uv, valid_screen = screen.intersect(ray)[1:]
    #   mask = valid & valid_screen
    #   return screen.shading(uv, mask), mask
    uv, valid_screen = screen.intersect(ray)[1:]
    mask = valid & valid_screen
    return screen.shading(uv, mask), mask


if __name__ == "__main__":
    Is = []
    for wid, wl in enumerate(wavelengths):
        screen.update_texture(torch.Tensor(tex[..., wid]).to(device))
        I = 0.0
        Mcount = 0
        for _ in range(rays_per_pixel):
            I_i, mask = render_single(wl)
            I = I + I_i
            Mcount = Mcount + mask
        I = I / (Mcount + 1e-10)
        I = I.reshape(*np.flip(np.asarray(film_size))).permute(1, 0)
        Is.append(I.cpu())

    I_rgb = torch.stack(Is, axis=-1).numpy().astype(np.uint8)
    plt.imsave(out + "I_rendered.png", I_rgb)
    print("rendered", I_rgb.shape, "-> saved to", out)
