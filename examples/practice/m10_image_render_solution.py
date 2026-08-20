"""MODULE 10 - SOLUTION. Render an image through the lens (MTS backward tracing)."""
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

pixel_size = 6.45e-3 * 4
film_size = [128, 96]
lens.prepare_mts(pixel_size, film_size)

z0 = 10e3
tex_pixel = 1.1
tex = imread(os.path.join("..", "images", "squirrel.jpg")).astype(np.float32)
tex = np.flip(tex, axis=(0, 1)).copy()
tex_size = np.array(tex.shape[0:2])
screen = do.Screen(
    do.Transformation(np.eye(3), np.array([0, 0, z0])),
    tex_size * tex_pixel, torch.Tensor(tex).to(device), device=device,
)

wavelengths = [656.2725, 587.5618, 486.1327]
rays_per_pixel = 8


def render_single(wavelength):
    valid, ray = lens.sample_ray_sensor(wavelength)
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
