import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import diffoptics as do 

# initialize the lens

device = torch.device('cpu')
lens = do.Lensgroup(device=device)

save_dir = './pointSrc_demo/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

R_l = 12.7
#Spherical surfaces
surfaces = [
    do.Aspheric(R_l, 0.0, c=1/50.0 ,device=device),
    do.Aspheric(R_l, 6.5, c=0.0, device=device)
]

materials = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air')
]

lens.load(surfaces, materials)
lens.d_sensor = 40.0
lens.r_last = 12.7
# receiver film: render() reads film_size + pixel_size
R_receiver = 12.7                                       # [mm] half-size of receiver area
lens.film_size = [256, 256]
lens.pixel_size = 2 * R_receiver / lens.film_size[0]    # [mm/pixel]  (scalar, not list)

# make surface curvature a differentiable leaf (autodiff target)
lens.surfaces[0].c = torch.Tensor(np.array(1/50.0))
lens.surfaces[0].c.requires_grad = True


wavelength = torch.Tensor([532.8]).to(device) # [nm]
z0 = 40.0
o_pt = torch.Tensor([0.0, 0.0, -z0]).to(device) # [mm]

M = 20
# target-grid half-size on z=0 plane. Kept < aperture so every ray stays valid
# (no aperture miss / TIR) across the whole c-sweep -> no NaN in trace or Jacobian.
R_src = 8.0

def sample_pointsrc_ray():
    # rays diverge from o_pt toward a grid of targets on the z=0 plane
    x = torch.linspace(-R_src, R_src, M)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    targets = torch.stack((X, Y, torch.zeros_like(X)), axis=-1).reshape(-1, 3).to(device)
    o = o_pt.expand_as(targets).clone()
    d = do.normalize(targets - o_pt)
    return do.Ray(o, d, wavelength, device=device)

def render():
    # sensor hit points (x,y) -- for spot diagram + spot Jacobian
    ray_init = sample_pointsrc_ray()
    ps = lens.trace_to_sensor(ray_init)
    return ps[..., :2]

def trace_all():
    # sensor points + full paths -- for the raytrace layout figure
    ray_init = sample_pointsrc_ray()
    ps, oss = lens.trace_to_sensor_r(ray_init)
    return ps[..., :2], oss


def compute_Jacobian(ps):
    # dps/dc via reverse-mode autograd, one output element at a time (autodiff.py:45)
    Js = []
    for i in range(1):
        J = torch.zeros(torch.numel(ps))
        for j in range(torch.numel(ps)):
            mask = torch.zeros(torch.numel(ps))
            mask[j] = 1
            ps.backward(mask.reshape(ps.shape), retain_graph=True)
            J[j] = lens.surfaces[i].c.grad.item()
            lens.surfaces[i].c.grad.data.zero_()
        J = J.reshape(ps.shape)
    Js.append(J.cpu().detach().numpy())
    return Js


# --- sweep over surface curvatures ---
RMS = lambda ps: torch.sqrt(torch.mean(torch.sum(torch.square(ps), axis=-1)))
N = 20
cs = np.linspace(0.045, 0.063, N)
Iss = []
Jss = []
for index, c in enumerate(cs):
    index_string = str(index).zfill(3)

    # set curvature for this step (fresh differentiable leaf)
    lens.surfaces[0].c = torch.Tensor(np.array(c))
    lens.surfaces[0].c.requires_grad = True

    # layout raytrace
    ps, oss = trace_all()
    ax, fig = lens.plot_raytraces(oss, color='b-', show=False)
    ax.axis('off'); ax.set_title("")
    fig.savefig(save_dir + "layout_trace_" + index_string + ".png", bbox_inches='tight')
    plt.close(fig)

    # spot diagram + RMS
    ps = render()
    print(f'[{index_string}] c={c:.4f}  RMS={RMS(ps):.4f}')
    lens.spot_diagram(ps, xlims=[-4, 4], ylims=[-4, 4],
                      savepath=save_dir + "spotdiagram_" + index_string + ".png", show=False)

    # spot Jacobian dps/dc + flow quiver
    Js = compute_Jacobian(ps)[0]
    ps_ = ps.cpu().detach().numpy()
    fig = plt.figure()
    x, y = ps_[:, 0], ps_[:, 1]
    plt.plot(x, y, 'b.', zorder=0)
    plt.quiver(x, y, Js[:, 0], Js[:, 1], color='b', zorder=1)
    plt.xlim(-4, 4); plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
    fig.savefig(save_dir + "flow_" + index_string + ".png", bbox_inches='tight')
    plt.close(fig)

    # irradiance image I + irradiance Jacobian dI/dc
    I = lens.render(sample_pointsrc_ray())
    I_np = I.cpu().detach().numpy()
    lm = do.LM(lens, ['surfaces[0].c'], 1e-2, option='diag')
    JI = lm.jacobian(lambda: lens.render(sample_pointsrc_ray())).squeeze()
    J_np = JI.abs().cpu().detach().numpy()

    Iss.append(I_np)
    Jss.append(J_np)
    plt.close()

Iss = np.array(Iss)
Jss = np.array(Jss)
for i in range(N):
    plt.imsave(save_dir + "I_" + str(i).zfill(3) + ".png", Iss[i], cmap='inferno')
    plt.imsave(save_dir + "J_" + str(i).zfill(3) + ".png", Jss[i], cmap='inferno')