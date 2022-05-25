import sys
sys.path.append("../src")
import numpy as np
from tweezer_force import Domain, Tweezers
import matlab.engine
from constants import units
import matplotlib.pyplot as plt


um = units["um"]["value"]
nm = units["nm"]["value"]
mW = units["mW"]["value"]
pN = units["pN"]["value"]
fN = units["fN"]["value"]


remake=False

if remake:
    run = True
    save = True
    load = False

else:
    run = False
    save = False
    load = True

powers = {"F": 70*mW, "B": 56*mW}

if run:
    C = Tweezers(wavelength0=1064*nm, n_medium=1.0)
    C.make_Tmatrix(R=1.585*um, n_particle=1.45)
    C.make_beam(name="F", waist=2*um, polarization=[1, 0], angle=0.000)
    C.make_beam(name="B", waist=2.2*um, polarization=[0, 1], angle=180.0)
    C.make_domain(xmin=-3*um, xmax=3*um, Nx=101,
                  ymin=-3*um, ymax=3*um, Ny=101,
                  zmin=-15*um, zmax=5*um, Nz=101)
    C.evaluate_force(name="F", offset=[0*um, 0*um, -12.5*um])
    C.evaluate_force(name="B", offset=[0*um, 0*um, 12.5*um])
    C.evaluate_net_force(powers=powers)
    C.make_interpolation(name="net")
    if save:
        C.save(nickname=None)



if load:
    C = Tweezers()
    C.load("../data/21d4619009086907a47a04a87dd52ccd94d0eacc")
    C.evaluate_net_force(powers=powers)
    C.make_interpolation(name="net")

Deltas = [100 * nm, 100 * nm, 700 * nm]
xyz0 = C.find_min()
print(xyz0)
meshspec = [[q0 - Delta, q0 + Delta, 21] for q0, Delta in zip(xyz0, Deltas)]
fit_domain = Domain(meshspec)
(xyz0, dxyz0), (kxyz, dkxyz) = C.fit_min(domain=fit_domain)
print("xyz0", xyz0)
print("kxyz", kxyz / (fN/nm))


Delta = 1000 * nm
meshspec = [[q0 - Delta, q0 + Delta, 101] for q0 in xyz0]
plot_domain = Domain(meshspec)
fig, axs = plt.subplots(3, 1)
x, y, z = plot_domain.axes

axs[0].plot(x/nm, -kxyz[0] * (x-xyz0[0])/fN)
axs[1].plot(y/nm, -kxyz[1] * (y-xyz0[1])/fN)
axs[2].plot(z/nm, -kxyz[2] * (z-xyz0[2])/fN)

C.plot_linecut(domain=plot_domain,
               plot_kwargs={"marker": ".", "ls": "none"},
               ax=axs[0],
               component="x",
               unit="nm",
               Funit="fN",
               line=["*", xyz0[1], xyz0[2]])
C.plot_linecut(domain=plot_domain,
               plot_kwargs={"marker": ".", "ls": "none"},
               ax=axs[1],
               component="y",
               unit="nm",
               Funit="fN",
               line=[xyz0[0], "*", xyz0[2]])
C.plot_linecut(domain=plot_domain,
               plot_kwargs={"marker": ".", "ls": "none"},
               ax=axs[2],
               component="z",
               unit="nm",
               Funit="fN",
               line=[xyz0[0], xyz0[1], "*"])
plt.show()

C.plot_linecuts(
    lines=[["*", xyz0[1], xyz0[2]],
           [xyz0[0], "*", xyz0[2]],
           [xyz0[0], xyz0[1], "*"]],
           unit="um",
           Funit="fN",
           )
plt.show()



planes = {k: [v] for k, v in zip("xyz", xyz0)}
C.plot_slices(unit="nm", quiver=False, contours=0,
Funit="pN", skip=10, planes=planes);
plt.tight_layout()
plt.show()
