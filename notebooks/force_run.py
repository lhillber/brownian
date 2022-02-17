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



#C = Tweezers(wavelength0=1064*nm, n_medium=1.0)
#C.make_Tmatrix(R=1.585*um, n_particle=1.2)
#C.make_beam(name="F", NA=0.70, polarization=[1, 0], angle=0.000)
#C.make_beam(name="B", NA=0.70, polarization=[0, 1], angle=180.0)
#C.make_domain(xmin=-3*um, xmax = 3*um, Nx=101,
#              ymin=-3*um, ymax = 3*um, Ny=101,
#              zmin=-3*um, zmax = 3*um, Nz=101)
#C.evaluate_force(name="F", offset=[0*um, 0*um, 0*um])
#C.evaluate_force(name="B", offset=[0*um, 0*um, 0*um])
#C.evaluate_net_force(powers={"F": 70*mW, "B": 70*mW})
#C.make_interpolation(name="net")
#C.save()

C = Tweezers()
C.load("../data/3c8b6964e7e8f728dce0a8e2735bab794b11daa4")


C.evaluate_net_force(powers={"F": 70*mW, "B": 70*mW})
C.make_interpolation(name="net")

Delta = 50 * nm
xyz0 = C.find_min()
meshspec = [[q0 - Delta, q0 + Delta, 101] for q0 in xyz0]
plot_domain = Domain(meshspec)
xyz0, kxyz = C.fit_min(domain=plot_domain)
xyz0, dxyz0 = xyz0
kxyz, dkxyz = kxyz
print(kxyz / (fN/nm))

#planes = {k: [v] for k, v in zip("xyz", xyz0)}
#C.plot_slices(unit="nm", domain=plot_domain,
#              Funit="pN", skip=10, planes=planes);
#plt.tight_layout()

fig, axs = plt.subplots(3, 1)
x, y, z = plot_domain.points.T
C.plot_linecut(domain=plot_domain,
               ax=axs[0],
               component="x",
               line=["*", xyz0[1], xyz0[2]])
C.plot_linecut(domain=plot_domain,
               ax=axs[1],
               component="y",
               line=[xyz0[0], "*", xyz0[2]])
C.plot_linecut(domain=plot_domain,
               ax=axs[2],
               component="z",
               line=[xyz0[0], xyz0[1], "*"])
axs[0].plot(x, -kxyz[0] * (x-xyz0[0]))
axs[1].plot(y, -kxyz[1] * (y-xyz0[1]))
axs[2].plot(z, -kxyz[2] * (z-xyz0[2]))
plt.show()
