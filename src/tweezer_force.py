import json
from uid import dict_uid
from os import path, pardir, getcwd, makedirs
from constants import units
from copy import deepcopy
from numpy.linalg import norm, inv
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matlab.engine

um = units["um"]["value"]
nm = units["nm"]["value"]
mW = units["mW"]["value"]
fN = units["fN"]["value"]

class Domain:
    """
    Define a 3D rectangular grid domain.

    Attributes
    ----------
        axes : List of 1D arrrays
            1D coordinate axes defined by each of the three rows of meshspec.

        deltas : list
            Step sizes of each of the three coordinate axis.

        grid : 4D array
            Three 3D coordinate arrays defined by the domain volume.

        shape : tuple
            Shape of the grid.

        coordinate_shape : tuple
            Shape of the coordinate grids.

        points : 2D array (Nx3)
            List of (x, y, z) points comprising the grid.

        points_shape : tuple
            Shape of the list of points comprising the grid.

    Methods
    -------
        interpolation(V):
            Interpolation function of scalar field V defined over the domain.

        vector_interpolation(VXYZ):
            List of interpolation functions for the components of vector
            field VXYZ defined over the domain.

    """

    def __init__(self, meshspec, spec="number"):

        """
        Constructs attributes for the Domain object.

        Parameters
        -----------
            meshspec : 3x3, array-like
                A sequence of min, max and specification values.
                [[xmin, xmax, xspec], [ymin, ymax, yspec], [zmin, zmax, zspec]]

            spec : str
                Interpretation of meshspec's third column.  'number' (default)
                specifies the number of points between the min and max. 'delta'
                specifies the step size between neighboring points.
        """

        if spec == "number":
            axes = [np.linspace(*min_max_spec) for min_max_spec in meshspec]
        elif spec == "delta":
            for min_max_spec in meshspec:
                min_max_spec[1] = min_max_spec[1] + min_max_spec[2]
            axes = [np.arange(*min_max_spec) for min_max_spec in meshspec]

        self.meshspec = meshspec
        self.bounds = [(minmaxspec[0], minmaxspec[1]) for minmaxspec in meshspec]
        self.axes = axes
        self.deltas = [x[1] - x[0] if len(x)>=2 else 0 for x in axes]
        self.grid = np.array(np.meshgrid(*axes, indexing="ij"))
        self.shape = self.grid.shape
        self.coordinate_shape = self.grid[0].shape
        self.points_shape = (3, np.prod(self.coordinate_shape))


    @property
    def points(self):
        """
        Reshape the domain grid into a list of points
        """
        return np.c_[[X.ravel() for X in self.grid]].T


    def interpolation(self, V):
        """
        Interpolation function of scalar field V defined over the domain.

        Parameters
        ----------
            V : 3D array
                Scalar field values at the domain grid points.

        Returns
        -------
            interp : scipy.interpolate.RegularGridInterpolator
                Linear interpolation function of V.
        """
        assert V.shape == self.coordinate_shape
        interp = RegularGridInterpolator(
            self.axes, V, method="linear", bounds_error=False, fill_value=0
        )
        return interp


    def vector_interpolation(self, VXYZ):
        """
        List of interpolation functions for the components of vector field VXYZ
        defined over the domain.

        Parameters
        ----------
            VXYZ : Component-wise list of 3D arrays
                Vector field values at the domain grid points.

        Returns
        -------
            interps : list
                Component-wise list of linear interpolation functions of VXYZ.
        """
        return [self.interpolation(V) for V in VXYZ]


def make_eng():
    """
    Create a matlab engine object

    Returns
    -------
        eng : matlab engine object
    """
    eng = matlab.engine.start_matlab()
    eng.ott.warning("once", nargout=0)
    eng.ott.change_warnings('off', nargout=0)
    return eng


def Xrot(deg):
    """
    Matrix representation of rotation by deg about the X axis

    Parameters
    ----------
        def : float
            Angle of rotation in degrees

    Returns
    -------
        Rmat : array (3x3)
            Rotation matrix
    """
    ang = deg*np.pi/180
    Rmat = np.array([
             [1, 0, 0],
             [0, np.cos(ang), -np.sin(ang)],
             [0, np.sin(ang), np.cos(ang)]
    ])
    return Rmat



class Tweezers:
    """
    Optical tweezer force calculations and visualizations.

    Mie scattering calculations based on the T matrix method and vector
    spherical harmonic basis functions are provided by the Optical Tweezers
    Toolbox (OTT) add on for Matlab. This class provides a Python interface
    to a small and specific set of OTT functions for calculating the force on
    a spherical dielectric particle trapped by two counter propagating, cross
    polarized, tightly focused TEM00-mode laser beams.

    Attributes
    ----------

        eng : matlab engine object
            This object proveds a python interface to matlab.

        params : dict
            Dictonary of parameters defining the system. Calling a `make_*`
            method updates the params attribute with the newly provided
            information.

        beams : dict
            A dictonary of matlab ott beam objects generated by the
            `make_beam()` method.  Keys correspond to the `name` kwarg passed
            to the `make_beam()`.

        FXYZs : dict
            A dictonary of arrays of force data generated by the
            `evaluate_force()` or `evaluate_net_force()` methods. Keys
            are either 'net' or correspond to the `name` kwarg passed to
            evaluate_force()

        interps : dict
            A dictonary of scipy.interpolate.RegularGridInterpolator objects
            generated by the `make_interpolation()` method. Keys are either
            'net' or correspond to the `name` kwarg passed to evaluate_force().
    """

    def __init__(self,
                 eng=None,
                 wavelength0=1064*nm,
                 n_medium=1.0,
                 c=3.0e8):
        """
        Initialize an instance of the Tweezer class.

        Parameters
        ----------
            eng : Matlab engine object

            wavelength0 : float, optional
                Wavelength of tweezer light. Default is 1064 nm.

            n_medium : float, optional
                Refractive index of the medium. Default is 1.0 (air).

            c : float, optional
                Speed of ligth; sets OTT units. Default is 3.0e8 m/s.

        Returns
        -------
            Tweezer object.

        """
        if eng is None:
            self.eng = make_eng()
        self.params = {"wavelength0": wavelength0, "n_medium": n_medium, "c": c}
        self.params["beam_names"] = []
        self.beams = {}
        self.FXYZs = {}
        self.interps = {}


    def file_path(self, der=None, nickname=None):
        """
        Create a unique file path (directory) for saving data.

        Parameters
        ----------
            der : str
                A base directory into which the project can create numerous
                additional data-filled directories. Default is "../data".

            nickname : str
                A uniqe name for the current state of the system. Default is
                a sha1 hash of the `params` attribute.

        Returns
        -------
            fpath : str
                A file path in which to save data. <der/nickname>
        """

        if der is None:
            base_der = path.join(getcwd(), pardir)
            der = path.join(base_der, "data")
        if nickname is None:
            nickname = dict_uid(self.params, uid_keys=self.params.keys())
        fpath = path.normpath(path.join(der, nickname))
        return fpath


    def make_domain(self,
                 xmin=-1.0 * um,
                 xmax=1.0 * um,
                 Nx=9,
                 ymin=-1.0 * um,
                 ymax=1.0 * um,
                 Ny=9,
                 zmin=-1.0 * um,
                 zmax=1.0 * um,
                 Nz=9):
        """
            Make a Domain object overwhich forces may be evaluated.

            Parameters
            ----------
                *min : float
                    Minimum value of the x, y, and z axes.

                *max : float
                    Maximum value of the x, y, and z axes.

                N* : int
                    The number of points along the x, y, and z axes.

            Returns
            -------
                domain : Domain object
                    A 3D rectangular grid domain
        """
        if True:
            meshspec = [[xmin, xmax, Nx],
                        [ymin, ymax, Ny],
                        [zmin, zmax, Nz]]
            self.params["meshspec"] = meshspec
            domain = Domain(meshspec)
            self.domain = domain
        return domain


    def make_Tmatrix(self, R=1.585*um, n_particle=1.2):
        """
        Create a matlab OTT simple Tmatrix object.

        Parameters
        ----------
            R : float
                Radius of the spherical dielectric scatterer.
                Default is 1.585 microns.

            n_particle
                Index of refraction of the scatterer. Default is 1.2 (Silica)

        Returns
        -------
            Tmatrix : ott.Tmatrix object
                T matrix for scattering calculations.
        """

        params_in = {"R": R, "n_particle": n_particle}
        self.params.update(params_in)
        Tmatrix = self.eng.ott.Tmatrix.simple('sphere', self.params["R"],
                                   'index_medium', self.params["n_medium"],
                                   'index_particle', self.params["n_particle"],
                                   'wavelength0', self.params["wavelength0"])
        self.Tmatrix = Tmatrix
        return Tmatrix


    def make_beam(self, name, waist, polarization, angle, add_name=True):
        """
        Create a matlab ott BscPmGauss object to model a focused, TEM00-mode
        laser beam. Each call will add the name and beam object as a key-value
        pair to the Tweezer object `beams` attribute.

        Parameters
        ----------
            name : str
                A name for this beam.

            waist : float
                Waist of the focused beam.

            polarization : list
                Polarization Jones vector. Default is [1, 0].

            angle : float
                Angle in degrees to rotate the propagation direction about the
                x axis. Default is 0 degrees.

            add_name : bool
                If True, adds the given name to a list stored in
                params["beam_names"].

        Returns
        -------
            beam : ott.BscPmGauss object
                A laser beam for scattering calculations
        """
        NA = self.params["n_medium"] * self.params["wavelength0"] / (np.pi * waist)
        params_in = {"waist": waist,
                     "polarization": polarization,
                     "angle": angle}
        for k, v in params_in.items():
            self.params[f"{k}_{name}"] = v
        beam = self.eng.ott.BscPmGauss(
                          'NA', NA,
                          'polarisation', matlab.double(polarization),
                          'index_medium', self.params["n_medium"],
                          'wavelength0', self.params["wavelength0"],
                          'angular_scaling', "sintheta")
        if angle > 0.0:
            beam = self.eng.rotate(beam, matlab.double(Xrot(angle).tolist()))
        self.beams[name] = beam
        if add_name:
            self.params["beam_names"].append(name)
        return beam


    def evaluate_force(self, name, offset=[0, 0, 0]):
        """
        Evaluate optical forces over the domain using a Tmatrix and a beam with
        a focus offset from the domain origin. Each call will add the name
        and 3-compnent force field array as a key-value pair to the Tweezer
        object's `FXYZs` attribute. Laser power is set to 1 W. Multiply the
        result by the desired laser power to rescale.

        Parameter
        ---------
            name : str
                Name of the beam for which to evaluate the force.

            offset : array-like (1x3)
                Displacement vector of the beams focus with respect to the
                domain's origin. Note that beam rotation has already occured

        Returns
        -------
            FXYZ : array (3 x Nx x Ny x Nz)
                3-component force field of the named beam
        """

        self.params["offset"] = offset
        offset_vector = np.array(offset)[:, np.newaxis]
        xyz = self.domain.points.T - offset_vector
        factor =  self.params["n_medium"] / self.params["c"]
        Fxyz = factor * np.array(
            self.eng.ott.forcetorque(
               self.beams[name], self.Tmatrix, 'position', matlab.double(xyz.tolist())
            )
        )

        FXYZ = Fxyz.reshape(self.domain.shape)
        self.FXYZs[name] = FXYZ
        return FXYZ


    def evaluate_net_force(self, powers):
        """
        Evalute the incoherent-sum of forces from all beams.  The coherent sum
        is not needed for cross-polarized beams (True?).  The result is stored
        in the Tweezer object's `FXYZs' attribute under the key 'net'.

        Parameters
        ----------
            powers : dict
                A dictionary with keys correponding to beam names and
                associated values corresponding with that laser beam's power.
        Returns
        -------
            FXYZ : array (3 x Nx x Ny x Nz)
                3-component force field from all available beams.
        """
        FXYZ = 0
        for name, F in self.FXYZs.items():
            if name != "net":
                FXYZ += powers[name] * F
        self.FXYZs["net"] = FXYZ
        return FXYZ


    def make_interpolation(self, name):
        """
        Perform a linear interpolation on the 3-compnent force field arrays
        evaluated over the domain. Each call will add the name and list of
        interpolation functions as a key-value pair to the Tweezer object's
        `interps` attribute. For individual beams, the result must be rescaled
        by the desired laser power. Net force laser powers must already be set
        through the evaluate_net_force method.

        Parameters
        ----------
            name :str
                Name of the force field to be interpolated.

        Returns
        -------
            interps : list
                Component-wise list of linear interpolation functions of the
                named force field.

        """
        interps = self.domain.vector_interpolation(self.FXYZs[name])
        self.interps[name] = interps
        return interps


    def interpolate_force(self, points=None, domain=None, name="net"):
        """
        Evaluate a named force field's interpolation functions at specified
        points or over a specified Domain. If neither domain nor points is
        specified the orginal interpolation domain is used.

        Parameters
        ----------
            points : array (N x 3)
                An array of coordinate points over which to evaluate the
                interpolated force.

            domain : Domain object
                A Domain object over which to evaluate the interpolated force.
                If points is not None, this arument is ignored.

            name : str, optional
                Name of the force intrepolation functions. Default is "net"


        Returns
        -------
            Fs : array (N x 3) or (3 x Nx x Ny x Nz)
                The force evaluated over the requested points or domain.

        """
        if points is None:
            if domain is None:
                domain = self.domain
                grid_reshape = True
            cshape = domain.coordinate_shape
            grid_reshape = True
            points = domain.points
        else:
            grid_reshape = False

        Fs = np.array([interp(points) for interp in self.interps[name]]).T
        if grid_reshape:
            FX, FY, FZ = Fs.T
            cshape = domain.coordinate_shape
            FX.shape = cshape
            FY.shape = cshape
            FZ.shape = cshape
            Fs = np.array([FX, FY, FZ])
        return Fs


    def find_min(self, name="net", domain=None):
        """
        Find the global minimum in force magnitude within a given domain.

        Parameters
        ----------
            name : str
                Name of the force field. Default is "net".

            domain : Domain object
                Domain over which to seek the minimum. Default is the
                interpolation domain,
        Returns
        -------
            min_point : array (1 x 3)
                Coordinate point within the given domain of the absolute
                minimum force magnitude.

        """

        if domain is None:
            domain = self.domain
        points = domain.points

        F = self.interpolate_force(points=points, name=name)
        Fmagnitude = np.sum(F*F, axis=1)
        mask = Fmagnitude > 0
        mindex = np.argmin(Fmagnitude[mask])
        min_point = domain.points[mask, :][mindex]
        return min_point


    def fit_min(self, name="net", domain=None):
        """
        Coordinate-and-component-wise fit the global minimum in force within a
        given domain fit to a linear function.

        Parameters
        ----------
            name : str
                Name of the force field. Default is "net".

            domain : Domain object
                Domain over which to fit the minimum. Default is the
                interpolation domain,

        Returns
        -------
            location : array (2, 3)
                coordiante point of the force minimum (row 0) and associated
                standard fitting error (row 1).

            strength : array (2, 3)
                Coordinate-wise spring constant (row 0) and associated
                standard fitting error (row 1).

        """
        if domain is None:
            domain = self.domain

        abs_min_xyz = self.find_min(name, domain)
        for iter in range(2):
            if iter == 0:
                fixed_values = abs_min_xyz
            else:
                fixed_values = location[0]
            axis_points = []
            all_indices = [0, 1, 2]
            for variable_index in all_indices:
                points = [0, 0, 0]
                axis = domain.axes[variable_index]
                points[variable_index] = axis
                remaining_indices = np.setdiff1d(all_indices, [variable_index])
                for fixed_index in remaining_indices:
                    constant_value = fixed_values[fixed_index]
                    points[fixed_index] = constant_value*np.ones_like(axis)
                axis_points.append(np.array(points))
            Fs = [self.interpolate_force(points=axis.T, name=name)[:, i]
                for i, axis in enumerate(axis_points)]
            Fs = np.array(Fs)
            def func(q, q0, k):
                return -k * (q - q0)
            location = np.zeros((2, 3))
            strength = np.zeros((2, 3))
            for i, (q, q0, F) in enumerate(zip(domain.axes, abs_min_xyz, Fs)):
                popt, pcov = curve_fit(func, q, F, p0=(q0, 10*fN/nm))
                perr = np.sqrt(np.diag(pcov))
                location[0, i] = popt[0]
                location[1, i] = perr[0]
                strength[0, i] = popt[1]
                strength[1, i] = perr[1]
        return location, strength

    def save(self, fpath=None, nickname=None):
        """
        Save the force field arrays and corresponding system params to
        disk. Forces are saved in binary format using numpy and params are
        saved as a txt file using json.

        Parameters
        ----------
            fpath : str
                A directory to which data is saved. If it does not exists, it
                is created.

        Returns
        -------
            None

        """
        if fpath is None:
            fpath = self.file_path(nickname=nickname)
        makedirs(fpath, exist_ok=True)
        force_path = path.join(fpath, "forces.npy")
        params_path = path.join(fpath, "params.txt")
        print("Data and parameters saved to:")
        print(fpath)
        Fs = {k: v for k, v in self.FXYZs.items() if k != "net"}
        np.save(force_path, Fs)
        with open(params_path, "w") as f:
            json.dump(self.params, f)


    def load(self, fpath=None, nickname=None):
        """
        Load the force field arrays and corresponding system params from
        disk and make Tweezer class attributes domain, Tmatrix, beams, interps.

        Parameters
        ----------
            fpath : str
                A directory from which data is loaded.

        Returns
        -------
            None

        """
        if fpath is None:
            fpath = self.file_path(nickname=nickname)
        force_path = path.join(fpath, "forces.npy")
        params_path = path.join(fpath, "params.txt")
        self.FXYZs = np.load(force_path, allow_pickle=True).item()
        with open(params_path, "r") as f:
            self.params = json.loads(f.read())
        self.domain = Domain(self.params["meshspec"])
        Tmatrix_args = ["R", "n_particle"]
        self.make_Tmatrix(*(self.params[k] for k in Tmatrix_args))
        beam_args = ["waist", "polarization", "angle"]
        for name in self.params["beam_names"]:
            self.make_beam(name, *(self.params[f"{k}_{name}"] for k in beam_args),
                           add_name=False)


    def plot_slices(
        self,
        name="net",
        domain=None,
        planes={"z": [0.0]},
        contours=4,
        label_contours=True,
        quiver=True,
        skip=1,
        unit="m",
        Funit="N",
        axs=None,
        figsize=None,
        imshow_kwargs={},
        contour_kwargs={},
        clabel_kwargs={},
        quiver_kwargs={}
    ):

        """
        Plot a named force field's interpolation functions as 2D-projected
        slices.  Plots can include force magnitude as color, contour lines of
        constant force magnitude, and a vector arrows of the projected force.
        Slices are parallel to coordinate planes and are defined by fixing the
        value of one coordinate, e.g. th xy-plane is defined by z=0.

        Parameters
        ----------
            name : str
                Name of the force intrepolation function to plot. Default is
                "net"

            domain : Domain object
                A domain over which to plot the slices. Default is the domain
                used to define the interpolation.

            planes : dict
                A dictonary of planes. Keys are a coordinate axis to be fixed
                ('x', 'y', or 'z') and the corresponding values is a list of
                constants at which to fix that axis. Defauls is {"z":[0.0]}
                (the xy-plane).

            contours : int
                The number of constant-force-magnitude contours to
                plot. Default is 4. If 0 no contours are plotted.

            label_contours : bool
                If True, includes text labels of the contour line
                values. Dafault is True

            quiver : bool
                If True, plots the projected force vectors as arrows.

            skip : int
                The stride over the domain at which to plot the arrows.

            unit : str
                Sets the spatial unit of plots. Default is "m" for SI.
                Set to "nm" for nanometers, etc.

            Funit : str
                Sets the Force unit of the plot. Default is "N". Set to "pN"
                for piconewtons, etc.

            axs : matplotlib axis objects or array of matplot plot axis objects
                A matplotlib axis to plot the planar slice data on. The
                coordinate axis along which a plane is defined (key of planes
                dict, 'x', 'y', or 'z') is held constant in rows (across
                columns) of plots. Each column corresponds to different
                constant values (values of the planes dict).

            figsize : tuple
                Size of the figure (horizontil inches, verticle inches).
                Default is 3 * the shape of axs.

            *_kwargs : dict
                Access to the matplotlib functions used: imshow_kwargs,
                contour_kwargs, clabel_kwargs, and quiver_qwargs.

            Returns
            -------
                Fig : matplotlib Figure

                axs : array of matplotlib axes
        """
        nrow = len(planes)
        ncol = np.max([len(p) for p in planes.values()])
        unit = units[unit]
        Funit = units[Funit]

        if figsize is None:
            figsize = (ncol*3, nrow*3)

        if axs is None:
            fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharey="row")
        else:
            fig = plt.gcf()

        if domain is None:
            domain = self.domain
        xyz = domain.axes
        cshape = domain.coordinate_shape
        interps = self.interps[name]
        vars_inds_dict = {0: [2, 1], 1: [2, 0], 2: [0, 1]}
        const_inds = ["xyz".index(k) for k in planes.keys()]
        vars_inds = [vars_inds_dict[c] for c in const_inds]

        for i, (const_str, const_vals) in enumerate(planes.items()):
            const_ind = "xyz".index(const_str)
            vars_inds = vars_inds_dict[const_ind]
            meshspec = [[0,0,0],[0,0,0],[0,0,0]]
            for vcoordi in vars_inds:
                for minmaxspeci in (0, 1, 2):
                    meshspec[vcoordi][minmaxspeci] = domain.meshspec[vcoordi][minmaxspeci]
            for j, const_val in enumerate(const_vals):
                for minmaxi in (0, 1):
                    meshspec[const_ind][minmaxi] = const_val
                meshspec[const_ind][2] = 1
                plot_domain = Domain(meshspec)
                if nrow > 1 and ncol > 1:
                    ax = axs[i, j]
                elif nrow > 1 or ncol > 1:
                    ax = axs[max(i, j)]
                else:
                    ax = axs
                VXYZ = self.interpolate_force(domain=plot_domain)
                VX = np.squeeze(VXYZ[vars_inds[0]])
                VY = np.squeeze(VXYZ[vars_inds[1]])
                X = np.squeeze(plot_domain.grid[vars_inds[0]])
                Y = np.squeeze(plot_domain.grid[vars_inds[1]])
                V = np.sqrt(VX**2 + VY**2)
                x = plot_domain.axes[vars_inds[0]]
                y = plot_domain.axes[vars_inds[1]]

                ax.imshow(
                    V/Funit["value"],
                    origin="lower",
                    interpolation="None",
                    extent=[x[0]/unit["value"], x[-1]/unit["value"],
                            y[0]/unit["value"], y[-1]/unit["value"]],
                    **imshow_kwargs
                )

                if contours:
                    c_kwargs = dict(colors="k")
                    c_kwargs.update(contour_kwargs)
                    cp = ax.contour(V/Funit["value"], contours,
                    origin="lower",
                    extent=[x[0]/unit["value"], x[-1]/unit["value"],
                            y[0]/unit["value"], y[-1]/unit["value"]],
                    **c_kwargs
                    )

                if contours and label_contours:
                    cl_kwargs = dict(inline=1, fontsize=11, fmt="%1.2f")
                    cl_kwargs.update(clabel_kwargs)
                    ax.clabel(cp, **cl_kwargs)

                if quiver:
                    q_kwargs = dict(pivot="middle",
                                    lw=0.5,
                                    color="k")
                    q_kwargs.update(quiver_kwargs)
                    ax.quiver(
                        X[::skip, ::skip]/unit["value"],
                        Y[::skip, ::skip]/unit["value"],
                        VX[::skip, ::skip]/Funit["value"],
                        VY[::skip, ::skip]/Funit["value"],
                        **q_kwargs
                    )

                ax.set_title(r"$%s = %.2f~(\rm %s)$" % (
                    "xyz"[const_ind], const_val/unit["value"], unit["label"]))
                ax.set_xlabel("xyz"[vars_inds[0]]+r"$~(\rm %s)$" % unit["label"])
                if j == 0:
                    ax.set_ylabel(
                        "xyz"[vars_inds[1]]+r"$~(\rm %s)$" % unit["label"]
                    )
        fig.tight_layout()
        title = r"$\bf{F}$" + r"$~(\rm %s)$" % Funit["label"]
        fig.suptitle(title)
        plt.subplots_adjust(
            hspace=0.5, wspace=0.1, top=0.93, bottom=0.08, left=0.10, right=0.9
        )
        return fig, axs


    def plot_linecut(
        self,
        name="net",
        domain=None,
        component="z",
        line=[0.0, 0.0, "*"],
        unit="m",
        Funit="N",
        ax=None,
        figsize=None,
        show_legend=True,
        plot_kwargs={},
    ):

        """
        Plot a component of a named force field's interpolation functions as 1D
        line cuts slices.

        Parameters
        ----------
            name : str
                Name of the force intrepolation function to plot. Default is
                "net".

            domain : Domain object
                A domain over which to plot the line cut. Default is the domain
                used to define the interpolation.

            component : str
                Specification of which force component to plot ("x", "y", or
                "z")

            line : list (1 x 3)
                Specification of the line cut to plot. Two elements of the list
                must be non-zero floats specifying the constant coordiante
                values and the third must be the string "*" specifying the
                variable coordinate, e.g.  [0, 0,"*" ] plots a line cut along
                the z-axis.

            unit : str
                Sets the spatial unit of plots. Default is "m" for SI.
                Set to "nm" for nanometers, etc.

            Funit : str
                Sets the Force unit of the plot. Default is "N". Set to "pN"
                for piconewtons, etc.

            ax : matplotlib axis object
                A matplotlib axis to plot the line cut.

            figsize : tuple
                Size of the figure (horizontil inches, verticle inches).
                Default is 3 * the shape of axs.

            show_legend : bool
                If True adds a legend element describing the constant
                coordinates of the line cut.

            *plot_kwargs : dict
                Access to the matplotlib plot function

            Returns
            -------
                Fig : matplotlib Figure

                ax : matplotlib axis

        """

        if figsize is None:
            figsize = (3, 2)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        else:
            fig = plt.gcf()

        unit = units[unit]
        Funit = units[Funit]
        interps = self.interps[name]
        if domain is None:
            domain = self.domain
        comp = "xyz".index(component)
        interp = interps[comp]
        xyz = domain.axes
        vari = line.index("*")
        vars = xyz[vari]
        label = ""
        points = np.zeros((len(vars), 3))
        for k, v in zip("xyz", line):
            idx = "xyz".index(k)
            if v == "*":
                points[::, idx] = vars
            else:
                points[::, idx] = v
                label_el = r"$%s = %s~{\rm %s}$" % (k, round(v/unit["value"],3), unit["label"])
                if label == "":
                    postfix = " \n"
                else:
                    postfix = ""
                label += label_el + postfix
        ylabel = r"$F_{%s}~{\rm (%s)}$" % ("xyz"[comp], Funit["label"])
        xlabel = r"$%s~{\rm (%s)}$" % ("xyz"[vari], unit["label"])
        p_kwargs = {"label": label}
        p_kwargs.update(plot_kwargs)
        ax.plot(vars/unit["value"], interp(points)/Funit["value"], **p_kwargs)
        ax.axhline(0, lw=1, c="k")
        if show_legend:
            ax.legend(fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        return fig, ax


    def plot_linecuts(
        self,
        name="net",
        domain = None,
        components="xyz",
        lines=[[0.0, 0.0, "*"], [0.0, "*", 0.0], ["*", 0.0, 0.0]],
        unit="m",
        Funit="N",
        axs = None,
        figsize=None,
        show_legend=True,
        plot_kwargs={}
    ):
        """
        Plot a several line cuts.

        Parameters
        ----------
            name : str
                Name of the force intrepolation function to plot. Default is
                "net".

            domain : Domain object
                A domain over which to plot the line cut. Default is the domain
                used to define the interpolation.

            component : str
                Specification of which force components to plot ("x", "y", "z",
                or a concetenation thereof). Components are constant across
                columns (vary down rows) of the plot arry.  Default is "xyz".

            lines : list (N x 3)
                Specification of the line cuts to plot. A list of `lines`
                (see plot_linecut). Lines are constant down rows (vary across
                columns) of the plot array. Default is the x, y, and z axes.

            unit : str
                Sets the spatial unit of plots. Default is "m" for SI.
                Set to "nm" for nanometers, etc.

            Funit : str
                Sets the Force unit of the plot. Default is "N". Set to "pN"
                for piconewtons, etc.


            figsize : tuple
                Size of the figure (horizontil inches, verticle inches).
                Default is 3 * the shape of axs.

            axs : matplotlib axis objects or array of matplot plot axis objects
                An array of matplotlib axes to plot the linecut data on.

            show_legend : bool
                If True adds a legend element describing the constant
                coordinates of the line cut.

            *plot_kwargs : dict
                Access to the matplotlib plot function

            Returns
            -------
                Fig : matplotlib Figure

                axs : array of matplotlib axes
        """

        varinds = [[], [], []]
        for i, line in enumerate(lines):
            varinds[line.index("*")] += [i]
        varinds = [vinds for vinds in varinds if vinds != []]
        nrow = len(components)
        ncol = len(varinds)

        if figsize is None:
            figsize = (ncol*3, nrow*3)

        if axs is None:
            fig, axs = plt.subplots(nrow, ncol, sharex="col", sharey="row", figsize=figsize)
        else:
            fig = plt.gcf()

        for comp in components:
            i = components.index(comp)
            for j, vinds in enumerate(varinds):
                if nrow > 1 and ncol > 1:
                    ax = axs[i, j]
                elif nrow > 1 or ncol > 1:
                    ax = axs[max(i, j)]
                else:
                    ax = axs
                for idx in vinds:
                    line = lines[idx]
                    self.plot_linecut(
                        domain=domain,
                        name=name,
                        unit=unit,
                        Funit=Funit,
                        ax=ax,
                        component=comp,
                        line=lines[idx],
                        show_legend=show_legend,
                        plot_kwargs=plot_kwargs)
                    ax.label_outer()
        return fig, axs
