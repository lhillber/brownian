# Simulate the Brownian motion of a damped, harmonically confined sphere.
#
# By Logan Hillberry


from time import time
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from matplotlib.mlab import stride_windows
from copy import copy
from numba import jit
from cmath import sqrt
from scipy.signal import welch
from scipy.integrate import simps

kB = 1.381e-23  # Boltzmann's constant

def get_eta(T, etap=1.83245e-5, Tp=23+273.15, S=110.4):
    """Sutherlands model for viscosity of air at temperature T"""
    return etap*(T/Tp)**(3/2) * (Tp + S) / (T+S)


def make_sim_params(K, rho, R, T):
    eta = get_eta(T)
    m = 4*np.pi*rho*R**3/3
    gamma = 6*np.pi*eta*R
    dim = len(K)
    params = dict(K=K,
                  rho=rho,
                  R=R,
                  eta=eta,
                  T=T,
                  m=m,
                  dim=dim,
                  gamma=gamma,
                  Gamma=gamma/m,
                  taup=m/gamma)
    return params


def acceleration(xs, vs, m, gamma, K, T, dt, **params):
    dim = len(K)
    G = np.sqrt(2*kB*T*gamma)
    return (-K*xs - gamma*vs + G*np.random.normal(size=dim)*dt**(-1/2)) / m


def setup(m, gamma, k, T, dt):
    D = kB * T / gamma
    a = gamma / (2 * m)
    b = sqrt(gamma**2 / (4*m*m) - k/m)
    lp = a + b
    lm = a - b
    up = np.array([-1, lp])
    um = np.array([1, -lm])
    cp = np.exp(-lp * dt)
    cm = np.exp(-lm * dt)
    A = sqrt(D/2) * (lp + lm) / (lp - lm)
    Ap = A * sqrt((1 - cp*cp) / lp)
    Am = A * sqrt((1 - cm*cm) / lm)
    alpha = 2 * np.sqrt(lp*lm) / (lp + lm)
    alpha *= (1 - cp*cm) / sqrt((1 - cp*cp) * (1 - cm*cm))
    Dxv1 = (Ap*up + Am*um) * sqrt(1 + alpha)
    Dxv2 = (Ap*up - Am*um) * sqrt(1 - alpha)
    expm = np.array(
            [[-lm*cp + lp*cm, -cp+cm],
             [lp*lm*(cp-cm), lp*cp - lm*cm]]
          ) / (lp - lm)
    return expm.real, Dxv1.real, Dxv2.real


def sim2(m, gamma, K, T, dt, tmax, x0=None, v0=None, **params):
    """ Second solver. TODO: Benchmark agains first"""
    dim = len(K)
    if x0 is None:
        x0 = np.array([np.random.normal(0, np.sqrt(kB*T/k)) for k in K])
    if v0 is None:
        v0 = np.random.normal(0, np.sqrt(kB*T/m), dim)
    ts = np.arange(0, tmax, dt)
    xs = np.zeros((len(ts), dim))
    vs = np.zeros((len(ts), dim))
    xs[0] = x0
    vs[0] = v0
    vs[1] = v0 + dt * acceleration(x0, v0, m, gamma, K, T, dt)
    xs[1] = x0 + dt*vs[1]
    for ti in range(1, len(ts)-1):
        xs[ti+1] = 2*xs[ti] - xs[ti-1] + dt * dt * \
            acceleration(xs[ti], vs[ti],  m, gamma, K, T, dt)
        vs[ti+1] = (xs[ti+1] - xs[ti-1]) / (2*dt)
    return xs, vs, ts


def sim(m, gamma, K, T, dt, tmax, x0=None, v0=None, **params):
    """Primary dynamical solver.
    Parameters (enter values in SI units):
        m: bead mass,
        gamma, Stokes friction coefficient,
        K: Vector of trap stiffness. Length of K sets spatial dimension.
        T: Enviromental temperature (Kelvin units)
        dt: Temporal time step
        tmax: Total simulation time
        x0: Initial position
        v0: Initial velocity
        """
    dim = len(K)
    ts = np.arange(0, tmax, dt)
    xvs = np.zeros((2, len(ts), dim))
    for coordi, k in enumerate(K):
        expm, Dxv1, Dxv2 = setup(m, gamma, k, T, dt)
        if x0 is None:
            x0 = np.random.normal(0, np.sqrt(kB*T/k))
        if v0 is None:
            v0 = np.random.normal(0, np.sqrt(kB*T/m))
        xvs[0, 0, coordi] = x0
        xvs[1, 0, coordi] = v0
        for ti in range(len(ts)-1):
            Ra, Rb = np.random.normal(0, 1, 2)
            xvs[:, ti+1, coordi] = expm.dot(xvs[:, ti, coordi]) + Dxv1*Ra + Dxv2*Rb
    return xvs[0,:,:], xvs[1,:,:], ts


def partition(xs, dt, taumax=None, noverlap=0):
    """Split xs into chunks overlapping by `noverlap` points.
    Each chunk is length taumax given for xs collected at a sample rate of 1/dt"""
    if taumax is None:
        taumax = (xs.size - 1)*dt
    Npts = min(len(xs), int(taumax/dt))
    xpart = stride_windows(xs, n=Npts, noverlap=noverlap, axis=0).T
    return xpart


def logbin_average(x, Npts=100, func=np.mean):
    ndecades = np.log10(x.size) - np.log10(1)
    npoints = int(ndecades) * Npts
    parts = np.logspace(
        np.log10(1), np.log10(x.size), num=npoints, endpoint=True, base=10
    )
    parts = np.unique(np.round(parts)).astype(np.int64)
    return np.array(
        [func(x[parts[i]: parts[i + 1]]) for i in range(len(parts) - 1)]
    )


def PSD(xs, dt, taumax=None, detrend="linear", window="hamming", noverlap=0):
    xpart = partition(xs, dt, taumax)
    Navg = len(xpart)
    freq = fft.rfftfreq(xpart[0].size, dt)
    psdavg = 0
    # average PSD of each partions
    for xp in xpart:
        # Bartlet PSD is welch PSD with zero overlap
        freq, psd = welch(xp, fs=1/dt, nperseg=xp.size, window=window,
                          detrend=detrend, noverlap=noverlap)
        psdavg += psd / Navg

    # clip zero freq bin = average of signal
    # psdavg = psdavg[1:]
    # freq = freq[1:]
    return freq, psdavg, Navg


def AVAR(xs, dt, func=np.mean, octave=True):
    """Allan Variance"""
    if octave:
        Npts_list = np.array([2**m for m in range(1, int(np.log2(xs.size/2)))])
    else:
        Npts_list = np.arange(2, int(xs.size/2))

    taus = dt * Npts_list
    avars = np.zeros_like(taus)
    davars = np.zeros_like(taus)
    for j, tau in enumerate(taus):
        xparts = partition(xs, dt, taumax=tau)
        vals = func(xparts, axis=1)
        avars[j] = np.mean((vals[1:] - vals[:-1])**2) / 2
        davars[j] = np.var(vals)
    return taus, avars, davars


def NVAR(xs, dt, func=np.mean, octave=True):
    """Normal variance"""
    if octave:
        Npts_list = np.array([2**m for m in range(0, int(np.log2(xs.size/2)))])
    else:
        Npts_list = np.arange(2, int(xs.size/2))

    taus = dt * Npts_list
    nvars = np.zeros_like(taus)
    dnvars = np.zeros_like(taus)
    m = np.mean(xs)
    for j, tau in enumerate(taus):
        xparts = partition(xs, dt=dt, taumax=tau)
        vals = np.array([func(x) for x in xparts])
        nvars[j] = np.mean((vals - m)**2)
        dnvars[j] = np.var(vals)
    return taus, nvars, dnvars


@jit(nopython=True)
def MSD_jit(x):
    N = x.size
    msd = np.zeros(N, dtype=np.float32)
    for lag in range(1, N):
        diffs = x[:-lag] - x[lag:]
        msd[lag - 1] = np.mean(diffs**2)
    return msd


MSD_jit(np.array([1.0, 2.9, 3.0, 4.0]))


def MSD(xs, dt, taumax=None):
    """Mean squared displacement"""
    xpart = partition(xs, dt, taumax)
    Navg = len(xpart)
    msdavg = 0
    # average MSD of each partion
    for xp in xpart:
        msdavg += MSD_jit(xp) / Navg

    tmsd = np.arange(1, len(msdavg)+1) * dt
    return tmsd, msdavg


def ACF(xs, dt, taumax=None):
    """Auto correlation function"""
    xpart = partition(xs, dt, taumax)
    Navg = len(xpart)
    acfavg = 0
    # average ACF of each partion
    m = xs.mean()
    for xp in xpart:
        m2 = np.mean(xp)
        n = len(xp)
        acf = np.correlate(
            xp, xp, mode='full')[-n:] / (np.arange(n, 0, -1))
        acfavg += acf
    acfavg /= Navg
    tacf = np.arange(0, len(acfavg)) * dt
    return tacf, acfavg


def S_statistic(p, q, freq, psd):
    return np.mean(freq ** (2 * p) * psd ** q)


def get_Smat(freq, psd):
    S02 = S_statistic(0, 2, freq, psd)
    S12 = S_statistic(1, 2, freq, psd)
    S22 = S_statistic(2, 2, freq, psd)
    S32 = S_statistic(3, 2, freq, psd)
    S42 = S_statistic(4, 2, freq, psd)
    S01 = S_statistic(0, 1, freq, psd)
    S11 = S_statistic(1, 1, freq, psd)
    S21 = S_statistic(2, 1, freq, psd)
    Smat = np.array([[S02, S12, S22], [S12, S22, S32], [S22, S32, S42]])
    Denom = (
        S02 * S22 * S42
        - S02 * S32 ** 2
        - S12 ** 2 * S42
        + 2 * S12 * S22 * S32
        - S22 ** 3
    )
    C0 = S22 * S42 - S32 ** 2
    C1 = S22 * S32 - S12 * S42
    C2 = S12 * S32 - S22 ** 2
    C3 = S12 * S22 - S02 * S32
    C4 = S02 * S22 - S12 ** 2
    C5 = S02 * S42 - S22 ** 2
    C = np.array([[C0, C1, C2], [C1, C5, C3], [C2, C3, C4]])
    Sinvmat = C / Denom
    Svec = np.array([S01, S11, S21])
    return Smat, Sinvmat, Svec


def get_krhoA(a, b, c, R, T, eta, **kwargs):
    d2 = b + 2 * np.sqrt(a*c)
    k = 12 * np.pi**2 * eta * R * np.sqrt(a/d2)
    rho = 9 * eta / (4*np.pi*R**2) * np.sqrt(c/d2)
    Ainv2 = 6*np.pi**3*eta*R / (kB*T * d2)
    A = np.sqrt(Ainv2)
    return k, rho, A


def abc_guess(freq, psd, n=1):
    Smat, Sinvmat, Svec = get_Smat(freq, psd)
    Sinvmat *= (n+1) / n
    popt0 = Sinvmat.dot(Svec)
    _, pSinvmat, _ = get_Smat(freq, psd_abc_func(freq, *popt0))
    pcov0 = (n+3) * pSinvmat / (n + 1) / freq.size
    return popt0, pcov0

def gaussian_func(x, var, mean=0):
    return np.exp(-(x - mean)**2 / (2 * var)) / (np.sqrt(2 * PI * var))

def psd_abc_func(f, a, b, c, **kwargs):
    return 1 / np.abs((a + b * f ** 2 + c * f ** 4))


def psdfunc(f, k, rho, T, R, eta=None, **kwargs):
    if eta is None:
        eta = get_eta(T)
    m = 4*np.pi*rho*R**3/3
    gamma = 6*np.pi*eta*R
    omega = 2*np.pi * f
    denom = (m*omega**2 - k)**2 + (gamma*omega)**2
    return 4 * kB * T * gamma / denom


def msdfunc(t, k, rho, T, R, eta=None, **kwargs):
    if eta is None:
        eta = get_eta(T)
    m = 4*np.pi*rho*R**3/3
    gamma = 6*np.pi*eta*R
    Omega = np.sqrt(k / m)
    Gamma = gamma / m
    taup = 1 / Gamma
    omega1 = sqrt(Omega**2 - Gamma**2/4)
    cs = np.cos(omega1 * t) + np.sin(omega1*t) / (2*omega1*taup)
    cs = cs.real
    return 2*kB*T/k * (1 - cs * np.exp(-t/(2*taup)))


def pacfunc(t, k, rho, T, R, eta=None, **kwargs):
    if eta is None:
        eta = get_eta(T)
    m = 4*np.pi*rho*R**3/3
    gamma = 6*np.pi*eta*R
    Omega = np.sqrt(k / m)
    Gamma = gamma / m
    taup = 1 / Gamma
    omega1 = sqrt(Omega**2 - Gamma**2/4)
    cs = np.cos(omega1 * t) + np.sin(omega1*t) / (2*omega1*taup)
    cs = cs.real
    return kB*T/k * cs * np.exp(-t/(2*taup))


def vacfunc(t, k, rho, T, R, eta=None, **kwargs):
    if eta is None:
        eta = get_eta(T)
    m = 4*np.pi*rho*R**3/3
    gamma = 6*np.pi*eta*R
    Omega = np.sqrt(k / m)
    Gamma = gamma / m
    taup = 1 / Gamma
    omega1 = sqrt(Omega**2 - Gamma**2/4)
    cs = np.cos(omega1 * t) - np.sin(omega1*t) / (2*omega1*taup)
    cs = cs.real
    return kB*T/m * cs * np.exp(-t/(2*taup))


def msd_from_pacf(pacf, T, k, **kwargs):
    return 2 * (kB * T / k) - 2*pacf


def pacf_from_msd(msd, T, k, **kwargs):
    return (kB * T / k) - msd/2


def dataplot(ax, x, y, Npts=0, color="k", **kwargs):
    if Npts > 0:
        plot_x = logbin_average(x, Npts=Npts)
        plot_y = logbin_average(y, Npts=Npts)
    else:
        plot_x = x
        plot_y = y
    plot_kwargs = dict(
                  marker = "o",
                  ls = "none",
                  mfc = "none",
                  mec = color,
                  color = color)
    plot_kwargs.update(kwargs)
    ax.plot(plot_x, plot_y, **plot_kwargs)
    return ax


if __name__ == "__main__":
    tmax = 5
    dt = 1/2e6
    load = False
    save = True

    params = make_sim_params(
                # testing different trap stiffnesses
                K=np.array([1.0, 50.0])*1e-6,
                rho=1740, # bead density
                R=3.17e-6 / 2, # bead radius
                T=273.15 + 25) # room temperature

    if not load:
        # generate position, velocity, and time data
        xs, vs, ts = sim(tmax=tmax, dt=dt, **params)
        if save:
            np.save("xs.npy", xs)
            np.save("vs.npy", vs)
            np.save("ts.npy", ts)

    else:
        xs = np.load("xs.npy")
        vs = np.load("vs.npy")
        ts = np.load("ts.npy")

    # loop through spatial dimensions
    for i in range(params["dim"]):
        params["k"] = params["K"][i] # set current trap stiffness
        fig, axs = plt.subplots(2, 2)

        # run analysis
        freq, psd, n = PSD(xs[:, i], dt, taumax=40e-3)
        tmsd, msd = MSD(xs[:, i], dt, taumax=40e-3)
        tpacf, pacf = ACF(xs[:, i], dt, taumax=40e-3)
        tvacf, vacf = ACF(vs[:, i], dt, taumax=40e-3)
        tavar, avar, davar = AVAR(vs[:, i], dt, func=np.mean)
        tnvar, nvar, dnvar = NVAR(vs[:, i], dt, func=np.mean)

        # Estimate parameters from PSD
        fmin = 75
        fmax = 1e4
        Npts = 25
        mask = np.logical_and(freq < fmax, freq > fmin)
        abc, pcov0 = abc_guess(freq[mask], psd[mask], n=n)
        kest, rhoest, Aest = get_krhoA(*abc, **params)

        # plot PSD fit
        #axs[0, 0].loglog(freq, psd_abc_func(freq, *abc), color="purple")
        #axs[0].axvline(fmin)
        #axs[0].axvline(fmax)

        # Report fitting uncertainty
        print(f"{'xyz'[i]}-coordinate:")
        print("   dk/k:", round(np.abs(kest-params["k"]) / params["k"], 3))
        print("   drho/rho", round(np.abs(rhoest-params["rho"]) / params["rho"], 3))
        print("   dA/A", round(np.abs(Aest-1)/1, 3))

        # PSD plot
        dataplot(axs[0, 0], freq, psd, Npts=Npts)
        axs[0, 0].plot(freq, psdfunc(freq, **params), c="r")
        axs[0, 0].set_ylabel(r"PSD ($\mathrm{m^2/Hz}$)")
        axs[0, 0].set_xlabel("$f$ (Hz)")
        axs[0, 0].set_yscale("log")
        axs[0, 0].set_xscale("log")

        # MSD plot
        dataplot(axs[0, 1], tmsd, msd, Npts=Npts)
        axs[0, 1].plot(tmsd, msdfunc(tmsd, **params), c="r")
        axs[0, 1].set_ylabel(r"MSD ($\mathrm{m^2}$)")
        axs[0, 1].set_xlabel(r"$\tau$ (s)")
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_xscale("log")

        # PACF plot
        dataplot(axs[1, 0], tpacf, pacf, Npts=Npts)
        axs[1, 0].plot(tpacf, pacfunc(tpacf, **params), c="r")
        axs[1, 0].set_ylabel("PACF ($\mathrm{m^2}$)")
        axs[1, 0].set_xlabel(r"$\tau$ (s)")
        axs[1, 0].set_xscale("log")

        # VACF plot
        #dataplot(axs[1, 1], tvacf, vacf, Npts=Npts)
        #axs[1, 1].plot(tpacf, vacfunc(tvacf, **params), c="r")
        #axs[1, 1].set_ylabel("VACF ($\mathrm{m^2/s^2}$)")
        #axs[1, 1].set_xlabel(r"$\tau$ (s)")
        #axs[1, 1].set_xscale("log")
        #



        # VAR plot
        dataplot(axs[1,1], tavar, avar, Npts=0)
        dataplot(axs[1,1], tnvar, nvar, Npts=0, marker="s")
        axs[1, 1].set_ylabel("Variance ($\mathrm{m^2}$)")
        axs[1, 1].set_xlabel(r"$\tau$ (s)")
        axs[1, 1].set_xscale("log")
        axs[1, 1].set_yscale("log")
        plt.tight_layout()
        plt.show()

