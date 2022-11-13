# Simulate the Brownian motion of a damped, harmonically confined sphere.
#
# By Logan Hillberry


from time import time
import numpy as np
from constants import kB
import matplotlib.pyplot as plt
from matplotlib.mlab import stride_windows
from copy import copy
from numba import jit
from cmath import sqrt
from joblib import Parallel, delayed
from scipy.integrate import simps
from scipy.signal import welch, detrend
sigdetrend = detrend # Alias


#def get_viscosity(T, etap=1.83245e-5, Tp=23+273.15, S=110.4):
#    """Sutherlands model for viscosity of air at temperature T"""
#    return etap*(T/Tp)**(3/2) * (Tp + S) / (T+S)

def get_air_density(T, RH=0.0):
    RHs = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    orders = np.array([1, 1e-3, 1e-5, 1e-7])
    coefs = np.array([
    [1.293076926, -4.668788752, 1.616496527, -0.270258690],
	[1.293108555, -4.755745337, 1.840833144, -0.768566623],
	[1.293140234, -4.842714106, 2.065234104, -1.266930718],
	[1.293172031, -4.929691585, 2.289622204, -1.765272412],
	[1.293203823, -5.016669259, 2.514022965, -2.263626682],
	[1.293235188, -5.10362474,  2.73838242,  -2.761958201],
	[1.293266904, -5.190564282, 2.962671829, -3.260235828],
	[1.293298927, -5.277566167, 3.187131352, -3.758628812],
	[1.293330395, -5.364522222, 3.411479567, -4.256947646],
	[1.293362125, -5.451485999, 3.635853785, -4.755289659],
	[1.293393662, -5.538444326, 3.860201577, -5.2536065]
    ])
    if T <= 273.15:
        t = T
        T = T + 273.15
    else:
        t = T - 273.15
    if RH < 1 and RH != 0:
        RH *= 100
    if RH in RHs:
        ind = RHs.index(RH)
        return np.poly1d((coefs[ind]*orders)[::-1])(t)
    else:
        ind2 = np.digitize(RH, RHs)
        ind1 = ind2 - 1
        rho1 = np.poly1d((coefs[ind1]*orders)[::-1])(t)
        rho2 = np.poly1d((coefs[ind1]*orders)[::-1])(t)
        return rho1 +  (RH - RHs[ind1]) * (rho2 - rho1) / (RHs[ind2] - RHs[ind1])

def get_viscosity(T, RH=0.0):
    RHs = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    orders = np.array([1e-5, 1e-8, 1e-10, 1e-12, 1e-14])
    coefs = np.array([
       [1.722618487, 4.946650018, -0.424120667,0.0601291585, -0.005756513],
       [1.721930268, 4.925271672, -0.762651605,0.2920526073, -0.880983538],
       [1.72124205, 4.90372862, -1.098203212,0.51295635, -1.740331579],
       [1.720553892, 4.88193357, -1.430354374,0.722263747, -2.583623145],
       [1.719866133, 4.85995929, -1.759483937,0.9205775182, -3.41122006],
       [1.719178571 , 4.837704954, -2.085178011,1.107458312, -4.223062267],
       [1.718491567, 4.815160916, -2.407370807,1.282796983, -5.019153423],
       [1.717805126, 4.792374125, -2.726332742 ,1.447098204, -5.799841843 ],
       [1.717118698 , 4.769355947, -3.04203167,1.600323725, -6.565188824],
       [1.716432329, 4.746124468, -3.354668975,1.742996867, -7.315608677],
       [1.715747771, 4.722402075, -3.663027156,1.873236686, -8.050218737]
       ])
    if T <= 273.15:
        t = T
        T = T + 273.15
    else:
        t = T - 273.15
    if RH < 1 and RH!=0:
        RH *= 100
    if RH in RHs:
        ind = RHs.index(RH)
        return np.poly1d((coefs[ind]*orders)[::-1])(t)
    else:
        ind2 = np.digitize(RH, RHs)
        ind1 = ind2 - 1
        mu1 = np.poly1d((coefs[ind1]*orders)[::-1])(t)
        mu2 = np.poly1d((coefs[ind1]*orders)[::-1])(t)
        return mu1 +  (RH - RHs[ind1]) * (mu2 - mu1) / (RHs[ind2] - RHs[ind1])

def get_sound_speed(T, RH=0, p=101325, xCo2=0):
    if T <= 273.15:
        t = T
        T = T + 273.15
    else:
        t = T - 273.15
    if RH > 1 and RH!=0:
        RH /= 100
    if p < 1000:
        p *= 1000
    a0 = 331.5024
    a1 = 0.603055
    a2 = -0.000528
    a3 = 51.471935
    a4 = 0.1495874
    a5 = -0.000782
    a6 = -1.82e-7
    a7 = 3.73e-8
    a8 = - 2.93e-10
    a9 = -85.20931
    a10 = -0.228525
    a11 = 5.91e-5
    a12 = -2.835194
    a13 = -2.15e-13
    a14 = 29.179762
    a15 = 0.000486

    f = 1.00062 + 3.14e-8*p + 5.6e-7*t*t
    psv = np.exp(1.2811805e-5*T*T - 1.9509874e-2*T + 34.04926034 - 6.3536311e3/T)
    xw = RH * f * psv / p
    c0 = a0 + a1*t + a2*t*t +(a3+a4*t+a5*t*t)*xw
    c0 += (a6+a7*t+a8*t*t)*p + (a9+a10*t+a11*t*t)*xCo2
    c0 += a12*xw*xw + a13*p*p + a14*xCo2*xCo2 + a15*xw*p*xCo2
    return c0


def get_mass(R, density):
    return 4/3 * np.pi * density * R**3

def get_gamma(R, viscosity):
    return 6 * np.pi * viscosity * R

def get_params_from_abcRT(a, b, c, R, T, RH=50):
    eta = get_viscosity(T, RH=RH)
    d2 = b + 2 * np.sqrt(a*c)
    k = 12 * np.pi**2 * eta * R * np.sqrt(a/d2)
    rho = 9 * eta / (4*np.pi*R**2) * np.sqrt(c/d2)
    cal_inv2 = 6*np.pi**3*eta*R / (kB*T * d2)
    cal = 1 / np.sqrt(cal_inv2)
    m = get_mass(R, rho)
    gamma = get_gamma(R, eta)
    taup = m / gamma
    w0 = np.sqrt(k/m)
    return {"a":a, "b":b, "c":c, "R":R, "Tavg":T, "k": k, "rho": rho, "cal": cal, "m": m, "gamma": gamma, "taup": taup, "w0": w0}

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
    Each chunk is length taumax for xs collected at a sample rate of 1/dt"""
    if taumax is None:
        taumax = (xs.size - 1)*dt
    Npts = min(len(xs), int(taumax/dt))
    xpart = stride_windows(xs, n=Npts, noverlap=noverlap, axis=0).T
    return xpart


def bin_func(xs, dt, taumax=None, func=np.mean):
    return func(partition(xs, dt=dt, taumax=taumax), axis=1)


def logbin_func(x, Npts=100, func=np.mean):
    ndecades = np.log10(x.size) - np.log10(1)
    npoints = int(ndecades) * Npts
    parts = np.logspace(
        np.log10(1), np.log10(x.size), num=npoints, endpoint=True, base=10
    )
    parts = np.unique(np.round(parts)).astype(np.int64)
    return np.array(
        [func(x[parts[i]: parts[i + 1]]) for i in range(len(parts) - 1)]
    )


def detrend(xs, dt, taumax=None, mode='constant'):
    xdetrend = np.zeros_like(xs)
    xpart = partition(xs, dt, taumax)
    lens = [len(xp) for xp in xpart]
    i=0
    for xp in xpart:
        L = len(xp)
        xdetrend[i:i+L] = sigdetrend(xp, type=mode)
        i += L
    return xdetrend

def PSD(xs, dt, taumax=None, tmin=None, tmax=None, detrend="linear", window="hann", noverlap=None):
    if tmin is not None or tmax is not None:
        ts = np.arange(len(xs)) * dt
        if tmin is None:
            tmin = ts[0]
        if tmax is None:
            tmax = ts[-1]
        mask = np.logical_and(ts>tmin, ts<tmax)
        xp = xs[mask]
        freq, psdavg = welch(xp, fs=1/dt, nperseg=xp.size, window=window,
                          detrend=detrend, noverlap=noverlap)
        Navg = 1
    else:
        xpart = partition(xs, dt, taumax)
        Navg = len(xpart)
        psdavg = 0
        # average PSD of each partions
        for xp in xpart:
            # Bartlet PSD is welch PSD with zero overlap
            freq, psd = welch(xp, fs=1/dt, nperseg=xp.size, window=window,
                              detrend=detrend, noverlap=noverlap)
            psdavg += psd / Navg
    return freq, psdavg, Navg


def AVAR(xs, dt, func=np.mean, octave=True, base=2, Nmin=20):
    """Allan Variance"""
    if octave:
        Npts_list = np.array([base**m for m in range(1, int(np.log2(xs.size/base)/np.log2(base)))])
    else:
        Npts_list = np.arange(Nmin, int(xs.size/2), Nmin)
    Npts_list = Npts_list[Npts_list>=Nmin]
    taus = dt * Npts_list
    avars = np.zeros_like(taus)
    davars = np.zeros_like(taus)
    for j, tau in enumerate(taus):
        xparts = partition(xs, dt, taumax=tau)
        try:
            vals = func(xparts, axis=1)
        except TypeError:
            vals = np.array([func(xp) for xp in xparts])
        avars[j] = np.mean((vals[1:] - vals[:-1])**2) / 2
        davars[j] = np.var(vals)
    Navg = 1
    return taus, avars, Navg


def NVAR(xs, dt, func=np.mean, octave=True, base=2, Nmin=20):
    """Normal variance"""
    if octave:
        Npts_list = np.array([base**m for m in range(1, int(np.log2(xs.size/base)/np.log2(base)))])
    else:
        Npts_list = np.arange(Nmin, int(xs.size/2), Nmin)
    Npts_list = Npts_list[Npts_list>=Nmin]
    taus = dt * Npts_list
    nvars = np.zeros_like(taus)
    dnvars = np.zeros_like(taus)
    m = np.mean(xs)
    for j, tau in enumerate(taus):
        xparts = partition(xs, dt=dt, taumax=tau)
        try:
            vals = func(xparts, axis=1)
        except TypeError:
            vals = np.array([func(xp) for xp in xparts])
        nvars[j] = np.mean((vals - m)**2)/2
        dnvars[j] = np.var(vals)
    Navg = 1
    return taus, nvars, Navg

def HIST(xs, dt, taumax=None, lb=None, ub=None, Nbins=45, density=True, remove_mean=False):
    xpart = partition(xs, dt, taumax)
    Navg = len(xpart)
    histavg = 0
    if lb is None:
        lb = xs.min()
    if ub is None:
        ub = xs.max()
    bins = np.linspace(lb, ub, Nbins, endpoint=True)
    for xp in xpart:
        if remove_mean:
            xpprime = xp - np.mean(xp)
        else:
            xpprime = xp
        hist, edges = np.histogram(xpprime, bins=bins, density=density)
        histavg += hist / Navg
    bins = edges[:-1] + np.diff(edges) / 2
    return bins, histavg, Navg



@jit(nopython=True)
def MSD_jit(x):
    N = x.size
    msd = np.zeros(N-1, dtype=np.float32)
    for lag in range(1, N):
        diffs = x[:-lag] - x[lag:]
        msd[lag - 1] = np.mean(diffs**2)
    return msd


MSD_jit(np.array([1.0, 2.9, 3.0, 4.0]))


def MSD(xs, dt, taumax=None, n_jobs=1):
    """Mean squared displacement"""
    xpart = partition(xs, dt, taumax)
    Navg = len(xpart)
    # average MSD of each partion
    res = Parallel(n_jobs=n_jobs)(delayed(MSD_jit)(xp) for xp in xpart)
    msdavg = np.mean(res, axis=0)
    tmsd = np.arange(1, len(msdavg)+1) * dt
    return tmsd, msdavg, Navg

def ACF_workload(xp):
    n = len(xp)
    corr = np.correlate(xp, xp, mode='full')[-n:]
    return corr / np.arange(n, 0, -1)

def ACF(xs, dt, taumax=None, n_jobs=1):
    """Auto correlation function"""
    xpart = partition(xs, dt, taumax)
    Navg = len(xpart)
    res = Parallel(n_jobs=n_jobs)(delayed(ACF_workload)(xp) for xp in xpart)
    acfavg = np.mean(res, axis=0)
    tacf = np.arange(0, len(acfavg)) * dt
    return tacf, acfavg, Navg


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


def get_krhoA(a, b, c, R, T, eta=None, RH=50, **kwargs):
    if eta is None:
        eta = get_viscosity(T, RH=RH)
    d2 = b + 2 * np.sqrt(a*c)
    k = 12 * np.pi**2 * eta * R * np.sqrt(a/d2)
    rho = 9 * eta / (4*np.pi*R**2) * np.sqrt(c/d2)
    Ainv2 = 6*np.pi**3*eta*R / (kB*T * d2)
    A = np.sqrt(Ainv2)
    return k, rho, A

def get_jac_krhoA(a, b, c, R, T, RH=50):
    """Jacobian of physical parameters w.r.t fitting parameters"""
    eta = get_viscosity(T, RH=RH)
    d1 = b + np.sqrt(a*c)
    d2 = b + 2 * np.sqrt(a*c)
    cof = 6 * np.pi**2 * eta * R / np.sqrt(a*d2**3)
    u1 = 3 / (16 * np.pi**3 * R**3)
    u2 = 1 / (2 * np.sqrt(6 * np.pi * eta * R * kB * T))
    jac = np.array([
    [d1, -a, -np.sqrt(a**3/c), 2*a*d2/R, 2*a*d2/eta, 0],
    [-u1*c, -u1*np.sqrt(a*c), u1*d1*np.sqrt(a/c),
        -4*u1*d2*np.sqrt(a*c)/R, 2*u1*d2*np.sqrt(a*c)/eta, 0],
     [-u2*np.sqrt(c), -u2*np.sqrt(a), -u2*a/np.sqrt(c),
        u2*d2*np.sqrt(a)/R, u2*d2*np.sqrt(a)/eta, -u2*d2*np.sqrt(a)/T]])
    return cof * jac

def get_m123(a, b, c, R, xvar, vvar, T, RH=50):
    """3 mass calculations from fitting parameters and variances"""
    eta = get_viscosity(T, RH=RH)
    d2 = b + 2 * np.sqrt(a*c)
    m1 = 3*eta*R * np.sqrt(c / d2)
    m2 = 6*np.pi**3*eta*R / (d2 * vvar)
    m3 = 12*np.pi**2*eta*R * np.sqrt(a/d2) * xvar/vvar
    return m1, m2, m3

def get_jac_m123(a, b, c, R, xvar, vvar, T, RH=50):
    """Jacobian of mass calculations"""
    eta = get_viscosity(T, RH=RH)
    d2 = b + 2 * np.sqrt(a*c)
    d1 = b + np.sqrt(a*c)
    cof = 6 * np.pi**2 * eta * R / np.sqrt(a*d2**3)
    v1 = 1 / (4*np.pi**2)
    v2 = np.pi / vvar
    v3 = xvar / vvar
    jac = np.array([
        [-v1*c, -v1*np.sqrt(a*c), v1*d1*np.sqrt(a/c), 0, 0,
            2*v1*d2*np.sqrt(a*c)/R, 2*v1*d2*np.sqrt(a*c)/eta],
         [-v2*np.sqrt(c / d2), -v2*np.sqrt(a / d2), -v2*a/np.sqrt(c*d2),
            -v2*np.sqrt(a*d2)/vvar, 0, v2*np.sqrt(a*d2)/R, v2*np.sqrt(a*d2)/eta],
         [v3*d1, -v3*a, -v3*np.sqrt(a**3/c), -2*v3*a*d2/vvar, 2*v3*a*d2/xvar,
            2*v3*a*d2/R, 2*v3*a*d2/eta] ])
    return cof * jac

def abc_guess(freq, psd, n=1):
    Smat, Sinvmat, Svec = get_Smat(freq, psd)
    Sinvmat *= (n+1) / n
    popt0 = Sinvmat.dot(Svec)
    _, pSinvmat, _ = get_Smat(freq, psd_abc_func(freq, *popt0))
    pcov0 = (n+3) * pSinvmat / (n + 1) / freq.size
    return popt0, pcov0

def gaussian_func(x, var, mean=0):
    return np.exp(-(x - mean)**2 / (2 * var)) / (np.sqrt(2 * np.pi * var))

def psd_abc_func(f, a, b, c, **kwargs):
    return 1 / np.abs((a + b * f ** 2 + c * f ** 4))


def psd_func(f, k, rho, T, R, eta=None, RH=50, **kwargs):
    if eta is None:
        eta = get_viscosity(T, RH=RH)
    m = 4*np.pi*rho*R**3/3
    gamma = 6*np.pi*eta*R
    omega = 2*np.pi * f
    denom = (m*omega**2 - k)**2 + (gamma*omega)**2
    return 4 * kB * T * gamma / denom


def msd_func(t, k, rho, T, R, eta=None, RH=50, **kwargs):
    if eta is None:
        eta = get_viscosity(T, RH=50)
    m = 4*np.pi*rho*R**3/3
    gamma = 6*np.pi*eta*R
    Omega = np.sqrt(k / m)
    Gamma = gamma / m
    taup = 1 / Gamma
    omega1 = sqrt(Omega**2 - Gamma**2/4)
    cs = np.cos(omega1 * t) + np.sin(omega1*t) / (2*omega1*taup)
    cs = cs.real
    return 2*kB*T/k * (1 - cs * np.exp(-t/(2*taup)))


def pac_func(t, k, rho, T, R, eta=None, RH=50, **kwargs):
    if eta is None:
        eta = get_viscosity(T, RH)
    m = 4*np.pi*rho*R**3/3
    gamma = 6*np.pi*eta*R
    Omega = np.sqrt(k / m)
    Gamma = gamma / m
    taup = 1 / Gamma
    omega1 = sqrt(Omega**2 - Gamma**2/4)
    cs = np.cos(omega1 * t) + np.sin(omega1*t) / (2*omega1*taup)
    cs = cs.real
    return kB*T/k * cs * np.exp(-t/(2*taup))


def vac_func(t, k, rho, T, R, eta=None, RH=50, **kwargs):
    if eta is None:
        eta = get_viscosity(T, RH=RH)
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
        plot_x = logbin_func(x, Npts=Npts)
        plot_y = logbin_func(y, Npts=Npts)
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
    load = True
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
        tmsd, msd, Navg = MSD(xs[:, i], dt, taumax=40e-3)
        tpacf, pacf, Navg = ACF(xs[:, i], dt, taumax=40e-3)
        tvacf, vacf, Navg = ACF(vs[:, i], dt, taumax=40e-3)
        tavar, avar, davar = AVAR(vs[:, i], dt, func=np.mean)
        tnvar, nvar, dnvar = NVAR(vs[:, i], dt, func=np.mean)

        # Estimate parameters from PSD
        fmin = 75
        fmax = 1e4
        Npts = 10
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
        axs[0, 0].plot(freq, psd_func(freq, **params), c="r")
        axs[0, 0].set_ylabel(r"PSD ($\mathrm{m^2/Hz}$)")
        axs[0, 0].set_xlabel("$f$ (Hz)")
        axs[0, 0].set_yscale("log")
        axs[0, 0].set_xscale("log")

        # MSD plot
        dataplot(axs[0, 1], tmsd, msd, Npts=Npts)
        axs[0, 1].plot(tmsd, msd_func(tmsd, **params), c="r")
        axs[0, 1].set_ylabel(r"MSD ($\mathrm{m^2}$)")
        axs[0, 1].set_xlabel(r"$\tau$ (s)")
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_xscale("log")

        # PACF plot
        dataplot(axs[1, 0], tpacf, pacf, Npts=Npts)
        axs[1, 0].plot(tpacf, pac_func(tpacf, **params), c="r")
        axs[1, 0].set_ylabel("PACF ($\mathrm{m^2}$)")
        axs[1, 0].set_xlabel(r"$\tau$ (s)")
        axs[1, 0].set_xscale("log")

        # VACF plot
        #dataplot(axs[1, 1], tvacf, vacf, Npts=Npts)
        #axs[1, 1].plot(tpacf, vac_func(tvacf, **params), c="r")
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

