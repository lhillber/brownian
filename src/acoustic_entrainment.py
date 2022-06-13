import numpy as np
from brownian import get_gamma, get_mass, get_viscosity, get_sound_speed, get_air_density
from constants import kB
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import get_window
from scipy.special import hankel2
from time_series import TimeSeries

class VelocityResponse:
    def __init__(self, R, rho, T, RH=0, k=0, c0=None, rho_fluid=None, mu=None):
        self.R = R
        self.k = k
        self.rho = rho
        self.T = T
        self.RH = RH
        if c0 is None:
            c0 = get_soundspeed(self.T, self.RH)
        if rho_fluid is None:
            rho_fluid = get_air_density(self.T, self.RH)
        if mu is None:
            mu = get_viscosity(self.T, self.RH)
        self.c0 = c0
        self.rho_fluid = rho_fluid
        self.mu = mu

        self.Z0 = self.rho_fluid*self.c0
        self.delta = self.rho_fluid / self.rho
        self.nu = self.mu / self.rho_fluid
        self.m = get_mass(self.R, self.rho)
        self.gamma = get_gamma(self.R, self.mu)
        self.taup = self.m / self.gamma
        self.Gamma = 1/self.taup
        self.w0=np.sqrt(self.k / self.m)


    def response(self, name, f):
        F, G, H, I = getattr(self, f"_FGHI_{name}")(f)
        return (F + 1j*G) / (H + 1j*I)


    def amplitude(self, name, f):
        return np.abs(self.response(name, f))


    def power(self, name, f):
        return np.abs(self.response(name, f))**2


    def phase(self, name, f):
        # NOTE: Using two-quadrant arctan instead of four-quadrant
        # (np.arctan2 or np.angle) because we can always remove a global phase
        # such that the relatve phase is between -pi/2 and pi/2
        F, G, H, I = getattr(self, f"_FGHI_{name}")(f)
        return np.arctan((G*H-I*F)/(F*H+I*G))


    def correct_signal(self, time, signal, tmin=None, tmax=None, r0=5e-2,
        correction="bassetbound", window="blackmanharris", impedance=None):
        if tmin is None:
            tmin =time[0]
        if tmax is None:
            tmax = time[-1]
        dt = time[1] - time[0]
        mask = np.logical_and(time>tmin, time<tmax)
        sig = signal[mask]
        t = time[mask]
        signal_length = len(sig)
        freq = rfftfreq(signal_length, dt)[1:]
        response = self.response(correction, freq)
        response = np.r_[1, response]
        if impedance is None:
            impedance = np.ones_like(response)
            name = "acoustic velocity"
        else:
            impedance = getattr(self, f"{impedance}_impedance")(freq, r0)
            name = "Pressure"
        win = get_window(window, signal_length)
        corr = np.sqrt(np.sum(win**2)/signal_length) #amp correction
        corrected_signal = irfft(rfft(sig * win) / corr / response, n=signal_length)
        return TimeSeries(corrected_signal, t, name=name)


    def w(self, f):
        return 2*np.pi*f


    def b(self, f):
        return 2 * np.pi * f * self.R / self.c0


    def y(self, f):
        return np.sqrt(2*np.pi*f * self.R**2 / (2*self.nu))


    def eps(self, f):
        return 3/2 * np.sqrt(self.delta*self.taup*2*np.pi*f)


    def _FGHI_stokesbound(self, f):
        F = self.Gamma * self.w(f)
        G = 0
        H = self.Gamma * self.w(f)
        I = -(self.w(f)**2 - self.w0**2)
        return F, G, H, I


    def _FGHI_partbassetbound(self, f):
        eps = self.eps(f)
        w0 = self.w0
        w = self.w(f)
        Gamma = 1/self.taup
        F = w*Gamma*(1 + eps)
        G = -w*eps*(3*eps/3 + Gamma)
        H = Gamma * w
        I = (w0*w0-w*w)
        return F, G, H, I


    def _FGHI_bassetbound(self, f):
        w = self.w(f)
        w0 = self.w0
        taup = self.taup
        d = self.delta
        eps = self.eps(f)
        F = w * (1 + eps)
        G = -w * eps * (1 + 2/3*eps)
        H = w * (1 + eps)
        I = taup*(w0*w0 - w*w*(1 + d/2)) - w*eps
        return F, G, H, I


    def _FGHI_exact(self, f):
        d = self.delta
        y = self.y(f)
        b = self.b(f)
        b_y2  = (b/y)**2
        b2_y = b*b / y
        F = (2*y*y + 3*y + b_y2*(1 + y))
        G = (3*(1 + y) - 2*b*b - b2_y)
        H = 2*y*y * (b*b - 2 - d) + y * (b*b * (1 + 2*d) - 9*d * (b + 1))
        H += 9*d*b * (2*d*b*b - 1) + 3*d*b2_y * (b - 1) - 3*d*b_y2
        I = 2*y*y*b * (2 + d) + y * (9*d * (b - 1) + b*b * (1 + 2*d))
        I += b*b * (1 + 4*d) - 9*d  + 3*d*b2_y * (b + 1) + 3*d*b*b_y2
        return 3*d* F, 3*d*G, H, I


    def plane_impedance(self, f, r0):
        return self.Z0


    def cyl_impedance(self, f, r0):
        kr = r0 * 2*np.pi*f/self.c0
        return self.Z0 * 1j*hankel2(0, kr)/hankel2(1, kr)


    def sph_impedance(self, f, r0):
        kr = r0 * 2*np.pi*f/self.c0
        return self.Z0*kr * (kr + 1j)/(1 + kr*kr)


    def Sx_basset(self, f, chi=0):
        eps = self.eps(f)
        w = self.w(f)
        G =self.Gamma
        d = self.delta
        num = 4 * kB * self.T * G / self.m * (1 + eps)
        denom = G*G*w*w*(1 + eps)**2
        denom += (self.w0**2 - w*w*(1+d/2) - G*w*eps)**2
        return num / denom + chi


    def Sv_basset(self, f, chi=0):
        w = self.w(f)
        return w*w * self.Sx_basset(f, chi)


    def Sx_stokes(self, f, chi=0):
        w = self.w(f)
        G = self.Gamma
        d = self.delta
        num = 4 * kB * self.T * G / self.m
        denom = G*G*w*w +(self.w0**2 - w*w)**2
        return num / denom + chi


    def Sv_stokes(self, f, chi=0):
        w = self.w(f)
        return w*w * self.Sx_stokes(f, chi)

    #def amplitude_ratio_basset(self, f):
    #    y = self.y(f)
    #    d = self.delta
    #    num = 4*y**4 + 12*y*y* y + 18*y*y + 18*y +9
    #    denom = 4*(2+d)**2*y**4 + 36*d*y*y*y*(2+d) + 81*d*d*(2*y*y + 2*y + 1)
    #    return 3*d*np.sqrt(num/denom)

    #def phase_basset(self, f):
    #    y = self.y(f)
    #    d = self.delta
    #    num = 12*(y+1)*y*y*(1-d)
    #    denom = 4*y**4*(2+d) + 12*y*y*y*(1 + 2*d) + 27*d*(2*y*y + 2*y + 1)
    #    return np.arctan2(num, denom)

    #def amplitude_ratio_stokes(self, f):
    #    return 1 / np.sqrt(1 + (2*np.pi*f * self.taup)**2)

    #def phase_stokes(self, f):
    #    return np.arctan2(2*np.pi*f * self.taup, 1)

    #def amplitude_ratio_exact(self, f):
    #    F, G, H, I = self._FGHI(f)
    #    return np.sqrt((F*F + G*G)/(H*H + I*I))


    #def phase_exact(self, f):
    #    F, G, H, I = self._FGHI(f)
    #    return np.arctan((G*H - I*F) / (F*H + I*G))


