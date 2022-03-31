import numpy as np
from brownian import get_gamma, get_mass, get_viscosity
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import get_window
from scipy.special import hankel2
from time_series import TimeSeries

class VelocityResponse:
    def __init__(self, R, rho, T, k=0, c0=344, rho_fluid=1.225):
        self.R = R
        self.k = k
        self.rho = rho
        self.T = T
        self.c0 = c0
        self.rho_fluid = rho_fluid

        self.delta = self.rho_fluid / self.rho
        self.mu = get_viscosity(self.T)
        self.nu = self.mu / self.rho_fluid
        self.mass = get_mass(self.R, self.rho)
        self.gamma = get_gamma(self.R, self.mu)
        self.taup = self.mass / self.gamma
        self.w0=np.sqrt(self.k / self.mass)

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
        amp = getattr(self, f"amplitude_ratio_{correction}")(freq)
        phase = getattr(self, f"phase_{correction}")(freq)
        if impedance is None:
            impedance = np.ones_like(amp)
        else:
            impedance = getattr(self, f"{impedance}_impedance")(freq, r0)
        response = amp * np.exp(1j * phase) / impedance
        response = np.r_[1, response]
        win = get_window(window, signal_length)
        corrected_signal = irfft(rfft(sig * win) / response, n=signal_length)
        return TimeSeries(corrected_signal, t, name="Corrected")

    def plane_impedance(self, f, r0):
        return self.c0 * self.rho_fluid

    def cyl_impedance(self, f, r0):
        kr = r0 * 2*np.pi*f/self.c0
        z0 = self.plane_impedance(f, r0)
        return z0 * 1j*hankel2(0, kr)/hankel2(1, kr)

    def sph_impedance(self, f, r0):
        kr = r0 * 2*np.pi*f/self.c0
        z0 = self.plane_impedance(f, r0)
        return z0*kr/(1+kr*kr) * (kr + 1j)

    def b(self, f):
        return 2 * np.pi * f * self.R / self.c0

    def y(self, f):
        return np.sqrt(2*np.pi*f * self.R**2 / (2*self.nu))

    def amplitude_ratio_stokesbound(self, f):
        w = 2 * np.pi * f
        w0 = self.w0
        return w / np.sqrt(w*w + self.taup**2 * (w*w - w0*w0)**2)

    def phase_stokesbound(self, f):
        w = 2 * np.pi * f
        w0 = self.w0
        return np.arctan2((w*w - w0*w0), (w/self.taup))

    def amplitude_ratio_basset(self, f):
        y = self.y(f)
        d = self.delta
        num = 4*y**4 + 12*y*y* y + 18*y*y + 18*y +9
        denom = 4*(2+d)**2*y**4 + 36*d*y*y*y*(2+d) + 81*d*d*(2*y*y + 2*y + 1)
        return 3*d*np.sqrt(num/denom)

    def phase_basset(self, f):
        y = self.y(f)
        d = self.delta
        num = 12*(y+1)*y*y*(1-d)
        denom = 4*y**4*(2+d) + 12*y*y*y*(1 + 2*d) + 27*d*(2*y*y + 2*y + 1)
        return np.arctan2(num, denom)

    def _FGHI_bassetbound(self, f):
        w = 2*np.pi * f
        w0 = self.w0
        taup = self.taup
        d = self.delta
        dwt = d*w*taup
        F = 3*w * (np.sqrt(dwt) + dwt)
        G = w * (3*np.sqrt(dwt) + 2)
        H = -2*taup*w0*w0 + 3*np.sqrt(dwt*w*w) + (2+d)*taup*w*w
        I = w*(2 + 3*np.sqrt(dwt))
        return F, G, H, I

    def amplitude_ratio_bassetbound(self, f):
        w0 = self.w0
        F, G, H, I = self._FGHI_bassetbound(f)
        return np.sqrt((F*F + G*G)/(H*H + I*I))

    def phase_bassetbound(self, f):
        w0 = self.w0
        F, G, H, I = self._FGHI_bassetbound(f)
        return np.arctan2((G*H - I*F), (F*H + I*G))

    def _FGHI_memorybound(self, f):
        w = 2*np.pi * f
        w0 = self.w0
        taup = self.taup
        d = self.delta
        dwt = d*w*taup
        F = w + 3 * np.sqrt(dwt*w*w)
        G = w
        H = F + taup*(w*w - w0*w0)
        I = w + taup*(-w*w + w0*w0)
        return F, G, H, I

    def amplitude_ratio_memorybound(self, f):
        F, G, H, I = self._FGHI_memorybound(f)
        return np.sqrt((F*F + G*G)/(H*H + I*I))

    def phase_memorybound(self, f):
        F, G, H, I = self._FGHI_memorybound(f)
        return np.arctan2((G*H - I*F), (F*H + I*G))

    def amplitude_ratio_stokes(self, f):
        return 1 / np.sqrt(1 + (2*np.pi*f * self.taup)**2)

    def phase_stokes(self, f):
        return np.arctan2(2*np.pi*f * self.taup, 1)

    def _FGHI(self, f):
        d = self.delta
        y = self.y(f)
        b = self.b(f)
        b_y2  = (b/y)**2
        b2_y = b*b / y
        F = 2*y*y + 3*y + b_y2*(1 + y)
        G = 3*(1 + y) - 2*b*b - b2_y
        H = 2*y*y * (b*b - 2 - d) + y * (b*b * (1 + 2*d) - 9*d * (b + 1))
        H += 9*d*b * (2*d*b*b - 1) + 3*d*b2_y * (b - 1) - 3*d*b_y2
        I = 2*y*y*b * (2 + d) + y * (9*d * (b - 1) + b*b * (1 + 2*d))
        I += b*b * (1 + 4*d) - 9*d  + 3*d*b2_y * (b + 1) + 3*d*b*b_y2
        return F, G, H, I

    def amplitude_ratio_exact(self, f):
        F, G, H, I = self._FGHI(f)
        return 3*self.delta * np.sqrt((F*F + G*G)/(H*H + I*I))


    def phase_exact(self, f):
        F, G, H, I = self._FGHI(f)
        return np.arctan((G*H - I*F) / (F*H + I*G))


