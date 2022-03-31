# Simulate the Brownian motion of a damped, harmonically confined sphere.
#
# By Logan Hillberry


import numpy as np
from constants import kB, units
from brownian import get_viscosity, get_mass, get_gamma

class Simulation:
    def __init__(self):
        self.params = {}

    def __getattr__(self, attr):
        return self.params[attr]

    def _add_params(self, dict_in):
        self.params.update(dict_in)

    def set_environment(self, T):
        self._add_params({"T": T})

    def set_particle(self, rho, R=None):
        self._add_params({"R": R, "rho": rho})

    def set_harmonic_trap(self, K):
        self._add_params({"K": K, "dim": len(K)})

    def set_tweezers(self, fname, power_F, power_B):
        from tweezer_force import Tweezers, Domain
        tweezers = Tweezers()
        tweezers.load(fname)
        tweezers.evaluate_net_force(powers={"F": power_F, "B": power_B})
        tweezers.make_interpolation(name="net")
        xyz0 = tweezers.find_min()
        nm = units["nm"]["value"]
        Deltas = [100 * nm, 100 * nm, 700 * nm]
        meshspec = [[q0 - Delta, q0 + Delta, 21] for q0, Delta in zip(xyz0, Deltas)]
        fit_domain = Domain(meshspec)
        (xyz0, dxyz0), (kxyz, dkxyz) = tweezers.fit_min(domain=fit_domain)
        self.tweezers = tweezers
        self.set_harmonic_trap(kxyz)
        self._add_params({"R": tweezers.params["R"]})
        self._add_params({"tweezers_fname": fname,
                          "power_F": power_F,
                          "power_B": power_B})


    def set_external_velocity_pulse(self, speed, width, origin, peak, shape, axis_index=0):
        self._add_params({"pulse_speed": speed,
                          "pulse_width": width,
                          "pulse_origin": origin,
                          "pulse_peak": peak,
                          "pulse_axis_index": axis_index,
                          "pulse_shape": shape})
        self.external_velocity_pulse = getattr(self, f"{shape}_pulse")


    def initialize_state(self, position=None, velocity=None):
        if position is None:
            position = np.array([
                np.random.normal(0, np.sqrt(kB * self.T / k)) for k in self.K])
        if velocity is None:
            velocity = np.random.normal(0, np.sqrt(kB * self.T / self.mass), self.dim)
        state = np.zeros(2*self.dim)
        state[:3] = position
        state[3:] = velocity
        self._add_params({"initial_state": state})
        self.state = state


    def initialize_time(self, dt, time_steps):
        max_time = (time_steps - 1) * dt
        self.t = 0
        self._add_params({
            "dt": dt, "time_steps": time_steps, "max_time": max_time
            })

    @property
    def position(self):
        return self.state[:3]

    @property
    def velocity(self):
        return self.state[3:]

    @property
    def mass(self):
        return get_mass(self.R, self.rho)

    @property
    def viscosity(self):
        return get_viscosity(self.T)

    @property
    def gamma(self):
        return get_gamma(self.R, self.viscosity)

    @property
    def taup(self):
        return self.mass / self.gamma

    def fullsin_pulse(self, t, position):
        coordinate = position[self.pulse_axis_index]
        xprime = coordinate - self.pulse_origin - self.pulse_speed * t
        if np.abs(xprime) <= self.pulse_width / 2:
            val = self.pulse_peak * np.cos(
                np.pi*xprime / (self.pulse_width / 2)
                )
        else:
            val = 0.0
        pulse = np.zeros(self.dim)
        pulse[self.pulse_axis_index] = val
        return pulse


    def Nwave_pulse(self, t, position):
        coordinate = position[self.pulse_axis_index]
        xprime = coordinate - self.pulse_origin - self.pulse_speed * t
        if np.abs(xprime) <= self.pulse_width / 2:
            val = self.pulse_peak * xprime / self.pulse_width
        else:
            val = 0.0
        pulse = np.zeros(self.dim)
        pulse[self.pulse_axis_index] = val
        return pulse

    def dgaussian_pulse(self, t, position):
        coordinate = position[self.pulse_axis_index]
        xprime = coordinate - self.pulse_origin - self.pulse_speed * t
        sigma = self.pulse_width/2
        val = np.exp(1/2 - xprime**2/(2*sigma**2))
        val *= -xprime/sigma
        pulse = np.zeros(self.dim)
        pulse[self.pulse_axis_index] = val
        return pulse

    def d2gaussian_pulse(self, t, position):
        coordinate = position[self.pulse_axis_index]
        xprime = coordinate - self.pulse_origin - self.pulse_speed * t
        sigma = self.pulse_width/2
        val = np.exp(3/2-xprime**2/(2*sigma**2))
        val *= (sigma**2 - xprime**2)
        val *=-self.pulse_peak / (2 * sigma**2)
        pulse = np.zeros(self.dim)
        pulse[self.pulse_axis_index] = val
        return pulse


    def ABAflattop_pulse(self, t, position):
        coordinate = position[self.pulse_axis_index]
        xprime = coordinate - self.pulse_origin - self.pulse_speed * t
        if np.abs(xprime) <= self.pulse_width / 2:
            if np.abs(xprime) <= self.pulse_width / 4:
                val = -self.pulse_peak
            else:
                val = self.pulse_peak
        else:
            val = 0.0
        pulse = np.zeros(self.dim)
        pulse[self.pulse_axis_index] = val
        return pulse

    def trapping_force(self, position, harmonic=False):
        if harmonic:
            trapping_force = -self.K * position
        else:
            trapping_force = np.squeeze(self.tweezers.interpolate_force(position))
        return trapping_force

    def drag_force(self, t, state):
        position = state[:3]
        velocity = state[3:]
        return -self.gamma * (velocity - self.external_velocity_pulse(t, position))


    def thermal_force(self, rand_normal):
        thermal_force =  rand_normal / np.sqrt(self.dt)
        thermal_force *=  np.sqrt(2 * kB * self.T * self.gamma)
        return thermal_force


    def acceleration(self, t, state, rand_normal):
        position = state[:3]
        a = self.trapping_force(position)
        a += self.drag_force(t, state)
        a += self.thermal_force(rand_normal)
        a /= self.mass
        return a


    def slope_function(self, t, state, rand_normal):
        velocity = state[3:]
        return np.r_[velocity, self.acceleration(t, state, rand_normal)]


    def rk4_step(self):
        rand_normal = np.random.normal(size=self.dim)
        K1 = self.slope_function(self.t, self.state, rand_normal)
        K2 = self.slope_function(self.t+self.dt/2.0, self.state + self.dt * K1/2.0, rand_normal)
        K3 = self.slope_function(self.t+self.dt/2.0, self.state + self.dt * K2/2.0, rand_normal)
        K4 = self.slope_function(self.t+self.dt, self.state + self.dt * K3, rand_normal)
        self.t += self.dt
        self.state += self. dt / 6.0 * (K1 + 2*K2 + 2*K3 + K4)
        return self.t, self.state


    def run(self):
        states = np.zeros((self.time_steps+1, 2*self.dim))
        times = np.zeros(self.time_steps+1)
        states[0] = self.initial_state
        for step_index in range(self.time_steps):
            t, state = self.rk4_step()
            times[step_index+1] = t
            states[step_index+1] = state
        return times, states


def acceleration(xs, vs, t, m, gamma, K, T, dt, field, **params):
    dim = len(K)
    G = np.sqrt(2*kB*T*gamma)
    a = (-K*xs - gamma*vs + G*np.random.normal(size=dim)*dt**(-1/2)) / m
    if field is not None:
        a += field(xs, t, **params) / m
    return a


def sim2(m, gamma, K, T, dt, tmax, field, x0=None, v0=None, **params):
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
    vs[1] = v0 + dt * acceleration(x0, v0, ts[0], m, gamma, K, T, dt, field, **params)
    xs[1] = x0 + dt*vs[1]
    for ti in range(1, len(ts)-1):
        xs[ti+1] = 2*xs[ti] - xs[ti-1] + dt * dt * \
            acceleration(xs[ti], vs[ti], ts[ti], m, gamma, K, T, dt, field, **params)
        vs[ti+1] = (xs[ti+1] - xs[ti-1]) / (2*dt)
    return xs, vs, ts
