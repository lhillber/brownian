# Simulate the Brownian motion of a damped, harmonically confined sphere.
#
# By Logan Hillberry


import numpy as np
from constants import kB, units
from brownian import get_mass, get_gamma

class Simulation:
    def __init__(self):
        self.params = {}

    def __getattr__(self, attr):
        return self.params[attr]

    def _add_params(self, dict_in):
        self.params.update(dict_in)

    def set_environment(self, T, eta):
        self._add_params({"T": T, "viscosity": eta})

    def set_particle(self, rho, R=None):
        self._add_params({"R": R, "rho": rho})

    def set_over_damped(self):
        self._add_params({"over_damped": True})

    def set_under_damped(self):
        self._add_params({"over_damped": False})

    def set_harmonic_trap(self, K, modulation_function=None):
        if modulation_function is None:
            modulation_function = lambda t: 1
        if type(K) == list:
            dim = len(K)
            K = np.array(K)
        if type(K) in (int, float):
            dim = 1
            K = np.array([K]* dim)
        self.harmonic = True
        self._add_params({"K": K,
                          "modulation_function": modulation_function,
                          "dim": dim})


    def set_external_force(self, external_force=None):
        if external_force is None:
            external_force = lambda t, state: 0
        self._add_params({"external_force": external_force})


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
        self.harmonic = False
        self.tweezers = tweezers
        self.set_harmonic_trap(kxyz)
        self._add_params({"R": tweezers.params["R"]})
        self._add_params({"tweezers_fname": fname,
                          "power_F": power_F,
                          "power_B": power_B})


    @property
    def mass(self):
        return get_mass(self.R, self.rho)


    @property
    def gamma(self):
        return get_gamma(self.R, self.viscosity)


    @property
    def taup(self):
        return self.mass / self.gamma


    def initialize_state(self, position=None, velocity=None):
        if position is None:
            position = np.array([
                np.random.normal(0, np.sqrt(kB * self.T / k)) for k in self.K])
        if self.over_damped:
            state = position
        else:
            if velocity is None:
                velocity = np.random.normal(0, np.sqrt(kB * self.T / self.mass), self.dim)
            state = np.zeros(2*self.dim)
            state[:self.dim] = position
            state[self.dim:] = velocity
        self._add_params({"initial_state": state})
        self.state = state


    def initialize_time(self, dt, time_steps):
        max_time = (time_steps - 1) * dt
        self.t = 0
        self._add_params({
            "dt": dt, "time_steps": time_steps, "max_time": max_time
            })


    def position(self, state=None):
        if state is None:
            state = self.state
        return self.state[:self.dim]


    def velocity(self, t=0, state=None, rand_normal=None):
        if not self.over_damped:
            if state is None:
                state = self.state
            v = self.state[self.dim:]
        else:
            v = self.trapping_force(t, state)
            v += self.thermal_force(rand_normal)
            v += self.external_force(t, state)
            v /= self.gamma
        return v

    def acceleration(self, t, state, rand_normal):
        a = self.trapping_force(t, state)
        a += self.drag_force(t, state)
        a += self.thermal_force(rand_normal)
        a += self.external_force(t, state)
        a /= self.mass
        return a


    def trapping_force(self, t, state):
        if self.harmonic:
            trapping_force = -self.K * self.position(state)
        else:
            trapping_force = np.squeeze(
            self.tweezers.interpolate_force(self.position(state)))
        return self.modulation_function(t) * trapping_force


    def drag_force(self, t, state):
        return -self.gamma * (self.velocity(state))


    def thermal_force(self, rand_normal=None):
        if rand_normal is None:
            rand_normal = np.random.normal(size=self.dim)
        thermal_force =  rand_normal / np.sqrt(self.dt)
        thermal_force *=  np.sqrt(2 * kB * self.T * self.gamma)
        return thermal_force


    def under_damped_slope_function(self, t, state, rand_normal):
        return np.r_[self.velocity(state),
                     self.acceleration(t, state, rand_normal)]

    def rk4_step(self):
        if self.over_damped:
            slope_function = self.velocity
        else:
            slope_function = self.under_damped_slope_function
        rand_normal = np.random.normal(size=self.dim)
        K1 = slope_function(self.t, self.state, rand_normal)
        K2 = slope_function(self.t+self.dt/2.0, self.state + self.dt * K1/2.0, rand_normal)
        K3 = slope_function(self.t+self.dt/2.0, self.state + self.dt * K2/2.0, rand_normal)
        K4 = slope_function(self.t+self.dt, self.state + self.dt * K3, rand_normal)
        self.t += self.dt
        self.state += self. dt / 6.0 * (K1 + 2*K2 + 2*K3 + K4)
        return self.t, self.state



    def euler_step(self):
        if self.over_damped:
            slope_function = self.velocity
        else:
            slope_function = self.under_damped_slope_function
        rand_normal = np.random.normal(size=self.dim)
        self.t += self.dt
        self.state += self.dt * slope_function(self.t, self.state, rand_normal)
        return self.t, self.state


    def run(self, stepper="rk4"):
        stepper = getattr(self, stepper+"_step")
        states = np.zeros((self.time_steps+1, len(self.initial_state)))
        times = np.zeros(self.time_steps+1)
        states[0] = self.initial_state
        for step_index in range(self.time_steps):
            t, state = stepper()
            times[step_index+1] = t
            states[step_index+1] = state
        self.times = times
        self.states = states
        return times, states



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from brownian import PSD, MSD

    ### initialize
    sim = Simulation()

    ### set constant parameters
    sim.set_environment(T=300, eta=1e-3) # Temperatre (K), viscosity (Pa s)
    sim.set_particle(rho=1700, R=3e-6/2) # Mass density (kg/m^3), radius (m)
    k = 2e-4 # Mean trapping potential 2 pN /nm
    K = [k, k] # List of k values sets dimension of simulation
    dt = sim.taup # Inverse sample rate
    time_steps = int(1e4) # Number of simulation time steps

    ### set modulation and external force parameters
    mod_depth = 0.05
    mod_ang_frequency = np.sqrt(10)*np.sqrt(k/sim.mass)

    def mod_func(t):
        return (1 + mod_depth*np.sin(mod_ang_frequency*t))

    def external_force(t, state):
        return np.array([k*100e-6, 0]) * np.sin(mod_ang_frequency*t)

    sim.set_harmonic_trap(K=K, modulation_function=mod_func)
    sim.set_external_force(external_force)

    ### run the simulation
    sim.initialize_state()
    sim.initialize_time(dt=dt, time_steps=time_steps)
    t, states = sim.run()
    print(states.shape)
    r, vr = states[:, :sim.dim], states[:, sim.dim:] # Unpack pos and vel

    ### analysis and visualization
    fig, axs = plt.subplots(3, 1)

    mask = t>2e-3
    t = t[mask]
    r = r[mask, :]
    for x, k in zip(r.T, K):
        # position time series
        axs[0].plot(t*1e3, x*1e9)
        axs[0].set_ylabel("Position (nm)")
        axs[0].set_xlabel("Time (ms)")

        # position power spectral density
        freq, psd, Navg = PSD(x, dt)
        axs[1].loglog(freq[1:], psd[1:]*1e9*1e9)
        axs[1].axvline(k/sim.gamma/ (2*np.pi), ls="--", c="r")
        axs[1].axvline(mod_ang_frequency/(2*np.pi), ls="-", c="k")
        axs[1].set_ylabel(r"PSD (${\rm nm^2/Hz}$)")
        axs[1].set_xlabel("frequency (Hz)")

        # mean squared displacement
        tau, msd, Navg = MSD(x, dt, taumax=sim.max_time*0.05)
        axs[2].loglog(tau*1e3, msd*1e9*1e9)
        axs[2].set_ylabel(r"MSD (${\rm nm^2}$)")
        axs[2].set_xlabel("Lag time (ms)")

    plt.show()
