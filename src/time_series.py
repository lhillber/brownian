#!/usr/bin/env python
# coding: utf-8
# Compute measures on bownian motion time series data.
# By Logan Hillberry

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import stride_windows
from constants import units
from copy import copy
from scipy.integrate import simps
from scipy.optimize import curve_fit, minimize
from nptdms import TdmsFile
from scipy.signal import butter, sosfiltfilt, sosfilt, get_window
from scipy.fft import rfft, rfftfreq, irfft
from joblib import Parallel, delayed
from brownian import (
    partition, bin_func, detrend, PSD, MSD, ACF, AVAR, NVAR, HIST, logbin_func
    )




# File I/O
# ========

def find_files(inder):
    """Return the full path to all files in directory inder."""
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(inder):
        for file in f:
            files.append(os.path.join(r, file))
    return files


def find_ders(inder):
    """Return the full path to all directories in directory der."""
    ders = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(inder):
        for der in d:
            if len(r.split(os.path.sep)) == len(inder.split(os.path.sep)):
                ders.append(os.path.join(r, der))

    return ders


# Helper functions
# ================

def tkey(key):
    """Return key name for indipendent variable given key name of depndent variable"""
    if key == "psd":
        return "freq"
    if key == "hist":
        return "bins"
    else:
        return "t"+key


def firstdiff(tx, acc=8, r=None, t=None):
    """Finite difference approximation for first derivative up to 8th order accuracy"""
    if type(tx) ==  tuple:
        t, x = tx
    else:
        x = tx
    if t is None:
        assert r is not None
        t = np.arange(0, x[:].size) / r
    dt = t[1] - t[0]
    assert acc in [2, 4, 6, 8]
    coeffs = [
        [1.0 / 2],
        [2.0 / 3, -1.0 / 12],
        [3.0 / 4, -3.0 / 20, 1.0 / 60],
        [4.0 / 5, -1.0 / 5, 4.0 / 105, -1.0 / 280],
    ]
    v = np.sum(
        np.array(
            [
                (
                    coeffs[acc // 2 - 1][k - 1] * x[k * 2 :]
                    - coeffs[acc // 2 - 1][k - 1] * x[: -k * 2]
                )[acc // 2 - k : len(x) - (acc // 2 + k)]
                / dt
                for k in range(1, acc // 2 + 1)
            ]
        ),
        axis=0,
    )
    tv = t[acc // 2 : -acc // 2]
    return tv, v


def parse_fname(fname):
    """
    Returns dictionary of parameters encoded in file name.
    Assumes structure:
    der/<name1>-<unit1><val1>_<name2>-<unit2><val2>_<note1>_<note2>/trial_v<version>.txt
    No numbers are allowed in the name, unit, or note fields
    """
    params = {}
    if ".dat" in fname:
        der, trial = os.path.split(fname)
        trial = trial.split("_v")[1]
        trial = int(trial.split(".dat")[0])
        params["trial"] = trial
    else:
        der = fname
    day, der = os.path.split(der)
    day = os.path.split(day)[1].split("_")[0]
    name = os.path.basename(der)
    notes = []
    nsplit = name.split("_")
    for puv in nsplit:
        for i, c in enumerate(puv):
            if c.isdigit():
                break
        if i == len(puv) - 1:
            notes.append(puv)
        else:
            params[puv[:i]] = float(puv[i:])
    params["notes"] = notes
    params["day"] = day
    return params

def load_weighing(fname, norm="Det-V"):
    params = parse_fname(fname)
    if norm is None:
        norm = 1
    if type(norm) == str:
        try:
            norm = self.params[norm]
        except:
            print(f"No param {norm}.")
            norm = 1.0
    x = np.fromfile(fname, dtype=">d")
    x -= np.mean(x)
    x /= norm
    x = np.array(x, dtype=np.float32)
    return x, params


# Object-oriented interfaces
# ==========================

class TimeSeries:
    def __init__(self, tx, t=None, r=None, name="x"):
        if type(tx) ==  tuple:
            t, x = tx
        else:
            x = tx
        if t is None:
            assert r is not None
            t = np.arange(0, x[:].size) / r
        self.name = name
        self.t = t
        self.x = x[:]
        self._x_bak = copy(x[:])
        self._t_bak = copy(self.t)

    def __call__(self):
        return self.t, self.x

    @property
    def size(self):
        return self.x.size

    @property
    def r(self):
        return 1 / (self.t[1] - self.t[0])

    def restore(self):
        self.x = self._x_bak
        self.t = self._t_bak
        return self.t, self.x


    def calibrate(self, cal, inplace=False):
        x2  = cal * self.x
        if inplace:
            self.x = x2
        return x2

    def delay(self, t0, inplace=False):
        t2 = self.t - t0
        if inplace:
            self.t = t2
        return t2


    def bin_func(self, Npts, func=np.mean, inplace=False):
        if Npts in (1, None):
            t2, x2 = self.t, self.x
        else:
            x2 = bin_func(self.x, dt=1/self.r, taumax=(Npts-1)/self.r)
            t2 = bin_func(self.t, dt=1/self.r, taumax=(Npts-1)/self.r)
            if inplace:
                self.t = t2
                self.x = x2
        return t2, x2


    def bin_average(self, Npts, inplace=False):
        return self.bin_func(Npts, inplace=inplace)


    def detrend(self, taumax=None, mode='constant', inplace=False):
        x2 = detrend(self.x, 1/self.r, taumax=taumax, mode=mode)
        if inplace:
            self.x = x2
        return self.t, x2


    def filter(self, cutoff, order=3, btype="lowpass", inplace=False):
        if cutoff in (None, "inf"):
            return self.t, self.x
        sos = butter(order, cutoff, fs=self.r, btype=btype, output="sos")
        x2 = sosfiltfilt(sos, self.x)
        if inplace:
            self.x = x2
        return self.t, x2


    def lowpass(self, cutoff, order=3, inplace=False):
        return self.filter(cutoff=cutoff, order=order, btype="lowpass", inplace=inplace)


    def time_gate(self, tmin=None, tmax=None, inplace=False):
        if tmin is None:
            tmin =self.t[0]
        if tmax is None:
            tmax = self.t[-1]
        mask = np.logical_and(self.t>=tmin, self.t<=tmax)
        x2 = self.x[mask]
        t2 = self.t[mask]
        if inplace:
            self.x = x2
            self.t = t2
        return t2, x2


    def correct(self, response, tmin=None, tmax=None,
        window="boxcar", differentiate=False, name=None):

        if name is None:
            name = self.name+" corrected"
        dt = 1/self.r
        t, sig = self.time_gate(tmin=tmin, tmax=tmax)
        signal_length = len(sig)
        freq = rfftfreq(signal_length, dt)[1:]
        resp = response(freq)
        resp = np.r_[1, resp]
        freq = np.r_[0, freq]
        win = get_window(window, signal_length)
        corr = np.sqrt(np.sum(win**2)/signal_length) #amp correction
        fft_vals = rfft(sig * win)
        if differentiate:
            fft_vals *= 1j * np.sin(2*np.pi*freq*dt)/dt
            #fft_vals *= 1j * (2*np.pi*freq)
        corrected_signal = irfft(fft_vals / corr / resp, n=signal_length)
        D = TimeSeries(corrected_signal, t, name=name)
        return D


    def shift(self, tau, inplace=False):
        t2 = self.t - tau
        if inplace:
            self.t = t2
        return t2, self.x


    def differentiate(self, inplace=False):
        t2, x2 = firstdiff((self.t, self.x))
        if inplace:
            self.x = x2
            self.t = t2
            self.name = "d_dt"+self.name
        return t2, x2


    def PSD(self, taumax=None, tmin=None, tmax=None, detrend="linear", window="hann", noverlap=None):
        """ Power spectral density """
        freq, psd, Navg = PSD(self.x, 1/self.r, taumax=taumax, tmin=tmin, tmax=tmax, detrend=detrend, window=window, noverlap=noverlap)
        self.freq = freq
        self.psd = psd
        self.Navg_psd = Navg
        return freq, psd, Navg


    def ACF(self, taumax=None, n_jobs=1):
        """ Autocorrelation function """
        tacf, acf, Navg = ACF(self.x, 1/self.r, taumax=taumax, n_jobs=n_jobs)
        self.tacf = tacf
        self.acf = acf
        self.Navg_acf = Navg
        return tacf, acf, Navg


    def MSD(self, taumax=None, n_jobs=1):
        """ Mean-squared displacement """
        tmsd, msd, Navg = MSD(self.x, 1/self.r, taumax=taumax, n_jobs=n_jobs)
        self.tmsd = tmsd
        self.msd = msd
        self.Navg_msd = Navg
        return tmsd, msd, Navg


    def AVAR(self, func=np.mean, octave=True, Nmin=20, base=2):
        """ Alan variance """
        tavar, avar, Navg = AVAR(self.x, 1/self.r, func=func, octave=octave, Nmin=Nmin, base=base)
        self.tavar = tavar
        self.avar = avar
        self.Navg_avar = Navg
        return tavar, avar, Navg


    def NVAR(self, func=np.mean, octave=True, base=2, Nmin=20):
        """ Normal variance """
        tnvar, nvar, Navg = NVAR(self.x, 1/self.r, func=func, octave=octave)
        self.tnvar = tnvar
        self.nvar = nvar
        self.Navg_nvar = Navg
        return tnvar, nvar, Navg


    def HIST(self, taumax=None, lb=None, ub=None, Nbins=45, density=True, remove_mean=False):
        bins, hist, Navg = HIST(self.x, 1/self.r, lb=lb, ub=ub, Nbins=Nbins, taumax=taumax, density=density, remove_mean=remove_mean)
        self.bins = bins
        self.hist = hist
        self.Navg_hist = Navg
        return bins, hist, Navg


    def plot(self, tmin=None, tmax=None, vshift=0, tshift=0,
            ax=None, figsize=(9,4), unit="V", tunit="s",
             return_data=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = plt. gcf()
        if type(unit) == str:
            unit = units[unit]
        elif type(unit) in (float, int):
            unit = {"value":unit, "label":""}
        if type(tunit) == str:
            tunit = units[tunit]
        elif type(tunit) in (float, int):
            tunit = {"value":tunit, "label":""}
        t, x = self.time_gate(tmin=tmin, tmax=tmax)
        ax.plot(tshift+t/tunit["value"], vshift+x/unit["value"], **kwargs)
        ax.set_ylabel(r"%s ($\rm %s$)" % (self.name, unit["label"]))
        ax.set_xlabel(r"Time ($\rm %s$)" % tunit['label'])
        if return_data:
            return fig, ax, t, x
        else:
            return fig, ax


class Collection:
    def __init__(self, timeseries_list):
        self.collection = timeseries_list
        r = self.collection[0].r
        size = self.collection[0].size
        same_r = np.all([D.r == r for D in self.collection])
        same_size = np.all([D.sixe == size for D in self.collection])
        assert same_r and same_size

    @property
    def Nrecords(self):
        return len(self.collection)

    @property
    def r(self):
        return self.collection[-1].r

    @property
    def size(self):
        return self.collection[-1].size

    @property
    def t(self):
        return self.collection[-1].t


    def average(self, method_str, n_jobs=1, **kwargs):
        def workload(timeseries):
            t, x, Navg, *dx = getattr(timeseries, method_str)(**kwargs)
            return t, x, Navg
        key = method_str.lower()
        res = Parallel(n_jobs=n_jobs)(delayed(workload)(C) for C in self.collection)
        t = res[0][0]
        Navg = res[0][2]
        avg = np.mean(np.array([r[1] for r in res]), axis=0)
        setattr(self, key, avg)
        setattr(self, tkey(key), t)
        setattr(self, "Navg_"+key, self.Nrecords*Navg)
        return t, avg


    def aggrigate(self, collection_slice=None):
        if collection_slice is None:
            collection_slice = slice(0, self.Nrecords, 1)
        if type(collection_slice) == int:
            collection_slice = slice(collection_slice, self.Nrecords, 1)
        agg = 0
        Cs = self.collection[collection_slice]
        for C in Cs:
            t, x = C()
            agg += x / len(Cs)
        agg = agg
        self.agg = TimeSeries((t, agg), name=f"{self.collection_name}_aggrigate")
        return t, agg


    def apply_func(self, func, n_jobs=1, **kwargs):
        def workload(timeseries):
            return func(timeseries.t, timeseries.x, **kwargs)
        Cs = Parallel(n_jobs=n_jobs)(delayed(workload)(C) for C in self.collection)
        self.collection = Cs


    def apply(self, method_str, n_jobs=1, recollect=False, **kwargs):
        def workload(timeseries):
            func = getattr(timeseries, method_str)(**kwargs)
            return func
        ret = Parallel(n_jobs=n_jobs)(delayed(workload)(C) for C in self.collection)
        if recollect:
            self.collection = ret

class CollectionTDMS_bak(Collection):
    def __init__(self, fname):
        tdms_file = TdmsFile(fname)["main"]
        try:
            t0 = tdms_file["t0"][:]
        except:
            t0 = tdms_file["Untitled"][:]
        t0 -= t0[0]
        t0 /= 1e6
        self.tdms_file = tdms_file
        self.t0 = t0
        self.colletion_name = "Collection channel is not set!"
        self.collection = []

    def __getattr__(self, attr):
        if attr[-1] == "s" and attr not in ("pos", "fos", "tdms"):
            Navailable = 1 + int(max([ch.split("_")[1] if len(ch.split("_"))>1 else 0 for
                                        ch in self.available_channels], key=int))
            return np.array([getattr(self, attr[:-1]+f"_{idx}") for idx in range(Navailable)])
        try:
            return self.tdms_file[attr]
        except KeyError:
            try:
                return self.params[attr]
            except KeyError:
                print(f"No property or channel '{attr}'")

    @property
    def params(self):
        return self.tdms_file.properties

    @property
    def available_channels(self):
        return [str(ch).split("/")[-1][1:-2] for ch in self.tdms_file.channels()]

    @property
    def channels(self):
        return sorted(list(set([ch.split("_")[0] for ch in self.available_channels])))

    def set_collection(self, name="x"):
        if name[-1].upper() in ("X", "Y"):
            r = self.params['r']
            name = name.upper()
        else:
            r = self.params['r2']
        Cs = [TimeSeries(C, r=r, name=name) for C in getattr(self, name+"s")]
        self.collection = Cs
        self.collection_name = name


class CollectionTDMS(Collection):
    def __init__(self, fname):
        self.fname = fname
        with TdmsFile.open(self.fname) as tdms_file:
            try:
                t0 = tdms_file["main"]["t0"][:]
            except:
                t0 = tdms_file["main"]["Untitled"][:]
            t0 -= t0[0]
            t0 /= 1e6
            self.params = tdms_file["main"].properties

            self.t0 = t0
            self.colletion_name = "Collection channel is not set!"
            self.collection = []

    def __getattr__(self, attr):
        try:
            return self.params[attr]
        except:
            with TdmsFile.open(self.fname) as tdms_file:
                return tdms_file["main"][attr][:]

    def set_collection(self, name="x", tmin=None, tmax=None):
        r = self.params['r']
        if tmin is None:
            lb = 0
        else:
            lb = int(tmin*r)
        if tmax is None:
            ub = -1
        else:
            ub = int(tmax*r)
        name = name.upper()
        with TdmsFile.open(self.fname) as tdms_file:
            available_channels = [str(ch).split("/")[-1][1:-2]
                for ch in tdms_file["main"].channels()]
            Navailable = 1 + int(max([ch.split("_")[1]
                        if len(ch.split("_"))>1 else 0 for
                        ch in available_channels], key=int))

            Cs = []
            for trial in range(Navailable):
                channel = tdms_file["main"][f"{name.upper()}_{trial}"]
                Cs.append(TimeSeries(channel[lb:ub], r=r, name=name))
        self.collection = Cs
        self.collection_name = name


class CollectionWeighing(Collection):
    def __init__(self, fnames, norm=1):
        self.fnames = fnames
        self.collection_name = "X"
        self.trials = []
        self.collection = []
        for fname in fnames:
            x, params = load_weighing(fname, norm=norm)
            r = params["r-Sps"]
            self.trials.append(params["trial"])
            self.collection.append(TimeSeries(x, r=r, name="X"))
        del params["trial"]
        self._params = params

    @property
    def Nrecords(self):
        return len(self.trials)

    def __getattr__(self, attr):
        try:
            return self.params[attr]
        except KeyError:
            print(f"No property or channel '{attr}'")

    @property
    def params(self):
        return self._params
