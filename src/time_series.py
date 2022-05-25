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
from scipy.signal import detrend, butter, sosfiltfilt
from joblib import Parallel, delayed
from brownian import (
    partition, bin_func, detrend, PSD, MSD, ACF, AVAR, NVAR, HIST, logbin_func
    )




# File I/O
# ========

def find_files(der):
    """Return the full path to all files in directory der."""
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(der):
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


def multipage(fname, figs=None, clf=True, dpi=300, clip=True, extra_artist=False):
    """Save multi-page PDFs"""
    pp = PdfPages(fname)
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        if clip is True:
            fig.savefig(
                pp, format="pdf", bbox_inches="tight", bbox_extra_artist=extra_artist
            )
        else:
            fig.savefig(pp, format="pdf", bbox_extra_artist=extra_artist)
        if clf == True:
            fig.clf()

    pp.close()


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
        self.x = cal*x2

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


    def differentiate(self, inplace=False):
        t2, x2 = firstdiff((self.t, self.x))
        if inplace:
            self.x = x2
            self.t = t2
            self.name = "d_dt"+self.name
        return t2, x2

    def PSD(self, taumax=None, detrend="linear", window="hann", noverlap=None):
        """ Power spectral density """
        freq, psd, Navg = PSD(self.x, 1/self.r, taumax=taumax, detrend=detrend, window=window, noverlap=noverlap)
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

    def AVAR(self, func=np.mean, octave=True):
        """ Alan variance """
        tavar, avar, Navg = AVAR(self.x, 1/self.r, func=func, octave=octave)
        self.tavar = tavar
        self.avar = avar
        self.Navg_avar = Navg
        return tavar, avar, Navg

    def NVAR(self, func=np.mean, octave=True):
        """ Normal variance """
        tnvar, nvar, Navg = NVAR(self.x, 1/self.r, func=func, octave=octave)
        self.tnvar = tnvar
        self.nvar = nvar
        self.Navg_nvar = Navg
        return tnvar, nvar, Navg

    def HIST(self, taumax=None, Nbins=45, density=True):
        bins, hist, Navg = HIST(self.x, 1/self.r, Nbins=Nbins, taumax=taumax, density=density)
        self.bins = bins
        self.hist = hist
        self.Navg_hist = Navg
        return bins, hist, Navg

    def plot(self, tmin=0, tmax=None, ax=None, figsize=(9,4), unit="nm", tunit="ms", **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = plt. gcf()
        if tmax is None:
            tmax = self.t[-1]
        if type(unit) == str:
            unit = units[unit]
        elif type(unit) in (float, int):
            unit = {"value":unit, "label":""}
        if type(tunit) == str:
            tunit = units[tunit]
        elif type(tunit) in (float, int):
            tunit = {"value":tunit, "label":""}

        mask = np.logical_and(self.t <= tmax, self.t >= tmin)
        ax.plot(self.t[mask]/tunit["value"], self.x[mask]/unit["value"], **kwargs)
        ax.set_ylabel(r"%s ($\rm %s$)" % (self.name, unit["label"]))
        ax.set_xlabel(r"Time ($\rm %s$)" % tunit['label'])
        return fig, ax


class Collection:
    def __init__(self, fname):
        tdms_file = TdmsFile(fname)["main"]
        t0 = tdms_file["t0"][:]
        t0 -= t0[0]
        t0 /= 1e6
        self.tdms_file = tdms_file
        self.t0 = t0
        self.colletion_name = "Collection channel is not set!"
        self.collection = []

    @property
    def Nrecords(self):
        return len(self.t0)

    def __getattr__(self, attr):
        if attr[-1] == "s" and attr != "pos":
            return np.array([getattr(self, attr[:-1]+f"_{idx}") for idx in range(self.Nrecords)])
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

    @property
    def r(self):
        return self.collection[-1].r

    @property
    def size(self):
        return self.collection[-1].size

    @property
    def t(self):
        return self.collection[-1].t

    def set_collection(self, name="x", lowpass=None, bin_average=None):
        if name[-1].upper() in ("X", "Y"):
            r = self.params['r']
            name = name.upper()
        else:
            r = self.params['r2']

        Cs = [TimeSeries(C, r=r, name=name) for C in getattr(self, name+"s")]
        self.collection = Cs
        self.collection_name = name


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


    def aggrigate(self, power=1):
        agg = 0
        for C in self.collection:
            t, x = C()
            agg += x**power / self.Nrecords
        agg = agg**(1/power)
        self.agg_power = power
        self.agg = TimeSeries((t, agg), name=f"{self.collection_name}_aggrigate")
        return t, agg


    def apply_func(self, func, n_jobs=1, **kwargs):
        def workload(timeseries):
            return func(timeseries.t, timeseries.x, **kwargs)
        Cs = Parallel(n_jobs=n_jobs)(delayed(workload)(C) for C in self.collection)
        self.collection = Cs

    def apply(self, method_str, n_jobs=1, **kwargs):
        def workload(timeseries):
            func = getattr(timeseries, method_str)(**kwargs)
        Parallel(n_jobs=n_jobs)(delayed(workload)(C) for C in self.collection)
