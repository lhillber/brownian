#!/usr/bin/env python
# coding: utf-8
# Compute measures on bownian motion time series data.
# By Logan Hillberry

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import stride_windows
from copy import copy
from scipy.integrate import simps
from scipy.optimize import curve_fit, minimize
from nptdms import TdmsFile
from scipy.signal import detrend
from joblib import Parallel, delayed
from brownian import partition, detrend, PSD, MSD, ACF, AVAR, NVAR, HIST

# Constants
# =========

PI = np.pi
kB = 1.382e-23


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

    def bin_average(self, Npts, inplace=False):
        if Npts in (1, None):
            t2, x2 = self.t, self.x
        else:
            x2 = np.mean(partition(self.x, dt=1/self.r, taumax=Npts/self.r), axis=1)
            t2 = np.mean(partition(self.t, dt=1/self.r, taumax=Npts/self.r), axis=1)
            if inplace:
                self.t = t2
                self.x = x2
        return t2, x2

    def detrend(self, taumax, mode='constant', inplace=False):
        x2 = detrend(self.x, 1/self.r, taumax=taumax, mode=mode)
        if inplace:
            self.x = x2
        return self.t, x2

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

    def plot(self, tmin=0, tmax=None, ax=None, figsize=(9,4), **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = plt. gcf()
        if tmax is None:
            tmax = self.t[-1]
        mask = np.logical_and(self.t <= tmax, self.t >= tmin)
        ax.plot(self.t[mask], self.x[mask], **kwargs)
        ax.set_ylabel(f"Coordinate {self.name}")
        ax.set_xlabel("Time (s)")
        return fig, ax


class Collection:
    def __init__(self, fname, r=None, coord="x", bin_average=1):
        TF = TdmsFile(fname)["main"]
        t0s = TF["t0"][:]
        t0s -= t0s[0]
        t0s /= 1e6
        self.t0s = t0s
        self.R = TF.properties["R"]
        self.Is = TF["I"][:]
        self.Vs = TF["V"][:]
        self.Ts = TF["T"][:]
        self.PDFs = TF["PDF"][:]
        self.PDBs = TF["PDB"][:]
        self.Nrecords = len(self.t0s)
        self.coord = coord
        r = TF.properties["r"]
        Cs = [TimeSeries(TF[f"{coord.upper().split('V')[-1]}_{idx}"], r=r, name=coord) for idx in range(self.Nrecords)]
        [C.bin_average(Npts=bin_average, inplace=True) for C in Cs]
        if coord[0].lower() == "v":
            Cs = [TimeSeries( firstdiff( C() ), name=coord) for C in Cs]
        self.collection = Cs

    @property
    def r(self):
        return self.collection[-1].r

    @property
    def size(self):
        return self.collection[-1].size

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

    def apply(self, method_str, n_jobs=1, **kwargs):
        def workload(timeseries):
            return getattr(timeseries, method_str)(**kwargs)
        Parallel(n_jobs=n_jobs)(delayed(workload)(C) for C in self.collection)
