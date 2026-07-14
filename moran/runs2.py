import numpy as np
import os
from pathlib import Path
import sys
import numpy as np
from numba import njit , prange
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from joblib import Parallel, delayed
from scipy.stats import skew
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert
import pandas as pd
from tqdm import tqdm
import progressbar 
import os, json, time, gc, io, contextlib, math
from numba import set_num_threads, get_num_threads
from operator import mod
from itertools import product
from statsmodels.tsa.stattools import adfuller
from itertools import permutations
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
import glob
import imageio.v2 as imageio
from scipy.integrate import solve_ivp
from math import comb
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture

from contextlib import contextmanager
import joblib


@njit()
def weighted_choice(weights):
    cdf = np.cumsum(weights)
    r = np.random.rand() * cdf[-1]
    return np.searchsorted(cdf, r)

@njit
def hist_dynamic_minmax(x, nbins):
    # returns: counts (nbins,), edges (nbins+1,)
    xmin = np.min(x)
    xmax = np.max(x)

    counts = np.zeros(nbins, dtype=np.int64)
    edges  = np.empty(nbins + 1, dtype=np.float64)

    # Handle degenerate case
    if xmax == xmin:
        # Put all mass in the middle bin (or bin 0 if nbins==1)
        bw = 1.0
        start = xmin - 0.5
        for i in range(nbins + 1):
            edges[i] = start + i * bw
        mid = nbins // 2
        counts[mid] = x.size
        return counts, edges

    bw = (xmax - xmin) / nbins
    for i in range(nbins + 1):
        edges[i] = xmin + i * bw

    # Manual binning
    invbw = 1.0 / bw
    for i in range(x.size):
        b = int((x[i] - xmin) * invbw)
        if b < 0:
            b = 0
        elif b >= nbins:
            b = nbins - 1
        counts[b] += 1

    return counts, edges

def generate_trait_distribution_with_hump(n_individuals, mean=0.0, variance=1.0, hump_mass_fraction=0.15, hump_position_sigma=2.5, hump_width=0.5, seed=None):
    """
    Generate trait values with a Gaussian core + right-tail hump.
    
    Parameters
    ----------
    n_individuals : int
        Total number of individuals
    mean : float
        Mean of the main Gaussian distribution
    variance : float
        Variance of the main Gaussian distribution
    hump_mass_fraction : float
        Fraction of individuals in the hump (0 to 1)
    hump_position_sigma : float
        Position of hump center in units of std from mean
    hump_width : float
        Width (std) of the hump Gaussian
    seed : int or None
        Random seed for reproducibility
        
    Returns
    -------
    trait_values : ndarray
        Array of trait values with shape (n_individuals,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    std = np.sqrt(variance)
    
    # Number of individuals in each component
    n_hump = int(n_individuals * hump_mass_fraction)
    n_core = n_individuals - n_hump
    
    # Generate core Gaussian
    core_traits = np.random.normal(mean, std, n_core)
    
    # Generate hump (positioned in right tail)
    hump_center = mean + hump_position_sigma * std
    hump_traits = np.random.normal(hump_center, hump_width, n_hump)
    
    # Combine and shuffle
    trait_values = np.concatenate([core_traits, hump_traits])
    np.random.shuffle(trait_values)
    
    return trait_values


@njit()  #Main3D, 3rd Moment
def Quad_Sim_V00(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values):
    n_individuals = len(trait_values)
    n_out = len(indices)

    # Main arrays
    Main_3D = np.zeros((3, n_out))  # 0: mean, 1: skew, 2: std
    Moments = np.zeros(n_out)  # 0: mu1, 1: mu2, 2: mu3, 3: mu4

    k=0
    cum_mean_trait_value=0
    for t in range(1, tmax*n_individuals):
        tv2 = trait_values ** 2
        wb = 1 + b1_rate * trait_values + b2_rate * tv2
        wd = 1 - d1_rate * trait_values - d2_rate * tv2
        wb_eff = np.clip(wb, 0.0, np.inf)
        wd_eff = np.clip(wd, 0.0, np.inf)
        indice_birth = weighted_choice(wb_eff)
        indice_death = weighted_choice(wd_eff)
        birth_trait = trait_values[indice_birth] + np.random.normal(0, 1)
        trait_values[indice_death] = birth_trait
        current_mean = np.mean(trait_values)
        cum_mean_trait_value += current_mean
        trait_values -= current_mean

        if k < n_out and t == indices[k]:
  
            Moments[k] = np.mean(trait_values**3)

            Main_3D[0, k] = cum_mean_trait_value
            Main_3D[2, k] = np.std(trait_values)
            if Main_3D[2, k] == 0: Main_3D[1, k] = 0
            else: Main_3D[1, k] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * Main_3D[2, k]**3)

            k += 1

    return Main_3D, Moments

@njit()  #Main_3D, Moments, AVG_HIST
def Quad_Sim_V01(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values):
    n_individuals = len(trait_values)
    n_out = len(indices)
    Main_3D = np.zeros((3, n_out))
    Moments = np.zeros(n_out)
    dx = 0.01
    max_bins = 20000          # spans ±1000 units from origin
    origin = -max_bins * dx / 2 # = -1000.0  (bin 0 starts here)
    AVG_HIST = np.zeros(max_bins)

    k = 0
    cum_mean_trait_value = 0.0

    for t in range(1, tmax * n_individuals):
        tv2 = trait_values ** 2
        wb = 1 + b1_rate * trait_values + b2_rate * tv2
        wd = 1 - d1_rate * trait_values - d2_rate * tv2
        wb_eff = np.clip(wb, 0.0, np.inf)
        wd_eff = np.clip(wd, 0.0, np.inf)
        indice_birth = weighted_choice(wb_eff)
        indice_death = weighted_choice(wd_eff)
        birth_trait  = trait_values[indice_birth] + np.random.normal(0, 1)
        trait_values[indice_death] = birth_trait
        current_mean = np.mean(trait_values)
        cum_mean_trait_value += current_mean
        trait_values -= current_mean

        if k < n_out and t == indices[k]:
            Moments[k] = np.mean(trait_values ** 3)
            Main_3D[0, k] = cum_mean_trait_value
            Main_3D[2, k] = np.std(trait_values)
            if Main_3D[2, k] == 0:
                Main_3D[1, k] = 0.0
            else:
                Main_3D[1, k] = (np.sum((trait_values - np.mean(trait_values)) ** 3)
                                 / (n_individuals * Main_3D[2, k] ** 3))

            # --- accumulate into AVG_HIST ---
            for val in trait_values:
                bin_idx = int((val - origin) / dx)
                if 0 <= bin_idx < max_bins:
                    AVG_HIST[bin_idx] += 1.0

            k += 1

    AVG_HIST /= (n_out * n_individuals * dx)

    return Main_3D, Moments, AVG_HIST




def run_single_sim(n_individuals, b1_rate, d1_rate, tmax, indices, t_lag):
    Main_3D, Moments = Quad_Sim_V00(b1_rate, 0, d1_rate, 0, tmax, indices, np.zeros(n_individuals))
    stdl = Main_3D[2, :]
    mean_trait_values = Main_3D[0, :]

    Mdot = np.gradient(mean_trait_values, 1) * n_individuals / t_lag
    Vdot = np.gradient(stdl**2, 1) * n_individuals / t_lag
    Sdot = np.gradient(Moments[:], 1) * n_individuals / t_lag

    return mean_trait_values, Mdot, Vdot, Sdot



n_individuals = 20000
b1_rate = 0.5
d1_rate = 0.5
skip = 5
tmax = 100
t_lag = 200
indices = np.arange(skip*n_individuals, tmax*n_individuals, t_lag)
n_out = len(indices)


n_ensemble = 1 
n_jobs = 1


# /flash/DieckmannU/Abolfazl
save_dir=f"C:/Results/Fitnesswaves/4D_N({n_individuals})_b1({b1_rate})d1({d1_rate})_tmax({tmax})_n_ensemble({n_ensemble})"
os.makedirs(save_dir, exist_ok=True)


# results = Parallel(n_jobs=n_jobs)(delayed(run_single_sim)(n_individuals=n_individuals, b1_rate=b1_rate, d1_rate=d1_rate, tmax=tmax, indices=indices, t_lag=t_lag)    for i in range(n_ensemble))


@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

# usage
with tqdm_joblib(tqdm(total=n_ensemble)) as progress_bar:
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_sim)(n_individuals=n_individuals, b1_rate=b1_rate, d1_rate=d1_rate, tmax=tmax, indices=indices, t_lag=t_lag)
        for i in range(n_ensemble)
    )


M_all    = np.array([r[0] for r in results])
Mdot_all = np.array([r[1] for r in results])
Vdot_all = np.array([r[2] for r in results])
Sdot_all = np.array([r[3] for r in results])

np.save(f"{save_dir}/M.npy", M_all)
np.save(f"{save_dir}/Mdot.npy", Mdot_all)
np.save(f"{save_dir}/Vdot.npy", Vdot_all)
np.save(f"{save_dir}/Sdot.npy", Sdot_all)



