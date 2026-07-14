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
import hints
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from pathlib import Path
from loguru import logger
import typer
import sys
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from scipy.stats import rayleigh, gamma, lognorm
from scipy.stats import ecdf

ಠ_ಠ = "hmmm..."

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


########################################################################################################################################

@njit()  #Main3D, 3rd Moment
def Quad_Sim_V00(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values):
    n_individuals = len(trait_values)
    n_out = len(indices)

    # Main arrays
    Main_3D = np.zeros((3, n_out))  # 0: mean, 1: skew, 2: std
    Moments = np.zeros(n_out)  # mu3

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

@njit()  #Main3D, Clipp, Moments, Moments_right_tail, Moments_left_tail
def Quad_Sim_V0(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values):
    n_individuals = len(trait_values)
    n_out = len(indices)

    # Main arrays
    Main_3D = np.zeros((3, n_out))  # 0: mean, 1: skew, 2: std
    Clipp = np.zeros((6, n_out)) # 0: birth_clipped_count, 1: death_clipped_count, 2: birth_clip_mass, 3: death_clip_mass, 4: wb eff mass, 5: wd eff mass
    Moments_right_tail = np.zeros((4, n_out)) 
    Moments_left_tail = np.zeros((4, n_out))
    Moments = np.zeros((4, n_out))  # 0: mu1, 1: mu2, 2: mu3, 3: mu4

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
            wb_neg_mass = np.sum(np.maximum(-wb, 0))
            wd_neg_mass = np.sum(np.maximum(-wd, 0))
            wb_eff_mass = np.sum(wb_eff)
            wd_eff_mass = np.sum(wd_eff)

            Clipp[0, k] = np.sum(wb < 0) / n_individuals
            Clipp[1, k] = np.sum(wd < 0) / n_individuals
            Clipp[2, k] = wb_neg_mass / (wb_eff_mass+wb_neg_mass)
            Clipp[3, k] = wd_neg_mass / (wd_eff_mass+wd_neg_mass)
            Clipp[4, k] = wb_eff_mass
            Clipp[5, k] = wd_eff_mass

            h = 0.5 * (np.max(trait_values) - np.min(trait_values)) / np.sqrt(n_individuals)
            if b1_rate != 0.0:
                tv_b_cut = np.where(trait_values < -1.0/b1_rate, trait_values, np.nan)  # left tail
            else:
                tv_b_cut = np.full_like(trait_values, np.nan)
            if d1_rate != 0.0:
                tv_d_cut = np.where(trait_values >  1.0/d1_rate, trait_values, np.nan)  # right tail
            else:
                tv_d_cut = np.full_like(trait_values, np.nan)

            Moments_right_tail[0, k] = np.nansum(tv_d_cut)
            Moments_right_tail[1, k] = np.nansum(tv_d_cut**2)
            Moments_right_tail[2, k] = np.nansum(tv_d_cut**3)
            Moments_right_tail[3, k] = np.nansum(tv_d_cut**4)

            Moments_left_tail[0, k] = np.nansum(tv_b_cut)
            Moments_left_tail[1, k] = np.nansum(tv_b_cut**2)
            Moments_left_tail[2, k] = np.nansum(tv_b_cut**3)
            Moments_left_tail[3, k] = np.nansum(tv_b_cut**4)

            Moments[0, k] = np.mean(trait_values**1)
            Moments[1, k] = np.mean(trait_values**2)
            Moments[2, k] = np.mean(trait_values**3)
            Moments[3, k] = np.mean(trait_values**4)

            Main_3D[0, k] = cum_mean_trait_value
            Main_3D[2, k] = np.std(trait_values)
            if Main_3D[2, k] == 0: Main_3D[1, k] = 0
            else: Main_3D[1, k] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * Main_3D[2, k]**3)

            k += 1

    return Main_3D, Clipp, Moments, Moments_right_tail, Moments_left_tail

@njit()  #Main3D, Clipp, Moments, Moments_right_tail, Moments_left_tail, Hist_counts, Hist_edges
def Quad_Sim_V1(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values, nbins=128):
    n_individuals = len(trait_values)
    n_out = len(indices)

    # Main arrays
    Main_3D = np.zeros((3, n_out))  # 0: mean, 1: skew, 2: std
    Clipp = np.zeros((6, n_out)) # 0: birth_clipped_count, 1: death_clipped_count, 2: birth_clip_mass, 3: death_clip_mass, 4: wb eff mass, 5: wd eff mass
    Moments_right_tail = np.zeros((4, n_out)) 
    Moments_left_tail = np.zeros((4, n_out))
    Moments = np.zeros((4, n_out))  # 0: mu1, 1: mu2, 2: mu3, 3: mu4

    # NEW: histogram storage
    Hist_counts = np.zeros((n_out, nbins), dtype=np.int64)
    Hist_edges  = np.zeros((n_out, nbins + 1), dtype=np.float64)

    k=0
    cum_mean_trait_value=0
    for t in range(1, tmax):
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
            wb_neg_mass = np.sum(np.maximum(-wb, 0))
            wd_neg_mass = np.sum(np.maximum(-wd, 0))
            wb_eff_mass = np.sum(wb_eff)
            wd_eff_mass = np.sum(wd_eff)

            Clipp[0, k] = np.sum(wb < 0) / n_individuals
            Clipp[1, k] = np.sum(wd < 0) / n_individuals
            Clipp[2, k] = wb_neg_mass / (wb_eff_mass+wb_neg_mass)
            Clipp[3, k] = wd_neg_mass / (wd_eff_mass+wd_neg_mass)
            Clipp[4, k] = wb_eff_mass
            Clipp[5, k] = wd_eff_mass


            h = 0.5 * (np.max(trait_values) - np.min(trait_values)) / np.sqrt(n_individuals)
            if b1_rate != 0.0:
                tv_b_cut = np.where(trait_values < -1.0/b1_rate, trait_values, np.nan)  # left tail
            else:
                tv_b_cut = np.full_like(trait_values, np.nan)
            if d1_rate != 0.0:
                tv_d_cut = np.where(trait_values >  1.0/d1_rate, trait_values, np.nan)  # right tail
            else:
                tv_d_cut = np.full_like(trait_values, np.nan)

            Moments_right_tail[0, k] = np.nansum(tv_d_cut)
            Moments_right_tail[1, k] = np.nansum(tv_d_cut**2)
            Moments_right_tail[2, k] = np.nansum(tv_d_cut**3)
            Moments_right_tail[3, k] = np.nansum(tv_d_cut**4)

            Moments_left_tail[0, k] = np.nansum(tv_b_cut)
            Moments_left_tail[1, k] = np.nansum(tv_b_cut**2)
            Moments_left_tail[2, k] = np.nansum(tv_b_cut**3)
            Moments_left_tail[3, k] = np.nansum(tv_b_cut**4)

            Moments[0, k] = np.mean(trait_values**1)
            Moments[1, k] = np.mean(trait_values**2)
            Moments[2, k] = np.mean(trait_values**3)
            Moments[3, k] = np.mean(trait_values**4)

            Main_3D[0, k] = cum_mean_trait_value
            Main_3D[2, k] = np.std(trait_values)
            if Main_3D[2, k] == 0: Main_3D[1, k] = 0
            else: Main_3D[1, k] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * Main_3D[2, k]**3)

            # NEW: store dynamic histogram at this output time
            c, e = hist_dynamic_minmax(trait_values, nbins)
            Hist_counts[k, :] = c
            Hist_edges[k, :]  = e

            k += 1

    return Main_3D, Clipp, Moments, Moments_right_tail, Moments_left_tail, Hist_counts, Hist_edges

@njit() # All_tv, Main_3D, Clipp, Moments, Moments_right_tail, Moments_left_tail, Hist_counts, Hist_edges
def Quad_Sim_V2(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values, nbins=128):
    n_individuals = len(trait_values)
    n_out = len(indices)

    All_tv = np.zeros((n_out, n_individuals))
    Main_3D = np.zeros((3, n_out))  # 0: mean, 1: skew, 2: std
    Clipp = np.zeros((6, n_out)) # 0: birth_clipped_count, 1: death_clipped_count, 2: birth_clip_mass, 3: death_clip_mass, 4: wb eff mass, 5: wd eff mass
    Moments_right_tail = np.zeros((4, n_out)) 
    Moments_left_tail = np.zeros((4, n_out))
    Moments = np.zeros((4, n_out))  # 0: mu1, 1: mu2, 2: mu3, 3: mu4
 

    # NEW: histogram storage
    Hist_counts = np.zeros((n_out, nbins), dtype=np.int64)
    Hist_edges  = np.zeros((n_out, nbins + 1), dtype=np.float64)

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
            wb_neg_mass = np.sum(np.maximum(-wb, 0))
            wd_neg_mass = np.sum(np.maximum(-wd, 0))
            wb_eff_mass = np.sum(wb_eff)
            wd_eff_mass = np.sum(wd_eff)

            Clipp[0, k] = np.sum(wb < 0) / n_individuals
            Clipp[1, k] = np.sum(wd < 0) / n_individuals
            Clipp[2, k] = wb_neg_mass / (wb_eff_mass+wb_neg_mass)
            Clipp[3, k] = wd_neg_mass / (wd_eff_mass+wd_neg_mass)
            Clipp[4, k] = wb_eff_mass
            Clipp[5, k] = wd_eff_mass


            h = 0.5 * (np.max(trait_values) - np.min(trait_values)) / np.sqrt(n_individuals)
            if b1_rate != 0.0:
                tv_b_cut = np.where(trait_values < -1.0/b1_rate, trait_values, np.nan)  # left tail
            else:
                tv_b_cut = np.full_like(trait_values, np.nan)
            if d1_rate != 0.0:
                tv_d_cut = np.where(trait_values >  1.0/d1_rate, trait_values, np.nan)  # right tail
            else:
                tv_d_cut = np.full_like(trait_values, np.nan)

            Moments_right_tail[0, k] = np.nansum(tv_d_cut)
            Moments_right_tail[1, k] = np.nansum(tv_d_cut**2)
            Moments_right_tail[2, k] = np.nansum(tv_d_cut**3)
            Moments_right_tail[3, k] = np.nansum(tv_d_cut**4)

            Moments_left_tail[0, k] = np.nansum(tv_b_cut)
            Moments_left_tail[1, k] = np.nansum(tv_b_cut**2)
            Moments_left_tail[2, k] = np.nansum(tv_b_cut**3)
            Moments_left_tail[3, k] = np.nansum(tv_b_cut**4)

            Moments[0, k] = np.mean(trait_values**1)
            Moments[1, k] = np.mean(trait_values**2)
            Moments[2, k] = np.mean(trait_values**3)
            Moments[3, k] = np.mean(trait_values**4)

            Main_3D[0, k] = cum_mean_trait_value
            Main_3D[2, k] = np.std(trait_values)
            if Main_3D[2, k] == 0: Main_3D[1, k] = 0
            else: Main_3D[1, k] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * Main_3D[2, k]**3)

            # NEW: store dynamic histogram at this output time
            c, e = hist_dynamic_minmax(trait_values, nbins)
            Hist_counts[k, :] = c
            Hist_edges[k, :]  = e

            All_tv[k, :] = trait_values
            k += 1

    return All_tv, Main_3D, Clipp, Moments, Moments_right_tail, Moments_left_tail, Hist_counts, Hist_edges

def Metadata_Quad_Sim_V2(nlist, b1list, b2list, d1list, d2list, tmax, skip, t_lag, save_dir, nbins, nansa=10, n_jobs=6):
    combinations = list(product(*[nlist] + [b1list] + [b2list] + [d1list] + [d2list]))
    n_combos = len(combinations)

    def process_one_combo(i):   
        params = combinations[i]
        n_individuals = params[0];         b1_rate = params[1];        b2_rate = params[2];        d1_rate = params[3];        d2_rate = params[4]
        indices = np.arange(skip*n_individuals, tmax*n_individuals, t_lag) 
        n_out = len(indices)

        All_tv, Main_3D, Clipp, Moments, Moments_right_tail, Moments_left_tail, Hist_counts, Hist_edges = Quad_Sim_V2(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, np.zeros(n_individuals), nbins=nbins)

        skwl = Main_3D[1, :]
        stdl = Main_3D[2, :]
        mean_trait_values = Main_3D[0, :]
        ALL_b_clip_count = Clipp[0, :]
        ALL_d_clip_count = Clipp[1, :]
        ALL_b_clip_mass  = Clipp[2, :]
        ALL_d_clip_mass  = Clipp[3, :]
        ALL_wb_eff_mass  = Clipp[4, :]
        ALL_wd_eff_mass  = Clipp[5, :]
        ALL_mu1 = Moments[0, :]
        ALL_mu2 = Moments[1, :]
        ALL_mu3 = Moments[2, :]
        ALL_mu4 = Moments[3, :]
        ALL_moments_right_tail_1 = Moments_right_tail[0, :]
        ALL_moments_right_tail_2 = Moments_right_tail[1, :]
        ALL_moments_right_tail_3 = Moments_right_tail[2, :]
        ALL_moments_right_tail_4 = Moments_right_tail[3, :]
        ALL_moments_left_tail_1 = Moments_left_tail[0, :]
        ALL_moments_left_tail_2 = Moments_left_tail[1, :]
        ALL_moments_left_tail_3 = Moments_left_tail[2, :]
        ALL_moments_left_tail_4 = Moments_left_tail[3, :]

        varl = stdl**2
        Amp_2D = np.sqrt(skwl**2 + varl)
        Phase_2D = np.arctan2(skwl - np.mean(skwl), stdl - np.mean(stdl))
        Freq_2D = np.diff(np.unwrap(Phase_2D))
        Freq_2D = np.concatenate([Freq_2D[:1], Freq_2D])/(t_lag)*n_individuals

        std_Hilbert = hilbert(stdl)
        Amp_Hil_std = np.abs(std_Hilbert)
        std_Hilbert = hilbert(stdl - np.mean(stdl))
        Phase_Hil_std = np.angle(std_Hilbert)
        Freq_Hil_std = np.diff(np.unwrap(Phase_Hil_std))
        Freq_Hil_std = np.concatenate([Freq_Hil_std[:1], Freq_Hil_std])/(t_lag)*n_individuals

        skw_Hilbert = hilbert(skwl)
        Amp_Hil_skw = np.abs(skw_Hilbert)
        skw_Hilbert = hilbert(skwl - np.mean(skwl))
        Phase_Hil_skw = np.angle(skw_Hilbert)
        Freq_Hil_skw = np.diff(np.unwrap(Phase_Hil_skw))
        Freq_Hil_skw = np.concatenate([Freq_Hil_skw[:1], Freq_Hil_skw])/(t_lag)*n_individuals

        # Mean_ts = mean_trait_values - mean_trait_values[0]
        # dt = 1
        # T = Mean_ts.shape[1]
        # ts = np.arange(T) * t_lag

        # Mean_Slope  = np.zeros(nansa)
        # Mean_Intercept = np.zeros(nansa)

        rtspeed = np.gradient(mean_trait_values)*n_individuals/(t_lag)
        Eff_slope = b1_rate + d1_rate - ALL_b_clip_count*b1_rate - ALL_d_clip_count*d1_rate
        NEW_b1_rate = b1_rate * n_individuals / ALL_wb_eff_mass
        NEW_d1_rate = d1_rate * n_individuals / ALL_wd_eff_mass
        NEW_Eff_slope = NEW_b1_rate + NEW_d1_rate - ALL_b_clip_count*NEW_b1_rate - ALL_d_clip_count*NEW_d1_rate


        Mdot = np.gradient(mean_trait_values, 1) * n_individuals / (t_lag)
        Vdot = np.gradient(stdl**2, 1) * n_individuals / (t_lag)
        Sdot = np.gradient(ALL_mu3, 1) * n_individuals / (t_lag) 

        Nb = n_individuals*(1-ALL_b_clip_count)
        Nd = n_individuals*(1-ALL_d_clip_count)
        mub = (ALL_mu1*n_individuals - ALL_moments_left_tail_1)/Nb
        mud = (ALL_mu1*n_individuals - ALL_moments_right_tail_1)/Nd
        beff = b1_rate/(1+b1_rate*mub)
        deff = d1_rate/(1-d1_rate*mud)
        Vb = (ALL_mu2*n_individuals - ALL_moments_left_tail_2)/Nb - mub**2
        Vd = (ALL_mu2*n_individuals - ALL_moments_right_tail_2)/Nd - mud**2
        Db3 = (ALL_mu3*n_individuals - ALL_moments_left_tail_3)
        Dd3 =  (ALL_mu3*n_individuals - ALL_moments_right_tail_3)
        Db4 = (ALL_mu4*n_individuals - ALL_moments_left_tail_4)
        Dd4 =  (ALL_mu4*n_individuals - ALL_moments_right_tail_4)
        Sb = Db3/Nb - mub**3 - 3*mub*Vb
        Sd = Dd3/Nd - mud**3 - 3*mud*Vd
        Kb = Db4/Nb - mub**4 -6*mub**2*Vb - 4*mub*Sb
        Kd = Dd4/Nd - mud**4 -6*mud**2*Vd - 4*mud*Sd

        M_dot = (mub - mud)  + beff*Vb + deff*Vd
        V_dot = (Vb-Vd) + (mub**2-mud**2)+ beff*(2*mub*Vb+Sb) + deff*(2*mud*Vd+Sd)+1
        S_dot = ((Sb-Sd) + 3*(mub*Vb-mud*Vd) + (mub**3-mud**3) + beff*(Kb + 3*(mub**2*Vb+mub*Sb)) + deff*(Kd + 3*(mud**2*Vd+mud*Sd)) + (mub+beff*Vb)) - 3*M_dot*ALL_mu2 - 3*ALL_mu1*V_dot - 3*ALL_mu1**2*M_dot
        
        # for i in range(nansa):
        #     Mean_Slope[i], Mean_Intercept[i] = np.polyfit(ts, Mean_ts[i], 1)



        # MVBD = np.nanmean(rtspeed / (stdl**2 * ( ( b1_rate + 2*b2_rate*mean_trait_values ) + ( d1_rate + 2*d2_rate*mean_trait_values ) )), axis=1)
        if b1_rate + d1_rate == 0 : MVBD_1_ar = MVBD_2_ar = MVBD_4_ar = MVBD_5_ar = MVBD_7_ar = np.nan
        else: 
            MVBD_1_ar = rtspeed / (varl * ( b1_rate + d1_rate ))
            MVBD_2_ar = rtspeed / (varl * ( Eff_slope ))
            MVBD_4_ar = rtspeed / (varl * ( NEW_b1_rate + NEW_d1_rate ))
            MVBD_5_ar = rtspeed / (varl * ( NEW_Eff_slope ))
            MVBD_7_ar = rtspeed / (M_dot)



        bimodal_flag = np.zeros(n_out, dtype=bool)
        hump_left = np.full(n_out, np.nan)
        hump_right = np.full(n_out, np.nan)
        hump_weight_left = np.full(n_out, np.nan)
        hump_weight_right = np.full(n_out, np.nan)

        for k in range(0, n_out):
            trait_values = All_tv[k, :].reshape(-1,1)
            gmm1 = GaussianMixture(n_components=1)
            gmm1.fit(trait_values)
            bic1 = gmm1.bic(trait_values)
            gmm2 = GaussianMixture(n_components=2)
            gmm2.fit(trait_values)
            bic2 = gmm2.bic(trait_values)
            mean2 = gmm2.means_.flatten()
            std2 = np.sqrt(gmm2.covariances_.flatten())
            weights2 = gmm2.weights_.flatten()
            sep = abs(mean2[0] - mean2[1]) / np.sqrt(std2[0]**2 + std2[1]**2)
            x_range = np.linspace(Hist_edges[k, 0], Hist_edges[k, -1], 128).reshape(-1,1)
            if bic2 < bic1 and sep > 1.5:
                bimodal_flag[k] = True
                # sort humps left/right
                order = np.argsort(mean2)
                hump_left[k] = mean2[order[0]]
                hump_right[k] = mean2[order[1]]
                hump_weight_left[k] = weights2[order[0]]
                hump_weight_right[k] = weights2[order[1]]

                # logprob = gmm2.score_samples(x_range)
                # pdf_hump = np.exp(logprob)
            # else:
            #     logprob = gmm1.score_samples(x_range)
            #     pdf_hump = np.exp(logprob)

        # Detect hump nucleation events
        hump_events = np.where((~bimodal_flag[:-1]) & (bimodal_flag[1:]))[0] + 1 

        # Waiting times between hump events
        if len(hump_events) > 1:
            waiting_times = np.diff(indices[hump_events])
            mean_waiting = np.mean(waiting_times)
            std_waiting = np.std(waiting_times)
        else:
            mean_waiting = np.nan
            std_waiting = np.nan

        # Where humps nucleate
        nucleation_left = hump_left[hump_events]
        nucleation_right = hump_right[hump_events]
        hump_distance = hump_right[hump_events] - hump_left[hump_events]

        # Statistics
        mean_nucl_left = np.nanmean(nucleation_left)
        mean_nucl_right = np.nanmean(nucleation_right)
        std_nucl_left = np.nanstd(nucleation_left)
        std_nucl_right = np.nanstd(nucleation_right)
        mean_hump_distance = np.nanmean(hump_distance)

        metadata = {
            "N": int(n_individuals),
            "b1": float(b1_rate),
            "b2": float(b2_rate),
            "d1": float(d1_rate),
            "d2": float(d2_rate)
        }
        metadata["Avg_bclipped"] = np.mean(ALL_b_clip_count)
        metadata["Avg_dclipped"] = np.mean(ALL_d_clip_count)
        metadata["Avg_bdclipped"] = np.mean(ALL_b_clip_count + ALL_d_clip_count)

        metadata["Avg_b_clip_mass"] = np.mean(ALL_b_clip_mass)
        metadata["Avg_d_clip_mass"] = np.mean(ALL_d_clip_mass)
        metadata["Avg_bd_clip_mass"] = np.mean(ALL_b_clip_mass + ALL_d_clip_mass)

        metadata["Avg_Skw"] = np.mean(skwl)
        metadata["std_Skw"] = np.std(skwl)

        metadata["Avg_Std"] = np.mean(stdl)
        metadata["std_Std"] = np.std(stdl)

        metadata["Avg_Var"] = np.mean(stdl**2)
        metadata["std_Var"] = np.std(stdl**2)

        metadata["Avg_Amp_2D"] = np.mean(Amp_2D)
        metadata["Avg_Phase_2D"] = np.mean(Phase_2D)
        metadata["Avg_Freq_2D"] = np.mean(Freq_2D)
        metadata["Std_Freq_2D"] = np.std(Freq_2D)
        metadata["Avg_Abs_Freq_2D"] = np.mean(np.abs(Freq_2D))
        
        metadata["Avg_Amp_Hil_std"] = np.mean(Amp_Hil_std)
        metadata["Avg_Phase_Hil_std"] = np.mean(Phase_Hil_std)
        metadata["Avg_Freq_Hil_std"] = np.mean(Freq_Hil_std)
        metadata["Avg_Abs_Freq_Hil_std"] = np.mean(np.abs(Freq_Hil_std))
        metadata["Std_Freq_Hil_std"] = np.std(Freq_Hil_std)

        metadata["Avg_Amp_Hil_skw"] = np.mean(Amp_Hil_skw)
        metadata["Avg_Phase_Hil_skw"] = np.mean(Phase_Hil_skw)
        metadata["Avg_Freq_Hil_skw"] =np.mean(Freq_Hil_skw)
        metadata["Avg_Abs_Freq_Hil_skw"] = np.mean(np.abs(Freq_Hil_skw))
        metadata["Std_Freq_Hil_skw"] = np.std(Freq_Hil_skw)

        # metadata["final_mean_trait"] = mean_trait_values[:, -1]
        # metadata["Mean_Slope"] = np.mean(Mean_Slope) * n_individuals
        # metadata["Mean_Intercept"] = np.mean(Mean_Intercept) * n_individuals
        metadata["Avg_M_dot_rt"] = np.mean(rtspeed)
        metadata["Std_M_dot_rt"] = np.std(rtspeed)



        metadata["MVBD_1_ar"] = np.nanmean(MVBD_1_ar)
        metadata["MVBD_1_ra"] = metadata["Avg_M_dot_rt"] / (metadata["Avg_Var"] * ( b1_rate + d1_rate ) )

        metadata["MVBD_2_ar"] = np.nanmean(MVBD_2_ar)
        metadata["MVBD_2_ra"] = metadata["Avg_M_dot_rt"] / (metadata["Avg_Var"] * np.mean(Eff_slope) )

        metadata["MVBD_4_ar"] = np.nanmean(MVBD_4_ar)
        metadata["MVBD_4_ra"] = metadata["Avg_M_dot_rt"] / (metadata["Avg_Var"] * np.mean(( NEW_b1_rate + NEW_d1_rate ) ) )

        metadata["MVBD_5_ar"] = np.nanmean(MVBD_5_ar)
        metadata["MVBD_5_ra"] = metadata["Avg_M_dot_rt"] / (metadata["Avg_Var"] * np.mean(NEW_Eff_slope) )

        metadata["MVBD_7_ar"] = np.nanmean(MVBD_7_ar)
        metadata["MVBD_7_ra"] = metadata["Avg_M_dot_rt"] / (np.mean(M_dot ) )

        metadata["Vdot_ar"] = np.mean(Vdot/V_dot)
        metadata["Vdot_ra"] = np.mean(Vdot) / np.mean(V_dot)

        metadata["Sdot_ar"] = np.mean(Sdot/S_dot)
        metadata["Sdot_ra"] = np.mean(Sdot) / np.mean(S_dot)

        metadata["Num_hump_events"] = len(hump_events)

        metadata["Mean_waiting_time"] = mean_waiting
        metadata["Std_waiting_time"] = std_waiting

        metadata["Mean_nucleation_left"] = mean_nucl_left
        metadata["Mean_nucleation_right"] = mean_nucl_right

        metadata["Std_nucleation_left"] = std_nucl_left
        metadata["Std_nucleation_right"] = std_nucl_right

        metadata["Mean_hump_distance"] = mean_hump_distance
        

        return metadata

    results = Parallel(n_jobs=n_jobs, verbose=0)( delayed(process_one_combo)(i) for i in tqdm(range(n_combos), total=len(combinations), desc="Simulating", ncols=100) )    # (n_jobs=n_jobs, backend='loky') , backend="threading"
    summary_path = os.path.join(save_dir, f"{t_lag}_ALL_summaries.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n✅ All summaries saved to {summary_path}")

def GMM(b1_rate, d1_rate, TV, Hist_edges, t_lag, sep_threshold=1.5, xic="bic"):  

    n_out = TV.shape[0]
    Nb1 = np.zeros(n_out)
    Nb2 = np.zeros(n_out)
    Nd1 = np.zeros(n_out)
    Nd2 = np.zeros(n_out)
    Db1_1 = np.zeros(n_out)
    Db1_2 = np.zeros(n_out)
    Dd1_1 = np.zeros(n_out)
    Dd1_2 = np.zeros(n_out)
    Db2_1 = np.zeros(n_out)
    Db2_2 = np.zeros(n_out)
    Dd2_1 = np.zeros(n_out)
    Dd2_2 = np.zeros(n_out)
    Db3_1 = np.zeros(n_out)
    Db3_2 = np.zeros(n_out)
    Dd3_1 = np.zeros(n_out)
    Dd3_2 = np.zeros(n_out)


    bimodal_flag = np.zeros(n_out, dtype=bool)
    hump_left = np.full(n_out, np.nan)
    hump_right = np.full(n_out, np.nan)
    hump_weight_left = np.full(n_out, np.nan)
    hump_weight_right = np.full(n_out, np.nan)
    hump_std_left = np.full(n_out, np.nan)
    hump_std_right = np.full(n_out, np.nan)

    for k in range(0, n_out):
        trait_values = TV[k, :].reshape(-1,1)
        gmm1 = GaussianMixture(n_components=1)
        gmm1.fit(trait_values)
        xic1 = getattr(gmm1, xic)(trait_values)
        gmm2 = GaussianMixture(n_components=2)
        gmm2.fit(trait_values)
        xic2 = getattr(gmm2, xic)(trait_values)
        mean2 = gmm2.means_.flatten()
        std2 = np.sqrt(gmm2.covariances_.flatten())
        weights2 = gmm2.weights_.flatten()
        sep = abs(mean2[0] - mean2[1]) / np.sqrt(std2[0]**2 + std2[1]**2)
        x_range = np.linspace(Hist_edges[k, 0], Hist_edges[k, -1], 128).reshape(-1,1)
        resp = gmm2.predict_proba(trait_values)
        labels = np.argmax(resp, axis=1)
        if xic2 < xic1 and sep > sep_threshold:
            bimodal_flag[k] = True
            # sort humps left/right
            order = np.argsort(mean2)
            hump_left[k] = mean2[order[0]]
            hump_right[k] = mean2[order[1]]
            hump_weight_left[k] = weights2[order[0]]
            hump_weight_right[k] = weights2[order[1]]
            hump_std_left[k] = std2[order[0]]
            hump_std_right[k] = std2[order[1]]
            order = np.argsort(mean2)
            left_comp = order[0]
            right_comp = order[1]
            labels = np.where(labels == left_comp, 1, 2)
            # logprob = gmm2.score_samples(x_range)
            # pdf_hump = np.exp(logprob)
        # else:
        #     logprob = gmm1.score_samples(x_range)
        #     pdf_hump = np.exp(logprob)
        x = trait_values.flatten()
        birth_adm = (1 + b1_rate * x) > 0
        death_adm = (1 - d1_rate * x) > 0
        m1 = labels == 1
        m2 = labels == 2

        Nb1[k] = np.sum(m1 & birth_adm)
        Nb2[k] = np.sum(m2 & birth_adm)

        Nd1[k] = np.sum(m1 & death_adm)
        Nd2[k] = np.sum(m2 & death_adm)

        Db1_1[k] = np.sum(x[m1 & birth_adm])
        Db1_2[k] = np.sum(x[m2 & birth_adm])
        Dd1_1[k] = np.sum(x[m1 & death_adm])
        Dd1_2[k] = np.sum(x[m2 & death_adm])

        Db2_1[k] = np.sum(x[m1 & birth_adm]**2)
        Db2_2[k] = np.sum(x[m2 & birth_adm]**2)
        Dd2_1[k] = np.sum(x[m1 & death_adm]**2)
        Dd2_2[k] = np.sum(x[m2 & death_adm]**2)

        Db3_1[k] = np.sum(x[m1 & birth_adm]**3)
        Db3_2[k] = np.sum(x[m2 & birth_adm]**3)
        Dd3_1[k] = np.sum(x[m1 & death_adm]**3)
        Dd3_2[k] = np.sum(x[m2 & death_adm]**3)

    results = {
        "num_bimodal": np.sum(bimodal_flag),
        "nb1": Nb1,
        "nb2": Nb2,
        "nd1": Nd1,
        "nd2": Nd2,
        "Db1_1": Db1_1,
        "Db1_2": Db1_2,
        "Dd1_1": Dd1_1,
        "Dd1_2": Dd1_2,
        "Db2_1": Db2_1,
        "Db2_2": Db2_2,
        "Dd2_1": Dd2_1,
        "Dd2_2": Dd2_2,
        "Db3_1": Db3_1,
        "Db3_2": Db3_2,
        "Dd3_1": Dd3_1,
        "Dd3_2": Dd3_2,
        "bimodal_flag": bimodal_flag,
        "hump_left": hump_left,
        "hump_right": hump_right,
        "hump_weight_left": hump_weight_left,
        "hump_weight_right": hump_weight_right,
        "hump_std_left": hump_std_left,
        "hump_std_right": hump_std_right
    }

    return results



########################################################################################################################################
# Zero Skew Runs
import numpy as np
from numba import njit

ZERO_NONE = 0
ZERO_PAIR_RANDOM = 1
ZERO_PAIR_TAILS = 2
ZERO_GAUSSIAN_PAIRS = 3
ZERO_CUBIC = 4
ZERO_LOCAL_TRIPLE = 5


@njit
def _mean(x):
    s = 0.0
    for i in range(x.size):
        s += x[i]
    return s / x.size


@njit
def _center_inplace(x):
    m = _mean(x)
    for i in range(x.size):
        x[i] -= m


@njit
def _var_about_zero(x):
    s = 0.0
    for i in range(x.size):
        s += x[i] * x[i]
    return s / x.size


@njit
def _third_about_mean(x):
    m = _mean(x)
    s = 0.0
    for i in range(x.size):
        z = x[i] - m
        s += z * z * z
    return s / x.size


@njit
def _enforce_mean_var_inplace(x, target_var):
    """
    Force mean 0 and variance target_var.
    Scaling does not change skewness sign or zero-skewness.
    """
    _center_inplace(x)

    if target_var <= 0.0:
        for i in range(x.size):
            x[i] = 0.0
        return

    v = _var_about_zero(x)

    if v <= 1e-300:
        n = x.size
        for i in range(n):
            x[i] = 0.0

        m = n // 2
        for j in range(m):
            x[2 * j] = 1.0
            x[2 * j + 1] = -1.0

        v = _var_about_zero(x)

    scale = np.sqrt(target_var / v)

    for i in range(x.size):
        x[i] *= scale

    _center_inplace(x)


@njit
def _shuffle_inplace(x):
    n = x.size
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = x[i]
        x[i] = x[j]
        x[j] = tmp

# no
@njit
def _pair_from_current_random_inplace(x, target_var):
    """
    Method 1:
    Use random magnitudes from the current population.
    Then create exact mirrored pairs: +a, -a.
    """
    n = x.size

    if n < 2 or target_var <= 0.0:
        _enforce_mean_var_inplace(x, target_var)
        return

    m = n // 2

    mags = np.empty(n)
    for i in range(n):
        mags[i] = abs(x[i])

    _shuffle_inplace(mags)

    for i in range(n):
        x[i] = 0.0

    for j in range(m):
        x[2 * j] = mags[j]
        x[2 * j + 1] = -mags[j]

    _enforce_mean_var_inplace(x, target_var)
    _shuffle_inplace(x)


@njit
def _pair_from_current_tails_inplace(x, target_var):
    """
    Method 2:
    Use the largest current magnitudes.
    Then create exact mirrored pairs.
    This keeps the scale of the tails better than random pairing.
    """
    n = x.size

    if n < 2 or target_var <= 0.0:
        _enforce_mean_var_inplace(x, target_var)
        return

    m = n // 2

    absx = np.empty(n)
    for i in range(n):
        absx[i] = abs(x[i])

    order = np.argsort(absx)  # ascending

    mags = np.empty(m)
    for j in range(m):
        mags[j] = absx[order[n - 1 - j]]

    for i in range(n):
        x[i] = 0.0

    for j in range(m):
        x[2 * j] = mags[j]
        x[2 * j + 1] = -mags[j]

    _enforce_mean_var_inplace(x, target_var)
    _shuffle_inplace(x)


@njit
def _gaussian_pairs_inplace(x, target_var):
    """
    Method 3:
    Replace the population by paired Gaussian values: +z, -z.
    This is closest to generating a new symmetric Gaussian sample.
    Mean = 0, skew = 0 exactly by construction.
    Variance is then rescaled to the old variance.
    """
    n = x.size

    if n < 2 or target_var <= 0.0:
        _enforce_mean_var_inplace(x, target_var)
        return

    m = n // 2

    for i in range(n):
        x[i] = 0.0

    for j in range(m):
        z = np.random.normal(0.0, 1.0)
        x[2 * j] = z
        x[2 * j + 1] = -z

    _enforce_mean_var_inplace(x, target_var)
    _shuffle_inplace(x)


@njit
def _cubic_m3_raw(x, v, a):
    """
    Third moment of y = x + a * (x^2 - v).
    If x has mean 0 and v = mean(x^2), then mean(y) is also 0.
    """
    s = 0.0

    for i in range(x.size):
        q = x[i] * x[i] - v
        y = x[i] + a * q
        s += y * y * y

    return s / x.size


@njit
def _cubic_correction_inplace(x, target_var):
    """
    Method 4:
    Smoothly transform every individual:

        y_i = x_i + a * (x_i^2 - Var)

    Because mean(x_i^2 - Var) = 0, this preserves mean 0 before rescaling.
    We choose a so that the third moment becomes zero.
    Then we rescale to preserve the old variance.

    This is usually the most natural method if you want a gentle deformation.
    """
    n = x.size

    if n < 3 or target_var <= 0.0:
        _enforce_mean_var_inplace(x, target_var)
        return False

    _enforce_mean_var_inplace(x, target_var)

    f0 = _cubic_m3_raw(x, target_var, 0.0)

    if abs(f0) <= 1e-14 * (target_var ** 1.5 + 1.0):
        return True

    scale = np.sqrt(target_var) + 1e-15
    A = 1.0 / scale

    lo = -A
    hi = A

    flo = _cubic_m3_raw(x, target_var, lo)
    fhi = _cubic_m3_raw(x, target_var, hi)

    found = False

    for _ in range(60):
        if flo == 0.0:
            hi = lo
            found = True
            break

        if fhi == 0.0:
            lo = hi
            found = True
            break

        if flo * fhi < 0.0:
            found = True
            break

        A *= 2.0
        lo = -A
        hi = A

        flo = _cubic_m3_raw(x, target_var, lo)
        fhi = _cubic_m3_raw(x, target_var, hi)

        if abs(flo) > 1e250 or abs(fhi) > 1e250:
            break

    if not found:
        return False

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = _cubic_m3_raw(x, target_var, mid)

        if abs(fm) <= 1e-13 * (target_var ** 1.5 + 1.0):
            lo = mid
            hi = mid
            break

        if flo * fm <= 0.0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm

    a = 0.5 * (lo + hi)

    for i in range(n):
        q = x[i] * x[i] - target_var
        x[i] = x[i] + a * q

    _enforce_mean_var_inplace(x, target_var)

    return True


@njit
def _attempt_triple_indices(x, i, j, k, total_p3):
    """
    Try to change only three individuals while preserving their sum and sum of squares.
    Therefore the total population mean and variance are preserved.

    We choose the new triple so that the total third moment becomes zero.
    """
    xi = x[i]
    xj = x[j]
    xk = x[k]

    S = xi + xj + xk
    Q = xi * xi + xj * xj + xk * xk

    oldC = xi * xi * xi + xj * xj * xj + xk * xk * xk

    # Need total_new_p3 = 0.
    # total_new_p3 = total_p3 - oldC + newC
    # so newC = oldC - total_p3.
    targetC = oldC - total_p3

    A = Q - S * S / 3.0

    if A < 0.0 and A > -1e-12:
        A = 0.0

    if A <= 1e-300:
        base = S * S * S / 9.0

        if abs(targetC - base) <= 1e-12 * (abs(base) + 1.0):
            y = S / 3.0
            x[i] = y
            x[j] = y
            x[k] = y
            return True

        return False

    m = S / 3.0

    # For three numbers with fixed sum and fixed sum of squares,
    # the possible third moments form a bounded interval.
    base = 3.0 * m * m * m + 3.0 * m * A

    R = np.sqrt(2.0 * A / 3.0)
    amp = 0.75 * R * R * R

    if amp <= 1e-300:
        return False

    c = (targetC - base) / amp

    if c < -1.0 - 1e-10 or c > 1.0 + 1e-10:
        return False

    if c < -1.0:
        c = -1.0

    if c > 1.0:
        c = 1.0

    theta = np.arccos(c) / 3.0
    two_pi_over_3 = 2.0943951023931953

    x[i] = m + R * np.cos(theta)
    x[j] = m + R * np.cos(theta + two_pi_over_3)
    x[k] = m + R * np.cos(theta + 2.0 * two_pi_over_3)

    return True


@njit
def _local_triple_correction_inplace(x, target_var):
    """
    Method 5:
    Try to zero skewness by moving only three individuals.

    This is the most local method.
    It may not always be possible with the first selected triple,
    so it tries a tail triple and then random triples.
    If all fail, the outer function falls back to the exact mirrored-tail method.
    """
    n = x.size

    if n < 3 or target_var <= 0.0:
        _enforce_mean_var_inplace(x, target_var)
        return False

    _enforce_mean_var_inplace(x, target_var)

    total_p3 = 0.0
    for idx in range(n):
        total_p3 += x[idx] * x[idx] * x[idx]

    if abs(total_p3 / n) <= 1e-14 * (target_var ** 1.5 + 1.0):
        return True

    # First try a deterministic tail triple.
    i_max = 0
    i_min = 0

    for idx in range(1, n):
        if x[idx] > x[i_max]:
            i_max = idx

        if x[idx] < x[i_min]:
            i_min = idx

    i_abs = -1
    best = -1.0

    for idx in range(n):
        if idx != i_max and idx != i_min:
            ax = abs(x[idx])
            if ax > best:
                best = ax
                i_abs = idx

    if i_abs >= 0:
        if _attempt_triple_indices(x, i_max, i_min, i_abs, total_p3):
            _enforce_mean_var_inplace(x, target_var)
            return True

    # Then try random triples.
    for _ in range(80):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        k = np.random.randint(0, n)

        if i == j or i == k or j == k:
            continue

        if _attempt_triple_indices(x, i, j, k, total_p3):
            _enforce_mean_var_inplace(x, target_var)
            return True

    return False


@njit
def zero_skew_population_inplace(x, zero_method):
    """
    Main zero-skewing dispatcher.

    zero_method:
        0: no zero-skewing
        1: random mirrored pairs from current population
        2: tail mirrored pairs from current population
        3: paired Gaussian reset
        4: cubic correction
        5: local three-individual correction

    All nonzero methods preserve mean 0 and old variance.
    If a soft method fails, it falls back to method 2.
    """
    _center_inplace(x)
    target_var = _var_about_zero(x)

    if target_var <= 1e-300 or x.size < 2:
        for i in range(x.size):
            x[i] = 0.0
        return

    if zero_method == ZERO_NONE:
        _enforce_mean_var_inplace(x, target_var)

    elif zero_method == ZERO_PAIR_RANDOM:
        _pair_from_current_random_inplace(x, target_var)

    elif zero_method == ZERO_PAIR_TAILS:
        _pair_from_current_tails_inplace(x, target_var)

    elif zero_method == ZERO_GAUSSIAN_PAIRS:
        _gaussian_pairs_inplace(x, target_var)

    elif zero_method == ZERO_CUBIC:
        ok = _cubic_correction_inplace(x, target_var)

        if not ok:
            _pair_from_current_tails_inplace(x, target_var)

    elif zero_method == ZERO_LOCAL_TRIPLE:
        ok = _local_triple_correction_inplace(x, target_var)

        if not ok:
            _pair_from_current_tails_inplace(x, target_var)

    else:
        _pair_from_current_tails_inplace(x, target_var)

    # Final safety check.
    # If numerical residue is too large, force exact symmetric construction.
    _enforce_mean_var_inplace(x, target_var)

    residual = abs(_third_about_mean(x))
    tol = 1e-10 * (target_var ** 1.5 + 1.0)

    if residual > tol:
        _pair_from_current_tails_inplace(x, target_var)



@njit()
def Quad_Sim_V00_zeroSkew(b1_rate,    b2_rate,    d1_rate,    d2_rate,    tmax,    indices,    trait_values,    zero_method,    zero_every_events):
    n_individuals = len(trait_values)
    n_out = len(indices)

    # Main arrays     0: mean, 1: skew, 2: std
    Main_3D = np.zeros((3, n_out))

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
        birth_trait = trait_values[indice_birth] + np.random.normal(0.0, 1.0)
        trait_values[indice_death] = birth_trait

        current_mean = np.mean(trait_values)
        cum_mean_trait_value += current_mean
        trait_values -= current_mean

        # zero_every_events = 0 means no zero-skewing
        if zero_every_events > 0:
            if t % zero_every_events == 0:
                zero_skew_population_inplace(trait_values, zero_method)

        if k < n_out and t == indices[k]:

            Main_3D[0, k] = cum_mean_trait_value
            Main_3D[2, k] = np.std(trait_values)

            if Main_3D[2, k] == 0.0:
                Main_3D[1, k] = 0.0
            else:
                m = np.mean(trait_values)
                Main_3D[1, k] = np.sum((trait_values - m) ** 3) / (
                    n_individuals * Main_3D[2, k] ** 3
                )

            k += 1

    return Main_3D

@njit()  # Main3D, Clipp, Moments, Moments_right_tail, Moments_left_tail, Hist_counts, Hist_edges
def Quad_Sim_V1_zeroSkew(    b1_rate,    b2_rate,    d1_rate,    d2_rate,    tmax,    indices,    trait_values,    nbins=128,    zero_method=0,    zero_every_events=0):
    n_individuals = len(trait_values)
    n_out = len(indices)

    # Main arrays
    Main_3D = np.zeros((3, n_out))  # 0: mean, 1: skew, 2: std

    Clipp = np.zeros((6, n_out))
    # 0: birth_clipped_count
    # 1: death_clipped_count
    # 2: birth_clip_mass
    # 3: death_clip_mass
    # 4: wb_eff_mass
    # 5: wd_eff_mass

    Moments_right_tail = np.zeros((4, n_out))
    Moments_left_tail = np.zeros((4, n_out))
    Moments = np.zeros((4, n_out))  # 0: mu1, 1: mu2, 2: mu3, 3: mu4

    # Histogram storage
    Hist_counts = np.zeros((n_out, nbins), dtype=np.int64)
    Hist_edges = np.zeros((n_out, nbins + 1), dtype=np.float64)

    k = 0
    cum_mean_trait_value = 0.0

    for t in range(1, tmax):

        tv2 = trait_values ** 2

        wb = 1 + b1_rate * trait_values + b2_rate * tv2
        wd = 1 - d1_rate * trait_values - d2_rate * tv2

        wb_eff = np.clip(wb, 0.0, np.inf)
        wd_eff = np.clip(wd, 0.0, np.inf)

        indice_birth = weighted_choice(wb_eff)
        indice_death = weighted_choice(wd_eff)

        birth_trait = trait_values[indice_birth] + np.random.normal(0.0, 1.0)

        trait_values[indice_death] = birth_trait

        # Original recentering step
        current_mean = np.mean(trait_values)
        cum_mean_trait_value += current_mean
        trait_values -= current_mean

        # ------------------------------------------------------------
        # NEW: zero-skewing step
        # ------------------------------------------------------------
        # zero_every_events = 0 means no zero-skewing.
        # zero_every_events = 1 means after every event.
        # zero_every_events = n_individuals means once per unit time.
        #
        # This preserves the current variance and mean=0.
        # ------------------------------------------------------------
        if zero_every_events > 0:
            if t % zero_every_events == 0:
                zero_skew_population_inplace(trait_values, zero_method)

        if k < n_out and t == indices[k]:

            wb_neg_mass = np.sum(np.maximum(-wb, 0.0))
            wd_neg_mass = np.sum(np.maximum(-wd, 0.0))

            wb_eff_mass = np.sum(wb_eff)
            wd_eff_mass = np.sum(wd_eff)

            Clipp[0, k] = np.sum(wb < 0.0) / n_individuals
            Clipp[1, k] = np.sum(wd < 0.0) / n_individuals

            Clipp[2, k] = wb_neg_mass / (wb_eff_mass + wb_neg_mass)
            Clipp[3, k] = wd_neg_mass / (wd_eff_mass + wd_neg_mass)

            Clipp[4, k] = wb_eff_mass
            Clipp[5, k] = wd_eff_mass

            h = 0.5 * (np.max(trait_values) - np.min(trait_values)) / np.sqrt(n_individuals)

            if b1_rate != 0.0:
                tv_b_cut = np.where(trait_values < -1.0 / b1_rate,                    trait_values,                    np.nan                )  # left tail
            else:
                tv_b_cut = np.full_like(trait_values, np.nan)

            if d1_rate != 0.0:
                tv_d_cut = np.where(trait_values > 1.0 / d1_rate,                    trait_values,                    np.nan                )  # right tail
            else:
                tv_d_cut = np.full_like(trait_values, np.nan)

            Moments_right_tail[0, k] = np.nansum(tv_d_cut)
            Moments_right_tail[1, k] = np.nansum(tv_d_cut ** 2)
            Moments_right_tail[2, k] = np.nansum(tv_d_cut ** 3)
            Moments_right_tail[3, k] = np.nansum(tv_d_cut ** 4)

            Moments_left_tail[0, k] = np.nansum(tv_b_cut)
            Moments_left_tail[1, k] = np.nansum(tv_b_cut ** 2)
            Moments_left_tail[2, k] = np.nansum(tv_b_cut ** 3)
            Moments_left_tail[3, k] = np.nansum(tv_b_cut ** 4)

            Moments[0, k] = np.mean(trait_values)
            Moments[1, k] = np.mean(trait_values ** 2)
            Moments[2, k] = np.mean(trait_values ** 3)
            Moments[3, k] = np.mean(trait_values ** 4)

            Main_3D[0, k] = cum_mean_trait_value
            Main_3D[2, k] = np.std(trait_values)

            if Main_3D[2, k] == 0.0:
                Main_3D[1, k] = 0.0
            else:
                m = np.mean(trait_values)
                Main_3D[1, k] = np.sum((trait_values - m) ** 3) / (
                    n_individuals * Main_3D[2, k] ** 3
                )

            # Store dynamic histogram at this output time
            c, e = hist_dynamic_minmax(trait_values, nbins)
            Hist_counts[k, :] = c
            Hist_edges[k, :] = e

            k += 1

    return Main_3D, Clipp, Moments, Moments_right_tail, Moments_left_tail, Hist_counts, Hist_edges  

########################################################################################################################################
# Emergence study

@njit() 
def Quad_Sim_V3(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values, nbins=128):
    n_individuals = len(trait_values)
    n_out = len(indices)

    All_tv = np.zeros((n_out, n_individuals))
    Main_3D = np.zeros((3, n_out))  # 0: mean, 1: skew, 2: std

    Hist_counts = np.zeros((n_out, nbins), dtype=np.int64)
    Hist_edges  = np.zeros((n_out, nbins + 1), dtype=np.float64)

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
            Main_3D[0, k] = cum_mean_trait_value
            Main_3D[2, k] = np.std(trait_values)
            if Main_3D[2, k] == 0: Main_3D[1, k] = 0
            else: Main_3D[1, k] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * Main_3D[2, k]**3)
            c, e = hist_dynamic_minmax(trait_values, nbins)
            Hist_counts[k, :] = c
            Hist_edges[k, :]  = e
            All_tv[k, :] = trait_values
            k += 1

    return All_tv, Main_3D, Hist_counts, Hist_edges

def Metadata_Quad_Sim_V3(nlist, b1list, b2list, d1list, d2list, tmax, skip, t_lag, save_dir, nbins, nansa=10, n_jobs=6):
    combinations = list(product(*[nlist] + [b1list] + [b2list] + [d1list] + [d2list]))
    n_combos = len(combinations)

    def process_one_combo(i):   
        params = combinations[i]
        n_individuals = params[0];         b1_rate = params[1];        b2_rate = params[2];        d1_rate = params[3];        d2_rate = params[4]
        indices = np.arange(skip*n_individuals, tmax*n_individuals, t_lag) 
        n_out = len(indices)

        All_tv, Main_3D, Hist_counts, Hist_edges = Quad_Sim_V2(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, np.zeros(n_individuals), nbins=nbins)

        skwl = Main_3D[1, :]
        stdl = Main_3D[2, :]
        mean_trait_values = Main_3D[0, :]

        Mdot = np.gradient(mean_trait_values, 1) * n_individuals / (t_lag)
        Vdot = np.gradient(stdl**2, 1) * n_individuals / (t_lag)
    
        bimodal_flag = np.zeros(n_out, dtype=bool)
        hump_left = np.full(n_out, np.nan)
        hump_right = np.full(n_out, np.nan)
        hump_weight_left = np.full(n_out, np.nan)
        hump_weight_right = np.full(n_out, np.nan)

        for k in range(0, n_out):
            trait_values = All_tv[k, :].reshape(-1,1)
            gmm1 = GaussianMixture(n_components=1)
            gmm1.fit(trait_values)
            bic1 = gmm1.bic(trait_values)
            gmm2 = GaussianMixture(n_components=2)
            gmm2.fit(trait_values)
            bic2 = gmm2.bic(trait_values)
            mean2 = gmm2.means_.flatten()
            std2 = np.sqrt(gmm2.covariances_.flatten())
            weights2 = gmm2.weights_.flatten()
            sep = abs(mean2[0] - mean2[1]) / np.sqrt(std2[0]**2 + std2[1]**2)
            x_range = np.linspace(Hist_edges[k, 0], Hist_edges[k, -1], 128).reshape(-1,1)
            if bic2 < bic1 and sep > 1.5:
                bimodal_flag[k] = True
                # sort humps left/right
                order = np.argsort(mean2)
                hump_left[k] = mean2[order[0]]
                hump_right[k] = mean2[order[1]]
                hump_weight_left[k] = weights2[order[0]]
                hump_weight_right[k] = weights2[order[1]]

        hump_events = np.where((~bimodal_flag[:-1]) & (bimodal_flag[1:]))[0] + 1 

        # Waiting times between hump events
        if len(hump_events) > 1:
            waiting_times = np.diff(indices[hump_events])
            mean_waiting = np.mean(waiting_times)
            std_waiting = np.std(waiting_times)
        else:
            mean_waiting = np.nan
            std_waiting = np.nan

        # Where humps nucleate
        nucleation_left = hump_left[hump_events]
        nucleation_right = hump_right[hump_events]
        hump_distance = hump_right[hump_events] - hump_left[hump_events]

        # Statistics
        mean_nucl_left = np.nanmean(nucleation_left)
        mean_nucl_right = np.nanmean(nucleation_right)
        std_nucl_left = np.nanstd(nucleation_left)
        std_nucl_right = np.nanstd(nucleation_right)
        mean_hump_distance = np.nanmean(hump_distance)

        metadata = {
            "N": int(n_individuals),
            "b1": float(b1_rate),
            "b2": float(b2_rate),
            "d1": float(d1_rate),
            "d2": float(d2_rate)
        }
        metadata["Avg_Skw"] = np.mean(skwl)
        metadata["std_Skw"] = np.std(skwl)

        metadata["Avg_Std"] = np.mean(stdl)
        metadata["std_Std"] = np.std(stdl)

        metadata["Avg_Var"] = np.mean(stdl**2)
        metadata["std_Var"] = np.std(stdl**2)

        metadata["Num_hump_events"] = len(hump_events)

        metadata["Mean_waiting_time"] = mean_waiting
        metadata["Std_waiting_time"] = std_waiting

        metadata["Mean_nucleation_left"] = mean_nucl_left
        metadata["Mean_nucleation_right"] = mean_nucl_right

        metadata["Std_nucleation_left"] = std_nucl_left
        metadata["Std_nucleation_right"] = std_nucl_right

        metadata["Mean_hump_distance"] = mean_hump_distance
        

        return metadata

    results = Parallel(n_jobs=n_jobs, verbose=0)( delayed(process_one_combo)(i) for i in tqdm(range(n_combos), total=len(combinations), desc="Simulating", ncols=100) )    # (n_jobs=n_jobs, backend='loky') , backend="threading"
    summary_path = os.path.join(save_dir, f"{t_lag}_ALL_summaries.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n✅ All summaries saved to {summary_path}")







def Metadata_Quad_Sim_V3(nlist, b1list, b2list, d1list, d2list, nansa_list, tmax, skip, t_lag, save_dir, nbins, n_jobs=6):

    combinations = list(product(nlist, b1list, b2list, d1list, d2list, nansa_list))
    n_combos = len(combinations)

    def process_one_combo(i):

        params = combinations[i]
        n_individuals, b1_rate, b2_rate, d1_rate, d2_rate, nansa = params

        indices = np.arange(skip*n_individuals, tmax*n_individuals, t_lag)
        n_out = len(indices)

        # ---- initial condition depends on nansa ----
        init_traits = np.random.normal(0, nansa, n_individuals)

        All_tv, Main_3D, Hist_counts, Hist_edges = Quad_Sim_V3( b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, init_traits, nbins=nbins)

        skwl = Main_3D[1, :]
        stdl = Main_3D[2, :]
        mean_trait_values = Main_3D[0, :]

        # ---- GMM detection ----
        bimodal_flag = np.zeros(n_out, dtype=bool)
        hump_left = np.full(n_out, np.nan)
        hump_right = np.full(n_out, np.nan)
        hump_weight_left = np.full(n_out, np.nan)
        hump_weight_right = np.full(n_out, np.nan)

        for k in range(n_out):
            tv = All_tv[k].reshape(-1,1)

            gmm1 = GaussianMixture(1).fit(tv)
            gmm2 = GaussianMixture(2).fit(tv)

            if gmm2.bic(tv) < gmm1.bic(tv):
                means = gmm2.means_.flatten()
                stds = np.sqrt(gmm2.covariances_.flatten())
                weights = gmm2.weights_.flatten()

                sep = abs(means[0]-means[1]) / np.sqrt(stds[0]**2 + stds[1]**2)

                if sep > 1.5:
                    bimodal_flag[k] = True
                    order = np.argsort(means)

                    hump_left[k] = means[order[0]]
                    hump_right[k] = means[order[1]]
                    hump_weight_left[k] = weights[order[0]]
                    hump_weight_right[k] = weights[order[1]]


        # ---- EVENT DETECTION ----
        event_idx = np.where((~bimodal_flag[:-1]) & (bimodal_flag[1:]))[0] + 1
        event_data = []

        for j, k in enumerate(event_idx):

            time = indices[k]

            # waiting time
            if j == 0:
                waiting_time = np.nan
            else:
                waiting_time = indices[k] - indices[event_idx[j-1]]

            left = hump_left[k]
            right = hump_right[k]
            dist = right - left

            w_left = hump_weight_left[k]
            w_right = hump_weight_right[k]

            # identify NEW morph (smaller weight)
            if w_left < w_right:
                new_weight = w_left
                new_pos = left
                old_pos = right
            else:
                new_weight = w_right
                new_pos = right
                old_pos = left

            event_data.append({
                "N": n_individuals,
                "b1": b1_rate,
                "b2": b2_rate,
                "d1": d1_rate,
                "d2": d2_rate,
                "nansa": nansa,

                "time": time,
                "waiting_time": waiting_time,

                "left": left,
                "right": right,
                "distance": dist,

                "new_morph_position": new_pos,
                "old_morph_position": old_pos,
                "new_morph_weight": new_weight
            })

        # ---- SUMMARY ----
        if len(event_idx) > 1:
            waiting_times = np.diff(indices[event_idx])
            mean_waiting = np.mean(waiting_times)
            std_waiting = np.std(waiting_times)
        else:
            mean_waiting = np.nan
            std_waiting = np.nan

        metadata = {
            "N": n_individuals,
            "b1": b1_rate,
            "b2": b2_rate,
            "d1": d1_rate,
            "d2": d2_rate,
            "nansa": nansa,

            "Avg_Std": np.mean(stdl),
            "Std_Std": np.std(stdl),

            "Num_events": len(event_idx),
            "Mean_waiting_time": mean_waiting,
            "Std_waiting_time": std_waiting
        }

        return metadata, event_data

    results = Parallel(n_jobs=n_jobs)( delayed(process_one_combo)(i) for i in tqdm(range(n_combos)) )

    # ---- unpack ----
    summaries = []
    all_events = []

    for meta, events in results:
        summaries.append(meta)
        all_events.extend(events)

    summary_df = pd.DataFrame(summaries)
    events_df = pd.DataFrame(all_events)

    summary_df.to_csv(os.path.join(save_dir, "summaries.csv"), index=False)
    events_df.to_csv(os.path.join(save_dir, "events.csv"), index=False)

    print("✅ Saved summaries + event-level data")

    return summary_df, events_df





########################################################################################################################################
## Loss function

def r2_score(V_true, V_pred):
    ss_res = np.sum((V_true - V_pred) ** 2)
    ss_tot = np.sum((V_true - np.mean(V_true)) ** 2)
    if ss_tot == 0:
        return -np.inf
    return -(1.0 - ss_res / ss_tot)

def L1_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def L2_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def huber_loss(y_true, y_pred, delta=5.0):
    err = y_true - y_pred
    mask = np.abs(err) < delta
    return np.mean(     np.where(mask, 0.5 * err**2, delta * (np.abs(err) - 0.5 * delta))    )

def get_loss_function(name):
    if name == "L1":
        return L1_loss
    elif name == "L2":
        return L2_loss
    elif name == "r2":
        return r2_score
    elif name == "huber":
        return lambda y, yhat: huber_loss(y, yhat, delta=5.0)
    else:
        raise ValueError("Unknown loss")

########################################################################################################################################    

def zero_crossing_pos_to_neg(signal):
    zc = np.where((signal[:-1] > 0) & (signal[1:] <= 0))[0]

    peaks, _ = find_peaks(signal)
    troughs, _ = find_peaks(-signal)

    valid_zc = []

    for i in zc:
        # last peak before crossing
        prev_peaks = peaks[peaks < i]
        if len(prev_peaks) == 0:
            continue
        last_peak = prev_peaks[-1]

        # first trough after crossing
        next_troughs = troughs[troughs > i]
        if len(next_troughs) == 0:
            continue
        next_trough = next_troughs[0]

        # ensure correct ordering
        if last_peak < i < next_trough:
            valid_zc.append(i)

    # print(f"Found {len(zc)} zero crossings, {len(valid_zc)} are valid pos-to-neg crossings.")

    # Require exactly one valid crossing
    if len(valid_zc) != 1:
        return None

    return valid_zc[0]

def zero_crossing(signal):
    smooth_signal = gaussian_filter1d(signal, sigma=30)
    zc = np.where( (smooth_signal[:-1] > 0) & (smooth_signal[1:] <= 0) )[0]
    if len(zc) == 0 or len(zc) > 1:
        return None

    return zc[0]

def get_exterma(Mdotdata, Vdotdata, Sdotdata, sigma, dt):
    M_s = gaussian_filter1d(Mdotdata, sigma=sigma)
    V_s = gaussian_filter1d(Vdotdata, sigma=sigma)
    T_s = gaussian_filter1d(Sdotdata, sigma=sigma)

    M_max = np.max(M_s)
    T_min = np.min(T_s)
    V_max = np.max(V_s)
    V_min = np.min(V_s)
    i_Vmax = np.argmax(V_s)
    i_Vmin = np.argmin(V_s)
    dt_V = (i_Vmin - i_Vmax) * dt

    return M_max, T_min, V_max, V_min, dt_V


def detect_events(Mdot, Vdot, Sdot, skip, tmax, t_lag, n_individuals, tune_params):

    M_sigma = tune_params["M_sigma"]
    M_height_mean = tune_params["M_height_mean"]
    M_height_std = tune_params["M_height_std"]
    M_distance = tune_params["M_distance"]
    V_sigma = tune_params["V_sigma"]
    V_height_mean = tune_params["V_height_mean"]
    V_height_std = tune_params["V_height_std"]
    V_distance = tune_params["V_distance"]
    T_sigma = tune_params["T_sigma"]
    T_height_mean = tune_params["T_height_mean"]
    T_height_std = tune_params["T_height_std"]
    T_distance = tune_params["T_distance"]
    time_tol = tune_params["time_tol"]

    tarray = np.arange(skip*n_individuals, tmax*n_individuals, t_lag) / n_individuals
    M_s = gaussian_filter1d(Mdot[:int((tmax-skip)*n_individuals/t_lag)], sigma=M_sigma)
    V_s = gaussian_filter1d(Vdot[:int((tmax-skip)*n_individuals/t_lag)], sigma=V_sigma)
    T_s = gaussian_filter1d(Sdot[:int((tmax-skip)*n_individuals/t_lag)], sigma=T_sigma)

    M_s2 = gaussian_filter1d(Mdot[:int((tmax-skip)*n_individuals/t_lag)], sigma=6)
    V_s2 = gaussian_filter1d(Vdot[:int((tmax-skip)*n_individuals/t_lag)], sigma=6)
    T_s2 = gaussian_filter1d(Sdot[:int((tmax-skip)*n_individuals/t_lag)], sigma=6)

    M_peaks, _ = find_peaks( M_s, height=M_height_mean*np.mean( M_s) + M_height_std*np.std( M_s), distance=M_distance)

    T_peaks, _ = find_peaks(-T_s, height=T_height_mean*np.mean(-T_s) + T_height_std*np.std(-T_s), distance=T_distance)

    V_max_idx, _ = find_peaks(V_s, height=V_height_mean*np.mean(V_s) + V_height_std*np.std(V_s), distance=V_distance)
    V_min_idx, _ = find_peaks(-V_s, height=V_height_mean*np.mean(-V_s) + V_height_std*np.std(-V_s), distance=V_distance)

    extrema = np.sort(np.concatenate([V_max_idx, V_min_idx]))
    is_max = {i: True for i in V_max_idx}
    is_min = {i: True for i in V_min_idx}
    sign = np.sign(V_s)
    zc_idx = np.where(np.diff(sign))[0]
    cycles = []
    i = 0
    while i < len(extrema) - 1:
        e1 = extrema[i]
        e2 = extrema[i+1]
        if not (e1 in is_max and e2 in is_min):
            i += 1
            continue
        z_candidates = zc_idx[(zc_idx > e1) & (zc_idx < e2)]
        if len(z_candidates) == 0:
            i += 1
            continue
        mid = (e1 + e2) / 2
        z = z_candidates[np.argmin(np.abs(z_candidates - mid))]
        cycles.append((e1, z, e2))
        i += 2
    V_peaks = np.array([c[0] for c in cycles])
    V_zeros = np.array([c[1] for c in cycles])
    V_troughs = np.array([c[2] for c in cycles])

    events = []
    event_M_idx = []
    event_T_idx = []
    event_V0_idx = []
    event_Vpk_idx = []
    event_Vtr_idx = []

    for e, mp in zip(events, M_peaks[:len(events)]): 
        pass

    for mp in M_peaks:
        t_m = tarray[mp]
        t_candidates = T_peaks[np.abs(tarray[T_peaks] - t_m) < time_tol]
        if len(t_candidates) == 0:
            continue
        tp = t_candidates[np.argmin(np.abs(tarray[t_candidates] - t_m))]
        v_candidates = V_zeros[np.abs(tarray[V_zeros] - t_m) < time_tol]
        if len(v_candidates) == 0:
            continue
        vz = v_candidates[np.argmin(np.abs(tarray[v_candidates] - t_m))]
        v_peak_near = V_peaks[np.argmin(np.abs(tarray[V_peaks] - t_m))]
        # v_trough_near = V_troughs[np.argmin(np.abs(tarray[V_troughs] - t_m))]
        v_trough_near = V_troughs[np.where(V_troughs > v_peak_near)[0][0]] if np.any(V_troughs > v_peak_near) else v_peak_near

        dt_M_T = tarray[tp] - t_m
        dt_M_V0 = tarray[vz] - t_m
        dt_Vpk_Vtr = tarray[v_trough_near] - tarray[v_peak_near]

        events.append({
            "mp": mp,
            "tp": tp,
            "vz": vz,
            "vpk": v_peak_near,
            "vtr": v_trough_near,
            "t_event": t_m,
            "M_peak": M_s2[mp],
            "T_min": T_s2[tp],
            "V_max": V_s2[v_peak_near],
            "V_min": V_s2[v_trough_near],
            "dt_M_T": dt_M_T,
            "dt_M_V0": dt_M_V0,
            "dt_Vpk_Vtr": dt_Vpk_Vtr
        })
    event_times = np.array([e["t_event"] for e in events])

    M_vals = np.array([e["M_peak"] for e in events])
    T_vals = np.array([e["T_min"] for e in events])
    V_max_vals = np.array([e["V_max"] for e in events])
    V_min_vals = np.array([e["V_min"] for e in events])

    dt_M_T = np.array([e["dt_M_T"] for e in events])
    dt_M_V0 = np.array([e["dt_M_V0"] for e in events])
    dt_Vpk_Vtr = np.array([e["dt_Vpk_Vtr"] for e in events])

    waiting_times = np.diff(event_times)

    print(f"Number of events: {len(events)}")
    print(f"Mean waiting time: {np.mean(waiting_times):.3f}")

    event_M_idx  = np.array([e["mp"]  for e in events])
    event_T_idx  = np.array([e["tp"]  for e in events])
    event_V0_idx = np.array([e["vz"]  for e in events])
    event_Vpk_idx = np.array([e["vpk"] for e in events])
    event_Vtr_idx = np.array([e["vtr"] for e in events])

    return tarray, M_s, V_s, T_s, M_peaks, V_peaks, V_zeros, V_troughs, T_peaks, event_M_idx, event_T_idx, event_V0_idx, event_Vpk_idx, event_Vtr_idx, event_times, M_vals, T_vals, V_max_vals, V_min_vals, dt_M_T, dt_M_V0, dt_Vpk_Vtr, waiting_times

def detect_events_ensemble(Mdot, Vdot, Sdot, skip, tmax, t_lag, n_individuals):
    n_ens = Mdot.shape[0]

    # aggregated outputs (same order as original return)
    outputs = [[] for _ in range(23)]  # your function returns 23 items

    for k in range(n_ens):
        res = detect_events(
            Mdot[k], Vdot[k], Sdot[k],
            skip, tmax, t_lag, n_individuals
        )

        for i, r in enumerate(res):
            outputs[i].append(r)

    # optional: convert what we safely can to arrays
    outputs[0] = outputs[0][0]  # tarray (same for all)
    # everything else stays list-of-arrays (since lengths differ)

    return tuple(outputs)

def plot_events(tarray, M_s, V_s, T_s, M_peaks, V_peaks, V_zeros, V_troughs, T_peaks, event_M_idx, event_T_idx, event_V0_idx, event_Vpk_idx, event_Vtr_idx, event_times):
    # ---- plotting ----
    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

    # Mdot
    axs[0].plot(tarray, M_s, label="Mdot")
    axs[0].scatter(tarray[M_peaks], M_s[M_peaks], color="red", label="M peaks")

    # Vdot
    axs[1].plot(tarray, V_s, label="Vdot")
    axs[1].scatter(tarray[V_peaks], V_s[V_peaks], color="red", label="V peaks")
    axs[1].scatter(tarray[V_troughs], V_s[V_troughs], color="blue", label="V troughs")
    axs[1].scatter(tarray[V_zeros], V_s[V_zeros], color="green", label="V zeros")


    # Tdot
    axs[2].plot(tarray, T_s, label="Tdot")
    axs[2].scatter(tarray[T_peaks], T_s[T_peaks], color="purple", label="T minima")

    # ---- vertical lines for alignment ----
    # for ax in axs:
    #     for p in M_peaks:
    #         ax.axvline(tarray[p], color="red", linestyle="--", alpha=0.3)
    #     for p in T_peaks:
    #         ax.axvline(tarray[p], color="purple", linestyle="--", alpha=0.3)
        # for z in V_zeros:
        #     ax.axvline(tarray[z], color="green", linestyle=":", alpha=0.2)



    for ax in axs:
        for t in event_times:
            ax.axvline(t, color="red", linestyle="--", alpha=0.3)



    # Mdot (highlight matched ones)
    axs[0].scatter(tarray[event_M_idx], M_s[event_M_idx],
                s=120, facecolors='none', edgecolors='black',
                linewidths=2, label="Matched M peaks")

    # Vdot
    axs[1].scatter(tarray[event_Vpk_idx], V_s[event_Vpk_idx],
                s=120, facecolors='none', edgecolors='black', linewidths=2,
                label="Matched V peaks")

    axs[1].scatter(tarray[event_Vtr_idx], V_s[event_Vtr_idx],
                s=120, facecolors='none', edgecolors='cyan', linewidths=2,
                label="Matched V troughs")

    axs[1].scatter(tarray[event_V0_idx], V_s[event_V0_idx],
                s=120, facecolors='none', edgecolors='lime', linewidths=2,
                label="Matched V zeros")

    # Tdot
    axs[2].scatter(tarray[event_T_idx], T_s[event_T_idx],
                s=120, facecolors='none', edgecolors='black',
                linewidths=2, label="Matched T minima")

    # labels
    axs[0].set_ylabel("Mdot")
    axs[1].set_ylabel("Vdot")
    axs[2].set_ylabel("Tdot")
    axs[2].set_xlabel("Time")

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_events2(tarray, M_s, V_s, T_s, M_peaks, V_peaks, V_zeros, V_troughs, T_peaks, event_M_idx, event_T_idx, event_V0_idx, event_Vpk_idx, event_Vtr_idx, event_times, plot_length=100):

    # index cutoff
    t_end = tarray[0] + plot_length
    cut   = np.searchsorted(tarray, t_end)

    def in_range(idx): return idx[idx < cut]

    M_peaks_p      = in_range(M_peaks)
    V_peaks_p      = in_range(V_peaks)
    V_troughs_p    = in_range(V_troughs)
    V_zeros_p      = in_range(V_zeros)
    T_peaks_p      = in_range(T_peaks)
    event_M_idx_p  = in_range(event_M_idx)
    event_T_idx_p  = in_range(event_T_idx)
    event_V0_idx_p = in_range(event_V0_idx)
    event_Vpk_idx_p= in_range(event_Vpk_idx)
    event_Vtr_idx_p= in_range(event_Vtr_idx)
    event_times_p  = event_times[event_times <= t_end]

    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

    axs[0].plot(tarray[:cut], M_s[:cut], label="Mdot")
    axs[0].scatter(tarray[M_peaks_p], M_s[M_peaks_p], color="red", label="M peaks")

    axs[1].plot(tarray[:cut], V_s[:cut], label="Vdot")
    axs[1].scatter(tarray[V_peaks_p],   V_s[V_peaks_p],   color="red",   label="V peaks")
    axs[1].scatter(tarray[V_troughs_p], V_s[V_troughs_p], color="blue",  label="V troughs")
    axs[1].scatter(tarray[V_zeros_p],   V_s[V_zeros_p],   color="green", label="V zeros")

    axs[2].plot(tarray[:cut], T_s[:cut], label="Tdot")
    axs[2].scatter(tarray[T_peaks_p], T_s[T_peaks_p], color="purple", label="T minima")

    for ax in axs:
        for t in event_times_p:
            ax.axvline(t, color="red", linestyle="--", alpha=0.3)

    axs[0].scatter(tarray[event_M_idx_p],  M_s[event_M_idx_p],
                   s=120, facecolors='none', edgecolors='black', linewidths=2, label="Matched M peaks")

    axs[1].scatter(tarray[event_Vpk_idx_p], V_s[event_Vpk_idx_p],
                   s=120, facecolors='none', edgecolors='black', linewidths=2, label="Matched V peaks")
    axs[1].scatter(tarray[event_Vtr_idx_p], V_s[event_Vtr_idx_p],
                   s=120, facecolors='none', edgecolors='cyan',  linewidths=2, label="Matched V troughs")
    axs[1].scatter(tarray[event_V0_idx_p],  V_s[event_V0_idx_p],
                   s=120, facecolors='none', edgecolors='lime',  linewidths=2, label="Matched V zeros")

    axs[2].scatter(tarray[event_T_idx_p], T_s[event_T_idx_p],
                   s=120, facecolors='none', edgecolors='black', linewidths=2, label="Matched T minima")

    axs[0].set_ylabel("Mdot")
    axs[1].set_ylabel("Vdot")
    axs[2].set_ylabel("Tdot")
    axs[2].set_xlabel("Time")

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_events3(tarray, M, M_s, V_s, T_s, M_peaks, V_peaks, V_zeros, V_troughs, T_peaks, event_M_idx, event_T_idx, event_V0_idx, event_Vpk_idx, event_Vtr_idx, event_times, plot_length=100):

    # index cutoff
    t_end = tarray[0] + plot_length
    cut   = np.searchsorted(tarray, t_end)

    def in_range(idx): return idx[idx < cut]

    M_peaks_p      = in_range(M_peaks)
    V_peaks_p      = in_range(V_peaks)
    V_troughs_p    = in_range(V_troughs)
    V_zeros_p      = in_range(V_zeros)
    T_peaks_p      = in_range(T_peaks)
    event_M_idx_p  = in_range(event_M_idx)
    event_T_idx_p  = in_range(event_T_idx)
    event_V0_idx_p = in_range(event_V0_idx)
    event_Vpk_idx_p= in_range(event_Vpk_idx)
    event_Vtr_idx_p= in_range(event_Vtr_idx)
    event_times_p  = event_times[event_times <= t_end]

    fig, axs = plt.subplots(4, 1, figsize=(20, 10), sharex=True)

    axs[0].plot(tarray[:cut], M[:cut], label="M")

    axs[1].plot(tarray[:cut], M_s[:cut], label="Mdot")
    axs[1].scatter(tarray[M_peaks_p], M_s[M_peaks_p], color="red", label="M peaks")

    axs[2].plot(tarray[:cut], V_s[:cut], label="Vdot")
    axs[2].scatter(tarray[V_peaks_p],   V_s[V_peaks_p],   color="red",   label="V peaks")
    axs[2].scatter(tarray[V_troughs_p], V_s[V_troughs_p], color="blue",  label="V troughs")
    axs[2].scatter(tarray[V_zeros_p],   V_s[V_zeros_p],   color="green", label="V zeros")

    axs[3].plot(tarray[:cut], T_s[:cut], label="Tdot")
    axs[3].scatter(tarray[T_peaks_p], T_s[T_peaks_p], color="purple", label="T minima")

    for ax in axs:
        for t in event_times_p:
            ax.axvline(t, color="red", linestyle="--", alpha=0.3)

    axs[1].scatter(tarray[event_M_idx_p],  M_s[event_M_idx_p],
                   s=120, facecolors='none', edgecolors='black', linewidths=2, label="Matched M peaks")

    axs[2].scatter(tarray[event_Vpk_idx_p], V_s[event_Vpk_idx_p],
                   s=120, facecolors='none', edgecolors='black', linewidths=2, label="Matched V peaks")
    axs[2].scatter(tarray[event_Vtr_idx_p], V_s[event_Vtr_idx_p],
                   s=120, facecolors='none', edgecolors='cyan',  linewidths=2, label="Matched V troughs")
    axs[2].scatter(tarray[event_V0_idx_p],  V_s[event_V0_idx_p],
                   s=120, facecolors='none', edgecolors='lime',  linewidths=2, label="Matched V zeros")

    axs[3].scatter(tarray[event_T_idx_p], T_s[event_T_idx_p],
                   s=120, facecolors='none', edgecolors='black', linewidths=2, label="Matched T minima")

    axs[0].set_ylabel("M")
    axs[1].set_ylabel("Mdot")
    axs[2].set_ylabel("Vdot")
    axs[3].set_ylabel("Tdot")
    axs[3].set_xlabel("Time")

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()


def Plot_Moment_dynamics_simple( n_individuals, indices, Mdot, Vdot, Tdot, fig_dir, ex_name, figsize=(25, 15), dpi=600):
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
    ax0 = axes[0]
    ax0.plot(indices/n_individuals, Mdot, label="M dot")
    ax0.set_ylabel(r"$\dot M$", fontsize=18)
    ax0.legend(fontsize=18)
    ax1 = axes[1]
    ax1.plot(indices/n_individuals, Vdot, label="V dot")
    ax1.set_ylabel(r"$\dot V$", fontsize=18)
    ax1.legend(fontsize=18)
    ax2 = axes[2]
    ax2.plot(indices/n_individuals, Tdot, label="T dot")
    ax2.set_xlabel("Time", fontsize=18)
    ax2.set_ylabel(r"$\dot T$", fontsize=18)
    ax2.legend(fontsize=18)

    fig.savefig(os.path.join(fig_dir, ex_name, f"moment_dynamics.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def Mean_f(t, muR0, VR0, deltaV0, deltaMu0, PR0, bpd, g):
    exp_term = np.exp( bpd * t * (bpd * t * deltaV0 + deltaMu0) )
    denominator = ( 1 + (exp_term - 1) * PR0 )
    correction = ( (PR0 - 1) * (bpd * t * deltaV0 + deltaMu0) / denominator )

    return ( muR0 + 0.5 * bpd * t * (g**2 * t + 2 * VR0) + correction )

def Mdot_f(t, VR0, deltaV0, deltaMu0, PR0, bpd, g):
    exp_term = np.exp(bpd * t * (bpd * t * deltaV0 + deltaMu0))
    denom = (1 + (exp_term - 1) * PR0)**2
    num = ( (PR0 - 1) * ( deltaV0 - PR0 * deltaV0 - exp_term * PR0 * ( deltaV0 * ( -1 + 2 * bpd**2 * t**2 * deltaV0 ) + 3 * bpd * t * deltaV0 * deltaMu0 + deltaMu0**2 ) ) )
    
    return bpd * (g**2 * t + VR0 + num / denom)

def Vdot_f(t, dV0, dmu0, PR0, bpd, g):
    exp_term = np.exp(bpd * t * (bpd * t * dV0 + dmu0))
    
    numerator = (
        bpd * exp_term * (PR0 - 1) * PR0 *
        (
            exp_term * PR0 * (
                2*bpd**3*t**3*dV0**3
                - 3*dV0*dmu0
                + 5*bpd**2*t**2*dV0**2*dmu0
                + dmu0**3
                + 4*bpd*t*dV0*(-dV0 + dmu0**2)
            )
            + (PR0 - 1) * (
                2*bpd**3*t**3*dV0**3
                + 3*dV0*dmu0
                + 5*bpd**2*t**2*dV0**2*dmu0
                + dmu0**3
                + 4*bpd*t*dV0*(dV0 + dmu0**2)
            )
        )
    )
    
    denominator = (1 + (exp_term - 1)*PR0)**3
    
    return g**2 + numerator / denominator

def Tdot_f(t, dV0, dmu0, PR0, bpd):
    exp_term = np.exp(bpd * t * (bpd * t * dV0 + dmu0))

    term1 = (        4 * bpd**4 * t**4 * dV0**4
        + 14 * bpd**3 * t**3 * dV0**3 * dmu0
        + 10 * bpd * t * dV0 * dmu0**3
        + 2 * dmu0**4
        + 3 * dV0**2 * (-1 + 6 * bpd**2 * t**2 * dmu0**2)    )

    term2 = (        2 * bpd**4 * t**4 * dV0**4
        + dmu0**4
        + dV0 * dmu0**2 * (-6 + 5 * bpd * t * dmu0)
        + bpd**2 * t**2 * dV0**3 * (-9 + 7 * bpd * t * dmu0)
        + 3 * dV0**2 * (
            1 + bpd * t * dmu0 * (-5 + 3 * bpd * t * dmu0)        )    )

    term3 = (        2 * bpd**4 * t**4 * dV0**4
        + dmu0**4
        + dV0 * dmu0**2 * (6 + 5 * bpd * t * dmu0)
        + bpd**2 * t**2 * dV0**3 * (9 + 7 * bpd * t * dmu0)
        + 3 * dV0**2 * (
            1 + bpd * t * dmu0 * (5 + 3 * bpd * t * dmu0)        )    )

    numerator = ( bpd * exp_term * (PR0 - 1) * PR0 * 
            ( -2 * exp_term * (PR0 - 1) * PR0 * term1
            - exp_term**2 * PR0**2 * term2
            - (PR0 - 1)**2 * term3        )    )

    denominator = (1 + (exp_term - 1) * PR0) ** 4

    return numerator / denominator


def loss_4D(muR0, VR0, dV0, dmu0, Mean_data, T_data, V_data, M_data, loss_fn, PR0, bpd, g, dt):
    t_model = np.arange(len(V_data)*4) * dt
    mean_model = Mean_f(t_model, muR0, VR0, dV0, dmu0, PR0, bpd, g)
    M_model = Mdot_f(t_model, VR0, dV0, dmu0, PR0, bpd, g)
    V_model = Vdot_f(t_model, dV0, dmu0, PR0, bpd, g)
    T_model = Tdot_f(t_model, dV0, dmu0, PR0, bpd)
    
    t0_data = zero_crossing(gaussian_filter1d(V_data, sigma=15))
    t0_model = zero_crossing_pos_to_neg(V_model)
    if t0_data is None or t0_model is None:
        return 1e10
    shift =  t0_model - t0_data
    if shift<0:
        return 1e10
    start_model = int(shift)
    end_model = int(shift + len(V_data))

    mean_model_aligned = mean_model[start_model:end_model]
    mean_data_aligned = Mean_data.copy()
    Mmodel_aligned = M_model[start_model:end_model]
    Mdata_aligned = M_data.copy()
    Vmodel_aligned = V_model[start_model:end_model]
    Vdata_aligned = V_data.copy()
    Tmodel_aligned = T_model[start_model:end_model]
    Tdata_aligned = T_data.copy()


    loss_mean = loss_fn(mean_data_aligned, mean_model_aligned)
    loss_M = loss_fn(Mdata_aligned, Mmodel_aligned)
    loss_V = loss_fn(Vdata_aligned, Vmodel_aligned)
    loss_T = loss_fn(Tdata_aligned, Tmodel_aligned)

    scale_mean = np.std(mean_data_aligned) + 1e-8
    scale_M = np.std(Mdata_aligned) + 1e-8
    scale_V = np.std(Vdata_aligned) + 1e-8
    scale_T = np.std(Tdata_aligned) + 1e-8


    loss_total = (loss_mean / scale_mean) + (loss_V / scale_V) + (loss_M / scale_M) + (loss_T / scale_T)

    return loss_total

def extract_Mdot_peak(t, M):
    i = np.argmax(M)
    return {
        "t_max": t[i],
        "M_max": M[i]
    }


################################################################################################


def detect_events_fast(Mdot, Vdot, Sdot, skip, tmax, t_lag, n_individuals, tune_params):
    
    M_sigma = tune_params["M_sigma"]
    M_height_mean = tune_params["M_height_mean"]
    M_height_std = tune_params["M_height_std"]
    M_distance = tune_params["M_distance"]
    V_sigma = tune_params["V_sigma"]
    V_height_mean = tune_params["V_height_mean"]
    V_height_std = tune_params["V_height_std"]
    V_distance = tune_params["V_distance"]
    T_sigma = tune_params["T_sigma"]
    T_height_mean = tune_params["T_height_mean"]
    T_height_std = tune_params["T_height_std"]
    T_distance = tune_params["T_distance"]
    time_tol = tune_params["time_tol"]

    tarray = np.arange(skip*n_individuals, tmax*n_individuals, t_lag) / n_individuals
    M_s = gaussian_filter1d(Mdot[:int((tmax-skip)*n_individuals/t_lag)], sigma=M_sigma)
    V_s = gaussian_filter1d(Vdot[:int((tmax-skip)*n_individuals/t_lag)], sigma=V_sigma)
    T_s = gaussian_filter1d(Sdot[:int((tmax-skip)*n_individuals/t_lag)], sigma=T_sigma)

    M_peaks, _ = find_peaks( M_s, height=M_height_mean*np.mean( M_s) + M_height_std*np.std( M_s), distance=M_distance)

    T_peaks, _ = find_peaks(-T_s, height=T_height_mean*np.mean(-T_s) + T_height_std*np.std(-T_s), distance=T_distance)

    V_max_idx, _ = find_peaks(V_s, height=V_height_mean*np.mean(V_s) + V_height_std*np.std(V_s), distance=V_distance)
    V_min_idx, _ = find_peaks(-V_s, height=V_height_mean*np.mean(-V_s) + V_height_std*np.std(-V_s), distance=V_distance)
    extrema = np.sort(np.concatenate([V_max_idx, V_min_idx]))
    is_max = {i: True for i in V_max_idx}
    is_min = {i: True for i in V_min_idx}
    sign = np.sign(V_s)
    zc_idx = np.where(np.diff(sign))[0]
    cycles = []
    i = 0
    while i < len(extrema) - 1:
        e1 = extrema[i]
        e2 = extrema[i+1]
        if not (e1 in is_max and e2 in is_min):
            i += 1
            continue
        z_candidates = zc_idx[(zc_idx > e1) & (zc_idx < e2)]
        if len(z_candidates) == 0:
            i += 1
            continue
        mid = (e1 + e2) / 2
        z = z_candidates[np.argmin(np.abs(z_candidates - mid))]
        cycles.append((e1, z, e2))
        i += 2
    V_peaks = np.array([c[0] for c in cycles])
    V_zeros = np.array([c[1] for c in cycles])
    V_troughs = np.array([c[2] for c in cycles])

    events = []

    for e, mp in zip(events, M_peaks[:len(events)]): 
        pass

    for mp in M_peaks:
        t_m = tarray[mp]
        t_candidates = T_peaks[np.abs(tarray[T_peaks] - t_m) < time_tol]
        if len(t_candidates) == 0:
            continue
        tp = t_candidates[np.argmin(np.abs(tarray[t_candidates] - t_m))]
        v_candidates = V_zeros[np.abs(tarray[V_zeros] - t_m) < time_tol]
        if len(v_candidates) == 0:
            continue
        vz = v_candidates[np.argmin(np.abs(tarray[v_candidates] - t_m))]
        v_peak_near = V_peaks[np.argmin(np.abs(tarray[V_peaks] - t_m))]
        # v_trough_near = V_troughs[np.argmin(np.abs(tarray[V_troughs] - t_m))]
        v_trough_near = V_troughs[np.where(V_troughs > v_peak_near)[0][0]] if np.any(V_troughs > v_peak_near) else v_peak_near

        dt_M_T = tarray[tp] - t_m
        dt_M_V0 = tarray[vz] - t_m
        dt_Vpk_Vtr = tarray[v_trough_near] - tarray[v_peak_near]

        events.append({
            "mp": mp,
            "tp": tp,
            "vz": vz,
            "vpk": v_peak_near,
            "vtr": v_trough_near,
            "t_event": t_m,
            "M_peak": M_s[mp],
            "T_min": T_s[tp],
            "V_max": V_s[v_peak_near],
            "V_min": V_s[v_trough_near],
            "dt_M_T": dt_M_T,
            "dt_M_V0": dt_M_V0,
            "dt_Vpk_Vtr": dt_Vpk_Vtr
        })
    event_times = np.array([e["t_event"] for e in events])
    dt_Vpk_Vtr = np.array([e["dt_Vpk_Vtr"] for e in events])

    return event_times, dt_Vpk_Vtr

def detect_events_ensemble_fast(Mdot, Vdot, Sdot, skip, tmax, t_lag, n_individuals, tune_params):
    n_ens = Mdot.shape[0]
    outputs = [[] for _ in range(2)]  

    for k in range(n_ens):
        res = detect_events_fast(Mdot[k], Vdot[k], Sdot[k], skip, tmax, t_lag, n_individuals, tune_params)
        for i, r in enumerate(res):
            outputs[i].append(r)

    outputs[0] = outputs[0] 

    return tuple(outputs)

################################################################################################
################ Step by Step
def obj_sbs_V0(dV0, dmu0, V_data, PR0, bpd, g, dt, loss_fn):
    t = np.arange(len(V_data)*300) * dt
    V_model = Vdot_f(t, dV0, dmu0, PR0, bpd, g)
    i_data = zero_crossing(gaussian_filter1d(V_data, sigma=15))
    i_model = zero_crossing_pos_to_neg(V_model)
    shift = i_model - i_data
    start_model = int(shift)
    end_model = int(shift + len(V_data))
    Vmodel_aligned = V_model[start_model:end_model]
    Vdata_aligned = V_data.copy()
    loss_V = loss_fn(Vdata_aligned, Vmodel_aligned)
    return loss_V

def opt_sbs_dV0(V_data, dmu0, bpd, g, dt, PR0, loss_type):
    loss_fn = get_loss_function(loss_type)
    result = minimize_scalar(lambda dV0: obj_sbs_V0(dV0, dmu0, V_data, PR0, bpd, g, dt, loss_fn), bounds=(0, 5), method='bounded' )
    return result.x, result.fun

def obj_sbs_dmu0(VR0, dV0, dmu0, PR0, M_data, bpd, g, dt, loss_fn):
    t = np.arange(len(M_data)*300) * dt
    M_model = Mdot_f(t, VR0, dV0, dmu0, PR0, bpd, g)
    i_data = np.argmax(gaussian_filter1d(M_data, sigma=15))
    i_model = np.argmax(M_model)
    shift = i_model - i_data
    start_model = int(shift)
    end_model = int(shift + len(M_data))
    Mmodel_aligned = M_model[start_model:end_model]
    Mdata_aligned = M_data.copy()
    loss_M = loss_fn(Mdata_aligned, Mmodel_aligned)
    return loss_M

def opt_sbs_VR0(M_data, dV0, dmu0, bpd, g, dt, PR0, loss_type):
    loss_fn = get_loss_function(loss_type)
    result = minimize_scalar( lambda VR0: obj_sbs_dmu0(VR0, dV0, dmu0, PR0, M_data, bpd, g, dt, loss_fn), bounds=(0, 20), method='bounded' )
    return result.x, result.fun

def event_sbs(Mdotdata, Vdotdata, Sdotdata, bpd, g, dt, PR0=0.01, loss_type="L2"):
    M_max, T_min, V_max, V_min, dt_V = get_exterma(Mdotdata, Vdotdata, Sdotdata, sigma=3, dt=dt)
    deltamu0_estimates = []
    deltamu0_1 = (-T_min*8/bpd)**(1/4)
    deltamu0_2 = ((V_max - V_min) * 3**(3/2) / bpd)**(1/3)
    deltamu0_3 = np.log(7+4*np.sqrt(3))/(bpd*dt_V)
    deltamu0_estimates.extend([deltamu0_1, deltamu0_2, deltamu0_3, np.mean([deltamu0_1, deltamu0_2]), np.mean([deltamu0_1, deltamu0_2, deltamu0_3])])

    results = []

    for i, dmu0 in enumerate(deltamu0_estimates):
        dV0, err = opt_sbs_dV0(Vdotdata, dmu0, bpd, g, dt, PR0, loss_type)
        VR0, err_VR0 = opt_sbs_VR0(Mdotdata, dV0, dmu0, bpd, g, dt, PR0, loss_type)
        print(f"dmu0: {dmu0:.4f} --> dV0: {dV0:.4f}, loss: {err:.4f} --> VR0: {VR0:.4f}, loss_VR0: {err_VR0:.4f}")
        results.append({"estimate_id": i, "dmu0": dmu0, "dV0": dV0, "loss_dV0": -err, "VR0": VR0, "loss_VR0": -err_VR0})

    return results

################################################################################################
# DV=0
def obj_vdot_dmu0(dmu0, V_data, PR0, bpd, g, dt, loss_fn):
    dV0 = 0; dmu0 = dmu0[0]
    t_model = np.arange(len(V_data)*15) * dt
    # t_model = np.arange(len(V_data)*4) * dt
    # t_model = np.arange(max([np.log(7 + 4*np.sqrt(3)) / (bpd * dmu0)*100*5, len(V_data)*10])) * dt 
    V_model = Vdot_f(t_model, dV0, dmu0, PR0, bpd, g)
    t0_data = zero_crossing(gaussian_filter1d(V_data, sigma=30))
    t0_model = zero_crossing_pos_to_neg(V_model)

    if t0_data is None or t0_model is None:
        return 1e10
    shift =  t0_model - t0_data
    if shift<0:
        return 1e10
    start_model = int(shift)
    end_model = int(shift + len(V_data))
    Vmodel_aligned = V_model[start_model:end_model]
    Vdata_aligned = V_data.copy()
    loss_V = loss_fn(Vdata_aligned, Vmodel_aligned)
    scale_V = np.std(Vdata_aligned) + 1e-8
    loss_total = (loss_V / scale_V)
    return loss_total

def obj_tdot_dmu0(dmu0, V_data, T_data, PR0, bpd, g, dt, loss_fn):
    dV0 = 0
    t_model = np.arange(len(T_data)*4) * dt
    t_model = np.arange(max([np.log(7 + 4*np.sqrt(3)) / (bpd * dmu0)*100*5, len(T_data)*10])) * dt 
    V_model = Vdot_f(t_model, dV0, dmu0, PR0, bpd, g)
    T_model = Tdot_f(t_model, dV0, dmu0, PR0, bpd)
    t0_data = zero_crossing(gaussian_filter1d(V_data, sigma=15)) 
    t0_model = zero_crossing_pos_to_neg(V_model)

    if t0_data is None or t0_model is None:
        return 1e10
    shift =  t0_model - t0_data
    if shift<0:
        return 1e10
    start_model = int(shift)
    end_model = int(shift + len(T_data))
    Tmodel_aligned = T_model[start_model:end_model]
    Tdata_aligned = T_data.copy()
    loss_T = loss_fn(Tdata_aligned, Tmodel_aligned)
    scale_T = np.std(Tdata_aligned) + 1e-8
    loss_total = (loss_T / scale_T)
    return loss_total


def obj_vdotTdot_dmu0(dmu0, V_data, T_data, PR0, bpd, g, dt, loss_fn):
    dV0 = 0; dmu0 = dmu0[0]
    t_model = np.arange(len(V_data)*15) * dt
    # t_model = np.arange(max([np.log(7 + 4*np.sqrt(3)) / (bpd * dmu0)*100*5, len(V_data)*10])) * dt 
    V_model = Vdot_f(t_model, dV0, dmu0, PR0, bpd, g)
    T_model = Tdot_f(t_model, dV0, dmu0, PR0, bpd)
    t0_data = zero_crossing(gaussian_filter1d(V_data, sigma=30)) 
    t0_model = zero_crossing_pos_to_neg(V_model)

    if t0_data is None or t0_model is None:
        return 1e10
    shift =  t0_model - t0_data
    if shift<0:
        return 1e10
    start_model = int(shift)
    end_model = int(shift + len(V_data))
    Vmodel_aligned = V_model[start_model:end_model]
    Vdata_aligned = V_data.copy()

    Tmodel_aligned = T_model[start_model:end_model]
    Tdata_aligned = T_data.copy()

    loss_V = loss_fn(Vdata_aligned, Vmodel_aligned)
    loss_t = loss_fn(Tdata_aligned, Tmodel_aligned)
    scale_V = np.std(Vdata_aligned) + 1e-8
    scale_T = np.std(Tdata_aligned) + 1e-8

    loss_total = (loss_V / scale_V) + (loss_t / scale_T)
    return loss_total

def opt_dv0_1(V_data, M_data, T_data, x0, bounds, PR0, bpd, g, dt, loss_type="L2"):
    loss_fn = get_loss_function(loss_type)
    result = minimize( lambda dmu0: obj_vdot_dmu0(dmu0, V_data, PR0, bpd, g, dt, loss_fn), x0, bounds=bounds, method='L-BFGS-B' )
    dmu0 = result.x[0]; total_loss = result.fun
    return dmu0, total_loss 

def opt_dv0_2(V_data, M_data, T_data, x0, bounds, PR0, bpd, g, dt, loss_type="L2"):
    loss_fn = get_loss_function(loss_type)
    result = minimize( lambda dmu0: obj_tdot_dmu0(dmu0, V_data, T_data, PR0, bpd, g, dt, loss_fn), x0, bounds=bounds, method='L-BFGS-B' )
    dmu0 = result.x[0]
    total_loss = result.fun
    return dmu0, total_loss 


def opt_dv0_3(V_data, M_data, T_data, x0, bounds, PR0, bpd, g, dt, loss_type="L2"):
    loss_fn = get_loss_function(loss_type)
    result = minimize( lambda dmu0: obj_vdotTdot_dmu0(dmu0, V_data, T_data, PR0, bpd, g, dt, loss_fn), x0, bounds=bounds, method='L-BFGS-B' )
    dmu0 = result.x[0]
    total_loss = result.fun
    return dmu0, total_loss 


def plot_comparison_dv0(V_data, T_data, dmu0, PR0, bpd, g, dt):
    dV0=0

    t_model = np.arange(len(V_data)*6) * dt
    V_model = Vdot_f(t_model, dV0, dmu0, PR0, bpd, g)
    T_model = Tdot_f(t_model, dV0, dmu0, PR0, bpd)

    t0_data = zero_crossing(gaussian_filter1d(V_data, sigma=15))
    t0_model = zero_crossing_pos_to_neg(V_model)

    print(t0_data, t0_model)

    shift =  t0_model - t0_data
    start_model = int(shift)
    end_model = int(shift + len(V_data))

    Vmodel_aligned = V_model[start_model:end_model]
    Tmodel_aligned = T_model[start_model:end_model]
    t_data = np.arange(len(V_data)) * 0.01
    t_model_aligned = t_model - shift * 0.01

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(t_data, V_data, label="V data")
    axs[0].plot(t_model_aligned, V_model, label="V model")
    axs[0].legend()
    axs[1].plot(t_data, T_data, label="T data")
    axs[1].plot(t_model_aligned, T_model, label="T model")
    axs[1].legend()
    plt.show()




########################################################################################################################################
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def centered_moving_average(arr, window, axis=-1):
    """
    Parameters
    ----------
    arr : np.ndarray
        Input array of any shape.
    window : int
        Size of the moving average window (must be odd).
    axis : int
        Axis along which to compute the moving average.
    Returns
    -------
    np.ndarray
        Array of the same shape as arr, with centered moving averages.
        Edge points are averaged with the available data (no padding).
    """
    arr = np.asarray(arr)
    if window % 2 == 0:
        raise ValueError("Window size must be odd for a centered average.")

    # Move the target axis to the last dimension
    arr_moved = np.moveaxis(arr, axis, -1)
    N = arr_moved.shape[-1]
    half = window // 2

    # Output array
    out = np.zeros_like(arr_moved, dtype=float)

    # Compute moving average for each index
    for t in range(N):
        start = max(0, t - half)
        end = min(N, t + half + 1)
        out[..., t] = arr_moved[..., start:end].mean(axis=-1)

    # Move axis back to original position
    return np.moveaxis(out, -1, axis)

########################################################################################################################################
def dattonpy(nansa, nlist, blist, dlist, tmax, transient, t_lag, base_dir):
    if transient[0] ==0: skip = int(tmax * transient[1] / 100)
    elif transient[0] ==1: skip = int(transient[1])
    indices = np.arange(skip, tmax, t_lag) 
    shape = [nansa] + [len(nlist)] + [len(blist)] + [len(dlist)] + [len(indices)]
    shape = tuple(shape)
    ALL_skw = np.memmap(os.path.join(base_dir, "ALL_skw_ansa.dat"), dtype='float64', mode='r', shape=shape)
    ALL_std = np.memmap(os.path.join(base_dir, "ALL_std_ansa.dat"), dtype='float64', mode='r', shape=shape)
    ALL_mean = np.memmap(os.path.join(base_dir, "ALL_mean_trait_values_ansa.dat"), dtype='float64', mode='r', shape=shape)
    np.save(os.path.join(base_dir, f"ALL_skw.npy"), ALL_skw)
    np.save(os.path.join(base_dir, f"ALL_std.npy"), ALL_std)
    np.save(os.path.join(base_dir, f"ALL_mean_trait_values.npy"), ALL_mean)



