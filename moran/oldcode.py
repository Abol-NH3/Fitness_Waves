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
import os, json, time, gc, io, contextlib, math
from numba import set_num_threads, get_num_threads
from operator import mod
from itertools import product
from statsmodels.tsa.stattools import adfuller
from itertools import permutations
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
import plotly.graph_objects as go
import glob
import imageio.v2 as imageio
from scipy.integrate import solve_ivp
from math import comb

import hints
from scipy.interpolate import interp1d



@njit()
def weighted_choice(weights):
    cdf = np.cumsum(weights)
    r = np.random.rand() * cdf[-1]
    return np.searchsorted(cdf, r)

########################################################################################################################################



@njit()
def Quadratic_simulate_evolution_clip_count_effectiveM(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values):
    n_individuals = len(trait_values)
    skwl = np.zeros(tmax)
    stdl = np.zeros(tmax)
    mean_trait_values = np.zeros(tmax)
    cum_mean_trait_value=0
    b_clip_count = np.zeros(tmax)
    d_clip_count = np.zeros(tmax)
    b_clip_mass = np.zeros(tmax)
    d_clip_mass = np.zeros(tmax)
    mu_b = np.zeros(tmax)
    mu_d = np.zeros(tmax)

    for t in range(1, tmax):
        tv2 = trait_values**2
        wb = 1 + b1_rate * trait_values + b2_rate * tv2;  wb_neg_mass = np.sum(np.maximum(-(wb), 0))
        wd = 1 - d1_rate * trait_values - d2_rate * tv2;  wd_neg_mass = np.sum(np.maximum(-(wd), 0))
        wb_eff = np.where(wb > 0.0, wb, 0.0);        wb_eff_mass = np.sum(wb_eff)
        wd_eff = np.where(wd > 0.0, wd, 0.0);        wd_eff_mass = np.sum(wd_eff)

        if b1_rate != 0.0:
            tv_b_cut = np.where(trait_values < -1.0/b1_rate, trait_values, np.nan)
        else:
            tv_b_cut = np.full_like(trait_values, np.nan)

        if d1_rate != 0.0:
            tv_d_cut = np.where(trait_values >  1.0/d1_rate, trait_values, np.nan)
        else:
            tv_d_cut = np.full_like(trait_values, np.nan)
        
        b_clip_count[t] = np.sum(wb < 0) / n_individuals
        d_clip_count[t] = np.sum(wd < 0) / n_individuals
        b_clip_mass[t] = wb_neg_mass / (wb_eff_mass+wb_neg_mass)
        d_clip_mass[t] = wd_neg_mass / (wd_eff_mass+wd_neg_mass)

        mu_d[t] = np.nansum(1/tv_d_cut)
        mu_b[t] = np.nansum(1/np.abs(tv_b_cut))

        birth_weight = np.clip(wb, 0, np.inf)  
        death_weight = np.clip(wd, 0, np.inf) 
        indice_birth = weighted_choice(birth_weight)
        indice_death = weighted_choice(death_weight)
        birth_trait = trait_values[indice_birth] + np.random.normal(0, 1)
        trait_values[indice_death] = birth_trait 
        current_mean_trait_value = np.mean(trait_values)
        cum_mean_trait_value += current_mean_trait_value
        trait_values -= current_mean_trait_value
        mean_trait_values[t] = cum_mean_trait_value
        stdl[t] = np.std(trait_values)
        if stdl[t] == 0: skwl[t] = 0
        else: skwl[t] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * stdl[t]**3)
    stdl[0] = 1e-8;                 skwl[0] = 0
    return mean_trait_values[indices], skwl[indices], stdl[indices], b_clip_count[indices], d_clip_count[indices], b_clip_mass[indices], d_clip_mass[indices], mu_b[indices], mu_d[indices]

########################################################################################################################################
@njit()
def Quadratic_simulate_evolution(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values):
    n_individuals = len(trait_values)
    skwl = np.zeros(tmax)
    stdl = np.zeros(tmax)
    mean_trait_values = np.zeros(tmax)
    cum_mean_trait_value=0
    for t in range(1, tmax):
        birth_weight = np.clip(1 + b1_rate * trait_values + b2_rate * trait_values**2, 0, np.inf)
        death_weight = np.clip(1 - d1_rate * trait_values - d2_rate * trait_values**2, 0, np.inf)
        indice_birth = weighted_choice(birth_weight)
        indice_death = weighted_choice(death_weight)
        birth_trait = trait_values[indice_birth] + np.random.normal(0, 1)
        trait_values[indice_death] = birth_trait 
        current_mean_trait_value = np.mean(trait_values)
        cum_mean_trait_value += current_mean_trait_value
        trait_values -= current_mean_trait_value
        mean_trait_values[t] = cum_mean_trait_value
        stdl[t] = np.std(trait_values)
        if stdl[t] == 0: skwl[t] = 0
        else: skwl[t] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * stdl[t]**3)
    stdl[0] = 1e-8;                 skwl[0] = 0
    return mean_trait_values[indices], skwl[indices], stdl[indices]

def Quadratic_analysis(nlist, b1list, b2list, d1list, d2list, tmax, transient, t_lag, save_dir, nansa=10, n_jobs=6):
    if transient[0] ==0: skip = int(tmax * transient[1] / 100)
    elif transient[0] ==1: skip = int(transient[1])
    indices = np.arange(skip, tmax, t_lag) 
    combinations = list(product(*[nlist] + [b1list] + [b2list] + [d1list] + [d2list]))
    n_timepoints = len(indices)
    n_combos = len(combinations)
    n_timepoints = len(indices)

    def process_one_combo(i):   
        params = combinations[i]
        n_individuals = params[0];         b1_rate = params[1];        b2_rate = params[2];        d1_rate = params[3];        d2_rate = params[4]
        ALL_skw = np.zeros((nansa, n_timepoints));        ALL_std = np.zeros((nansa, n_timepoints));        ALL_mean = np.zeros((nansa, n_timepoints))

        for j in range(nansa):
            trait_values = np.zeros(n_individuals)
            mean_trait_values, skwl, stdl = Quadratic_simulate_evolution(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values)
            ALL_skw[j] = skwl
            ALL_std[j] = stdl
            ALL_mean[j] = mean_trait_values

        skwl = np.copy(ALL_skw)
        stdl = np.copy(ALL_std)
        mean_trait_values = np.copy(ALL_mean)
        
        Amp_2D = np.sqrt(skwl**2 + stdl**2)
        Phase_2D = np.arctan2(skwl - np.mean(skwl, axis=1)[:, None], stdl - np.mean(stdl, axis=1)[:, None])
        Freq_2D = np.diff(np.unwrap(Phase_2D, axis=1), axis=1)
        Freq_2D = np.concatenate([Freq_2D[:, :1], Freq_2D], axis=1)/(t_lag)*n_individuals

        std_Hilbert = hilbert(stdl, axis=1)
        Amp_Hil_std = np.abs(std_Hilbert)
        std_Hilbert = hilbert(stdl - np.mean(stdl, axis=1)[:, None], axis=1)
        Phase_Hil_std = np.angle(std_Hilbert)
        Freq_Hil_std = np.diff(np.unwrap(Phase_Hil_std, axis=1), axis=1)
        Freq_Hil_std = np.concatenate([Freq_Hil_std[:, :1], Freq_Hil_std], axis=1)/(t_lag)*n_individuals

        skw_Hilbert = hilbert(skwl, axis=1)
        Amp_Hil_skw = np.abs(skw_Hilbert)
        skw_Hilbert = hilbert(skwl - np.mean(skwl, axis=1)[:, None], axis=1)
        Phase_Hil_skw = np.angle(skw_Hilbert)
        Freq_Hil_skw = np.diff(np.unwrap(Phase_Hil_skw, axis=1), axis=1)
        Freq_Hil_skw = np.concatenate([Freq_Hil_skw[:, :1], Freq_Hil_skw], axis=1)/(t_lag)*n_individuals

        Mean_ts = mean_trait_values - mean_trait_values[:, 0][:, None]
        dt = 1
        T = Mean_ts.shape[1]
        ts = np.arange(T) * t_lag
        Mean_Slope  = np.zeros(nansa)
        Mean_Intercept = np.zeros(nansa)
        for i in range(nansa):
            Mean_Slope[i], Mean_Intercept[i] = np.polyfit(ts, Mean_ts[i], 1)

        rtspeed = np.diff(mean_trait_values, axis=1)
        rtspeed = np.concatenate([rtspeed[:, :1], rtspeed], axis=1)
        MVBD = np.nanmean(rtspeed / (stdl**2 * ( ( b1_rate + 2*b2_rate*mean_trait_values ) + ( d1_rate + 2*d2_rate*mean_trait_values ) )), axis=1)

        metadata = {
            "N": int(n_individuals),
            "b1": float(b1_rate),
            "b2": float(b2_rate),
            "d1": float(d1_rate),
            "d2": float(d2_rate)
        }
        metadata["Avg_Skw"] = np.mean(np.mean(skwl, axis=1))
        metadata["Avg_Std"] = np.mean(np.mean(stdl, axis=1))
        metadata["Avg_Var"] = np.mean(np.mean(stdl**2, axis=1))
        metadata["Avg_Amp_2D"] = np.mean(np.mean(Amp_2D, axis=1))
        metadata["Avg_Phase_2D"] = np.mean(np.mean(Phase_2D, axis=1))
        metadata["Avg_Freq_2D"] = np.mean(np.mean(Freq_2D, axis=1))
        metadata["Avg_Abs_Freq_2D"] = np.mean(np.mean(np.abs(Freq_2D), axis=1))
        metadata["Std_Freq_2D"] = np.mean(np.std(Freq_2D, axis=1))
        metadata["Avg_Amp_Hil_std"] = np.mean(np.mean(Amp_Hil_std, axis=1))
        metadata["Avg_Phase_Hil_std"] = np.mean(np.mean(Phase_Hil_std, axis=1))
        metadata["Avg_Freq_Hil_std"] = np.mean(np.mean(Freq_Hil_std, axis=1))
        metadata["Avg_Abs_Freq_Hil_std"] = np.mean(np.mean(np.abs(Freq_Hil_std), axis=1))
        metadata["Std_Freq_Hil_std"] = np.mean(np.std(Freq_Hil_std, axis=1))
        metadata["Avg_Amp_Hil_skw"] = np.mean(np.mean(Amp_Hil_skw, axis=1))
        metadata["Avg_Phase_Hil_skw"] = np.mean(np.mean(Phase_Hil_skw, axis=1))
        metadata["Avg_Freq_Hil_skw"] = np.mean(np.mean(Freq_Hil_skw, axis=1))
        metadata["Avg_Abs_Freq_Hil_skw"] = np.mean(np.mean(np.abs(Freq_Hil_skw), axis=1))
        metadata["Std_Freq_Hil_skw"] = np.mean(np.std(Freq_Hil_skw, axis=1))
        metadata["final_mean_trait"] = np.mean(mean_trait_values[:, -1])
        metadata["Mean_Slope"] = np.mean(Mean_Slope) * n_individuals
        metadata["Mean_Intercept"] = np.mean(Mean_Intercept) * n_individuals
        metadata["MVbd"] = np.mean(MVBD)/t_lag * n_individuals

        if nansa>1 :
            metadata["Avg_Skw_Err"] = np.std(np.mean(skwl, axis=1))/np.sqrt(nansa)
            metadata["Avg_Std_Err"] = np.std(np.mean(stdl, axis=1))/np.sqrt(nansa)
            metadata["Avg_Var_Err"] = np.std(np.mean(stdl**2, axis=1))/np.sqrt(nansa)          
            metadata["Avg_Amp_2D_Err"] = np.std(np.mean(Amp_2D, axis=1))/np.sqrt(nansa)
            metadata["Avg_Phase_2D_Err"] = np.std(np.mean(Phase_2D, axis=1))/np.sqrt(nansa)
            metadata["Avg_Freq_2D_Err"] = np.std(np.mean(Freq_2D, axis=1))/np.sqrt(nansa)
            metadata["Avg_Abs_Freq_2D_Err"] = np.std(np.mean(np.abs(Freq_2D), axis=1))/np.sqrt(nansa)
            metadata["Std_Freq_2D_Err"] = np.std(np.std(Freq_2D, axis=1))/np.sqrt(nansa)
            metadata["Avg_Amp_Hil_std_Err"] = np.std(np.mean(Amp_Hil_std, axis=1))/np.sqrt(nansa)
            metadata["Avg_Phase_Hil_std_Err"] = np.std(np.mean(Phase_Hil_std, axis=1))/np.sqrt(nansa)
            metadata["Avg_Freq_Hil_std_Err"] = np.std(np.mean(Freq_Hil_std, axis=1))/np.sqrt(nansa)
            metadata["Avg_Abs_Freq_Hil_std_Err"] = np.std(np.mean(np.abs(Freq_Hil_std), axis=1))/np.sqrt(nansa)
            metadata["Std_Freq_Hil_std_Err"] = np.std(np.std(Freq_Hil_std, axis=1))/np.sqrt(nansa)
            metadata["Avg_Amp_Hil_skw_Err"] = np.std(np.mean(Amp_Hil_skw, axis=1))/np.sqrt(nansa)
            metadata["Avg_Phase_Hil_skw_Err"] = np.std(np.mean(Phase_Hil_skw, axis=1))/np.sqrt(nansa)
            metadata["Avg_Freq_Hil_skw_Err"] = np.std(np.mean(Freq_Hil_skw, axis=1))/np.sqrt(nansa)
            metadata["Avg_Abs_Freq_Hil_skw_Err"] = np.std(np.mean(np.abs(Freq_Hil_skw), axis=1))/np.sqrt(nansa)
            metadata["Std_Freq_Hil_skw_Err"] = np.std(np.std(Freq_Hil_skw, axis=1))/np.sqrt(nansa)
            metadata["final_mean_trait_Err"] = np.std(mean_trait_values[:, -1])/np.sqrt(nansa)
            metadata["Mean_Slope_Err"] = np.std(Mean_Slope)/np.sqrt(nansa) * n_individuals
            metadata["Mean_Intercept_Err"] = np.std(Mean_Intercept)/np.sqrt(nansa) * n_individuals
            metadata["MVbd_Err"] = np.std(MVBD)/np.sqrt(nansa)/t_lag * n_individuals


        return metadata

    results = Parallel(n_jobs=n_jobs, verbose=0)( delayed(process_one_combo)(i) for i in tqdm(range(n_combos), total=len(combinations), desc="Simulating", ncols=100) )    # (n_jobs=n_jobs, backend='loky')
    summary_path = os.path.join(save_dir, f"{t_lag}_ALL_summaries.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n✅ All summaries saved to {summary_path}")

########################################################################################################################################
@njit()
def Quadratic_simulate_evolution_dist(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values):
    n_individuals = len(trait_values)
    skwl = np.zeros(tmax)
    stdl = np.zeros(tmax)
    mean_trait_values = np.zeros(tmax)
    all_trait_values = np.zeros((tmax, n_individuals))
    cum_mean_trait_value=0
    for t in range(1, tmax):
        birth_weight = np.clip(1 + b1_rate * trait_values + b2_rate * trait_values**2, 0, np.inf)
        death_weight = np.clip(1 - d1_rate * trait_values - d2_rate * trait_values**2, 0, np.inf)
        indice_birth = weighted_choice(birth_weight)
        indice_death = weighted_choice(death_weight)
        birth_trait = trait_values[indice_birth] + np.random.normal(0, 1)
        trait_values[indice_death] = birth_trait 
        current_mean_trait_value = np.mean(trait_values)
        cum_mean_trait_value += current_mean_trait_value
        trait_values -= current_mean_trait_value
        all_trait_values[t] = trait_values
        mean_trait_values[t] = cum_mean_trait_value
        stdl[t] = np.std(trait_values)
        if stdl[t] == 0: skwl[t] = 0
        else: skwl[t] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * stdl[t]**3)
    stdl[0] = 1e-8;                 skwl[0] = 0
    return all_trait_values[indices], mean_trait_values[indices], skwl[indices], stdl[indices]

# Quads Clipp count
########################################################################################################################################
@njit()
def Quadratic_simulate_evolution_clip_count(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values):
    n_individuals = len(trait_values)
    skwl = np.zeros(tmax)
    stdl = np.zeros(tmax)
    mean_trait_values = np.zeros(tmax)
    cum_mean_trait_value=0
    n_birth_clipped = np.zeros(tmax)
    n_death_clipped = np.zeros(tmax)
    b_clip_mass = np.zeros(tmax)
    d_clip_mass = np.zeros(tmax)
    b_clip_mass_scaled = np.zeros(tmax)
    d_clip_mass_scaled = np.zeros(tmax)
    for t in range(1, tmax):
        n_birth_clipped[t] = np.sum(1 + b1_rate * trait_values + b2_rate * trait_values**2 < 0)
        n_death_clipped[t] = np.sum(1 - d1_rate * trait_values - d2_rate * trait_values**2 < 0)
        b_clip_mass[t] = np.sum(np.maximum(-(1 + b1_rate * trait_values + b2_rate * trait_values**2), 0))
        d_clip_mass[t] = np.sum(np.maximum(-(1 - d1_rate * trait_values - d2_rate * trait_values**2), 0))
        if n_birth_clipped[t] > 0:
            b_clip_mass_scaled[t] = b_clip_mass[t] / (n_birth_clipped[t])
        if n_death_clipped[t] > 0:
            d_clip_mass_scaled[t] = d_clip_mass[t] / (n_death_clipped[t])
        birth_weight = np.clip(1 + b1_rate * trait_values + b2_rate * trait_values**2, 0, np.inf)  
        death_weight = np.clip(1 - d1_rate * trait_values - d2_rate * trait_values**2, 0, np.inf) 
        indice_birth = weighted_choice(birth_weight)
        indice_death = weighted_choice(death_weight)
        birth_trait = trait_values[indice_birth] + np.random.normal(0, 1)
        trait_values[indice_death] = birth_trait 
        current_mean_trait_value = np.mean(trait_values)
        cum_mean_trait_value += current_mean_trait_value
        trait_values -= current_mean_trait_value
        mean_trait_values[t] = cum_mean_trait_value
        stdl[t] = np.std(trait_values)
        if stdl[t] == 0: skwl[t] = 0
        else: skwl[t] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * stdl[t]**3)
    stdl[0] = 1e-8;                 skwl[0] = 0
    return mean_trait_values[indices], skwl[indices], stdl[indices], n_birth_clipped[indices], n_death_clipped[indices], b_clip_mass[indices], d_clip_mass[indices], b_clip_mass_scaled[indices], d_clip_mass_scaled[indices]

def Quadratic_analysis_clip_count(nlist, b1list, b2list, d1list, d2list, tmax, transient, t_lag, save_dir, nansa=10, n_jobs=6):
    if transient[0] ==0: skip = int(tmax * transient[1] / 100)
    elif transient[0] ==1: skip = int(transient[1])
    indices = np.arange(skip, tmax, t_lag) 
    combinations = list(product(*[nlist] + [b1list] + [b2list] + [d1list] + [d2list]))
    n_timepoints = len(indices)
    n_combos = len(combinations)
    n_timepoints = len(indices)

    def process_one_combo(i):   
        params = combinations[i]
        n_individuals = params[0];         b1_rate = params[1];        b2_rate = params[2];        d1_rate = params[3];        d2_rate = params[4]
        ALL_skw = np.zeros((nansa, n_timepoints));        ALL_std = np.zeros((nansa, n_timepoints));        ALL_mean = np.zeros((nansa, n_timepoints));        ALL_n_birth_clipped = np.zeros((nansa, n_timepoints));        ALL_n_death_clipped = np.zeros((nansa, n_timepoints))
        ALL_b_clip_mass = np.zeros((nansa, n_timepoints));        ALL_d_clip_mass = np.zeros((nansa, n_timepoints));        ALL_b_clip_mass_scaled = np.zeros((nansa, n_timepoints));        ALL_d_clip_mass_scaled = np.zeros((nansa, n_timepoints))
        for j in range(nansa):
            trait_values = np.zeros(n_individuals)
            mean_trait_values, skwl, stdl, n_birth_clipped, n_death_clipped, b_clip_mass, d_clip_mass, b_clip_mass_scaled, d_clip_mass_scaled = Quadratic_simulate_evolution_clip_count(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values)
            ALL_skw[j] = skwl
            ALL_std[j] = stdl
            ALL_mean[j] = mean_trait_values
            ALL_n_birth_clipped[j] = n_birth_clipped
            ALL_n_death_clipped[j] = n_death_clipped
            ALL_b_clip_mass[j] = b_clip_mass
            ALL_d_clip_mass[j] = d_clip_mass
            ALL_b_clip_mass_scaled[j] = b_clip_mass_scaled
            ALL_d_clip_mass_scaled[j] = d_clip_mass_scaled

        skwl = np.copy(ALL_skw)
        stdl = np.copy(ALL_std)
        mean_trait_values = np.copy(ALL_mean)
        
        Amp_2D = np.sqrt(skwl**2 + stdl**2)
        Phase_2D = np.arctan2(skwl - np.mean(skwl, axis=1)[:, None], stdl - np.mean(stdl, axis=1)[:, None])
        Freq_2D = np.diff(np.unwrap(Phase_2D, axis=1), axis=1)
        Freq_2D = np.concatenate([Freq_2D[:, :1], Freq_2D], axis=1)/(t_lag)*n_individuals

        std_Hilbert = hilbert(stdl, axis=1)
        Amp_Hil_std = np.abs(std_Hilbert)
        std_Hilbert = hilbert(stdl - np.mean(stdl, axis=1)[:, None], axis=1)
        Phase_Hil_std = np.angle(std_Hilbert)
        Freq_Hil_std = np.diff(np.unwrap(Phase_Hil_std, axis=1), axis=1)
        Freq_Hil_std = np.concatenate([Freq_Hil_std[:, :1], Freq_Hil_std], axis=1)/(t_lag)*n_individuals

        skw_Hilbert = hilbert(skwl, axis=1)
        Amp_Hil_skw = np.abs(skw_Hilbert)
        skw_Hilbert = hilbert(skwl - np.mean(skwl, axis=1)[:, None], axis=1)
        Phase_Hil_skw = np.angle(skw_Hilbert)
        Freq_Hil_skw = np.diff(np.unwrap(Phase_Hil_skw, axis=1), axis=1)
        Freq_Hil_skw = np.concatenate([Freq_Hil_skw[:, :1], Freq_Hil_skw], axis=1)/(t_lag)*n_individuals

        Mean_ts = mean_trait_values - mean_trait_values[:, 0][:, None]
        dt = 1
        T = Mean_ts.shape[1]
        ts = np.arange(T) * t_lag
        Mean_Slope  = np.zeros(nansa)
        Mean_Intercept = np.zeros(nansa)
        for i in range(nansa):
            Mean_Slope[i], Mean_Intercept[i] = np.polyfit(ts, Mean_ts[i], 1)

        rtspeed = np.diff(mean_trait_values, axis=1)
        rtspeed = np.concatenate([rtspeed[:, :1], rtspeed], axis=1)
        MVBD = np.nanmean(rtspeed / (stdl**2 * ( ( b1_rate + 2*b2_rate*mean_trait_values ) + ( d1_rate + 2*d2_rate*mean_trait_values ) )), axis=1)

        metadata = {
            "N": int(n_individuals),
            "b1": float(b1_rate),
            "b2": float(b2_rate),
            "d1": float(d1_rate),
            "d2": float(d2_rate)
        }
        metadata["Avg_bclipped"] = np.mean(np.mean(ALL_n_birth_clipped, axis=1))
        metadata["Avg_dclipped"] = np.mean(np.mean(ALL_n_death_clipped, axis=1))
        metadata["Avg_bdclipped"] = np.mean(np.mean(ALL_n_birth_clipped + ALL_n_death_clipped, axis=1))

        metadata["Avg_b_clip_mass"] = np.mean(np.mean(ALL_b_clip_mass, axis=1))
        metadata["Avg_d_clip_mass"] = np.mean(np.mean(ALL_d_clip_mass, axis=1))
        metadata["Avg_bd_clip_mass"] = np.mean(np.mean(ALL_b_clip_mass + ALL_d_clip_mass, axis=1))

        metadata["Avg_b_clip_mass_scaled"] = np.mean(np.mean(ALL_b_clip_mass_scaled, axis=1))
        metadata["Avg_d_clip_mass_scaled"] = np.mean(np.mean(ALL_d_clip_mass_scaled, axis=1))
        metadata["Avg_bd_clip_mass_scaled"] = np.mean(np.mean(ALL_b_clip_mass_scaled + ALL_d_clip_mass_scaled, axis=1))

        metadata["Avg_Skw"] = np.mean(np.mean(skwl, axis=1))
        metadata["Avg_Std"] = np.mean(np.mean(stdl, axis=1))
        metadata["Avg_Var"] = np.mean(np.mean(stdl**2, axis=1))
        metadata["Avg_Amp_2D"] = np.mean(np.mean(Amp_2D, axis=1))
        metadata["Avg_Phase_2D"] = np.mean(np.mean(Phase_2D, axis=1))
        metadata["Avg_Freq_2D"] = np.mean(np.mean(Freq_2D, axis=1))
        metadata["Avg_Abs_Freq_2D"] = np.mean(np.mean(np.abs(Freq_2D), axis=1))
        metadata["Std_Freq_2D"] = np.mean(np.std(Freq_2D, axis=1))
        metadata["Avg_Amp_Hil_std"] = np.mean(np.mean(Amp_Hil_std, axis=1))
        metadata["Avg_Phase_Hil_std"] = np.mean(np.mean(Phase_Hil_std, axis=1))
        metadata["Avg_Freq_Hil_std"] = np.mean(np.mean(Freq_Hil_std, axis=1))
        metadata["Avg_Abs_Freq_Hil_std"] = np.mean(np.mean(np.abs(Freq_Hil_std), axis=1))
        metadata["Std_Freq_Hil_std"] = np.mean(np.std(Freq_Hil_std, axis=1))
        metadata["Avg_Amp_Hil_skw"] = np.mean(np.mean(Amp_Hil_skw, axis=1))
        metadata["Avg_Phase_Hil_skw"] = np.mean(np.mean(Phase_Hil_skw, axis=1))
        metadata["Avg_Freq_Hil_skw"] = np.mean(np.mean(Freq_Hil_skw, axis=1))
        metadata["Avg_Abs_Freq_Hil_skw"] = np.mean(np.mean(np.abs(Freq_Hil_skw), axis=1))
        metadata["Std_Freq_Hil_skw"] = np.mean(np.std(Freq_Hil_skw, axis=1))
        metadata["final_mean_trait"] = np.mean(mean_trait_values[:, -1])
        metadata["Mean_Slope"] = np.mean(Mean_Slope) * n_individuals
        metadata["Mean_Intercept"] = np.mean(Mean_Intercept) * n_individuals
        metadata["MVbd"] = np.mean(MVBD)/t_lag * n_individuals

        if nansa>1 :
            metadata["Avg_bclipped_Err"] = np.std(np.mean(ALL_n_birth_clipped, axis=1))/np.sqrt(nansa)
            metadata["Avg_dclipped_Err"] = np.std(np.mean(ALL_n_death_clipped, axis=1))/np.sqrt(nansa)
            metadata["Avg_bdclipped_Err"] = np.std(np.mean(ALL_n_birth_clipped + ALL_n_death_clipped, axis=1))/np.sqrt(nansa)

            metadata["Avg_Skw_Err"] = np.std(np.mean(skwl, axis=1))/np.sqrt(nansa)
            metadata["Avg_Std_Err"] = np.std(np.mean(stdl, axis=1))/np.sqrt(nansa)
            metadata["Avg_Var_Err"] = np.std(np.mean(stdl**2, axis=1))/np.sqrt(nansa)          
            metadata["Avg_Amp_2D_Err"] = np.std(np.mean(Amp_2D, axis=1))/np.sqrt(nansa)
            metadata["Avg_Phase_2D_Err"] = np.std(np.mean(Phase_2D, axis=1))/np.sqrt(nansa)
            metadata["Avg_Freq_2D_Err"] = np.std(np.mean(Freq_2D, axis=1))/np.sqrt(nansa)
            metadata["Avg_Abs_Freq_2D_Err"] = np.std(np.mean(np.abs(Freq_2D), axis=1))/np.sqrt(nansa)
            metadata["Std_Freq_2D_Err"] = np.std(np.std(Freq_2D, axis=1))/np.sqrt(nansa)
            metadata["Avg_Amp_Hil_std_Err"] = np.std(np.mean(Amp_Hil_std, axis=1))/np.sqrt(nansa)
            metadata["Avg_Phase_Hil_std_Err"] = np.std(np.mean(Phase_Hil_std, axis=1))/np.sqrt(nansa)
            metadata["Avg_Freq_Hil_std_Err"] = np.std(np.mean(Freq_Hil_std, axis=1))/np.sqrt(nansa)
            metadata["Avg_Abs_Freq_Hil_std_Err"] = np.std(np.mean(np.abs(Freq_Hil_std), axis=1))/np.sqrt(nansa)
            metadata["Std_Freq_Hil_std_Err"] = np.std(np.std(Freq_Hil_std, axis=1))/np.sqrt(nansa)
            metadata["Avg_Amp_Hil_skw_Err"] = np.std(np.mean(Amp_Hil_skw, axis=1))/np.sqrt(nansa)
            metadata["Avg_Phase_Hil_skw_Err"] = np.std(np.mean(Phase_Hil_skw, axis=1))/np.sqrt(nansa)
            metadata["Avg_Freq_Hil_skw_Err"] = np.std(np.mean(Freq_Hil_skw, axis=1))/np.sqrt(nansa)
            metadata["Avg_Abs_Freq_Hil_skw_Err"] = np.std(np.mean(np.abs(Freq_Hil_skw), axis=1))/np.sqrt(nansa)
            metadata["Std_Freq_Hil_skw_Err"] = np.std(np.std(Freq_Hil_skw, axis=1))/np.sqrt(nansa)
            metadata["final_mean_trait_Err"] = np.std(mean_trait_values[:, -1])/np.sqrt(nansa)
            metadata["Mean_Slope_Err"] = np.std(Mean_Slope)/np.sqrt(nansa) * n_individuals
            metadata["Mean_Intercept_Err"] = np.std(Mean_Intercept)/np.sqrt(nansa) * n_individuals
            metadata["MVbd_Err"] = np.std(MVBD)/np.sqrt(nansa)/t_lag * n_individuals


        return metadata

    results = Parallel(n_jobs=n_jobs, verbose=0)( delayed(process_one_combo)(i) for i in tqdm(range(n_combos), total=len(combinations), desc="Simulating", ncols=100) )    # (n_jobs=n_jobs, backend='loky')
    summary_path = os.path.join(save_dir, f"{t_lag}_ALL_summaries.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n✅ All summaries saved to {summary_path}")























########################################################################################################################################

exit()





# Not tested
# single pass over trait values in each step
# No data save
# old version
@njit(fastmath=True, nogil=True)
def Quadratic_simulate_evolution_clip_count_effectiveM(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values):
    n_individuals = len(trait_values)
    
    skwl = np.zeros(tmax)
    stdl = np.zeros(tmax)
    mean_trait_values = np.zeros(tmax)
    cum_mean_trait_value=0
    b_clip_count = np.zeros(tmax)
    d_clip_count = np.zeros(tmax)
    b_clip_mass = np.zeros(tmax)
    d_clip_mass = np.zeros(tmax)

    for t in range(1, tmax):
        tv2 = trait_values**2
        wb = 1 + b1_rate * trait_values + b2_rate * tv2
        wd = 1 - d1_rate * trait_values - d2_rate * tv2

        sb = 0.0        
        sd = 0.0
        neg_b = 0.0        
        neg_d = 0.0
        cnt_b = 0       
        cnt_d = 0
        for i in range(n_individuals):
            if wb[i] > 0:
                sb += wb[i]
            else:
                neg_b -= wb[i]
                cnt_b += 1
                wb[i] = 0
            if wd[i] > 0:
                sd += wd[i]
            else:
                neg_d -= wd[i]
                cnt_d += 1
                wd[i] = 0
        b_clip_count[t] = cnt_b / n_individuals
        d_clip_count[t] = cnt_d / n_individuals
        b_clip_mass[t] = neg_b / (sb + neg_b)
        d_clip_mass[t] = neg_d / (sd + neg_d)
        cdf = np.cumsum(wb); r = np.random.rand() * cdf[-1]
        indice_birth = np.searchsorted(cdf, r)
        cdf = np.cumsum(wd); r = np.random.rand() * cdf[-1]
        indice_death = np.searchsorted(cdf, r)
        birth_trait = trait_values[indice_birth] + np.random.normal(0, 1)
        trait_values[indice_death] = birth_trait 
        current_mean_trait_value = np.mean(trait_values)
        cum_mean_trait_value += current_mean_trait_value
        trait_values -= current_mean_trait_value
        mean_trait_values[t] = cum_mean_trait_value

        stdl[t] = np.std(trait_values)
        if stdl[t] == 0: skwl[t] = 0
        else: skwl[t] = np.sum((trait_values) ** 3) / (n_individuals * stdl[t]**3)
    stdl[0] = 1e-8;                 skwl[0] = 0
    return mean_trait_values[indices], skwl[indices], stdl[indices], b_clip_count[indices], d_clip_count[indices], b_clip_mass[indices], d_clip_mass[indices]

def Quad_sim(nlist, b1list, b2list, d1list, d2list, tmax, transient, t_lag, save_dir, nansa=10, n_jobs=6):
    if transient[0] ==0: skip = int(tmax * transient[1] / 100)
    elif transient[0] ==1: skip = int(transient[1])
    indices = np.arange(skip, tmax, t_lag) 
    combinations = list(product(*[nlist] + [b1list] + [b2list] + [d1list] + [d2list]))
    n_timepoints = len(indices)
    # shape = tuple([nansa] + [len(nlist)] + [b1list] + [b2list] + [d1list] + [d2list] + [n_timepoints])
    shape = (    nansa,    len(nlist),    len(b1list),    len(b2list),    len(d1list),    len(d2list),    n_timepoints)


    ALL_skw  = np.memmap(f"{save_dir}/ALL_skw_ansa.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_std  = np.memmap(f"{save_dir}/ALL_std_ansa.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_mean = np.memmap(f"{save_dir}/ALL_mean_trait_values_ansa.dat", dtype='float64', mode='w+', shape=shape)
    ALL_b_clip_count = np.memmap(f"{save_dir}/ALL_b_clip_count.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_d_clip_count = np.memmap(f"{save_dir}/ALL_d_clip_count.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_b_clip_mass  = np.memmap(f"{save_dir}/ALL_b_clip_mass.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_d_clip_mass  = np.memmap(f"{save_dir}/ALL_d_clip_mass.dat",               dtype='float64', mode='w+', shape=shape)

    def run_parallel(i, params):
        n = params[0]
        b1_rate = params[1]
        b2_rate = params[2]
        d1_rate = params[3]
        d2_rate = params[4]
        idxs = np.unravel_index(i, shape[1:-1])
        for j in range(nansa):
            trait_values = np.zeros(n)
            mean_trait_values, skwl, stdl, n_birth_clipped, n_death_clipped, b_clip_mass, d_clip_mass = Quadratic_simulate_evolution_clip_count_effectiveM(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values)
            ALL_skw[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = skwl
            ALL_std[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = stdl
            ALL_mean[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = mean_trait_values
            ALL_b_clip_count[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = n_birth_clipped
            ALL_d_clip_count[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = n_death_clipped
            ALL_b_clip_mass[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]]  = b_clip_mass
            ALL_d_clip_mass[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]]  = d_clip_mass

    Parallel(n_jobs=n_jobs, backend="threading")(delayed(run_parallel)(i, params) for i, params in tqdm(enumerate(combinations), total=len(combinations), desc="Simulating", ncols=100))
    ALL_skw.flush()
    ALL_std.flush()
    ALL_mean.flush()
    ALL_b_clip_count.flush()
    ALL_d_clip_count.flush()
    ALL_b_clip_mass.flush()
    ALL_d_clip_mass.flush()
    print("✅ All data saved as memmap files in", save_dir)














# rnd ... in analysis function
def Con_Avg(x, y, nbins):
    mean_x = np.zeros(nbins)
    std_x = np.zeros(nbins)
    sem_x = np.zeros(nbins)
    counts = np.zeros(nbins)

    bins = np.linspace(np.min(y), np.max(y), nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for i in range(nbins):
        mask = (y >= bins[i]) & (y < bins[i + 1])
        if np.any(mask):
            xi = x[mask]
            counts[i] = len(xi)
            mean_x[i] = np.mean(xi)
            std_x[i] = np.std(xi, ddof=0) if counts[i] > 1 else 0.0
            sem_x[i] = std_x[i] / np.sqrt(counts[i])
        else:
            mean_x[i] = np.nan
            sem_x[i] = np.nan
            counts[i] = 0

    return bin_centers, mean_x, sem_x, counts

@njit()
def weighted_choice(weights):
    cdf = np.cumsum(weights)
    r = np.random.rand() * cdf[-1]
    return np.searchsorted(cdf, r)

@njit()
def Quadratic_simulate_evolution_clip_count_effectiveM(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values):
    n_individuals = len(trait_values)
    skwl = np.zeros(tmax)
    stdl = np.zeros(tmax)
    mean_trait_values = np.zeros(tmax)
    cum_mean_trait_value=0
    b_clip_count = np.zeros(tmax)
    d_clip_count = np.zeros(tmax)
    b_clip_mass = np.zeros(tmax)
    d_clip_mass = np.zeros(tmax)
    mu_b = np.zeros(tmax)
    mu_d = np.zeros(tmax)

    for t in range(1, tmax):
        tv2 = trait_values**2
        wb = 1 + b1_rate * trait_values + b2_rate * tv2;  wb_neg_mass = np.sum(np.maximum(-(wb), 0))
        wd = 1 - d1_rate * trait_values - d2_rate * tv2;  wd_neg_mass = np.sum(np.maximum(-(wd), 0))
        wb_eff = np.where(wb > 0.0, wb, 0.0);        wb_eff_mass = np.sum(wb_eff)
        wd_eff = np.where(wd > 0.0, wd, 0.0);        wd_eff_mass = np.sum(wd_eff)

        if b1_rate != 0.0:
            tv_b_cut = np.where(trait_values < -1.0/b1_rate, trait_values, np.nan)
        else:
            tv_b_cut = np.full_like(trait_values, np.nan)

        if d1_rate != 0.0:
            tv_d_cut = np.where(trait_values >  1.0/d1_rate, trait_values, np.nan)
        else:
            tv_d_cut = np.full_like(trait_values, np.nan)
        
        b_clip_count[t] = np.sum(wb < 0) / n_individuals
        d_clip_count[t] = np.sum(wd < 0) / n_individuals
        b_clip_mass[t] = wb_neg_mass / (wb_eff_mass+wb_neg_mass)
        d_clip_mass[t] = wd_neg_mass / (wd_eff_mass+wd_neg_mass)

        mu_d[t] = np.nansum(1/tv_d_cut)
        mu_b[t] = np.nansum(1/np.abs(tv_b_cut))

        birth_weight = np.clip(wb, 0, np.inf)  
        death_weight = np.clip(wd, 0, np.inf) 
        indice_birth = weighted_choice(birth_weight)
        indice_death = weighted_choice(death_weight)
        birth_trait = trait_values[indice_birth] + np.random.normal(0, 1)
        trait_values[indice_death] = birth_trait 
        current_mean_trait_value = np.mean(trait_values)
        cum_mean_trait_value += current_mean_trait_value
        trait_values -= current_mean_trait_value
        mean_trait_values[t] = cum_mean_trait_value
        stdl[t] = np.std(trait_values)
        if stdl[t] == 0: skwl[t] = 0
        else: skwl[t] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * stdl[t]**3)
    stdl[0] = 1e-8;                 skwl[0] = 0
    return mean_trait_values[indices], skwl[indices], stdl[indices], b_clip_count[indices], d_clip_count[indices], b_clip_mass[indices], d_clip_mass[indices], mu_b[indices], mu_d[indices]

def Quad_sim(nlist, b1list, b2list, d1list, d2list, tmax, transient, t_lag, save_dir, nansa=10, n_jobs=6):
    if transient[0] ==0: skip = int(tmax * transient[1] / 100)
    elif transient[0] ==1: skip = int(transient[1])
    indices = np.arange(skip, tmax, t_lag) 
    combinations = list(product(*[nlist] + [b1list] + [b2list] + [d1list] + [d2list]))
    n_timepoints = len(indices)
    shape = (nansa,    len(nlist),    len(b1list),    len(b2list),    len(d1list),    len(d2list),    n_timepoints)

    ALL_skw  = np.memmap(f"{save_dir}/ALL_skw_ansa.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_std  = np.memmap(f"{save_dir}/ALL_std_ansa.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_mean = np.memmap(f"{save_dir}/ALL_mean_trait_values_ansa.dat", dtype='float64', mode='w+', shape=shape)
    ALL_b_clip_count = np.memmap(f"{save_dir}/ALL_b_clip_count.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_d_clip_count = np.memmap(f"{save_dir}/ALL_d_clip_count.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_b_clip_mass  = np.memmap(f"{save_dir}/ALL_b_clip_mass.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_d_clip_mass  = np.memmap(f"{save_dir}/ALL_d_clip_mass.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_mu_b  = np.memmap(f"{save_dir}/ALL_mu_b.dat",               dtype='float64', mode='w+', shape=shape)
    ALL_mu_d  = np.memmap(f"{save_dir}/ALL_mu_d.dat",               dtype='float64', mode='w+', shape=shape)

    def run_parallel(i, params):
        n = params[0]
        b1_rate = params[1]
        b2_rate = params[2]
        d1_rate = params[3]
        d2_rate = params[4]
        idxs = np.unravel_index(i, shape[1:-1])
        for j in range(nansa):
            trait_values = np.zeros(n)
            mean_trait_values, skwl, stdl, n_birth_clipped, n_death_clipped, b_clip_mass, d_clip_mass, mu_b, mu_d = Quadratic_simulate_evolution_clip_count_effectiveM(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values)
            ALL_skw[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = skwl
            ALL_std[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = stdl
            ALL_mean[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = mean_trait_values
            ALL_b_clip_count[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = n_birth_clipped
            ALL_d_clip_count[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = n_death_clipped
            ALL_b_clip_mass[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]]  = b_clip_mass
            ALL_d_clip_mass[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]]  = d_clip_mass
            ALL_mu_b[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = mu_b
            ALL_mu_d[j, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = mu_d

    Parallel(n_jobs=n_jobs)(delayed(run_parallel)(i, params) for i, params in tqdm(enumerate(combinations), total=len(combinations), desc="Simulating", ncols=100))   # , backend="threading"
    ALL_skw.flush()
    ALL_std.flush()
    ALL_mean.flush()
    ALL_b_clip_count.flush()
    ALL_d_clip_count.flush()
    ALL_b_clip_mass.flush()
    ALL_d_clip_mass.flush()
    ALL_mu_b.flush()
    ALL_mu_d.flush()
    print("✅ All data saved as memmap files in", save_dir)

def Load_data_Main_calc_Quadratic_analysis_clip_count_effectiveM(nlist, b1list, b2list, d1list, d2list, tmax, transient, t_lag, t_lag2, save_dir, nansa=10, n_jobs=6, Load=True):
    if transient[0] ==0: skip = int(tmax * transient[1] / 100)
    elif transient[0] ==1: skip = int(transient[1])
    indices = np.arange(skip, tmax, t_lag) 
    combinations = list(product(*[nlist] + [b1list] + [b2list] + [d1list] + [d2list]))
    n_timepoints = len(indices)
    n_combos = len(combinations)
    n_timepoints = len(indices)
    shape = (    nansa,    len(nlist),    len(b1list),    len(b2list),    len(d1list),    len(d2list),    n_timepoints)


    ALL_skw = np.memmap(os.path.join(save_dir, "ALL_skw_ansa.dat"), dtype='float64', mode='r', shape=shape)
    ALL_std = np.memmap(os.path.join(save_dir, "ALL_std_ansa.dat"), dtype='float64', mode='r', shape=shape)
    ALL_mean = np.memmap(os.path.join(save_dir, "ALL_mean_trait_values_ansa.dat"), dtype='float64', mode='r', shape=shape)
    ALL_B_clip_count = np.memmap(os.path.join(save_dir, "ALL_b_clip_count.dat"), dtype='float64', mode='r', shape=shape)
    ALL_D_clip_count = np.memmap(os.path.join(save_dir, "ALL_d_clip_count.dat"), dtype='float64', mode='r', shape=shape)
    ALL_B_clip_mass  = np.memmap(os.path.join(save_dir, "ALL_b_clip_mass.dat"), dtype='float64', mode='r', shape=shape)
    ALL_D_clip_mass  = np.memmap(os.path.join(save_dir, "ALL_d_clip_mass.dat"), dtype='float64', mode='r', shape=shape)
    ALL_mu_B  = np.memmap(os.path.join(save_dir, "ALL_mu_b.dat"), dtype='float64', mode='r', shape=shape)
    ALL_mu_D  = np.memmap(os.path.join(save_dir, "ALL_mu_d.dat"), dtype='float64', mode='r', shape=shape)

    def process_one_combo(i):   
        params = combinations[i]
        n_individuals = params[0];         b1_rate = params[1];        b2_rate = params[2];        d1_rate = params[3];        d2_rate = params[4]
        idxs = np.unravel_index(i, shape[1:-1])

        skwl = np.array(ALL_skw[:, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], ::t_lag2])
        stdl = np.array(ALL_std[:, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], ::t_lag2])
        mean_trait_values = np.array(ALL_mean[:, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], ::t_lag2])
        ALL_b_clip_count = np.array(ALL_B_clip_count[:, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], ::t_lag2])
        ALL_d_clip_count = np.array(ALL_D_clip_count[:, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], ::t_lag2])
        ALL_b_clip_mass  = np.array(ALL_B_clip_mass[:, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], ::t_lag2])
        ALL_d_clip_mass  = np.array(ALL_D_clip_mass[:, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], ::t_lag2])
        ALL_mu_b = np.array(ALL_mu_B[:, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], ::t_lag2])
        ALL_mu_d = np.array(ALL_mu_D[:, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], ::t_lag2])
       
        Amp_2D = np.sqrt(skwl**2 + stdl**2)
        Phase_2D = np.arctan2(skwl - np.mean(skwl, axis=1)[:, None], stdl - np.mean(stdl, axis=1)[:, None])
        Freq_2D = np.diff(np.unwrap(Phase_2D, axis=1), axis=1)
        Freq_2D = np.concatenate([Freq_2D[:, :1], Freq_2D], axis=1)/(t_lag*t_lag2)*n_individuals

        std_Hilbert = hilbert(stdl, axis=1)
        Amp_Hil_std = np.abs(std_Hilbert)
        std_Hilbert = hilbert(stdl - np.mean(stdl, axis=1)[:, None], axis=1)
        Phase_Hil_std = np.angle(std_Hilbert)
        Freq_Hil_std = np.diff(np.unwrap(Phase_Hil_std, axis=1), axis=1)
        Freq_Hil_std = np.concatenate([Freq_Hil_std[:, :1], Freq_Hil_std], axis=1)/(t_lag*t_lag2)*n_individuals

        skw_Hilbert = hilbert(skwl, axis=1)
        Amp_Hil_skw = np.abs(skw_Hilbert)
        skw_Hilbert = hilbert(skwl - np.mean(skwl, axis=1)[:, None], axis=1)
        Phase_Hil_skw = np.angle(skw_Hilbert)
        Freq_Hil_skw = np.diff(np.unwrap(Phase_Hil_skw, axis=1), axis=1)
        Freq_Hil_skw = np.concatenate([Freq_Hil_skw[:, :1], Freq_Hil_skw], axis=1)/(t_lag*t_lag2)*n_individuals

        Mean_ts = mean_trait_values - mean_trait_values[:, 0][:, None]
        dt = 1
        T = Mean_ts.shape[1]
        ts = np.arange(T) * t_lag*t_lag2

        Mean_Slope  = np.zeros(nansa)
        Mean_Intercept = np.zeros(nansa)

        Delta_odd = np.zeros(nansa)
        Peak_height = np.zeros(nansa)
        curvature = np.zeros(nansa)
        Integrated_response = np.zeros(nansa)
        Weighted_asym = np.zeros(nansa)

        rtspeed = np.gradient(mean_trait_values, axis=1)*n_individuals/(t_lag*t_lag2)

        Eff_slope = b1_rate + d1_rate - ALL_b_clip_count*b1_rate - ALL_d_clip_count*d1_rate
        Eff_slope2 = (1-ALL_b_clip_count-ALL_d_clip_count)*(b1_rate + d1_rate) + ALL_d_clip_count*(b1_rate)  + ALL_mu_d/n_individuals + ALL_b_clip_count*(d1_rate) + ALL_mu_b/n_individuals

        # MVBD = np.nanmean(rtspeed / (stdl**2 * ( ( b1_rate + 2*b2_rate*mean_trait_values ) + ( d1_rate + 2*d2_rate*mean_trait_values ) )), axis=1)
        if b1_rate + d1_rate == 0 : MVBD = MVBD_Eff_Slope = MVBD_Eff_Slope2 = MVbd_eff_mass_clip_rt = MVbd_eff_count_clip_rt = np.nan
        else: 
            MVBD = np.nanmean(rtspeed / (stdl**2 * ( b1_rate + d1_rate )), axis=1)
            MVBD_Eff_Slope = np.nanmean(rtspeed / (stdl**2 * ( Eff_slope )), axis=1)
            MVBD_Eff_Slope2 = np.nanmean(rtspeed / (stdl**2 * ( Eff_slope2 )), axis=1)

            MVbd_eff_mass_clip_rt = np.nanmean(rtspeed / (stdl**2 * ( b1_rate + d1_rate - ALL_b_clip_mass - ALL_d_clip_mass)), axis=1)
            MVbd_eff_count_clip_rt = np.nanmean(rtspeed / (stdl**2 * ( b1_rate + d1_rate - ALL_b_clip_count - ALL_d_clip_count)), axis=1)

        for i in range(nansa):
            Mean_Slope[i], Mean_Intercept[i] = np.polyfit(ts, Mean_ts[i], 1)

            bin_centers, mean_x, sem_x, counts = Con_Avg(rtspeed[i],Phase_2D[i],121)
            mask = ~np.isnan(mean_x)
            Delta_odd[i] = np.trapz(mean_x[mask] * np.sign(bin_centers[mask]), bin_centers[mask])   # trapz   trapezoid

            baseline = np.nanmean(mean_x[np.abs(bin_centers) > 2.0])
            peak = np.nanmax(mean_x)
            Peak_height[i] = peak - baseline

            i0 = np.argmin(np.abs(bin_centers))
            curvature[i] = (mean_x[i0+1] - 2*mean_x[i0] + mean_x[i0-1]    ) / (bin_centers[1] - bin_centers[0])**2

            Integrated_response[i] = np.trapz(mean_x[mask], bin_centers[mask])

            valid = (sem_x > 0) & np.isfinite(mean_x)
            w = np.zeros_like(sem_x)
            w[valid] = 1.0 / sem_x[valid]**2
            if np.any(valid):
                Weighted_asym[i] = ( np.sum(w[valid] * mean_x[valid] * np.sign(bin_centers[valid])) / np.sum(w[valid]) )
            else:
                Weighted_asym[i] = np.nan


        metadata = {
            "N": int(n_individuals),
            "b1": float(b1_rate),
            "b2": float(b2_rate),
            "d1": float(d1_rate),
            "d2": float(d2_rate)
        }
        metadata["Avg_bclipped"] = np.mean(np.mean(ALL_b_clip_count, axis=1))
        metadata["Avg_dclipped"] = np.mean(np.mean(ALL_d_clip_count, axis=1))
        metadata["Avg_bdclipped"] = np.mean(np.mean(ALL_b_clip_count + ALL_d_clip_count, axis=1))

        metadata["Avg_b_clip_mass"] = np.mean(np.mean(ALL_b_clip_mass, axis=1))
        metadata["Avg_d_clip_mass"] = np.mean(np.mean(ALL_d_clip_mass, axis=1))
        metadata["Avg_bd_clip_mass"] = np.mean(np.mean( ALL_b_clip_mass + ALL_d_clip_mass, axis=1))

        metadata["Avg_Skw"] = np.mean(np.mean(skwl, axis=1))
        metadata["std_Skw"] = np.mean(np.std(skwl, axis=1))

        metadata["Avg_Std"] = np.mean(np.mean(stdl, axis=1))
        metadata["std_Std"] = np.mean(np.std(stdl, axis=1))

        metadata["Avg_Var"] = np.mean(np.mean(stdl**2, axis=1))
        metadata["std_Var"] = np.mean(np.std(stdl**2, axis=1))

        metadata["Avg_Amp_2D"] = np.mean(np.mean(Amp_2D, axis=1))
        metadata["Avg_Phase_2D"] = np.mean(np.mean(Phase_2D, axis=1))
        metadata["Avg_Freq_2D"] = np.mean(np.mean(Freq_2D, axis=1))
        metadata["Std_Freq_2D"] = np.mean(np.std(Freq_2D, axis=1))
        metadata["Avg_Abs_Freq_2D"] = np.mean(np.mean(np.abs(Freq_2D), axis=1))
        
        metadata["Avg_Amp_Hil_std"] = np.mean(np.mean(Amp_Hil_std, axis=1))
        metadata["Avg_Phase_Hil_std"] = np.mean(np.mean(Phase_Hil_std, axis=1))
        metadata["Avg_Freq_Hil_std"] = np.mean(np.mean(Freq_Hil_std, axis=1))
        metadata["Avg_Abs_Freq_Hil_std"] = np.mean(np.mean(np.abs(Freq_Hil_std), axis=1))
        metadata["Std_Freq_Hil_std"] = np.mean(np.std(Freq_Hil_std, axis=1))

        metadata["Avg_Amp_Hil_skw"] = np.mean(np.mean(Amp_Hil_skw, axis=1))
        metadata["Avg_Phase_Hil_skw"] = np.mean(np.mean(Phase_Hil_skw, axis=1))
        metadata["Avg_Freq_Hil_skw"] = np.mean(np.mean(Freq_Hil_skw, axis=1))
        metadata["Avg_Abs_Freq_Hil_skw"] = np.mean(np.mean(np.abs(Freq_Hil_skw), axis=1))
        metadata["Std_Freq_Hil_skw"] = np.mean(np.std(Freq_Hil_skw, axis=1))

        metadata["final_mean_trait"] = np.mean(mean_trait_values[:, -1])
        metadata["Mean_Slope"] = np.mean(Mean_Slope) * n_individuals
        metadata["Mean_Intercept"] = np.mean(Mean_Intercept) * n_individuals
        metadata["Avg_M_dot_rt"] = np.mean(np.mean(rtspeed, axis=1))
        metadata["Std_M_dot_rt"] = np.mean(np.std(rtspeed, axis=1))
        metadata["MVbd_rt"] = np.mean(MVBD)

        metadata["Eff_Slope"] = np.mean(np.mean(Eff_slope, axis=1))
        metadata["Eff_Slope2"] = np.mean(np.mean(Eff_slope2, axis=1))
        metadata["MVbd_rt_ES"] = np.mean(MVBD_Eff_Slope)
        metadata["MVbd_rt_ES2"] = np.mean(MVBD_Eff_Slope2)

        metadata["MVbd_eff_mass_clip_rt"] = np.mean(MVbd_eff_mass_clip_rt) 
        metadata["MVbd_eff_count_clip_rt"] = np.mean(MVbd_eff_count_clip_rt)     

        metadata["Delta_odd"] = np.mean(Delta_odd)
        metadata["Peak_height"] = np.mean(Peak_height)
        metadata["curvature"] = np.mean(curvature)
        metadata["Integrated_response"] = np.mean(Integrated_response)
        metadata["Weighted_asym"] = np.mean(Weighted_asym)

  
        # if nansa>1 :
        #     metadata["Avg_bclipped_Err"] = np.std(np.mean(ALL_b_clip_count, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_dclipped_Err"] = np.std(np.mean(ALL_d_clip_count, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_bdclipped_Err"] = np.std(np.mean(ALL_b_clip_count + ALL_d_clip_count, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Skw_Err"] = np.std(np.mean(skwl, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Std_Err"] = np.std(np.mean(stdl, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Var_Err"] = np.std(np.mean(stdl**2, axis=1))/np.sqrt(nansa)          
        #     metadata["Avg_Amp_2D_Err"] = np.std(np.mean(Amp_2D, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Phase_2D_Err"] = np.std(np.mean(Phase_2D, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Freq_2D_Err"] = np.std(np.mean(Freq_2D, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Abs_Freq_2D_Err"] = np.std(np.mean(np.abs(Freq_2D), axis=1))/np.sqrt(nansa)
        #     metadata["Std_Freq_2D_Err"] = np.std(np.std(Freq_2D, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Amp_Hil_std_Err"] = np.std(np.mean(Amp_Hil_std, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Phase_Hil_std_Err"] = np.std(np.mean(Phase_Hil_std, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Freq_Hil_std_Err"] = np.std(np.mean(Freq_Hil_std, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Abs_Freq_Hil_std_Err"] = np.std(np.mean(np.abs(Freq_Hil_std), axis=1))/np.sqrt(nansa)
        #     metadata["Std_Freq_Hil_std_Err"] = np.std(np.std(Freq_Hil_std, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Amp_Hil_skw_Err"] = np.std(np.mean(Amp_Hil_skw, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Phase_Hil_skw_Err"] = np.std(np.mean(Phase_Hil_skw, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Freq_Hil_skw_Err"] = np.std(np.mean(Freq_Hil_skw, axis=1))/np.sqrt(nansa)
        #     metadata["Avg_Abs_Freq_Hil_skw_Err"] = np.std(np.mean(np.abs(Freq_Hil_skw), axis=1))/np.sqrt(nansa)
        #     metadata["Std_Freq_Hil_skw_Err"] = np.std(np.std(Freq_Hil_skw, axis=1))/np.sqrt(nansa)
        #     metadata["final_mean_trait_Err"] = np.std(mean_trait_values[:, -1])/np.sqrt(nansa)
        #     metadata["Mean_Slope_Err"] = np.std(Mean_Slope)/np.sqrt(nansa) * n_individuals
        #     metadata["Mean_Intercept_Err"] = np.std(Mean_Intercept)/np.sqrt(nansa) * n_individuals

        return metadata

    results = Parallel(n_jobs=n_jobs, verbose=0, backend="threading")( delayed(process_one_combo)(i) for i in tqdm(range(n_combos), total=len(combinations), desc="Simulating", ncols=100) )    # (n_jobs=n_jobs, backend='loky')
    summary_path = os.path.join(save_dir, f"{t_lag}_ALL_summaries.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n✅ All summaries saved to {summary_path}")







