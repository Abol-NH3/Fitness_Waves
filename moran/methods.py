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


ಠ_ಠ = "hmmm..."


# optimize: single pass over trait values in each step, 
#     cal the out puts only in wanted indices 





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


########################################################################################################################################
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

@njit()
def Final_Quadratic_Sim(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values, nbins=128):
    n_individuals = len(trait_values)
    n_out = len(indices)

    # Main arrays
    Main_3D = np.zeros((3, n_out))  # 0: mean, 1: skew, 2: std
    Clipp = np.zeros((6, n_out)) # 0: birth_clipped_count, 1: death_clipped_count, 2: birth_clip_mass, 3: death_clip_mass, 4: wb eff mass, 5: wd eff mass
    Moments_right_tail = np.zeros((4, n_out)) 
    Moments_left_tail = np.zeros((4, n_out))
    Moments = np.zeros((8, n_out))  # 0: mu1, 1: mu2, 2: mu3, 3: mu4, 4: mu5, 5: mu6, 6: mu7, 7: mu8
    Theo_M_dot = np.zeros((3, n_out))  # 0: 

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
            Moments[4, k] = np.mean(trait_values**5)
            Moments[5, k] = np.mean(trait_values**6)
            Moments[6, k] = np.mean(trait_values**7)
            Moments[7, k] = np.mean(trait_values**8)

            Theo_M_dot[0, k] = 0.0 if wb_eff_mass == 0.0 or wd_eff_mass == 0.0 else np.sum(trait_values * wb_eff) / (wb_eff_mass) - np.sum(trait_values * wd_eff) / (wd_eff_mass)
            Theo_M_dot[1, k] = np.sum(trait_values * (wb_eff - wd_eff)) / n_individuals
            Theo_M_dot[2, k] = np.sum(trait_values * ( (wb_eff - wd_eff)  -  np.mean(wb_eff - wd_eff))) / n_individuals   # add Var - Sigma x**2 at tails

            Main_3D[0, k] = cum_mean_trait_value
            Main_3D[2, k] = np.std(trait_values)
            if Main_3D[2, k] == 0: Main_3D[1, k] = 0
            else: Main_3D[1, k] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * Main_3D[2, k]**3)

            # NEW: store dynamic histogram at this output time
            c, e = hist_dynamic_minmax(trait_values, nbins)
            Hist_counts[k, :] = c
            Hist_edges[k, :]  = e

            k += 1

    return Main_3D, Clipp, Moments, Moments_right_tail, Moments_left_tail, Theo_M_dot, Hist_counts, Hist_edges

@njit()
def Quad_Sim_tvout(b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, trait_values, nbins=128):
    n_individuals = len(trait_values)
    n_out = len(indices)

    # Main arrays
    GMM1 = []
    GMM2 = []
    BIC1 = []
    BIC2 = []


    All_tv = np.zeros((n_out, n_individuals))  # Store trait values of all individuals at each output time
    Main_3D = np.zeros((3, n_out))  # 0: mean, 1: skew, 2: std
    Clipp = np.zeros((6, n_out)) # 0: birth_clipped_count, 1: death_clipped_count, 2: birth_clip_mass, 3: death_clip_mass, 4: wb eff mass, 5: wd eff mass
    Moments_right_tail = np.zeros((4, n_out)) 
    Moments_left_tail = np.zeros((4, n_out))
    Moments = np.zeros((8, n_out))  # 0: mu1, 1: mu2, 2: mu3, 3: mu4, 4: mu5, 5: mu6, 6: mu7, 7: mu8
    Theo_M_dot = np.zeros((3, n_out))  # 0: 

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

        All_tv[k, :] = trait_values 

        gmm1 = GaussianMixture(n_components=1)
        gmm1.fit(trait_values.reshape(-1,1))
        bic1 = gmm1.bic(trait_values.reshape(-1,1))
        mean1 = gmm1.means_.flatten()[0]
        std1 = np.sqrt(gmm1.covariances_.flatten()[0])
        gmm2 = GaussianMixture(n_components=2)
        gmm2.fit(trait_values.reshape(-1,1))
        bic2 = gmm2.bic(trait_values.reshape(-1,1))
        mean2_1, mean2_2 = gmm2.means_.flatten()
        std2_1, std2_2 = np.sqrt(gmm2.covariances_.flatten())
        weights2_1, weights2_2 = gmm2.weights_.flatten()
        GMM1.append(gmm1)
        BIC1.append(bic1)
        GMM2.append(gmm2)
        BIC2.append(bic2)


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
            Moments[4, k] = np.mean(trait_values**5)
            Moments[5, k] = np.mean(trait_values**6)
            Moments[6, k] = np.mean(trait_values**7)
            Moments[7, k] = np.mean(trait_values**8)

            Theo_M_dot[0, k] = 0.0 if wb_eff_mass == 0.0 or wd_eff_mass == 0.0 else np.sum(trait_values * wb_eff) / (wb_eff_mass) - np.sum(trait_values * wd_eff) / (wd_eff_mass)
            Theo_M_dot[1, k] = np.sum(trait_values * (wb_eff - wd_eff)) / n_individuals
            Theo_M_dot[2, k] = np.sum(trait_values * ( (wb_eff - wd_eff)  -  np.mean(wb_eff - wd_eff))) / n_individuals   # add Var - Sigma x**2 at tails

            Main_3D[0, k] = cum_mean_trait_value
            Main_3D[2, k] = np.std(trait_values)
            if Main_3D[2, k] == 0: Main_3D[1, k] = 0
            else: Main_3D[1, k] = np.sum((trait_values - np.mean(trait_values)) ** 3) / (n_individuals * Main_3D[2, k]**3)

            # NEW: store dynamic histogram at this output time
            c, e = hist_dynamic_minmax(trait_values, nbins)
            Hist_counts[k, :] = c
            Hist_edges[k, :]  = e

            k += 1

    return GMM1, BIC1, GMM2, BIC2, Main_3D, Clipp, Moments, Moments_right_tail, Moments_left_tail, Theo_M_dot, Hist_counts, Hist_edges

def hump_detevtor(Hist_counts, Hist_edges, All_tv, b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, nbins=128):
    n_out, n_individuals = All_tv.shape

    for k in range(n_out):
        trait_values = All_tv[k, :]


        gmm1 = GaussianMixture(n_components=1)
        gmm1.fit(trait_values)
        bic1 = gmm1.bic(trait_values)
        mean1 = gmm1.means_.flatten()[0]
        std1 = np.sqrt(gmm1.covariances_.flatten()[0])

        gmm2 = GaussianMixture(n_components=2)
        gmm2.fit(trait_values)
        bic2 = gmm2.bic(trait_values)
        mean2_1, mean2_2 = gmm2.means_.flatten()
        std2_1, std2_2 = np.sqrt(gmm2.covariances_.flatten())
        weights2_1, weights2_2 = gmm2.weights_.flatten()


        if bic2 > bic1 or abs(mean2_1 - mean2_2)/np.sqrt(std2_1**2 + std2_2**2) < 2: continue





def save_hist_frames_hump(All_tv, Main_3D, indices, Hist_counts, Hist_edges, lag, n_individuals, out_dir, zname, histskip=10):

    frame_dir = os.path.join(out_dir, "vid2", zname)
    os.makedirs(frame_dir, exist_ok=True)

    mean = Main_3D[0, :]
    skw  = Main_3D[1, :]
    std  = Main_3D[2, :]

    var  = std**2
    Mdot = np.gradient(mean, 1) * n_individuals / lag
    Vdot = np.gradient(var, 1)  * n_individuals / lag

    n_out = Hist_counts.shape[0]

    # ---- Global limits for histogram ----
    global_xmin = np.min([edges[0] for edges in Hist_edges])
    global_xmax = np.max([edges[-1] for edges in Hist_edges])

    normalized_hists = []
    global_ymax = 0.0

    for k in range(n_out):
        counts = Hist_counts[k]
        edges  = Hist_edges[k]
        widths = np.diff(edges)
        total  = np.sum(counts)

        pdf = counts / (total * widths)
        normalized_hists.append(pdf)

        global_ymax = max(global_ymax, np.max(pdf))

    global_ymax *= 1.05

    # ---- Global limits for time series ----
    mean_lim = (np.min(mean), np.max(mean))
    var_lim  = (np.min(var),  np.max(var))
    skw_lim  = (np.min(skw),  np.max(skw))
    Mdot_lim = (np.min(Mdot), np.max(Mdot))
    Vdot_lim = (np.min(Vdot), np.max(Vdot))

    for k in range(0, n_out, histskip):

        trait_values = All_tv[k, :].reshape(-1,1)
        gmm1 = GaussianMixture(n_components=1)
        gmm1.fit(trait_values)
        bic1 = gmm1.bic(trait_values)
        mean1 = gmm1.means_.flatten()[0]
        std1 = np.sqrt(gmm1.covariances_.flatten()[0])
        gmm2 = GaussianMixture(n_components=2)
        gmm2.fit(trait_values)
        bic2 = gmm2.bic(trait_values)
        mean2_1, mean2_2 = gmm2.means_.flatten()
        std2_1, std2_2 = np.sqrt(gmm2.covariances_.flatten())
        weights2_1, weights2_2 = gmm2.weights_.flatten()

        x_range = np.linspace(Hist_edges[k, 0], Hist_edges[k, -1], 128)
        if bic2 < bic1 and abs(mean2_1 - mean2_2)/np.sqrt(std2_1**2 + std2_2**2) > 2:
            logprob = gmm2.score_samples(x_range)
            pdf_hump = np.exp(logprob)
        # else:
        #     logprob = gmm1.score_samples(x_range)
        #     pdf_hump = np.exp(logprob)

        edges   = Hist_edges[k]
        pdf     = normalized_hists[k]
        widths  = np.diff(edges)
        centers = edges[:-1] + widths/2

        kurt = np.sum(pdf * ((centers - mean[k])/std[k])**4 * widths) - 3.0

        # ----- Create figure -----
        fig = plt.figure(figsize=(7,10))
        gs = fig.add_gridspec(5, 1, height_ratios=[2, 1, 1, 1, 1])

        # ==========================
        # 1️⃣ Histogram panel
        # ==========================
        ax0 = fig.add_subplot(gs[0])
        ax0.stairs(pdf, edges)
        if bic2 < bic1 and abs(mean2_1 - mean2_2)/np.sqrt(std2_1**2 + std2_2**2) > 2:
            ax0.plot(x_range, pdf_hump, color='red', linestyle='-', label='GMM Fit')
            ax0.axvline(mean2_1, color='black', linestyle='--', label='mean1')
            ax0.axvline(mean2_2, color='black', linestyle='--', label='mean2')
            ax0.legend()

        ax0.set_xlim(global_xmin, global_xmax)
        ax0.set_ylim(0, global_ymax)
        ax0.set_title(f"t ={indices[k]/n_individuals:.2f}")
        ax0.set_ylabel("PDF")

        stats_text = (
            f"var = {var[k]:.3f}\n"
            f"skew = {skw[k]:.3f}\n"
            f"kurt = {kurt:.3f}\n"
            f"M_dot = {Mdot[k]:.3f}\n"
            f"V_dot = {Vdot[k]:.3f}"
        )
        ax0.text(
            0.98, 0.95,
            stats_text,
            transform=ax0.transAxes,
            va='top', ha='right',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9
        )

        # ==========================
        # 2️⃣ Moments time series
        # ==========================
        ax1 = fig.add_subplot(gs[1])
        ax1.plot(indices/n_individuals, var)
        ax1.axvline(indices[k]/n_individuals, linestyle="--")
        ax1.set_xlim(indices[0]/n_individuals-0.5, indices[-1]/n_individuals+0.5)
        ax1.set_ylabel("Var")
        ax1.set_xticklabels([])
        # ==========================
        ax2 = fig.add_subplot(gs[2])
        ax2.plot(indices/n_individuals, skw)
        ax2.axvline(indices[k]/n_individuals, linestyle="--")
        ax2.set_xlim(indices[0]/n_individuals-0.5, indices[-1]/n_individuals+0.5)
        ax2.set_ylabel("Skew")
        ax2.set_xticklabels([])
        # ==========================
        ax3 = fig.add_subplot(gs[3])
        ax3.plot(indices/n_individuals, Mdot)
        ax3.axvline(indices[k]/n_individuals, linestyle="--")
        ax3.set_xlim(indices[0]/n_individuals-0.5, indices[-1]/n_individuals+0.5)
        ax3.set_ylabel("Mdot")
        ax3.set_xticklabels([])
        # ==========================
        ax4 = fig.add_subplot(gs[4])
        ax4.plot(indices/n_individuals, Vdot)
        ax4.axvline(indices[k]/n_individuals, linestyle="--")
        ax4.set_xlim(indices[0]/n_individuals-0.5, indices[-1]/n_individuals+0.5)
        ax4.set_ylabel("Vdot")
        ax4.set_xlabel("Time")

        # plt.tight_layout()
        plt.savefig(os.path.join(frame_dir, f"frame_{k:08d}.png"), dpi=150)
        plt.close()







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



