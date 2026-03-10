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

########################################################################################################################################

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
        print(All_tv.shape)
        print(All_tv[0, :50])
  
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
            waiting_times = np.diff(indices[hump_events]) * t_lag
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

    results = Parallel(n_jobs=n_jobs, verbose=0, backend="threading")( delayed(process_one_combo)(i) for i in tqdm(range(n_combos), total=len(combinations), desc="Simulating", ncols=100) )    # (n_jobs=n_jobs, backend='loky')
    summary_path = os.path.join(save_dir, f"{t_lag}_ALL_summaries.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n✅ All summaries saved to {summary_path}")




########################################################################################################################################





def hump_detevtor(Hist_counts, Hist_edges, All_tv, b1_rate, b2_rate, d1_rate, d2_rate, tmax, indices, nbins=128):
    n_out, n_individuals = All_tv.shape

    for k in range(n_out): 
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

        x_range = np.linspace(Hist_edges[k, 0], Hist_edges[k, -1], 128).reshape(-1,1)
        if bic2 < bic1 and abs(mean2_1 - mean2_2)/np.sqrt(std2_1**2 + std2_2**2) > 1.5:
            logprob = gmm2.score_samples(x_range)
            pdf_hump = np.exp(logprob)
        # else:
        #     logprob = gmm1.score_samples(x_range)
        #     pdf_hump = np.exp(logprob)


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

        x_range = np.linspace(Hist_edges[k, 0], Hist_edges[k, -1], 128).reshape(-1,1)
        if bic2 < bic1 and abs(mean2_1 - mean2_2)/np.sqrt(std2_1**2 + std2_2**2) > 1.5:
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
        if bic2 < bic1 and abs(mean2_1 - mean2_2)/np.sqrt(std2_1**2 + std2_2**2) > 1.5:
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
            f"V_dot = {Vdot[k]:.3f}\n"
            f"bic1 = {bic1:.1f}\n"
            f"bic2 = {bic2:.1f}\n"
            f"ration = {abs(mean2_1 - mean2_2)/np.sqrt(std2_1**2 + std2_2**2):.3f}"
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

def make_video(save_dir, zname, frame_duration=0.5):
    images = sorted(glob.glob(save_dir + "vid2/" + zname + "/*.png"))
    with imageio.get_writer(save_dir + zname + "_video2.mp4", fps=1 / frame_duration) as writer:
        for image_file in images:
            img = imageio.imread(image_file)
            writer.append_data(img)





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



