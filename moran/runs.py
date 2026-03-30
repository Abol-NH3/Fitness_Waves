import numpy as np
import os
from pathlib import Path
import sys

# PROJECT_ROOT = Path().resolve().parents[0]
# REPORTS_DIR = PROJECT_ROOT / "reports"
# sys.path.append(str(PROJECT_ROOT ))


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

# import moran.methods as methods
# import moran.config as config

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

    results = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')( delayed(process_one_combo)(i) for i in tqdm(range(n_combos), total=len(combinations), desc="Simulating", ncols=100) )    # (n_jobs=n_jobs, backend='loky')   , backend="threading"
    summary_path = os.path.join(save_dir, f"{t_lag}_ALL_summaries.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n✅ All summaries saved to {summary_path}")


nlist = np.array([3000])  # , 10000    10, 50 ,100, 500, 1000, 5000
b1list = np.arange(0.1,1.001 ,0.1)   
d1list = np.arange(0.1,1.001 ,0.1) 
b2list = np.array([0])
d2list = np.array([0])
skip = 20
tmax = 1020 
nbins = 128
t_lag  = 100
nansa = 1

save_dir=f"/flash/DieckmannU/Abolfazl/Hump_({nlist[0]},{nlist[-1]})_b1({b1list[0]},{b1list[-1]})_b2({b2list[0]},{b2list[-1]})_d1({d1list[0]},{d1list[-1]})_d2({d2list[0]},{d2list[-1]})_skip({skip})_tmax({tmax})_tlag({t_lag})_nansa({nansa})"
# save_dir=REPORTS_DIR / f"Hump_({nlist[0]},{nlist[-1]})_b1({b1list[0]},{b1list[-1]})_b2({b2list[0]},{b2list[-1]})_d1({d1list[0]},{d1list[-1]})_d2({d2list[0]},{d2list[-1]})_skip({skip})_tmax({tmax})_tlag({t_lag})_nansa({nansa})"
os.makedirs(save_dir, exist_ok=True)
# methods.Metadata_Quad_Sim_V2(nlist, b1list, b2list, d1list, d2list, tmax, skip, t_lag, save_dir, nbins, nansa=nansa, n_jobs=6)
Metadata_Quad_Sim_V2(nlist, b1list, b2list, d1list, d2list, tmax, skip, t_lag, save_dir, nbins, nansa=nansa, n_jobs=100)

