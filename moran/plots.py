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


def Plot_Moment_dynamics(b1_rate, d1_rate, n_individuals, indices, Mdot, Vdot, Tdot, Mdotapp, Vdotapp, Tdotapp, fig_dir, ex_name, figsize=(25, 15), dpi=600):
    os.makedirs(os.path.join(fig_dir, ex_name), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
    fig.suptitle(f"Moment dynamics - b1={b1_rate}, d1={d1_rate}, N={n_individuals}", fontsize=16) # Main title
    ax0 = axes[0]
    ax0.plot(indices/n_individuals, Mdot, label="M dot")
    ax0.plot(indices/n_individuals, Mdotapp, label="M dot approx")
    ax0.set_ylabel(r"$\dot M$", fontsize=18)
    ax0.legend(fontsize=18)
    ax1 = axes[1]
    ax1.plot(indices/n_individuals, Vdot, label="V dot")
    ax1.plot(indices/n_individuals, Vdotapp, label="V dot approx")
    ax1.set_ylabel(r"$\dot V$", fontsize=18)
    ax1.legend(fontsize=18)
    ax2 = axes[2]
    ax2.plot(indices/n_individuals, Tdot, label="T dot")
    ax2.plot(indices/n_individuals, Tdotapp, label="T dot approx")
    ax2.set_xlabel("Time", fontsize=18)
    ax2.set_ylabel(r"$\dot T$", fontsize=18)
    ax2.legend(fontsize=18)

    fig.savefig(os.path.join(fig_dir, ex_name, f"moment_dynamics.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def Plot_Moments(b1_rate, d1_rate, n_individuals, indices, M, V, T, fig_dir, ex_name, figsize=(25, 15), dpi=600):
    os.makedirs(os.path.join(fig_dir, ex_name), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
    fig.suptitle(f"Moment dynamics - b1={b1_rate}, d1={d1_rate}, N={n_individuals}", fontsize=16) # Main title
    ax0 = axes[0]
    ax0.plot(indices/n_individuals, M, label="M")
    ax0.set_ylabel(r"$M$", fontsize=18)
    ax0.legend(fontsize=18)
    ax1 = axes[1]
    ax1.plot(indices/n_individuals, V, label="V")
    ax1.set_ylabel(r"$ V$", fontsize=18)
    ax1.legend(fontsize=18)
    ax2 = axes[2]
    ax2.plot(indices/n_individuals, T, label="T")
    ax2.set_xlabel("Time", fontsize=18)
    ax2.set_ylabel(r"$T$", fontsize=18)
    ax2.legend(fontsize=18)

    fig.savefig(os.path.join(fig_dir, ex_name, f"moments.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)



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

def save_hist_frames_hump(All_tv, Main_3D, indices, Hist_counts, Hist_edges, lag, n_individuals, fig_dir, ex_name, histskip=10):
    frame_dir = os.path.join(fig_dir, ex_name, "frames")
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



def GMM_compare_plot_frames(All_tv, indices, Hist_counts, Hist_edges, n_individuals, fig_dir, ex_name, xic, sep_threshold, histskip=10, dpi=150):
    frame_dir = os.path.join(fig_dir, ex_name, "gmm_frames")
    os.makedirs(frame_dir, exist_ok=True)
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


    for k in range(0, n_out, histskip):
        # ----- Create figure -----
        fig = plt.figure(figsize=(7,10))
        gs = fig.add_gridspec(len(xic), 1)
        fig.suptitle(f"t ={indices[k]/n_individuals:.2f}", fontsize=16) # Main title

        for j in range(len(xic)):

            trait_values = All_tv[k, :].reshape(-1,1)
            gmm1 = GaussianMixture(n_components=1)
            gmm1.fit(trait_values)
            xic1 = getattr(gmm1, xic[j])(trait_values)
            mean1 = gmm1.means_.flatten()[0]
            std1 = np.sqrt(gmm1.covariances_.flatten()[0])
            gmm2 = GaussianMixture(n_components=2)
            gmm2.fit(trait_values)
            xic2 = getattr(gmm2, xic[j])(trait_values)
            mean2 = gmm2.means_.flatten()
            std2 = np.sqrt(gmm2.covariances_.flatten())
            weights2 = gmm2.weights_.flatten()
            sep = abs(mean2[0] - mean2[1]) / np.sqrt(std2[0]**2 + std2[1]**2)
            x_range = np.linspace(Hist_edges[k, 0], Hist_edges[k, -1], 128).reshape(-1,1)
            resp = gmm2.predict_proba(trait_values)
            labels = np.argmax(resp, axis=1)
            if xic2 < xic1 and sep > sep_threshold[j]:
                logprob = gmm2.score_samples(x_range)
                pdf_hump = np.exp(logprob)

            # ==========================
            # 1️⃣ Histogram panel
            # ==========================
            ax0 = fig.add_subplot(gs[j])
            ax0.stairs(normalized_hists[k], Hist_edges[k])
            if xic2 < xic1 and sep > sep_threshold[j]:
                ax0.plot(x_range, pdf_hump, color='red', linestyle='-', label='GMM Fit')
                ax0.axvline(mean2[0], color='black', linestyle='--', label='mean1')
                ax0.axvline(mean2[1], color='black', linestyle='--', label='mean2')
                ax0.legend()

            ax0.set_xlim(global_xmin, global_xmax)
            ax0.set_ylim(0, global_ymax)
            ax0.set_title(f"{xic[j]}, sep={sep_threshold[j]:.2f}", fontsize=10)
            ax0.set_ylabel("PDF")
            if j != len(xic)-1: ax0.set_xticklabels([])
            if j == len(xic)-1: ax0.set_xlabel("Trait Value")

            if xic2 < xic1 and sep > sep_threshold[j]:
                stats_text = (
                f"{xic[j]}1 = {xic1:.1f}\n"
                f"{xic[j]}2 = {xic2:.1f}\n"
                f"sep = {sep:.3f}"
                f"mean_1 = {mean2[0]:.1f}\n"
                f"mean_2 = {mean2[1]:.1f}\n"
                )
            ax0.text(
                0.98, 0.95,
                stats_text,
                transform=ax0.transAxes,
                va='top', ha='right',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=9
            )

        plt.savefig(os.path.join(frame_dir, f"{k:05d}.png"), dpi=dpi)
        plt.close()



def make_video(fig_dir, ex_name, frame_duration=0.5):
    images = sorted(glob.glob(str(fig_dir) + "/" + ex_name + "/*frames/*.png"))
    with imageio.get_writer(str(fig_dir) + "/" + ex_name + "/video.mp4", fps=1 / frame_duration) as writer:
        for image_file in images:
            img = imageio.imread(image_file)
            writer.append_data(img)




def fit_best_gmm(trait_values, xic="bic", max_components=2):
    scores = []
    models = []
    for k in range(1, max_components+1):
        gmm = GaussianMixture(n_components=k)
        gmm.fit(trait_values)
        score = getattr(gmm, xic)(trait_values)
        scores.append(score)
        models.append(gmm)
    best_idx = np.argmin(scores)
    return models[best_idx], best_idx+1


def ridge_plot_pdf_with_gmm(All_tv, Hist_counts, Hist_edges, indices, times_to_plot, n_individuals, criterion="bic", sep_threshold=1.5, scale=0.4):

    fig, ax = plt.subplots(figsize=(6,8))

    for t_idx in times_to_plot:

        k = t_idx

        # ---- histogram → PDF ----
        counts = Hist_counts[k]
        edges = Hist_edges[k]
        widths = np.diff(edges)
        centers = 0.5*(edges[:-1] + edges[1:])

        pdf = counts / (np.sum(counts)*widths)

        # normalize for plotting width
        pdf_scaled = pdf / np.max(pdf) * scale

        time_val = indices[k] / n_individuals

        # ---- plot PDF (gray) ----
        ax.fill_betweenx(centers, time_val, time_val + pdf_scaled, color='gray', alpha=0.6)

        # ---- fit GMM ----
        trait_values = All_tv[k].reshape(-1,1)
        gmm, n_comp = fit_best_gmm(trait_values, criterion)

        if n_comp == 2:
            mean = gmm.means_.flatten()
            std = np.sqrt(gmm.covariances_.flatten())

            sep = abs(mean[0]-mean[1]) / np.sqrt(std[0]**2 + std[1]**2)

        else:
            sep = 0

        # ---- overlay GMM ----
        if (n_comp == 2 and sep > sep_threshold) or n_comp == 1:
            x_range = np.linspace(edges[0], edges[-1], 200).reshape(-1,1)
            pdf_gmm = np.exp(gmm.score_samples(x_range))
            pdf_gmm = pdf_gmm / np.max(pdf_gmm) * scale
            ax.plot(time_val + pdf_gmm, x_range.flatten(), color='red', linewidth=2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Trait value")
    plt.tight_layout()
    plt.show()

def ridge_plot_gmm_with_pdf(All_tv, Hist_counts, Hist_edges, indices, times_to_plot, n_individuals, criterion="bic", sep_threshold=1.5, scale=0.4):

    fig, ax = plt.subplots(figsize=(6,8))

    for t_idx in times_to_plot:

        k = t_idx

        counts = Hist_counts[k]
        edges = Hist_edges[k]
        widths = np.diff(edges)
        centers = 0.5*(edges[:-1] + edges[1:])

        pdf = counts / (np.sum(counts)*widths)
        pdf_scaled = pdf / np.max(pdf) * scale

        time_val = indices[k] / n_individuals

        # ---- fit GMM ----
        trait_values = All_tv[k].reshape(-1,1)
        gmm, n_comp = fit_best_gmm(trait_values, criterion)

        if n_comp == 2:
            mean = gmm.means_.flatten()
            std = np.sqrt(gmm.covariances_.flatten())
            sep = abs(mean[0]-mean[1]) / np.sqrt(std[0]**2 + std[1]**2)
        else:
            sep = 0

        x_range = np.linspace(edges[0], edges[-1], 200).reshape(-1,1)

        # ---- plot GMM (gray) ----
        if (n_comp == 2 and sep > sep_threshold) or n_comp == 1:

            pdf_gmm = np.exp(gmm.score_samples(x_range))
            pdf_gmm = pdf_gmm / np.max(pdf_gmm) * scale

            ax.fill_betweenx(x_range.flatten(), time_val, time_val + pdf_gmm, color='gray', alpha=0.6)

        # ---- overlay real PDF (red) ----
        ax.plot(time_val + pdf_scaled, centers, color='red', linewidth=2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Trait value")
    plt.tight_layout()
    plt.show()




















# from pathlib import Path

# from loguru import logger
# from tqdm import tqdm
# import typer

# from moran.config import FIGURES_DIR, PROCESSED_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = FIGURES_DIR / "plot.png",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Generating plot from data...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Plot generation complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
