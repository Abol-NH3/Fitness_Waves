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


import methods as mfw



# Fix saving ts directory and saving res direcory


nlist = np.array([50 ,100, 500, 1000, 5000])  # , 10000    10, 50 ,100, 500, 1000, 5000
b1list = np.arange(0,1.001 ,0.025)   
d1list = np.arange(0,1.001 ,0.025) 
b2list = np.array([0])
d2list = np.array([0])
tmax = 530000 
transient = np.array([1, 30000])            # transient[0, n]->skip first n precent -  transient[1, n]->skip first n steps
t_lag  = 10
t_lag2 = 5
nansa  = 10

save_dir=f"/flash/DieckmannU/Abolfazl/T3_Quad_N({nlist[0]},{nlist[-1]})_b1({b1list[0]},{b1list[-1]})_b2({b2list[0]},{b2list[-1]})_d1({d1list[0]},{d1list[-1]})_d2({d2list[0]},{d2list[-1]})_tmax({tmax})_tlag({t_lag})_transient({int(transient[0])},{int(transient[1])})_nansa({nansa})"
os.makedirs(save_dir, exist_ok=True)
mfw.Quad_sim(nlist, b1list, b2list, d1list, d2list, tmax, transient, t_lag, save_dir, nansa=nansa, n_jobs=128)

mfw.Load_data_Main_calc_Quadratic_analysis_clip_count_effectiveM(nlist, b1list, b2list, d1list, d2list, tmax, transient, t_lag, t_lag2, save_dir, nansa=nansa, n_jobs=128, Load=True)

