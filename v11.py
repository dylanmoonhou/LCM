# Highlights: mesh calculation to evaluate V (cost 360 min using MacPro)
# 1) State var: X (total wealth = Home + 401k + Liquid), H (homevalue, as % of X), K (401k saving, as % of X-H), , Ph/Ps (Perm ME path for head/spouse), N (num of hh members), A (SS adjusted)
# 2) Choice var: C (Consumption, as % of Liquid), D (401k distribution, as % of 401k $), S (stock invest, as % of Liquid), (sell home)
# 3) Shocks: re (Equity return), perm (perm ME), tran (transitory ME), ft (family transfer), sp (spouse alive), adj (SS benefit adjust)
# 4) integral of expectation with multivariate prob density by restorting to Gaussian quadrature integration (3 terms)
# 5) Policy function grid search by linear interp (ideally by CubicSpline along X and linear interp by other dimensions)
# Last modified @ Feb 4, 2020

# Preparation: import python packages and define program path & version
import os
import numpy as np
import pandas as pd
import itertools
from numba import vectorize
from interpolation.splines import CGrid, eval_linear
from scipy import ndimage
from scipy import interpolate
from scipy import stats
import math
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
num_workers = mp.cpu_count()
print('number of CPUs: %d' % num_workers)
output_path = '/Users/wenlianghou/Dropbox/output/'
version_name = 'v11_gamma5'

# Part 1: model setup
# Step 1-1: functions
gamma = 5
beta = 0.96
@vectorize(['float64(float64,float64,float64,float64)'], target='cpu',nopython=True)
def func_V(V0, surv_prob, EV1, Bequest, beta = 0.96):
    return V0 + beta * surv_prob * EV1 + (1-surv_prob) * Bequest
@vectorize(['float64(float64,int16,int16)'], target='cpu',nopython=True)
def func_U(consumption, N, gamma = 5): # equivalence scale of consumption = 1.52 for couple (Browning, Chiappori, and Lewbel, 2013)
    return np.power(consumption / N * (1+0.52*(N-1)),(1-gamma))/(1-gamma) # N = 1/2 single/couple; assume equal Pareto wgt for couple
@vectorize(['float64(float64,int16,int16)'], target='cpu',nopython=True)
def func_B(Wealth, bequest_motive = 0, gamma = 5):
    return bequest_motive * (Wealth/bequest_motive) ** (1-gamma)
def lognorm_para(mean,std): # solve lognormal parameters using mean and std
    sigma = (math.log(std ** 2 / mean ** 2 + 1)) ** 0.5
    mu = math.log(mean) - 0.5 * sigma ** 2
    return(mu,sigma)

# Step 1-2: grids
class GridPoints:   #grid points
    def __init__(self, vmin, vmax, step):
        self.vmin = vmin
        self.vmax = vmax
        self.step = step
        self.n = int((vmax - vmin) / step + 1)
        self.v = np.arange(self.n) * self.step + self.vmin


# health follows JPE (2010)


# Part 2: import data and set up parameters

GQ_wgt = [1 / 6, 2 / 3, 1 / 6] # 􏰕􏰕three-node Guass-Hermite quadrature method, QE(2011) Judd et al
GQ_factor = 1.73205080756887
grid_GQ = np.round(np.array([-GQ_factor, 0, GQ_factor]),3)

# Step 2-0 Household Finance Data from HRS (initial wealth and income in 2019 $000) : Y-SSB; K-401k&IRA; H-house; L-liquid assets
Yh = 17; Ys = 15; K_hh = 290; H_hh = 192; L_hh = 48; X_hh = K_hh + H_hh + L_hh
Yh_pct = Yh / X_hh; Ys_pct = Ys / X_hh

# Step 2-1(a) Longevity risk (cohort life table from SSA general population for age 65 @ 2019 by gender)
q_m = [0.015899, 0.016868, 0.018387, 0.019092, 0.020462, 0.022047, 0.023816, 0.025697, 0.027674, 0.029828, 0.032382,
       0.035360, 0.038579, 0.041991, 0.045724, 0.049974, 0.054971, 0.060902, 0.068009, 0.076327, 0.085749, 0.096120,
       0.107306, 0.119231, 0.131905, 0.145375, 0.159718, 0.175027, 0.191392, 0.208919, 0.226082, 0.242533, 0.257903,
       0.271821, 0.283929, 0.296581, 0.309802, 0.323618, 0.338056, 0.353144, 0.368912, 0.385389, 0.402610, 0.420607,
       0.439416, 0.459073, 0.479618, 0.501090, 0.523533, 0.546989, 0.571506, 0.597132, 0.623916, 0.651913, 0.681177]
q_f = [0.009285, 0.010041, 0.010879, 0.011795, 0.012807, 0.014005, 0.015371, 0.016805, 0.018288, 0.019898, 0.021817,
       0.024092, 0.026614, 0.029366, 0.032429, 0.035959, 0.040081, 0.044876, 0.050495, 0.057014, 0.064428, 0.072694,
       0.081757, 0.091586, 0.102186, 0.113584, 0.125823, 0.138960, 0.153050, 0.168151, 0.183270, 0.198142, 0.212480,
       0.225985, 0.238357, 0.251410, 0.265183, 0.279715, 0.295049, 0.311228, 0.328300, 0.346315, 0.365324, 0.385383,
       0.406550, 0.428887, 0.452459, 0.477334, 0.503585, 0.531289, 0.557688, 0.582742, 0.608930, 0.636306, 0.664924]
prob_h, prob_s = 1 - np.array(q_m), 1 - np.array(q_f)
T = len(p_h) # model horizon 65-119, 120 must die
# Step 2-1(b) alternative: no longevity risk (always live to life expectancy)
LE_h = int(np.sum(np.cumprod(prob_h)))
prob_h_norisk = np.array([1]*LE_h + [0]*(T-LE_h))

# Step 2-2(a): market risk (stock and housing returns in real terms)

# STATE VAR: total wealth X = H + K + L; H as % of X; K as % of (X-H)
grid_X = np.linspace(0,500,11)
grid_H = np.linspace(0,1,6)
grid_K = np.linspace(0,1,6)
# Regulations
L_min = 2; grid_X[0] = L_min  # In 2019 (most states) a single applicant aged 65+ is permitted up to $2K in countable assets to be eligible for nursing home Medicaid
K_share = np.concatenate((np.linspace(0.5, 0.3, num=8, endpoint=True), 0.3 * np.ones(T+1 - 8))) # TDF equity glid path (Vanguard)
rmd = [31.9, 31.1, 30.2, 29.2, 28.3, 27.4, 26.5, 25.6, 24.7, 23.8, 22.9, 22.0, 21.2, 20.3, 19.5, 18.7, 17.9, 17.1,
          16.3, 15.5, 14.8, 14.1, 13.4, 12.7, 12.0, 11.4, 10.8, 10.2, 9.6, 9.1, 8.6, 8.1, 7.6, 7.1, 6.7, 6.3, 5.9, 5.5,
          5.2, 4.9, 4.5, 4.2, 3.9, 3.7, 3.4, 3.1, 2.9, 2.6, 2.4, 2.1, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9] # RMD life span table
rmd_pct = np.round(1 / np.array(rmd),3); rmd_pct[0:72-65] = 0 # no requirement until age 72
rent = 0.04 # if not homeowners, need to pay 4% of housing price for rental
tax = 0.25
# shocks
rf = 1.01 # risk-free bond
re_mean, re_std = 1.045, 0.157  # return of equity follows lognormal
re_mu, re_sigma = lognorm_para(re_mean, re_std)
rh_mean, rh_std = 1.01, 0.05
rh_mu, rh_sigma = lognorm_para(rh_mean, rh_std)
r_corr = 0.0 # for simplicity, assume no correlation between stock and housing market return
grid_re = re_mu + re_sigma*grid_GQ
grid_re = np.round(np.exp(grid_re),3)
wgt_re = GQ_wgt
grid_rh = re_mh + rh_sigma*grid_GQ
grid_rh = np.round(np.exp(grid_rh),3)
wgt_rh = GQ_wgt

# Step 2-2(b) alternative: no market risk (always follows mean)
re_norisk = np.mean(K_share * re_mean + (1-K_share) * rf)
rh_norisk = rh_mean

# Step 2-3 (a): health risk: medical expenses from JPE(2010), defined as out-of-pocket expenses (sum of insurance premia, drug costs, hospital stays, nursing home care, doctor visits, dental visits, and outpatient care, excluding expenses covered by (public/private) insurance)
eta = 0.922 # AR(1)
perm_mu, perm_sigma = 0, 0.050 ** 0.5
tran_mu, tran_sigma = 0, 0.665 ** 0.5
grid_P = np.round(stats.norm.ppf([1/6,3/6,5/6],perm_mu,perm_sigma),3)
AR1_adj = [((1-(eta*eta)**t)/(1-eta*eta))**0.5 for t in range(T)] # convert ME path CDF to actual value (psi)
grid_P = [grid_P * AR1_adj[t] for t in range(T)]
me_mu_m = [5.729,5.770,5.811,5.852,5.894,5.937,6.015,6.081,6.137,6.184,6.223,6.257,6.285,6.310,6.332,6.353,6.373,6.394,6.416,6.440,6.466,6.495,6.528,6.564,6.605,6.649,6.698,6.751,6.808,6.869,6.934,7.002,7.073,7.146,7.221,7.297,7.374,7.449,7.298,7.340,7.381,7.422,7.464,7.505,7.546,7.587,7.629,7.670,7.711,7.753,7.794,7.835,7.877,7.918,7.959]
me_sigma_m = [1.075,1.112,1.147,1.181,1.214,1.376,1.398,1.415,1.428,1.438,1.446,1.453,1.459,1.466,1.474,1.483,1.495,1.508,1.524,1.543,1.564,1.588,1.615,1.644,1.675,1.708,1.743,1.779,1.815,1.852,1.889,1.925,1.960,1.993,2.024,2.053,2.078,2.099,2.041,2.061,2.080,2.099,2.118,2.136,2.155,2.173,2.191,2.209,2.227,2.245,2.262,2.280,2.297,2.314,2.331]
me_mu_f = [5.692,5.743,5.794,5.845,5.896,5.948,6.036,6.112,6.178,6.234,6.283,6.326,6.364,6.399,6.431,6.461,6.491,6.522,6.553,6.587,6.622,6.661,6.704,6.750,6.800,6.854,6.912,6.975,7.042,7.112,7.187,7.264,7.345,7.428,7.513,7.598,7.684,7.769,7.628,7.679,7.730,7.781,7.832,7.883,7.934,7.985,8.036,8.087,8.138,8.189,8.240,8.291,8.342,8.392,8.443]
me_sigma_f = [0.846,0.904,0.958,1.010,1.059,1.251,1.284,1.311,1.333,1.352,1.368,1.384,1.398,1.414,1.430,1.447,1.466,1.488,1.511,1.537,1.566,1.597,1.630,1.666,1.703,1.742,1.782,1.823,1.865,1.907,1.948,1.989,2.028,2.066,2.101,2.134,2.163,2.189,2.139,2.162,2.186,2.209,2.232,2.254,2.277,2.299,2.321,2.343,2.364,2.386,2.407,2.428,2.449,2.469,2.490]
me_mu_h,me_sigma_h,me_mu_s,me_sigma_s = np.array(me_mu_m),np.array(me_sigma_m),np.array(me_mu_f),np.array(me_sigma_f)
me_cpi = math.log(1 / 1000 * (255.657/163)) # from 1998 $ to 2019 $000 for log(me), only for mu, no need for sigma
me_mu_h += me_cpi
me_mu_s += me_cpi


mesh_Ph1 = Ph0 * eta + mesh_perm_h
mesh_Ps1 = Ps0 * eta + mesh_perm_s
mesh_me_h = np.exp(me_mu_h[t] + me_sigma_h[t] * (mesh_Ph1 + mesh_tran_h))
mesh_me_s = np.exp(me_mu_s[t] + me_sigma_s[t] * (mesh_Ps1 + mesh_tran_s))










# Step 2-3(b) alternative: no health risk. ME always takes mean

# 1-4: family risk:
#(1) (net) family transfer 65-74 ; 75-84; 85+
p_ft1, p_ft2, p_ft3 = 0.39, 0.33, 0.28
p_ft = np.array([p_ft1] * 10 + [p_ft2] * 10 + [p_ft3] * (T-10-10))
ft_mu1, ft_mu2, ft_mu3 = 13639, 14704, 16836
ft_mu = np.array([ft_mu1] * 10 + [ft_mu2] * 10 + [ft_mu3] * (T-10-10)) / 1000
#(2) spousal survival: p_s
# 1-5: policy risk: increasing prob of SSB COLA (2%) adjustment from 2020 (0%) to 2035 (100%)
p_SSadj = np.ones_like(p_h)
p_SSadj[0:15] = np.round(np.linspace(0,1,num=15),2)
SSadj = 0.02

# 1-7 Other regulation


# Step 2: model setup
# 2-0: special setup to eliminate risk
# None
# 2-1: model parameters
gamma = 5
beta = 0.96
inf_min = -1e+10


# 2-3: STATE VAR - def grid points (in $ and %), create grid mesh (mesh always in $)
step_X = 50.0
grid_X = np.rint(np.arange(0.0, 500.01, step_X)) # total wealth = L+K
grid_X[0] = L_min
grid_K = np.round(np.linspace(0.0, 1.0, num=10, endpoint=False),2) # K as % of X
grid_H = np.array([0,1]) # homeowner or not
#grid_P = stats.norm.ppf([0.1,0.3,0.5,0.7,0.9],perm_mu,perm_sigma) # CDF of perm part medical exp
grid_P = np.round(stats.norm.ppf([1/6,0.5,5/6],perm_mu,perm_sigma),3) # CDF of perm part medical exp
grid_N = np.array([1,2]) # number of family member
grid_A = np.array([0,1]) # SSB adjustment
grid_shape = [grid_X.shape[0],grid_K.shape[0], grid_H.shape[0], grid_P.shape[0], grid_P.shape[0], grid_N.shape[0], grid_A.shape[0]]
mesh_one = np.ones(grid_shape)
# 2-4 CHOICE VAR & policy function: optimal choice of C,D,S and optimal result V given state var
C_min = 1  # min consumption assumption
policy_C = [mesh_one] * (T + 1)
policy_D = [mesh_one] * (T + 1)
policy_S = [mesh_one] * (T + 1)
policy_V = [mesh_one] * (T + 1)
# 2-5 SHOCKS - def grid points using GQ
grid_GQ = np.round(np.array([-GQ_factor,0,GQ_factor]),3)
grid_re = re_mu + re_sigma*grid_GQ
grid_re = np.round(np.exp(grid_re),3)
grid_perm = np.round(np.array(perm_mu + perm_sigma * grid_GQ),3)
grid_tran = np.round(np.array(tran_mu + tran_sigma * grid_GQ),3)
grid_ft = np.stack([np.zeros_like(ft_mu),ft_mu],axis=1)
grid_ft = list(grid_ft.transpose().T)
grid_SSadj = np.array([0, SSadj])

# Step 3: Backward induction: @t0 decide C(L,K) and D(L,K) given optimized result V(L,K) @t1
# Terminal period: withdraw all and consume all if no bequest motive
def core_T(t):
    mesh_X, mesh_K, mesh_H, mesh_Ph, mesh_Ps, mesh_N, mesh_A = np.meshgrid(grid_X, grid_K, grid_H, grid_P, grid_P, grid_N, grid_A ,indexing='ij')
    mesh_K = mesh_X * mesh_K # K from % to $
    mesh_L = mesh_X - mesh_K
    mesh_C = mesh_L + mesh_K * (1-tax) + H_price[t] * mesh_H * (1-tax)
    mesh_V = utility(mesh_C,mesh_N)
    policy_C = mesh_one
    policy_D = mesh_one
    policy_S = mesh_one
    policy_V = mesh_V
    return(policy_C,policy_D,policy_S,policy_V)
policy_C[T],policy_D[T],policy_S[T],policy_V[T] = core_T(T)
# From T-1 backward induction in each period
def core(i, t, coords, policy_C, policy_D, policy_S, policy_V):
    # translate state var and set up choice var
    X0, K0, H0, Ph0, Ps0, N0, A0 = grid_X[coords[0]], grid_K[coords[1]], grid_H[coords[2]], grid_P[coords[3]], grid_P[coords[4]], grid_N[coords[5]], grid_A[coords[6]]
    if (X0==grid_X[0]) | ( (t>=15) & (A0==0) ): # after 2035, there will be sure SSB adjustment
        return(1,1,1,utility(X0,N0))
    K0 = X0 * K0 # K from % to $
    Ph0 = Ph0 * ((1-(eta*eta)**t)/(1-eta*eta))**0.5
    Ps0 = Ps0 * ((1-(eta*eta)**t)/(1-eta*eta))**0.5
    if (t >= T - 3) | (t == 71-65):
        if K0==0:
            grid_D = np.array([0])
        else:
            grid_D = np.arange(21)/20
            grid_D = np.extract(grid_D >= rmd_pct[t],grid_D)
        grid_C = np.arange(1,20)/20
        grid_S = np.arange(21)/20
    else:
        C_guess = policy_C[coords]
        D_guess = policy_D[coords]
        S_guess = policy_S[coords]
        grid_C = np.arange(max(.01, C_guess - .05), min(C_guess + .05, .99), .01)
        if K0==0:
            grid_D = np.array([0])
        else:
            grid_D = np.arange(max(rmd_pct[t], D_guess - .05), min(D_guess + .05, 1.0), .01) # not optimal to withdraw less than RMD
        grid_S = np.arange(max(0, S_guess - .05), min(S_guess + .05, 1.0), .01)
    # SHOCKS
    re, perm_h, tran_h, perm_s, tran_s, ft = grid_re, grid_perm, grid_tran, grid_perm, grid_tran, grid_ft[t]
    wgt_re, wgt_perm, wgt_tran = GQ_wgt, GQ_wgt, GQ_wgt
    wgt_ft = [1-p_ft[t],p_ft[t]]
    if N0 == 1:
        sp, wgt_sp = np.array([0]), [1]
    elif N0 == 2:
        sp, wgt_sp = np.array([0,1]), [1-p_s[t],p_s[t]]
    if A0 == 0:
        adj, wgt_adj = np.array([0,1]), [1-p_SSadj[t],p_SSadj[t]]
    elif A0 == 1:
        adj, wgt_adj = np.array([1]), [1]
    # evaluate at mesh grid for shocks
    mesh_C, mesh_D, mesh_S, mesh_re, mesh_perm_h, mesh_tran_h, mesh_perm_s, mesh_tran_s, mesh_ft, mesh_sp, mesh_adj = np.meshgrid(grid_C, grid_D, grid_S, re, perm_h, tran_h, perm_s, tran_s, ft, sp, adj, indexing='ij')
    mesh_shape = [grid_C.shape[0], grid_D.shape[0], grid_S.shape[0], re.shape[0], perm_h.shape[0], tran_h.shape[0], perm_s.shape[0], tran_s.shape[0], ft.shape[0], sp.shape[0], adj.shape[0]]
    mesh_Ph1 = Ph0 * eta + mesh_perm_h
    mesh_Ps1 = Ps0 * eta + mesh_perm_s
    mesh_me_h = np.exp(me_mu_h[t] + me_sigma_h[t] * (mesh_Ph1 + mesh_tran_h))
    mesh_me_s = np.exp(me_mu_s[t] + me_sigma_s[t] * (mesh_Ps1 + mesh_tran_s))
    mesh_L0 = X0 - K0 + (K0 * mesh_D) * (1-tax)  # comprehensive tax rule of SSB later (tax up to 50% or 85% of SSB based on combo inc)
    mesh_C = mesh_L0 * mesh_C
    mesh_L1 = mesh_L0 - mesh_C
    mesh_L1 *= (1 + (mesh_S * mesh_re + (1-mesh_S) * rf - 1) * (1-tax))
    mesh_L1 += (- H_price[t+1] * rent * (1-H0) - mesh_me_h - mesh_me_s * N0 - mesh_ft + (Y_h+Y_s*mesh_sp) * (1-mesh_adj*SSadj))
    mesh_K1 = K0 * (1-mesh_D) * (K_share[t] * mesh_re + (1-K_share[t]) * rf)
    # if L1 < L_min (medicaid limit), withdraw K to fill; if not enough and own home, sell home; if still not enough, claim medicaid
    mesh_DD = np.clip(L_min - mesh_L1,0,mesh_K1)
    mesh_K1 = mesh_K1 - mesh_DD
    mesh_L1 = mesh_L1 + mesh_DD
    mesh_H1 = np.ones_like(mesh_L1) * H0
    if H0 == 1:
        mesh_sellhome = mesh_L1 < L_min
        mesh_sellhome = mesh_sellhome.astype(int)
        mesh_L1 += H_price[t+1] * mesh_sellhome * (1-tax)
        mesh_H1 = mesh_H1 - mesh_sellhome
    # new state var grid at next period
    mesh_X1 = mesh_L1 + mesh_K1
    mesh_X1 = np.clip(mesh_X1,grid_X[0],grid_X[-1])
    mesh_K1 = np.clip(mesh_K1/mesh_X1,grid_K[0],grid_K[-1])
    grid_P1 = grid_P * ((1-(eta*eta)**(t+1))/(1-eta*eta))**0.5
    mesh_Ph1 = np.clip(mesh_Ph1, grid_P1[0], grid_P1[-1])
    mesh_Ps1 = np.clip(mesh_Ps1, grid_P1[0], grid_P1[-1])
    mesh_N1 = mesh_sp + 1
    mesh_A1 = mesh_adj

    grid = CGrid(grid_X,grid_K,grid_H,grid_P1,grid_P1,grid_N,grid_A)
    values = policy_V
    points = np.stack([mesh_X1.ravel(),mesh_K1.ravel(),mesh_H1.ravel(),mesh_Ph1.ravel(),mesh_Ps1.ravel(),mesh_N1.ravel(),mesh_A1.ravel()],axis=-1)
    mesh_EV1 = eval_linear(grid,values,points).reshape(mesh_shape)

    mesh_EV1 = np.average(mesh_EV1, axis=(-1), weights=wgt_adj)
    mesh_EV1 = np.average(mesh_EV1, axis=(-1), weights=wgt_sp)
    mesh_EV1 = np.average(mesh_EV1, axis=(-1), weights=wgt_ft)
    mesh_EV1 = np.average(mesh_EV1, axis=(-1), weights=wgt_tran)
    mesh_EV1 = np.average(mesh_EV1, axis=(-1), weights=wgt_perm)
    mesh_EV1 = np.average(mesh_EV1, axis=(-1), weights=wgt_tran)
    mesh_EV1 = np.average(mesh_EV1, axis=(-1), weights=wgt_perm)
    mesh_EV1 = np.average(mesh_EV1, axis=(-1), weights=wgt_re)
    mesh_C = mesh_C[:,:,:,0,0,0,0,0,0,0,0]
    mesh_V0 = utility(mesh_C, N0)
    mesh_V = mesh_V0 + beta * p_h[t] * mesh_EV1
    coord_C, coord_D, coord_S = np.unravel_index(np.nanargmax(mesh_V), mesh_V.shape)
    [C_max, D_max, S_max] = grid_C[coord_C], grid_D[coord_D], grid_S[coord_S]
    V_max = mesh_V[coord_C, coord_D, coord_S]
    return(i, C_max, D_max, S_max, V_max)
# parallel computing
coords = [(X,K,H,Ph,Ps,N,A) for X in range(grid_shape[0]) for K in range(grid_shape[1]) for H in range(grid_shape[2]) for Ph in range(grid_shape[3]) for Ps in range(grid_shape[4]) for N in range(grid_shape[5]) for A in range(grid_shape[6])]
start_time0 = time.time()
for t in range(T - 1, -1, -1):
    start_time = time.time()
    print('age = %d, t = %d' % (65 + t, t))
    pool = mp.Pool(processes=num_workers)
    results = pool.starmap_async(core,[(i, t, state, policy_C[t+1], policy_D[t+1], policy_S[t+1], policy_V[t+1]) for i, state in enumerate(coords)]).get()
    pool.close()
    pool.join()
    policy_V[t] = np.array([x[-1] for x in results]).reshape(grid_shape)
    policy_C[t] = np.array([x[1] for x in results]).reshape(grid_shape)
    policy_D[t] = np.array([x[2] for x in results]).reshape(grid_shape)
    policy_S[t] = np.array([x[3] for x in results]).reshape(grid_shape)
    if t>=15:
        policy_C[t][:,:,:,:,:,:,0] = policy_C[t][:,:,:,:,:,:,1]
        policy_D[t][:,:,:,:,:,:,0] = policy_D[t][:,:,:,:,:,:,1]
        policy_S[t][:,:,:,:,:,:,0] = policy_S[t][:,:,:,:,:,:,1]
        policy_V[t][:,:,:,:,:,:,0] = policy_V[t][:,:,:,:,:,:,1]
    print("--- %s mins for this age ---" % round((time.time() - start_time)/60, 2))
    np.savez(os.path.join(output_path,'policy_%s.npz' % version_name), T=t, C=policy_C, D=policy_D,S=policy_S,V=policy_V)
print("--- %s mins for whole program ---" % round((time.time() - start_time0)/60, 2))

# Step 4. Run simulations using policy function (all in $)
# initialzie simulation data
sim_num = 100000
sim_X = np.zeros([T+1,sim_num])
sim_K = np.zeros([T+1,sim_num])
sim_H = np.zeros([T+1,sim_num])
sim_Ph = np.zeros([T+1,sim_num])
sim_Ps = np.zeros([T+1,sim_num])
sim_N = np.zeros([T+1,sim_num])
sim_A = np.zeros([T+1,sim_num])
sim_C = np.zeros([T+1,sim_num])
sim_D = np.zeros([T+1,sim_num])
sim_S = np.zeros([T+1,sim_num])
sim_coord_X = np.zeros([T+1,sim_num])
sim_coord_K = np.zeros([T+1,sim_num])
sim_coord_H = np.zeros([T+1,sim_num])
sim_coord_Ph = np.zeros([T+1,sim_num])
sim_coord_Ps = np.zeros([T+1,sim_num])
sim_coord_N = np.zeros([T+1,sim_num])
sim_coord_A = np.zeros([T+1,sim_num])
sim_me_h = np.zeros([T+1,sim_num])
sim_me_s = np.zeros([T+1,sim_num])

rng = np.random.RandomState(seed=2)
sim_re = np.exp(rng.normal(re_mu,re_sigma,(T,sim_num)))
sim_perm_h = rng.normal(perm_mu,perm_sigma,(T,sim_num))
sim_tran_h = rng.normal(tran_mu,tran_sigma,(T,sim_num))
sim_perm_s = rng.normal(perm_mu,perm_sigma,(T,sim_num))
sim_tran_s = rng.normal(tran_mu,tran_sigma,(T,sim_num))
sim_ft = rng.binomial(1,p_ft,(sim_num,T))
sim_ft = sim_ft.transpose()
sim_sp = rng.binomial(1,p_s,(sim_num,T))
sim_sp = sim_sp.transpose()
sim_adj = rng.binomial(1,p_SSadj,(sim_num,T))
sim_adj = sim_adj.transpose()

# run simulation
sim_X[0,:] = L_h + K_h
sim_K[0,:] = K_h
sim_H[0,:] = 1
sim_Ph[0,:] = 0.0
sim_Ps[0,:] = 0.0
sim_N[0,:] = 2
sim_A[0,:] = 0
for t in range(0,T+1):
    X0 = sim_X[t,:]
    K0 = sim_K[t,:]
    H0 = sim_H[t,:]
    Ph0 = sim_Ph[t,:]
    Ps0 = sim_Ps[t,:]
    N0 = sim_N[t,:]
    A0 = sim_A[t,:]

    coord_X0 = interpolate.griddata(grid_X, np.arange(len(grid_X)), X0, method='linear')
    coord_K0 = interpolate.griddata(grid_K, np.arange(len(grid_K)), np.round(K0/X0,2), method='linear')
    coord_H0 = H0
    grid_P0 = grid_P * ((1-(eta*eta)**t)/(1-eta*eta))**0.5
    coord_Ph0 = interpolate.griddata(grid_P0, np.arange(len(grid_P0)), Ph0, method='linear')
    coord_Ps0 = interpolate.griddata(grid_P0, np.arange(len(grid_P0)), Ps0, method='linear')
    coord_N0 = N0 - 1
    coord_A0 = A0
    sim_coord_X[t,:] = coord_X0
    sim_coord_K[t,:] = coord_K0
    sim_coord_H[t,:] = coord_H0
    sim_coord_Ph[t,:] = coord_Ph0
    sim_coord_Ps[t,:] = coord_Ps0
    sim_coord_N[t,:] = coord_N0
    sim_coord_A[t,:] = coord_A0
    D = K0 * ndimage.map_coordinates(policy_D[t],[coord_X0, coord_K0, coord_H0, coord_Ph0, coord_Ps0, coord_N0, coord_A0], order=1, mode='constant', cval=inf_min)
    L0 = X0 - K0 + D * (1-tax)
    C = L0 * ndimage.map_coordinates(policy_C[t],[coord_X0, coord_K0, coord_H0, coord_Ph0, coord_Ps0, coord_N0, coord_A0], order=1, mode='constant', cval=inf_min)
    L0 = L0 - C
    S = L0 * ndimage.map_coordinates(policy_S[t],[coord_X0, coord_K0, coord_H0, coord_Ph0, coord_Ps0, coord_N0, coord_A0], order=1, mode='constant', cval=inf_min)
    sim_C[t,:] = C
    sim_D[t,:] = D
    sim_S[t,:] = S
    if t < T:
        Ph1 = eta * Ph0 + sim_perm_h[t,:]
        Ps1 = eta * Ps0 + sim_perm_s[t,:]
        me_h = np.exp(me_mu_h[t] + me_sigma_h[t]*(Ph1 + sim_tran_h[t,:]))
        me_s = np.exp(me_mu_s[t] + me_sigma_s[t]*(Ps1 + sim_tran_s[t,:]))
        sim_me_h[t,:] = me_h
        sim_me_s[t,:] = me_s
        sim_sp[t,:][N0==1] = 0
        N1 = sim_sp[t,:] + 1
        sim_adj[t,:][A0==1] = 1
        A1 = sim_adj[t,:]
        L1 = L0 + (S * sim_re[t,:] + (L0 - S) * rf - L0) * (1-tax)
        L1 += (- H_price[t+1] * rent * (1-H0) - me_h - me_s * (N0-1) - sim_ft[t,:] * ft_mu[t] + (Y_h + Y_s * sim_sp[t,:]) * (1 - SSadj*sim_adj[t,:]))
        K1 = (K0 - D) * (K_share[t] * sim_re[t,:] + (1-K_share[t]) * rf)
        D1 = np.clip(L_min - L1,0,K1)
        K1 = K1 - D1
        L1 += D1
        sellhome = (L1 < L_min) & (H0 == 1)
        sellhome = sellhome.astype(int)
        L1 += H_price[t+1] * sellhome * (1-tax)
        H1 = H0 - sellhome
        L1 = np.clip(L1,L_min,None)
        X1 = L1 + K1
        X1 = np.clip(X1,grid_X[0],grid_X[-1])
        K1 = np.clip(K1/X1,grid_K[0],grid_K[-1]) * X1
        grid_P1 = grid_P * ((1-(eta*eta)**(t+1))/(1-eta*eta))**0.5 # grid_P1 = grid_P * (1-eta**(t+1))/(1-eta)
        Ph1 = np.clip(Ph1,grid_P1[0],grid_P1[-1])
        Ps1 = np.clip(Ps1,grid_P1[0],grid_P1[-1])
        sim_X[t+1,:] = X1
        sim_K[t+1,:] = K1
        sim_H[t+1,:] = H1
        sim_Ph[t+1,:] = Ph1
        sim_Ps[t+1,:] = Ps1
        sim_N[t+1,:] = N1
        sim_A[t+1,:] = A1
np.savez(os.path.join(output_path,'sim_%s.npz' % version_name), X=sim_X,K=sim_K,H=sim_H,Ph=sim_Ph,Ps=sim_Ps,N=sim_N,A=sim_A,C=sim_C,D=sim_D,S=sim_S)
# Step 5. Plot figures
# life time avg
avg_X = np.average(sim_X,axis=-1)
#std_X = np.std(sim_X,axis=-1)
avg_K = np.average(sim_K,axis=-1)
avg_C = np.average(sim_C,axis=-1)
avg_D = np.average(sim_D,axis=-1)
#sim_D_pct = sim_D/sim_K
#avg_D_pct = np.average(sim_D_pct,axis=-1)
avg_me_h = np.average(sim_me_h,axis=-1)
avg_me_s = np.average(sim_me_s,axis=-1)
avg_Ph = np.average(sim_Ph,axis=-1)
avg_Ps = np.average(sim_Ps,axis=-1)
avg_H = np.average(sim_H,axis=-1)
avg_W = avg_H * H_price + avg_X
avg_N = np.average(sim_N,axis=-1)
avg_A = np.average(sim_A,axis=-1)
avg_me_hh = avg_me_h + avg_me_s * (avg_N-1)
# plot figure
plt.figure(figsize=(20,10))
plt.clf()
fig1_x = np.arange(T+1)+65
plt.plot(fig1_x, avg_W, marker='o', markerfacecolor='orange', markersize=1, color='orange', linewidth=1)
plt.plot(fig1_x, avg_X, marker='o', markerfacecolor='blue', markersize=1, color='skyblue', linewidth=1)
plt.plot(fig1_x, avg_K, marker='o', markerfacecolor='red', markersize=1, color='tomato', linewidth=1)
plt.plot(fig1_x, avg_C, marker='o', markerfacecolor='black', markersize=1, color='brown', linewidth=1)
plt.plot(fig1_x, avg_D, marker='o', markerfacecolor='green', markersize=1, color='yellowgreen', linewidth=1)
plt.plot(fig1_x, avg_me_hh, marker='o', markerfacecolor='gold', markersize=1, color='gold', linewidth=1)
plt.legend(['Total wealth (w/ Housing)','Total wealth (w/o Housing)', '401k','Consumption','401k Distribution','Medical Expense (HH)'], loc='upper right')
plt.xticks(np.arange(65,101,5))
plt.xlim(65, 101)
plt.yticks(np.arange(0,L_h+K_h+251,50))
plt.ylim(0, L_h+K_h+251)
plt.grid(color='black',axis = 'y')
plt.xlabel('Age')
plt.ylabel('Amount')

plt.savefig(os.path.join(output_path,'Figure_%s.pdf' % version_name))
# output data to EXCEL
data_avg = np.array([avg_X,avg_K,avg_C,avg_D,avg_me_h,avg_me_s,avg_me_hh,avg_Ph,avg_Ps,avg_H,avg_W,avg_N,avg_A])
df_avg = pd.DataFrame(np.transpose(data_avg),columns = ['X','K','C','D','me_h','me_s','me_hh','Ph','Ps','H','W','N','A'],index=range(65,121))
with pd.ExcelWriter(os.path.join(output_path,"LCM_%s.xlsx" % version_name)) as writer:
    df_avg.to_excel(writer, sheet_name = 'SimAvg', index=True)

