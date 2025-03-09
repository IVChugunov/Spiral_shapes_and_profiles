import numpy as np
from .spiral_funcs import *
from .ImfitModelFork import *

def main_func_quadr(X, o0, o1, I0, ih, gr, co, st, end, w0, w1):
    phi, r, rho_init = X

    offs = o0 + o1 * r
    rho = rho_init + offs

    g_gr = np.clip(1 - (((phi - st - gr) / (gr)) ** 2), 0, 1)
    g_e = np.clip(1 - (((phi - end + co) / (co)) ** 2), 0, 1)
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    hw_loc = (w0 + w1 * r) / 2
    I_perp = np.exp(-np.log(2) * np.power(np.abs(rho / hw_loc), 2))

    I = I_parallel * I_perp
    return I

def main_func_gaussian(X, o0, o1, I0, ih, gr, co, st, end, w0, w1):
    phi, r, rho_init = X

    offs = o0 + o1 * r
    rho = rho_init + offs

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    hw_loc = (w0 + w1 * r) / 2
    I_perp = np.exp(-np.log(2) * np.power(np.abs(rho / hw_loc), 2))

    I = I_parallel * I_perp
    return I

def main_func_linear(X, o0, o1, I0, ih, gr, co, st, end, w0, w1):
    phi, r, rho_init = X

    offs = o0 + o1 * r
    rho = rho_init + offs

    g_gr = np.clip((phi - st) / gr, 0, 1)
    g_e = np.clip((end - phi) / co, 0, 1)

    hvs = g_gr * g_e
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    hw_loc = (w0 + w1 * r) / 2
    I_perp = np.exp(-np.log(2) * np.power(np.abs(rho / hw_loc), 2))

    I = I_parallel * I_perp
    return I

def main_func_qsm(X, o0, o1, I0, ih, gr, co, st, end, w0, w1):
    phi, r, rho_init = X

    offs = o0 + o1 * r
    rho = rho_init + offs

    phstgr = np.clip((phi - st) / gr, 0, 1)
    endphico = np.clip((end - phi) / co, 0, 1)

    g_gr = 3 * np.power(phstgr, 2) - 2 * np.power(phstgr, 3)
    g_e = 3 * np.power(endphico, 2) - 2 * np.power(endphico, 3)

    hvs = g_gr * g_e
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    hw_loc = (w0 + w1 * r) / 2
    I_perp = np.exp(-np.log(2) * np.power(np.abs(rho / hw_loc), 2))

    I = I_parallel * I_perp
    return I

# ==============================================
# ============== Overly simple =================
# ==============================================

def o1_r_o1_r(X, I0, ih1, w0, w1): #profile along the arm _ width
    phi, r, rho = X

    I_parallel = I0 * np.exp(-(ih1 * r))
    w_loc = w0 + w1 * r
    I_bot = np.exp(-(rho) ** 2 / (2 * w_loc ** 2))
    
    I = I_parallel * I_bot
    return I