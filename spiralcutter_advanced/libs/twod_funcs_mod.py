import numpy as np
from .spiral_funcs import *
from .ImfitModelFork import *

def main_func_w_n(X, o0, o1, I0, ih, gr, co, st, end, w0, w1, S0, S1, n):
    phi, r, rho_init = X

    offs = o0 + o1 * r
    rho = rho_init + offs
    ninv = 1 / n

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    w_loc = w0 + w1 * r
    S_loc = S0 + S1 * r
    w_in = w_loc * (1 - S_loc) / 2
    w_out = w_loc * (1 + S_loc) / 2

    I_perp_in = np.exp(-np.log(2) * np.power(np.abs(rho / w_in), ninv))
    I_perp_out = np.exp(-np.log(2) * np.power(np.abs(rho / w_out), ninv))
    I_perp = np.where(rho > 0, I_perp_out, I_perp_in)

    I = I_parallel * I_perp
    return I

def main_func_w_asymm(X, o0, o1, I0, ih, gr, co, st, end, w0, w1, S0, S1):
    phi, r, rho_init = X

    offs = o0 + o1 * r
    rho = rho_init + offs

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    w_loc = w0 + w1 * r
    S_loc = S0 + S1 * r
    w_in = w_loc * (1 - S_loc) / 2
    w_out = w_loc * (1 + S_loc) / 2

    I_perp_in = np.exp(-np.log(2) * np.power(np.abs(rho / w_in), 2))
    I_perp_out = np.exp(-np.log(2) * np.power(np.abs(rho / w_out), 2))
    I_perp = np.where(rho > 0, I_perp_out, I_perp_in)

    I = I_parallel * I_perp
    return I

def main_func(X, o0, o1, I0, ih, gr, co, st, end, w0, w1):
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

def main_func_sharp(X, o0, o1, I0, ih, st, end, w0, w1):
    phi, r, rho_init = X
    offs = o0 + o1 * r
    rho = rho_init + offs

    hvs = np.where(phi < st, 0, 1)
    hvs = hvs * np.where(phi > end, 0, 1)
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    hw_loc = (w0 + w1 * r) / 2
    I_perp = np.exp(-np.log(2) * np.power(np.abs(rho / hw_loc), 2))

    I = I_parallel * I_perp
    return I

def main_func_cw(X, o0, o1, I0, ih, gr, co, st, end, w0):
    phi, r, rho_init = X

    offs = o0 + o1 * r
    rho = rho_init + offs

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    hw_loc = w0 / 2
    I_perp = np.exp(-np.log(2) * np.power(np.abs(rho / hw_loc), 2))

    I = I_parallel * I_perp
    return I


def main_func_nozpw(X, o0, o1, I0, ih, gr, co, st, end, w1):
    phi, r, rho_init = X

    offs = o0 + o1 * r
    rho = rho_init + offs

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    hw_loc = w1 * r / 2
    I_perp = np.exp(-np.log(2) * np.power(np.abs(rho / hw_loc), 2))

    I = I_parallel * I_perp
    return I

def main_func_cbr(X, o0, o1, I0, gr, co, st, end, w0, w1):
    phi, r, rho_init = X

    offs = o0 + o1 * r
    rho = rho_init + offs

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * hvs

    hw_loc = (w0 + w1 * r) / 2
    I_perp = np.exp(-np.log(2) * np.power(np.abs(rho / hw_loc), 2))

    I = I_parallel * I_perp
    return I

# ==============================================
# ================= With dips ==================
# ==============================================

def main_func_d1(X, o0, o1, I0, ih, gr, co, st, end, w0, w1, ad1, ld1, wd1):
    phi, r, rho_init = X
    offs = o0 + o1 * r
    rho = rho_init + offs

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    dips = 1 + ad1 * np.exp(-((phi - ld1) ** 2) / (2 * wd1 ** 2))
    I_parallel = I0 * np.exp(-(ih * r)) * hvs / dips

    hw_loc = (w0 + w1 * r) / 2
    I_perp = np.exp(-np.log(2) * np.power(np.abs(rho / hw_loc), 2))

    I = I_parallel * I_perp
    return I

def main_func_d2(X, o0, o1, I0, ih, gr, co, st, end, w0, w1, ad1, ld1, wd1, ad2, ld2, wd2):
    phi, r, rho_init = X
    offs = o0 + o1 * r
    rho = rho_init + offs

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    dips = 1 + ad1 * np.exp(-((phi - ld1) ** 2) / (2 * wd1 ** 2)) + ad2 * np.exp(-((phi - ld2) ** 2) / (2 * wd2 ** 2))
    I_parallel = I0 * np.exp(-(ih * r)) * hvs / dips

    hw_loc = (w0 + w1 * r) / 2
    I_perp = np.exp(-np.log(2) * np.power(np.abs(rho / hw_loc), 2))

    I = I_parallel * I_perp
    return I

# ==============================================
# ================ Add. params =================
# ==============================================

def main_func_add(X, o0, o1, I0, ih, ih2, gr, co, st, end, w0, w1, w2):
    phi, r, rho_init = X

    offs = o0 + o1 * r
    rho = rho_init + offs

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * r + ih2 * np.power(r, 2))) * hvs

    hw_loc = (w0 + w1 * r + w2 * np.power(r, 2)) / 2
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