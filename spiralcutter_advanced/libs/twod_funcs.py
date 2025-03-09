import numpy as np
from .spiral_funcs import *
from .ImfitModelFork import *

def o1_r_o1_r(X, I0, ih1, w0, w1): #profile along the arm _ width
    phi, r, rho = X

    I_parallel = I0 * np.exp(-(ih1 * r))
    w_loc = w0 + w1 * r
    I_bot = np.exp(-(rho) ** 2 / (2 * w_loc ** 2))
    
    I = I_parallel * I_bot
    return I

def fin_phi_o2_r(X, I0, ih, gr, co, st, end, w0, w1, w2):
    phi, r, rho = X

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * phi)) * hvs

    w_loc = w0 + w1 * r + w2 * r ** 2
    I_perp = np.exp(-np.power(rho, 2) / (2 * np.power(w_loc, 2)))

    I = I_parallel * I_perp
    return I

def fin_r_o1_r(X, I0, ih, gr, co, st, end, w0, w1):
    phi, r, rho = X

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    w_loc = w0 + w1 * r
    I_perp = np.exp(-np.power(rho, 2) / (2 * np.power(w_loc, 2)))

    I = I_parallel * I_perp
    return I

def fin_r_o2_r(X, I0, ih, gr, co, st, end, w0, w1, w2):
    phi, r, rho = X

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * r)) * hvs

    w_loc = w0 + w1 * r + w2 * r ** 2
    I_perp = np.exp(-np.power(rho, 2) / (2 * np.power(w_loc, 2)))

    I = I_parallel * I_perp
    return I

def fin_r_o1_r_d1(X, I0, ih, gr, co, st, end, w0, w1, ad1, ld1, wd1):
    phi, r, rho = X

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    dips = 1 + ad1 * np.exp(-((phi - ld1) ** 2) / (2 * wd1 ** 2))
    I_parallel = I0 * np.exp(-(ih * r)) * hvs / dips

    w_loc = w0 + w1 * r
    I_perp = np.exp(-np.power(rho, 2) / (2 * np.power(w_loc, 2)))

    I = I_parallel * I_perp
    return I

def fin_r_o1_r_d2(X, I0, ih, gr, co, st, end, w0, w1, ad1, ld1, wd1, ad2, ld2, wd2):
    phi, r, rho = X

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    dips = 1 + ad1 * np.exp(-((phi - ld1) ** 2) / (2 * wd1 ** 2)) + ad2 * np.exp(-((phi - ld2) ** 2) / (2 * wd2 ** 2))
    I_parallel = I0 * np.exp(-(ih * r)) * hvs / dips

    w_loc = w0 + w1 * r
    I_perp = np.exp(-np.power(rho, 2) / (2 * np.power(w_loc, 2)))

    I = I_parallel * I_perp
    return I

def o3_r_o3_r(X, I0, h1, h2, h3, w0, w1, w2, w3):
    phi, r, rho = X

    I_parallel = I0 * np.exp(-(h1 * r + h2 * r ** 2 + h3 * r ** 3))
    w_loc = w0 + w1 * r + w2 * r ** 2 + w3 * r ** 3
    I_bot = np.exp(-(rho) ** 2 / (2 * w_loc ** 2))
    
    I = I_parallel * I_bot
    return I

def o1_r_exp_phi(X, I0, h1, w0, w1):
    phi, r, rho = X

    I_parallel = I0 * np.exp(-(h1 * r))
    w_loc = w0 * np.exp(w1 * phi)
    I_bot = np.exp(-(rho) ** 2 / (2 * w_loc ** 2))
    
    I = I_parallel * I_bot
    return I