import numpy as np
from .spiral_funcs import *
from .ImfitModelFork import *

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def gaussian_c(x, A, sigma):
    return A * np.exp(-(x ** 2) / (2 * sigma ** 2))

def exponential(x, I0, ih):
    return I0 * np.exp(-(x * ih))

def exponential_mg_phi(X, I0, ih, gr, co, st, end):
    phi, r = X
    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I = I0 * np.exp(-(ih * r)) * hvs
    return I

def exp_mg_phi_wd1(X, I0, ih, gr, co, st, end, a1, l1, w1):
    phi, r = X
    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    a1 = np.abs(a1)
    dips = 1 + a1 * np.exp(-((phi - l1) ** 2) / (2 * w1 ** 2))
    I = I0 * np.exp(-(ih * r)) * hvs / dips
    return I

def exp_mg_phi_wd2(X, I0, ih, gr, co, st, end, a1, l1, w1, a2, l2, w2):
    phi, r = X
    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    a1 = np.abs(a1)
    a2 = np.abs(a2)
    dips = 1 + a1 * np.exp(-((phi - l1) ** 2) / (2 * w1 ** 2)) + a2 * np.exp(-((phi - l2) ** 2) / (2 * w2 ** 2))
    I = I0 * np.exp(-(ih * r)) * hvs / dips
    return I

def exponential_mg(x, I0, ih, gr, co, st, end):
    gr = np.abs(gr)
    co = np.abs(co)
    g_gr = np.exp(-((x - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((x - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(x < st + gr, g_gr, 1)
    hvs = hvs * np.where(x > end - co, g_e, 1)
    return I0 * np.exp(-(x * ih)) * hvs

def exp_o2_mg(x, I0, a1, a2, gr, co, st, end):
    gr = np.abs(gr)
    co = np.abs(co)
    g_gr = np.exp(-((x - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((x - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(x < st + gr, g_gr, 1)
    hvs = hvs * np.where(x > end - co, g_e, 1)
    return I0 * np.exp(-(a1 * x + a2 * x ** 2)) * hvs

def exp_mg_wd1(x, I0, ih, gr, co, st, end, a1, l1, w1):
    gr = np.abs(gr)
    co = np.abs(co)
    g_gr = np.exp(-((x - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((x - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(x < st + gr, g_gr, 1)
    hvs = hvs * np.where(x > end - co, g_e, 1)
    a1 = np.abs(a1)
    dips = 1 + a1 * np.exp(-((x - l1) ** 2) / (2 * w1 ** 2))
    return I0 * np.exp(-(x * ih)) * hvs / dips

def exp_mg_wd2(x, I0, ih, gr, co, st, end, a1, l1, w1, a2, l2, w2):
    gr = np.abs(gr)
    co = np.abs(co)
    g_gr = np.exp(-((x - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((x - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(x < st + gr, g_gr, 1)
    hvs = hvs * np.where(x > end - co, g_e, 1)
    a1 = np.abs(a1)
    a2 = np.abs(a2)
    dips = 1 + a1 * np.exp(-((x - l1) ** 2) / (2 * w1 ** 2)) + a2 * np.exp(-((x - l2) ** 2) / (2 * w2 ** 2))
    return I0 * np.exp(-(x * ih)) * hvs / dips

def exp_b1(x, I0, ih1, ih2, r_br):
    I_br = I0 * np.exp(-(r_br * ih1))
    I_i = I0 * np.exp(-(x * ih1))
    I_o = I_br * np.exp(-((x - r_br) * ih2))
    I = np.where(x > r_br, I_o, I_i)
    return I

def exp_b2(x, I0, ih1, ih2, ih3, r_br1, r_br2):
    if r_br1 > r_br2:
        r_br1, r_br2 = r_br2, r_br1
    I_br1 = I0 * np.exp(-(r_br1 / ih1))
    I_br2 = I_br1 * np.exp(-((r_br2 - r_br1) / ih2))
    I_1 = I0 * np.exp(-(x / ih1))
    I_2 = I_br1 * np.exp(-((x - r_br1) / ih2))
    I_3 = I_br2 * np.exp(-((x - r_br2) / ih3))
    I_12 = np.where(x > r_br1, I_2, I_1)
    I = np.where(x > r_br2, I_3, I_12)
    return I

def exp_o2(x, I0, a1, a2):
    return I0 * np.exp(-(a1 * x + a2 * x ** 2))

def exp_o3(x, I0, a1, a2, a3):
    return I0 * np.exp(-(a1 * x + a2 * x ** 2 + a3 * x ** 3))

def exp_o4(x, I0, a1, a2, a3, a4):
    return I0 * np.exp(-(a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4))

def sersic(x, I0, r_e, n):
    return I0 * np.exp(-((np.abs(x / r_e)) ** (1 / n)))

def linear(x, a, b):
    return a * x + b

def poly_2(x, a, b, c):
    return a * x ** 2 + b * x + c

def poly_3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def broken_exp(x, I0, h1, h2, r_br):
    I_br = I0 * np.exp(-(r_br / h1))
    I_i = I0 * np.exp(-(x / h1))
    I_o = I_br * np.exp(-((x - r_br) / h2))
    I = np.where(x > r_br, I_o, I_i)
    return I