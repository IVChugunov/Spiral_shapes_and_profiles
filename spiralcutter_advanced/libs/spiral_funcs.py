import numpy as np

def sp_o1(x, r0, k1):
    return r0 * np.exp(k1 * x)

def sp_o2(x, r0, k1, k2):
    return r0 * np.exp(k1 * x + k2 * x ** 2)

def sp_o3(x, r0, k1, k2, k3):
    return r0 * np.exp(k1 * x + k2 * x ** 2 + k3 * x ** 3)

def sp_o4(x, r0, k1, k2, k3, k4):
    return r0 * np.exp(k1 * x + k2 * x ** 2 + k3 * x ** 3 + k4 * x ** 4)

def sp_o5(x, r0, k1, k2, k3, k4, k5):
    return r0 * np.exp(k1 * x + k2 * x ** 2 + k3 * x ** 3 + k4 * x ** 4 + k5 * x ** 5)

def sp_wave(x, r0, k1, A, per, phi0):
    return r0 * np.exp(k1 * x) * (1 + A * np.sin((2 * np.pi * x / per) + phi0))

def sp_o1_br1(x, r0, xbr, ka1, kb1):
    rbr = r0 * np.exp(ka1 * xbr)
    ri = r0 * np.exp(ka1 * x)
    ro = rbr * np.exp(kb1 * (x - xbr))
    r = np.where(x > xbr, ro, ri)
    return r

def sp_o2_br1(x, r0, xbr, ka1, ka2, kb1, kb2):
    rbr = r0 * np.exp(ka1 * xbr + ka2 * xbr ** 2)
    ri = r0 * np.exp(ka1 * x + ka2 * x ** 2)
    ro = rbr * np.exp(kb1 * (x - xbr) + kb2 * (x - xbr) ** 2)
    r = np.where(x > xbr, ro, ri)
    return r

def sp_o3_br1(x, r0, xbr, ka1, ka2, ka3, kb1, kb2, kb3):
    rbr = r0 * np.exp(ka1 * xbr + ka2 * xbr ** 2 + ka3 * xbr ** 3)
    ri = r0 * np.exp(ka1 * x + ka2 * x ** 2 + ka3 * x ** 3)
    ro = rbr * np.exp(kb1 * (x - xbr) + kb2 * (x - xbr) ** 2 + kb3 * (x - xbr) ** 3)
    r = np.where(x > xbr, ro, ri)
    return r

def sp_o1_br2(x, r0, xbr1, xbr2, ka1, kb1, kc1):
    if xbr1 > xbr2:
        xbr1, xbr2 = xbr2, xbr1
    rbr1 = r0 * np.exp(ka1 * xbr1)
    rbr2 = rbr1 * np.exp(kb1 * (xbr2 - xbr1))
    ra = r0 * np.exp(ka1 * x)
    rb = rbr1 * np.exp(kb1 * (x - xbr1))
    rc = rbr2 * np.exp(kc1 * (x - xbr2))
    rab = np.where(x > xbr1, rb, ra)
    r = np.where(x > xbr2, rc, rab)
    return r

def sp_o2_br2(x, r0, xbr1, xbr2, ka1, ka2, kb1, kb2, kc1, kc2):
    if xbr1 > xbr2:
        xbr1, xbr2 = xbr2, xbr1
    rbr1 = r0 * np.exp(ka1 * xbr1 + ka2 * xbr1 ** 2)
    rbr2 = rbr1 * np.exp(kb1 * (xbr2 - xbr1) + kb2 * (xbr2 - xbr1) ** 2)
    ra = r0 * np.exp(ka1 * x + ka2 * x ** 2)
    rb = rbr1 * np.exp(kb1 * (x - xbr1) + kb2 * (x - xbr1) ** 2)
    rc = rbr2 * np.exp(kc1 * (x - xbr2) + kc2 * (x - xbr2) ** 2)
    rab = np.where(x > xbr1, rb, ra)
    r = np.where(x > xbr2, rc, rab)
    return r

def arch_sp(x, r0, k1):
    b = k1 * r0
    r = r0 + b * x
    return r

def arch_sp_o3(x, r0, k1, k2, k3):
    r = r0 + r0 * (k1 * x + k2 * x ** 2 + k3 * x ** 3)
    return r

def hyp_sp(x, r0, k1):
    phi_t = 1 / k1
    a = r0 * phi_t
    r = a / (phi_t - x)
    return r

def hyp_sp_o3(x, r0, k1, k2, k3):
    phi_t = 1 / k1
    a = r0 * phi_t
    x_c = (phi_t - x)
    r = a / (x_c + k2 * x_c ** 2 + k3 * x_c ** 3)
    return r