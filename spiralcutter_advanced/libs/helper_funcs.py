import numpy as np
import os
from .spiral_funcs import *
from .ImfitModelFork import *

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def gaussian_c(x, A, sigma):
    return A * np.exp(-(x ** 2) / (2 * sigma ** 2))

def gaussian_skewed(x, A, sigma_1, sigma_2):
    peak = (np.abs(sigma_1) - np.abs(sigma_2)) * np.sqrt(2 / np.pi)
    xp = x - peak
    y_1 = A * np.exp(-(xp ** 2) / (2 * sigma_1 ** 2))
    y_2 = A * np.exp(-(xp ** 2) / (2 * sigma_2 ** 2))
    y = np.where(xp > 0, y_2, y_1)
    return y

def sersic_asymm(x, A, w_1, w_2, n):
    y_1 = A * np.exp(-np.log(2) * (np.abs(x / w_1) ** (1 / n)))
    y_2 = A * np.exp(-np.log(2) * (np.abs(x / w_2) ** (1 / n)))
    y = np.where(x > 0, y_2, y_1)
    return y

def sersic_asymm_offs(x_init, A, w_1, w_2, n, off):
    x = x_init - off
    y_1 = A * np.exp(-np.log(2) * (np.abs(x / w_1) ** (1 / n)))
    y_2 = A * np.exp(-np.log(2) * (np.abs(x / w_2) ** (1 / n)))
    y = np.where(x > 0, y_2, y_1)
    return y

def gaussian_asymm_offs(x_init, A, w_1, w_2, off):
    x = x_init - off
    y_1 = A * np.exp(-np.log(2) * (np.abs(x / w_1) ** 2))
    y_2 = A * np.exp(-np.log(2) * (np.abs(x / w_2) ** 2))
    y = np.where(x > 0, y_2, y_1)
    return y

def sersic_asymm_offs_alt(x_init, A, w, sk, n, off):
    w_1 = w * (1 - sk) / 2
    w_2 = w * (1 + sk) / 2
    x = x_init - off
    y_1 = A * np.exp(-np.log(2) * (np.abs(x / w_1) ** (1 / n)))
    y_2 = A * np.exp(-np.log(2) * (np.abs(x / w_2) ** (1 / n)))
    y = np.where(x > 0, y_2, y_1)
    return y

def gaussian_asymm_offs_alt(x_init, A, w, sk, off):
    w_1 = w * (1 - sk) / 2
    w_2 = w * (1 + sk) / 2
    x = x_init - off
    y_1 = A * np.exp(-np.log(2) * (np.abs(x / w_1) ** 2))
    y_2 = A * np.exp(-np.log(2) * (np.abs(x / w_2) ** 2))
    y = np.where(x > 0, y_2, y_1)
    return y

def linear(x, a, b):
    return a * x + b

def find_fit_params(imfit_file):
    # find the biggest disc, then its center and position angle
    imfit_model = ImfitModel(imfit_file)
    disc = imfit_model.get_disc()
    xc, yc = disc.get_center()
    pa = disc.get_par_by_name("PA").value
    ell = disc.get_par_by_name("ell").value
    return xc, yc, pa, ell

def find_disc_params(imfit_file):
    imfit_model = ImfitModel(imfit_file)
    disc = imfit_model.get_disc()
    try:
        h = disc.get_par_by_name("h").value * 0.75
        r_br = np.nan
    except Exception:
        h = [disc.get_par_by_name("h1").value * 0.75, disc.get_par_by_name("h2").value * 0.75]
        r_br = disc.get_par_by_name("r_break").value * 0.75
    return h, r_br

def read_shapes_file(pa, fname = None, uv = False, rb = False, return_fname = False):
    col_arr = []
    phi_arr = []
    r_arr = []
    fn_arr = []
    if uv:
        this_gal = os.getcwd().split("/")[-1]
        path_sh = f"../../galaxies_images/{this_gal}/spiral_params/shapes.dat"
    elif rb:
        this_gal = os.getcwd().split("/")[-1]
        path_sh = f"../../rebinned_fit_results/{this_gal}/spiral_params/shapes.dat"
    else:
        path_sh = "spiral_params/shapes.dat"
    if fname is not None:
        path_sh = fname
    with open(path_sh, "r") as file:
        for line in file:
            col = line.split()[0]
            func_n = line.split()[1]
            phi0 = float(line.split()[2])
            phie = float(line.split()[3])
            sp_fit = np.array(line.split("[")[1][:-2].split()).astype(float)

            #phi_arm = np.radians(np.arange(0, phie - phi0, 0.5))
            if rb:
                phi_arm = np.radians(np.arange(-30, phie - phi0 + 30, 1))
            else:
                phi_arm = np.radians(np.arange(-30, phie - phi0 + 30, 0.5))

            for sp_func in [sp_o1, sp_o2, sp_o3, sp_o4, sp_o5, sp_o1_br1, sp_o2_br1, sp_o3_br1, sp_o1_br2, sp_o2_br2, arch_sp, arch_sp_o3, hyp_sp, hyp_sp_o3, sp_wave]:
                if sp_func.__name__ == func_n:
                    if np.isnan(sp_fit[0]):
                        r = np.nan
                    else:
                        r = sp_func(phi_arm, *sp_fit)
                    break

            phi_span = phi_arm + np.radians(phi0 - pa - 90)

            col_arr.append(col)
            phi_arr.append(phi_span)
            fn_arr.append(func_n)
            if uv:
                r = r / 2
            if rb:
                r_arr.append(r / 3)
            else:
                r_arr.append(r / 0.75) #in pixels
    if return_fname:
        return (col_arr, phi_arr, r_arr, fn_arr)
    else:
        return (col_arr, phi_arr, r_arr)

def squeeze(image, sigma, phi, r, sf):
    weights = 1 / (sigma ** 2)
    arm_w, arm_l = np.shape(image)
    arm_l_s = int(np.ceil(arm_l / sf))
    image_s = np.zeros((arm_w, arm_l_s))
    weights_s = np.zeros((arm_w, arm_l_s))
    phi_s = np.zeros(arm_l_s)
    r_s = np.zeros(arm_l_s)
    for i in range(len(r_s)):
        i_cut = image[:, sf * i:sf * (i + 1)]
        w_cut = weights[:, sf * i:sf * (i + 1)]
        phi_cut = phi[sf * i:sf * (i + 1)]
        r_cut = r[sf * i:sf * (i + 1)]
        image_s[:, i] = np.nansum(i_cut * w_cut, axis = 1) / np.nansum(w_cut, axis = 1)
        weights_s[:, i] = np.nansum(w_cut, axis = 1)
        phi_s[i] = np.nanmean(phi_cut)
        r_s[i] = np.nanmean(r_cut)
    sigma_s = np.sqrt(1 / weights_s)
    return image_s, sigma_s, phi_s, r_s
