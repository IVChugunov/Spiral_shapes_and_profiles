import os
from functools import partial
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline, CubicSpline
from scipy.ndimage import rotate
from libs.spiral_funcs import *
from libs.helper_funcs import *
from libs.profile_funcs import *
from libs.twod_funcs import *
from scipy.optimize import curve_fit
import pywt
from nfft import nfft

def lin_0(x, a):
    return a * x
    
def find_disc_params(gal, fn):
    path = f"../fit_results/{gal}/{fn}"
    disc_found = False
    h = []
    r_br = np.nan
    with open(path, "r") as file:
        for line in file:
            if line.startswith("FUNCTION Exponential") or line.startswith("FUNCTION BrokenExponential"):
                disc_found = True
            if not disc_found:
                continue
            if line.startswith("h"):
                h.append(float(line.split()[1]) * 0.75)
            if line.startswith("r_break"):
                r_br = float(line.split()[1]) * 0.75
    return (h, r_br)

def find_spiral_h(gal):
    path = f"../fit_results/{gal}/spiral_params/h_spiral.dat"
    with open(path, "r") as file:
        for line in file:
            h = float(line.split()[0])
    return h

def fit_profiles_1d(col, phi_span_o, r_o, flip, ax, color):
    this_gal = os.getcwd().split("/")[-1]
    if flip:
        phi_span_o = np.abs(phi_span_o - np.max(phi_span_o))
    else:
        phi_span_o = np.abs(phi_span_o - np.min(phi_span_o))

    if color == "b":
        arm_str_o = fits.getdata(f'str_arms_azavg/arm_str_{col}.fits')
        sigma_str_o = fits.getdata(f'str_arms_sigma/arm_str_{col}.fits')
    else:
        arm_str_o = fits.getdata(f'str_arms_azavg_masked/arm_str_{col}.fits')
        sigma_str_o = fits.getdata(f'str_arms_sigma_masked/arm_str_{col}.fits')

    #squeeze
    arm_str, sigma_str, phi_span, r = squeeze(arm_str_o, sigma_str_o, phi_span_o, r_o, 6)

    I = np.zeros_like(arm_str[0, :]) * np.nan
    I_e = np.zeros_like(arm_str[0, :]) * np.nan

    for j in range(len(arm_str[0, :])):
        y = arm_str[:, j]
        sigma = sigma_str[:, j]
        x = np.arange(len(y)) - (len(y) // 2)
        nona = ~np.isnan(y)
        try:
            p0 = [np.nanmax(y), 0, (np.max(x[nona]) - np.min(x[nona])) / 2]
            popt, pcov = curve_fit(gaussian, x[nona], y[nona], p0 = p0, sigma = sigma[nona])
            perr = np.sqrt(np.diag(pcov))
            if (popt[0] < 0) or (np.abs(popt[1]) > np.abs(popt[2]) / 2) or (np.abs(popt[2]) > np.max(r)):
                pass
            else:
                p0 = [popt[0], np.abs(popt[2])]
                popt, pcov = curve_fit(gaussian_c, x[nona], y[nona], p0 = p0, sigma = sigma[nona])
                perr = np.sqrt(np.diag(pcov))
                if (popt[0] < 0) or (np.abs(popt[1]) > np.max(r)):
                    pass
                else:
                    I[j] = popt[0]
                    I_e[j] = perr[0]
        except RuntimeError:
            pass
        except ValueError:
            pass
        except TypeError:
            pass

    r = r * 0.75

    if ((this_gal == "NGC1232") and (col == "red")) or ((this_gal == "NGC5247") and (col == "green")):
        I = np.where(phi_span > 0.8, I, np.nan)
        I_e = np.where(phi_span > 0.8, I_e, np.nan)

    ax.errorbar(r, I, yerr = I_e, c=color, marker = "o", markersize = 3, ls = "", elinewidth=0.5)
        
    nz = I > 0

    ax.set_title("I vs r")
    ax.set_xlabel("r, arcsec")
    ax.set_ylim(np.nanmin(I[nz]) * 0.8, np.nanmax(I[nz]) * 1.2)
    ax.set_yscale("log")
    #ax.legend(fontsize = 8)

    this_gal = os.getcwd().split("/")[-1]
    ax.set_title(f"{this_gal} arm")


gals_all = np.sort(glob.glob("*"))
os.chdir(gals_all[0])

gals = ["NGC0628", "NGC4321", "NGC5247"]
cols = ["red", "red", "green"]

fig, (ax0, ax1, ax2) = plt.subplots(figsize=[9, 3], nrows = 1, ncols = 3)
axs = (ax0, ax1, ax2)

for i in range(len(gals)):
    gal = gals[i]
    col = cols[i]

    os.chdir(f"../{gal}")
    imfit_path = "fit_nosp.imfit"
    xc, yc, pa, ell = find_fit_params(imfit_path)
    col_arr, phi_arr, r_arr = read_shapes_file(pa)

    i_gr = col_arr.index('green')
    if r_arr[i_gr][-1] > r_arr[i_gr][0]:
        flip = False
    else:
        flip = True

    i_this = col_arr.index(col)
    phi = phi_arr[i_this]
    r = r_arr[i_this]

    imfit_path_m = "masked/fit_masked.imfit"
    xc_m, yc_m, pa_m, ell_m = find_fit_params(imfit_path_m)
    col_arr_m, phi_arr_m, r_arr_m = read_shapes_file(pa_m, fname = "spiral_params/shapes_masked.dat")

    i_gr_m = col_arr_m.index('green')
    if r_arr_m[i_gr_m][-1] > r_arr_m[i_gr_m][0]:
        flip_m = False
    else:
        flip_m = True

    i_this_m = col_arr_m.index(col)
    phi_m = phi_arr_m[i_this_m]
    r_m = r_arr_m[i_this_m]

    fit_profiles_1d(col, phi_m, r_m, flip_m, axs[i], "r") # masked
    fit_profiles_1d(col, phi, r, flip, axs[i], "b") # original

    print(f"{gal} done")
os.chdir("..")

axs[0].set_ylabel("I, MJy/sterad")
axs[0].errorbar([], [], c="b", marker = "o", ls = "", markersize = 3, elinewidth=0.5, label = "disc as q = 0.1")
axs[0].errorbar([], [], c="r", marker = "o", ls = "", markersize = 3, elinewidth=0.5, label = "disc from decomposition")
axs[0].legend(fontsize = 8)
#axs[2].set_ylabel("I, MJy/sterad")

fig.tight_layout()
fig.savefig("../images/selected/profiles_examples", dpi = 300)