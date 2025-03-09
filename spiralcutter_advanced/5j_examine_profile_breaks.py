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

def fit_profiles_1d(col_arr, phi_arr, r_arr, flip, h, r_br):
    I_all = []
    I_e_all = []
    r_all = []

    this_gal = os.getcwd().split("/")[-1]
    arm_profiles = []
    arm_w_profiles = []

    for i in range(len(col_arr)):
        col = col_arr[i]
        phi_span_o = phi_arr[i]
        r_o = r_arr[i]
        max_hw = int(np.round(np.max(r_o) * 0.2))
        if flip:
            phi_span_o = np.abs(phi_span_o - np.max(phi_span_o))
        else:
            phi_span_o = np.abs(phi_span_o - np.min(phi_span_o))

        arm_str_o = fits.getdata(f'str_arms_azavg/arm_str_{col}.fits')
        sigma_str_o = fits.getdata(f'str_arms_sigma/arm_str_{col}.fits')

        lpsi = np.shape(arm_str_o[:, 60:-60])[1] / 2

        #squeeze
        arm_str, sigma_str, phi_span, r = squeeze(arm_str_o, sigma_str_o, phi_span_o, r_o, 6)

        I = np.zeros_like(arm_str[0, :]) * np.nan
        I_e = np.zeros_like(arm_str[0, :]) * np.nan
        w = np.zeros_like(arm_str[0, :]) * np.nan
        w_e = np.zeros_like(arm_str[0, :]) * np.nan

        #I_s = arm_str[len(arm_str[:, 0]) // 2, :]

        for j in range(len(arm_str[0, :])):
            y = arm_str[:, j]
            sigma = sigma_str[:, j]
            x = np.arange(len(y)) - (len(y) // 2)
            nona = ~np.isnan(y)
            try:
                p0 = [np.nanmax(y), 0, (np.max(x[nona]) - np.min(x[nona])) / 2]
                popt, pcov = curve_fit(gaussian, x[nona], y[nona], p0 = p0, sigma = sigma[nona], absolute_sigma = False)
                perr = np.sqrt(np.diag(pcov))
                if (popt[0] < 0) or (np.abs(popt[1]) > np.abs(popt[2]) / 2) or (np.abs(popt[2]) > np.max(r)):
                    pass
                else:
                    p0 = [popt[0], np.abs(popt[2])]
                    popt, pcov = curve_fit(gaussian_c, x[nona], y[nona], p0 = p0, sigma = sigma[nona], absolute_sigma = False)
                    perr = np.sqrt(np.diag(pcov))
                    if (popt[0] < 0) or (np.abs(popt[1]) > np.max(r)):
                        pass
                    else:
                        I[j] = popt[0]
                        I_e[j] = perr[0]
                        w[j] = np.abs(popt[1])
                        w_e[j] = perr[1]
            except RuntimeError:
                pass
            except ValueError:
                pass
            except TypeError:
                pass

        r = r * 0.75
        w = w * 0.75
        w_e = w_e * 0.75

        rmin = np.nanmin(r)
        rmax = np.nanmax(r)

        bad_w_e = (w_e > np.nanmedian(w_e) * 5)
        w_e = np.where(bad_w_e, np.nan, w_e)
        w = np.where(bad_w_e, np.nan, w)

        if ((this_gal == "NGC1232") and (col == "red")) or ((this_gal == "NGC5247") and (col == "green")):
            I = np.where(phi_span > 0.8, I, np.nan)
            I_e = np.where(phi_span > 0.8, I_e, np.nan)
            w = np.where(phi_span > 0.8, w, np.nan)
            w_e = np.where(phi_span > 0.8, w_e, np.nan)

        #I_e = sigma_str[len(y) // 2, :]

        I_all.extend(I[10:-10])
        I_e_all.extend(I_e[10:-10])
        r_all.extend(r[10:-10])

    I_all = np.array(I_all)
    I_e_all = np.array(I_e_all)
    r_all = np.array(r_all)
    nona = ~np.isnan(I_all)

    p0 = [np.nanmax(I_all), h[0], h[1], r_br]
    r_sp = np.linspace(np.min(r_all), np.max(r_all), 50)
    h_res = [np.nan, np.nan]
    r_br_res = np.nan
    try:
        popt, pcov = curve_fit(broken_exp, r_all[nona], I_all[nona], p0=p0, sigma = I_e_all[nona])
        h_res[0] = np.round(popt[1], 2)
        h_res[1] = np.round(popt[2], 2)
        r_br_res = np.round(popt[3], 2)
    except RuntimeError:
        pass

    with open("spiral_params/h_spiral_break.dat", "w") as file:
        file.write(f"h1: {h_res[0]} arcsec\n")
        file.write(f"h2: {h_res[1]} arcsec\n")
        file.write(f"r_br: {r_br_res} arcsec")
    print(h_res)
    print(r_br_res)


gals = np.sort(glob.glob("*"))
os.chdir(gals[0])
for gal in gals:
    os.chdir(f"../{gal}")
    imfit_path = "fit_nosp.imfit"
    xc, yc, pa, ell = find_fit_params(imfit_path)
    h, r_br = find_disc_params(imfit_path)
    if np.isnan(r_br):
        continue
    col_arr, phi_arr, r_arr = read_shapes_file(pa)

    i_gr = col_arr.index('green')
    if r_arr[i_gr][-1] > r_arr[i_gr][0]:
        flip = False
    else:
        flip = True

    fit_profiles_1d(col_arr, phi_arr, r_arr, flip, h, r_br)
    print(f"{gal} done")