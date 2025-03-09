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

def fit_profiles_1d(col_arr, phi_arr, r_arr, flip, d_cat):

    this_gal = os.getcwd().split("/")[-1]

    for i in range(len(col_arr)):
        col = col_arr[i]

        if col != "red":
            continue

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

        fwhm_f = 2 * np.sqrt(2 * np.log(2))

        r = r * 0.75
        w = w * 0.75 * fwhm_f
        w_e = w_e * 0.75 * fwhm_f

        rmin = np.nanmin(r)
        rmax = np.nanmax(r)

        bad_w_e = (w_e > np.nanmedian(w_e) * 5)
        w_e = np.where(bad_w_e, np.nan, w_e)
        w = np.where(bad_w_e, np.nan, w)

        #I_e = sigma_str[len(y) // 2, :]

        fig, axs = plt.subplots(figsize=[6,6], nrows = 2)

        axs[0].errorbar(r, w, yerr = w_e, c="k", marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[1].errorbar(np.degrees(phi_span), w, yerr = w_e, c="k", marker = ".", markersize = 10, ls = "", elinewidth=2)
        
        for xi in [0, 1]:
            if xi == 0:
                xarg = r
            else:
                xarg = np.degrees(phi_span)

            if np.isnan(I[0]):
                I[0] = 0
                I_e[0] = np.nanmax(I_e)
            nona = ~np.isnan(I) * ~np.isnan(I_e)

            xarg_cont = np.linspace(np.min(xarg), np.max(xarg), 500)

            p0 = [np.nanmax(I), 4 / np.max(xarg)]
            popt_i, pcov_i = curve_fit(exponential, xarg[nona], I[nona], p0=p0, sigma = I_e[nona], maxfev = 5000)

            fs = [exponential, exp_b1, exp_o2, exp_o2_mg, exp_o3, exponential_mg] #best function is last
            p0s = [
            [*popt_i],
            [popt_i[0], 0.9 * popt_i[1], 1.1 * popt_i[1], 0.5 * np.min(xarg) + 0.5 * np.max(xarg)],
            [*popt_i, 0],
            [*popt_i, 0, 0.1 * np.max(xarg), 0.1 * np.max(xarg), 0, np.max(xarg)],
            [*popt_i, 0, 0],
            [*popt_i, 0.1 * np.max(phi_span), 0.1 * np.max(phi_span), 0, np.max(phi_span)]
            ]
            if xi == 0:
                fs[-1] = exponential_mg_phi
            lss = ["-", "", "", "-.", ":", "--"]
            als = np.ones(len(fs))
            cs = [col] * len(fs)
            lws = np.zeros(len(fs)) + 0.5

            arm_check = f"{this_gal}_{col}"
            if arm_check in d_cat:
                d_locs = d_cat[arm_check]
                if len(d_locs) == 1:
                    if xi == 0:
                        fs.append(exp_mg_phi_wd1)
                    else:
                        fs.append(exp_mg_wd1)
                    p0s.append([*popt_i, 0.1 * np.max(phi_span), 0.1 * np.max(phi_span), 0, np.max(phi_span), 2, d_locs[0], 0.1 * np.max(phi_span)])
                else:
                    if xi == 0:
                        fs.append(exp_mg_phi_wd2)
                    else:
                        fs.append(exp_mg_wd2)
                    p0s.append([*popt_i, 0.1 * np.max(phi_span), 0.1 * np.max(phi_span), 0, np.max(phi_span), 2, d_locs[0], 0.1 * np.max(phi_span), 2, d_locs[1], 0.1 * np.max(phi_span)])
                lss.append("-")
                als = np.append(als, 0.3)
                cs.append("k")
                lws = np.append(lws, 3)
            else:
                lss[-1] = "-"
                als[-1] = 0.3
                cs[-1] = "k"
                lws[-1] = 3

            nona = ~np.isnan(w)
            try:
                p0 = [np.nanmax(w) / np.nanmax(xarg), 0]
                popt, pcov = curve_fit(linear, xarg[nona], w[nona], p0=p0, sigma = w_e[nona], maxfev = 800)
                chisq = np.sum(((linear(xarg[nona], *popt) - w[nona]) / w_e[nona]) ** 2) / (len(xarg[nona]) - len(p0))
                axs[xi].plot(xarg_cont, linear(xarg_cont, *popt), c=col, ls = "-", lw = 2,
                                label = f"Linear: $\\chi^2$ = {np.round(chisq, 2)}")

                popt_str = np.array2string(popt).replace('\n', '')
            except RuntimeError:
                pass

            if xi == 1:
                try:
                    p0 = [1, -4 / np.nanmax(xarg)]
                    popt, pcov = curve_fit(exponential, xarg[nona], w[nona], p0=p0, sigma = w_e[nona], maxfev = 800)
                    chisq = np.sum(((exponential(xarg[nona], *popt) - w[nona]) / w_e[nona]) ** 2) / (len(xarg[nona]) - len(p0))
                    axs[xi].plot(xarg_cont, exponential(xarg_cont, *popt), c=col, ls = ":", lw = 2,
                                    label = f"Exponential: $\\chi^2$ = {np.round(chisq, 2)}")
                    
                    popt_str = np.array2string(popt).replace('\n', '')
                except RuntimeError:
                    pass

        axs[0].set_title("Width as a function of radius")
        axs[0].set_xlabel("$r$, arcsec")
        axs[0].set_ylabel("$w$, arcsec")
        axs[0].set_ylim(np.nanmin(w) * 0.8, np.nanmax(w) * 1.1)
        axs[0].legend(fontsize = 8)

        axs[1].set_title("Width as a function of azimuthal angle")
        axs[1].set_xlabel("$\\psi$, deg")
        axs[1].set_ylabel("$w$, arcsec")
        axs[1].set_ylim(np.nanmin(w) * 0.8, np.nanmax(w) * 1.1)
        axs[1].legend(fontsize = 8)

        gal = os.getcwd().split("/")[-1]
        fig.suptitle(f'{this_gal}: "{col}" arm width variation')
        fig.tight_layout()
        fig.savefig(f"../../images/selected/NGC1042_red_width.png", dpi = 300)
        plt.close(fig)

gals = ["NGC1042"]
os.chdir(gals[0])
for gal in gals:
    os.chdir(f"../{gal}")
    imfit_path = "fit_nosp.imfit"
    xc, yc, pa, ell = find_fit_params(imfit_path)
    col_arr, phi_arr, r_arr = read_shapes_file(pa)

    i_gr = col_arr.index('green')
    if r_arr[i_gr][-1] > r_arr[i_gr][0]:
        flip = False
    else:
        flip = True

    d_cat = {
        "NGC0628_blue": [1.5],
        "NGC1042_green": [4],
        "NGC1042_red": [3.5],
        "NGC1073_green": [1.75],
        "NGC1232_green": [2.5, 5],
        "NGC1232_red": [4.5],
        "NGC1300_green": [2.25, 5.25],
        "NGC1300_red": [2.5],
        "NGC1566_green": [6],
        "NGC1566_red": [6],
        "NGC4123_cyan": [2],
        "NGC4321_red": [2.5, 5],
        "NGC4535_green": [2],
        "NGC4535_red": [2],
        "NGC5247_blue": [3, 6]
    }

    fit_profiles_1d(col_arr, phi_arr, r_arr, flip, d_cat)
    print(f"{gal} done")