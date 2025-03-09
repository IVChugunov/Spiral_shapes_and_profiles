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

def fit_profiles_1d(col_arr, phi_arr, r_arr, flip, d_cat, col_arr_rb, phi_arr_rb, r_arr_rb):
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

        #squeeze
        arm_str, sigma_str, phi_span, r = squeeze(arm_str_o, sigma_str_o, phi_span_o, r_o, 6)
        noise_level = np.nanmedian(sigma_str[np.isfinite(sigma_str)])

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

        fig, axs = plt.subplots(figsize=[32,12], nrows = 3, ncols = 4)

        axs[1, 0].errorbar(r, w, yerr = w_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[1, 1].errorbar(phi_span, w, yerr = w_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[2, 0].errorbar(r, I, yerr = I_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[2, 1].errorbar(phi_span, I, yerr = I_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        
        for xi in [0, 1]:
            if xi == 0:
                xarg = r
            else:
                xarg = phi_span

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

            r_dip = np.nan
            for j in range(len(fs)):
                if (xi == 0) and fs[j].__name__ in ["exponential_mg_phi", "exp_mg_phi_wd1", "exp_mg_phi_wd2"]:
                    xarg_f = (phi_span[nona], r[nona])
                    
                    phi_cont = np.linspace(phi_span[0], phi_span[-1], 500)
                    if flip:
                        r_cont_f = CubicSpline(np.flip(phi_span), np.flip(r))
                    else:
                        r_cont_f = CubicSpline(phi_span, r)
                    r_cont = r_cont_f(phi_cont)
                    xarg_cont_y = (phi_cont, r_cont)
                    xarg_cont_x = r_cont
                else:
                    xarg_f = xarg[nona]
                    xarg_cont_y = xarg_cont
                    xarg_cont_x = xarg_cont
                try:
                    p0 = p0s[j]
                    popt, pcov = curve_fit(fs[j], xarg_f, I[nona], p0=p0, sigma = I_e[nona], maxfev = 5000)
                    chisq = np.sum(((fs[j](xarg_f, *popt) - I[nona]) / I_e[nona]) ** 2) / (len(r[nona]) - len(p0))
                    axs[2, xi].plot(xarg_cont_x, fs[j](xarg_cont_y, *popt), c=cs[j], ls = lss[j], lw = lws[j], alpha = als[j],
                        label = f"{fs[j].__name__}: $\\chi^2$ = {np.round(chisq, 2)}")
                    if fs[j].__name__ in ["exponential_mg", "exponential_mg_phi"]:
                        if fs[-1].__name__ in ["exponential_mg_wd1", "exp_mg_phi_wd1"]:
                            p0s[-1] = [*popt, 2, d_locs[0], 0.1 * np.max(phi_span)]
                        elif fs[-1].__name__ in ["exponential_mg_wd2", "exp_mg_phi_wd2"]:
                            p0s[-1] = [*popt, 2, d_locs[0], 0.1 * np.max(phi_span), 2, d_locs[1], 0.1 * np.max(phi_span)]
                    if (xi == 0):
                        if fs[-1].__name__ in ["exp_mg_phi_wd1", "exp_mg_phi_wd2"]:
                            if fs[j].__name__ in ["exponential", "exp_mg_phi_wd1", "exp_mg_phi_wd2"]:
                                popt_str = np.array2string(popt).replace('\n', '')
                                pcov_str = np.array2string(pcov).replace('\n', '')
                                arm_profiles.append(f"{col} {fs[j].__name__} {popt_str} {pcov_str}")
                        else:
                            if fs[j].__name__ in ["exponential", "exponential_mg_phi"]:
                                popt_str = np.array2string(popt).replace('\n', '')
                                pcov_str = np.array2string(pcov).replace('\n', '')
                                arm_profiles.append(f"{col} {fs[j].__name__} {popt_str} {pcov_str}")

                except RuntimeError:
                    pass
                except TypeError:
                    pass

            nona = ~np.isnan(w)
            try:
                p0 = [np.nanmax(w) / np.nanmax(xarg), 0]
                popt, pcov = curve_fit(linear, xarg[nona], w[nona], p0=p0, sigma = w_e[nona], maxfev = 800)
                chisq = np.sum(((linear(xarg[nona], *popt) - w[nona]) / w_e[nona]) ** 2) / (len(xarg[nona]) - len(p0))
                axs[1, xi].plot(xarg_cont, linear(xarg_cont, *popt), c=col, ls = "-", lw = 0.5,
                                label = f"{linear.__name__}: $\\chi^2$ = {np.round(chisq, 2)}")
            except RuntimeError:
                pass

            try:
                p0 = [0, np.nanmax(w) / np.nanmax(xarg), 0]
                popt, pcov = curve_fit(poly_2, xarg[nona], w[nona], p0=p0, sigma = w_e[nona], maxfev = 800)
                chisq = np.sum(((poly_2(xarg[nona], *popt) - w[nona]) / w_e[nona]) ** 2) / (len(xarg[nona]) - len(p0))
                axs[1, xi].plot(xarg_cont, poly_2(xarg_cont, *popt), c=col, ls = "--", lw = 0.5,
                                label = f"quadratic: $\\chi^2$ = {np.round(chisq, 2)}")
            except RuntimeError:
                pass

            try:
                p0 = [1, -4 / np.nanmax(xarg)]
                popt, pcov = curve_fit(exponential, xarg[nona], w[nona], p0=p0, sigma = w_e[nona], maxfev = 800)
                chisq = np.sum(((exponential(xarg[nona], *popt) - w[nona]) / w_e[nona]) ** 2) / (len(xarg[nona]) - len(p0))
                axs[1, xi].plot(xarg_cont, exponential(xarg_cont, *popt), c=col, ls = ":", lw = 0.5,
                                label = f"exponential: $\\chi^2$ = {np.round(chisq, 2)}")
            except RuntimeError:
                pass

        axs[0, 0].imshow(arm_str_o, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
        axs[0, 0].set_title("Subtracting q = 0.1", fontsize = 20)

        axs[0, 1].set_visible(False)

        axs[1, 0].set_title("width vs r")
        axs[1, 0].set_ylabel("width, arcsec")
        axs[1, 0].set_ylim(np.nanmin(w) * 0.8, np.nanmax(w) * 1.1)
        axs[1, 0].legend(fontsize = 8)

        axs[1, 1].set_title("width vs phi")
        axs[1, 1].set_ylim(np.nanmin(w) * 0.8, np.nanmax(w) * 1.1)
        axs[1, 1].legend(fontsize = 8)
        
        nz = I > 0

        axs[2, 0].set_title("I vs r")
        axs[2, 0].set_xlabel("r, arcsec")
        axs[2, 0].set_ylabel("I, MJy/sterad")
        axs[2, 0].set_ylim(np.nanmin(I[nz]) * 0.8, np.nanmax(I[nz]) * 1.2)
        axs[2, 0].axhline(3 * noise_level, lw = 2, ls = "--", c = "k", label = f"3 sigma noise = {np.round(3 * noise_level, 3)}")
        axs[2, 0].set_yscale("log")
        axs[2, 0].legend(fontsize = 8)

        axs[2, 1].set_title("I vs phi")
        axs[2, 1].set_xlabel("phi, radians")
        axs[2, 1].set_ylim(np.nanmin(I[nz]) * 0.8, np.nanmax(I[nz]) * 1.2)
        axs[2, 1].axhline(3 * noise_level, lw = 2, ls = "--", c = "k", label = f"3 sigma noise = {np.round(3 * noise_level, 3)}")
        axs[2, 1].set_yscale("log")
        axs[2, 1].legend(fontsize = 8)

        I_lims = [np.nanmin(I[nz]) * 0.8, np.nanmax(I[nz]) * 1.2]
        w_lims = [np.nanmin(w) * 0.8, np.nanmax(w) * 1.1]








        try:
            k = col_arr_rb.index(col)
        except ValueError:
            continue

        phi_span_o = phi_arr_rb[k]
        r_o = r_arr_rb[k]
        max_hw = int(np.round(np.max(r_o) * 0.2))
        if flip:
            phi_span_o = np.abs(phi_span_o - np.max(phi_span_o))
        else:
            phi_span_o = np.abs(phi_span_o - np.min(phi_span_o))

        arm_str_o = fits.getdata(f'str_arms_azavg_masked/arm_str_{col}.fits')
        sigma_str_o = fits.getdata(f'str_arms_sigma_masked/arm_str_{col}.fits')

        lpsi = np.shape(arm_str_o[:, 60:-60])[1] / 2

        #squeeze
        arm_str, sigma_str, phi_span, r = squeeze(arm_str_o, sigma_str_o, phi_span_o, r_o, 6)
        noise_level = np.nanmedian(sigma_str[np.isfinite(sigma_str)])

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

        I_all.extend(I[10:-10])
        I_e_all.extend(I_e[10:-10])
        r_all.extend(r[10:-10])

        axs[1, 2].errorbar(r, w, yerr = w_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[1, 3].errorbar(phi_span, w, yerr = w_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[2, 2].errorbar(r, I, yerr = I_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[2, 3].errorbar(phi_span, I, yerr = I_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        
        for xi in [0, 1]:
            if xi == 0:
                xarg = r
            else:
                xarg = phi_span

            if np.isnan(I[0]):
                I[0] = 0
                I_e[0] = np.nanmax(I_e)
            nona = ~np.isnan(I) * ~np.isnan(I_e)

            xarg_cont = np.linspace(np.min(xarg), np.max(xarg), 500)

            p0 = [np.nanmax(I), 4 / np.max(xarg)]
            try:
                popt_i, pcov_i = curve_fit(exponential, xarg[nona], I[nona], p0=p0, sigma = I_e[nona], maxfev = 5000)
            except ValueError:
                continue
            except RuntimeError:
                continue

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

            r_dip = np.nan
            for j in range(len(fs)):
                if (xi == 0) and fs[j].__name__ in ["exponential_mg_phi", "exp_mg_phi_wd1", "exp_mg_phi_wd2"]:
                    xarg_f = (phi_span[nona], r[nona])
                    
                    phi_cont = np.linspace(phi_span[0], phi_span[-1], 500)
                    if flip:
                        r_cont_f = CubicSpline(np.flip(phi_span), np.flip(r))
                    else:
                        r_cont_f = CubicSpline(phi_span, r)
                    r_cont = r_cont_f(phi_cont)
                    xarg_cont_y = (phi_cont, r_cont)
                    xarg_cont_x = r_cont
                else:
                    xarg_f = xarg[nona]
                    xarg_cont_y = xarg_cont
                    xarg_cont_x = xarg_cont
                try:
                    p0 = p0s[j]
                    popt, pcov = curve_fit(fs[j], xarg_f, I[nona], p0=p0, sigma = I_e[nona], maxfev = 5000)
                    chisq = np.sum(((fs[j](xarg_f, *popt) - I[nona]) / I_e[nona]) ** 2) / (len(r[nona]) - len(p0))
                    axs[2, xi + 2].plot(xarg_cont_x, fs[j](xarg_cont_y, *popt), c=cs[j], ls = lss[j], lw = lws[j], alpha = als[j],
                        label = f"{fs[j].__name__}: $\\chi^2$ = {np.round(chisq, 2)}")
                    if fs[j].__name__ in ["exponential_mg", "exponential_mg_phi"]:
                        if fs[-1].__name__ in ["exponential_mg_wd1", "exp_mg_phi_wd1"]:
                            p0s[-1] = [*popt, 2, d_locs[0], 0.1 * np.max(phi_span)]
                        elif fs[-1].__name__ in ["exponential_mg_wd2", "exp_mg_phi_wd2"]:
                            p0s[-1] = [*popt, 2, d_locs[0], 0.1 * np.max(phi_span), 2, d_locs[1], 0.1 * np.max(phi_span)]
                    if (xi == 0):
                        if fs[-1].__name__ in ["exp_mg_phi_wd1", "exp_mg_phi_wd2"]:
                            if fs[j].__name__ in ["exponential", "exp_mg_phi_wd1", "exp_mg_phi_wd2"]:
                                popt_str = np.array2string(popt).replace('\n', '')
                                pcov_str = np.array2string(pcov).replace('\n', '')
                                arm_profiles.append(f"{col} {fs[j].__name__} {popt_str} {pcov_str}")
                        else:
                            if fs[j].__name__ in ["exponential", "exponential_mg_phi"]:
                                popt_str = np.array2string(popt).replace('\n', '')
                                pcov_str = np.array2string(pcov).replace('\n', '')
                                arm_profiles.append(f"{col} {fs[j].__name__} {popt_str} {pcov_str}")

                except RuntimeError:
                    pass
                except TypeError:
                    pass

            nona = ~np.isnan(w)
            try:
                p0 = [np.nanmax(w) / np.nanmax(xarg), 0]
                popt, pcov = curve_fit(linear, xarg[nona], w[nona], p0=p0, sigma = w_e[nona], maxfev = 800)
                chisq = np.sum(((linear(xarg[nona], *popt) - w[nona]) / w_e[nona]) ** 2) / (len(xarg[nona]) - len(p0))
                axs[1, xi + 2].plot(xarg_cont, linear(xarg_cont, *popt), c=col, ls = "-", lw = 0.5,
                                label = f"{linear.__name__}: $\\chi^2$ = {np.round(chisq, 2)}")

                popt_str = np.array2string(popt).replace('\n', '')
                if xi == 0:
                    arm_w_profiles.append(f"{col} r_linear {popt_str} {rmin} {rmax} {chisq} {lpsi} {len(w[nona])}")
                else:
                    arm_w_profiles.append(f"{col} phi_linear {popt_str} {rmin} {rmax} {chisq} {lpsi} {len(w[nona])}")
            except RuntimeError:
                pass
            except TypeError:
                pass

            try:
                p0 = [0, np.nanmax(w) / np.nanmax(xarg), 0]
                popt, pcov = curve_fit(poly_2, xarg[nona], w[nona], p0=p0, sigma = w_e[nona], maxfev = 800)
                chisq = np.sum(((poly_2(xarg[nona], *popt) - w[nona]) / w_e[nona]) ** 2) / (len(xarg[nona]) - len(p0))
                axs[1, xi + 2].plot(xarg_cont, poly_2(xarg_cont, *popt), c=col, ls = "--", lw = 0.5,
                                label = f"quadratic: $\\chi^2$ = {np.round(chisq, 2)}")
            except RuntimeError:
                pass
            except TypeError:
                pass

            try:
                p0 = [1, -4 / np.nanmax(xarg)]
                popt, pcov = curve_fit(exponential, xarg[nona], w[nona], p0=p0, sigma = w_e[nona], maxfev = 800)
                chisq = np.sum(((exponential(xarg[nona], *popt) - w[nona]) / w_e[nona]) ** 2) / (len(xarg[nona]) - len(p0))
                axs[1, xi + 2].plot(xarg_cont, exponential(xarg_cont, *popt), c=col, ls = ":", lw = 0.5,
                                label = f"exponential: $\\chi^2$ = {np.round(chisq, 2)}")
                
                popt_str = np.array2string(popt).replace('\n', '')
                if xi == 1:
                    arm_w_profiles.append(f"{col} phi_exp {popt_str} {rmin} {rmax} {chisq} {lpsi} {len(w[nona])}")
            except RuntimeError:
                pass
            except TypeError:
                pass

        axs[0, 2].imshow(arm_str_o, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
        axs[0, 2].set_title("Subtracting decomp model with masked arms", fontsize = 20)

        axs[0, 3].set_visible(False)

        axs[1, 2].set_title("width vs r")
        axs[1, 2].set_ylabel("width, arcsec")
        axs[1, 2].set_ylim(*w_lims)
        axs[1, 2].legend(fontsize = 8)

        axs[1, 3].set_title("width vs phi")
        axs[1, 3].set_ylim(*w_lims)
        axs[1, 3].legend(fontsize = 8)
        
        nz = I > 0

        axs[2, 2].set_title("I vs r")
        axs[2, 2].set_xlabel("r, arcsec")
        axs[2, 2].set_ylabel("I, MJy/sterad")
        axs[2, 2].set_ylim(*I_lims)
        axs[2, 2].axhline(3 * noise_level, lw = 2, ls = "--", c = "k", label = f"3 sigma noise = {np.round(3 * noise_level, 3)}")
        axs[2, 2].set_yscale("log")
        axs[2, 2].legend(fontsize = 8)

        axs[2, 3].set_title("I vs phi")
        axs[2, 3].set_xlabel("phi, radians")
        axs[2, 3].set_ylim(*I_lims)
        axs[2, 3].axhline(3 * noise_level, lw = 2, ls = "--", c = "k", label = f"3 sigma noise = {np.round(3 * noise_level, 3)}")
        axs[2, 3].set_yscale("log")
        axs[2, 3].legend(fontsize = 8)

        gal = os.getcwd().split("/")[-1]
        fig.suptitle(f"{this_gal} {col} arm profiles")
        fig.tight_layout()
        fig.savefig(f"../../images/grids/1d_comp_masked/{this_gal}_{col}.png", dpi = 300)
        plt.close(fig)

    I_all = np.array(I_all)
    I_e_all = np.array(I_e_all)
    r_all = np.array(r_all)
    nona = ~np.isnan(I_all)

    p0 = [np.nanmax(I_all), 4 / np.nanmax(r_all)]
    r_sp = np.linspace(np.min(r_all), np.max(r_all), 50)
    h = np.nan
    try:
        popt, pcov = curve_fit(exponential, r_all[nona], I_all[nona], p0=p0, sigma = I_e_all[nona])
        h = np.round(1 / popt[1], 1)
    except RuntimeError:
        pass

    with open("spiral_params/h_spiral_masked.dat", "w") as file:
        file.write(f"{h} arcsec")

    if os.path.exists("spiral_params/widths_1d_masked.dat"):
        os.remove("spiral_params/widths_1d_masked.dat")
    with open("spiral_params/widths_1d_masked.dat", "w") as file:
        for line in arm_w_profiles:
            file.write(f"{line}\n")

gals = np.sort(glob.glob("*"))
os.chdir(gals[0])
for gal in gals:
    os.chdir(f"../{gal}")
    imfit_path = "fit_nosp.imfit"
    xc, yc, pa, ell = find_fit_params(imfit_path)
    imfit_path_m = "masked/fit_masked.imfit"
    xc_m, yc_m, pa_m, ell_m = find_fit_params(imfit_path_m)
    col_arr, phi_arr, r_arr = read_shapes_file(pa)
    col_arr_m, phi_arr_m, r_arr_m = read_shapes_file(pa_m, fname = "spiral_params/shapes_masked.dat")

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
        "NGC1232_red": [5.5],
        "NGC1300_green": [2.25, 5.25],
        "NGC1300_red": [2.5],
        "NGC1566_green": [6],
        "NGC1566_red": [6],
        "NGC4123_cyan": [2],
        "NGC4535_green": [2],
        "NGC4535_red": [2],
        "NGC4321_red": [2.5, 5],
        "NGC5247_blue": [3, 6]
    }

    fit_profiles_1d(col_arr, phi_arr, r_arr, flip, d_cat, col_arr_m, phi_arr_m, r_arr_m)
    print(f"{gal} done")