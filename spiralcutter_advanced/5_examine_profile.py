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

def general_fit_str_arms(col_arr, phi_arr, r_arr, flip, d_cat):
    I_all = []
    I_e_all = []
    r_all = []

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=[12,8], ncols = 3, nrows = 2)
    for i in range(len(col_arr)):
        col = col_arr[i]
        phi_span = phi_arr[i]
        r = r_arr[i]

        # delete extension
        phi_span = phi_span[60:-60]
        r = r[60:-60]

        max_hw = int(np.round(np.max(r) * 0.2))

        arm_str = fits.getdata(f'str_arms_azavg/arm_str_{col}.fits')
        arm_str = arm_str[:, 60:-60]
        I = np.zeros_like(arm_str[0, :]) * np.nan
        I_e = np.zeros_like(arm_str[0, :]) * np.nan
        rho = np.zeros_like(arm_str[0, :]) * np.nan
        w = np.zeros_like(arm_str[0, :]) * np.nan
        w_e = np.zeros_like(arm_str[0, :]) * np.nan
        
        for i in range(len(arm_str[0, :])):
            #y = np.nanmean(arm_str[:, i - 2:i + 3], axis = 1)
            y = arm_str[:, i]
            x = np.arange(len(y)) - (len(y) // 2)
            nona = ~np.isnan(y)
            p0 = [np.nanmax(y), 0, 50]
            #print(i)
            try:
                popt, pcov = curve_fit(gaussian, x[nona], y[nona], p0 = p0)
                perr = np.sqrt(np.diag(pcov))
                if (popt[0] < 0) or (np.abs(popt[1]) > np.abs(popt[2]) / 2) or (np.abs(popt[2]) > np.max(r)):
                    pass
                else:
                    I[i] = popt[0]
                    I_e[i] = perr[0]
                    rho[i] = popt[1] * 0.75
                    w[i] = np.abs(popt[2]) * 0.75 * 2
                    w_e[i] = perr[2] * 0.75 * 2
            except RuntimeError:
                pass
            except ValueError:
                pass
            except TypeError:
                pass
        #print(I)
        #print(rho)
        #print(w)
        r = r * 0.75

        #ax1.scatter(phi_span, r, c=col, s = 3)
        ax1.scatter(r, w, c=col, s = 3)
        ax2.scatter(r, I, c=col, s = 3)
        ax3.scatter(r, rho, c=col, s = 3)
        ax4.scatter(np.degrees(phi_span), w, c=col, s = 3)
        ax5.scatter(np.degrees(phi_span), I, c=col, s = 3)

        nona = ~np.isnan(I)
        p0 = [np.nanmax(I), 4 / np.nanmax(r)]
        r_sp = np.linspace(np.min(r), np.max(r), 50)
        try:
            popt, pcov = curve_fit(exponential, r[nona], I[nona], p0=p0, sigma = I_e[nona])
            h = np.round(1 / popt[1], 1)
            ax2.plot(r_sp, exponential(r_sp, *popt), c=col, lw = 0.5, label = f"h = {h} arcsec")
        except RuntimeError:
            pass

        I_all.extend(I)
        I_e_all.extend(I_e)
        r_all.extend(r)

        #fit width
        p0 = [0.5, np.nanmin(w)]
        try:
            popt, pcov = curve_fit(linear, r[nona], w[nona], p0=p0, sigma = w_e[nona])
            c, w0 = np.round(popt, 2)
            ax1.plot(r_sp, linear(r_sp, *popt), c=col, lw = 0.5, label = f"w = {c} r + {w0} arcsec")
        except RuntimeError:
            pass

    I_all = np.array(I_all)
    I_e_all = np.array(I_e_all)
    r_all = np.array(r_all)
    nona = ~np.isnan(I_all)

    p0 = [np.nanmax(I_all), 4 / np.nanmax(r_all)]
    r_sp = np.linspace(np.min(r_all), np.max(r_all), 50)
    try:
        popt, pcov = curve_fit(exponential, r_all[nona], I_all[nona], p0=p0, sigma = I_e_all[nona])
        h = np.round(1 / popt[1], 1)
        ax2.plot(r_sp, exponential(r_sp, *popt), c="k", lw = 1, label = f"All spirals: {h} arcsec")
    except RuntimeError:
        pass
    #ax2.plot(None, None, c="w", lw = 1, label = f"Disc: {disc_h} arcsec")

    gal = os.getcwd().split("/")[-1]
    fig.suptitle(gal)
    ax1.legend(fontsize = 8)
    ax1.set_title("width vs r")
    ax2.set_yscale('log')
    ax2.legend(fontsize = 8)
    ax2.set_title("I vs r")
    ax3.set_title("rho vs r")
    ax4.set_title("w vs phi")
    ax5.set_title("I vs phi")
    ax5.set_yscale('log')
    ax6.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"../../images/grids/other/{gal}.png", dpi = 300)
    plt.close(fig)

def fit_profiles_1d(col_arr, phi_arr, r_arr, flip, d_cat):
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

        fig, axs = plt.subplots(figsize=[16,12], nrows = 3, ncols = 2)

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
                            if this_gal == "NGC1232" and col == "red":
                                p0s[-1] = [*popt, 2, d_locs[0], 0.03 * np.max(phi_span)]
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

                popt_str = np.array2string(popt).replace('\n', '')
                if xi == 0:
                    arm_w_profiles.append(f"{col} r_linear {popt_str} {rmin} {rmax} {chisq} {lpsi} {len(w[nona])}")
                else:
                    arm_w_profiles.append(f"{col} phi_linear {popt_str} {rmin} {rmax} {chisq} {lpsi} {len(w[nona])}")
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
                
                popt_str = np.array2string(popt).replace('\n', '')
                if xi == 1:
                    arm_w_profiles.append(f"{col} phi_exp {popt_str} {rmin} {rmax} {chisq} {lpsi} {len(w[nona])}")
            except RuntimeError:
                pass

        axs[0, 0].imshow(arm_str_o, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
        axs[0, 0].set_title("Linearized arm")

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
        axs[2, 0].set_yscale("log")
        axs[2, 0].legend(fontsize = 8)

        axs[2, 1].set_title("I vs phi")
        axs[2, 1].set_xlabel("phi, radians")
        axs[2, 1].set_ylim(np.nanmin(I[nz]) * 0.8, np.nanmax(I[nz]) * 1.2)
        axs[2, 1].set_yscale("log")
        axs[2, 1].legend(fontsize = 8)

        gal = os.getcwd().split("/")[-1]
        fig.suptitle(f"{this_gal} {col} arm profiles")
        fig.tight_layout()
        fig.savefig(f"../../images/grids/1d_fits/{this_gal}_{col}.png", dpi = 300)
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

    with open("spiral_params/h_spiral.dat", "w") as file:
        file.write(f"{h} arcsec")

    if os.path.exists("spiral_params/profiles_1d.dat"):
        os.remove("spiral_params/profiles_1d.dat")
    with open("spiral_params/profiles_1d.dat", "w") as file:
        for line in arm_profiles:
            file.write(f"{line}\n")

    if os.path.exists("spiral_params/widths_1d.dat"):
        os.remove("spiral_params/widths_1d.dat")
    with open("spiral_params/widths_1d.dat", "w") as file:
        for line in arm_w_profiles:
            file.write(f"{line}\n")

def fit_profile_2d(col_arr, phi_arr, r_arr, flip, d_cat):

    this_gal = os.getcwd().split("/")[-1]
    arm_profiles = []

    for i in range(len(col_arr)):
        col = col_arr[i]
        fn = f"../../images/grids/2d_fits/{this_gal}_{col}.png"
        if os.path.exists(fn):
            os.remove(fn)
        phi_span = phi_arr[i]
        r = r_arr[i]

        if flip:
            phi_span = np.abs(phi_span - np.max(phi_span))
        else:
            phi_span = np.abs(phi_span - np.min(phi_span))

        max_hw = int(np.round(np.max(r) * 0.2))

        arm_str = fits.getdata(f'str_arms_azavg/arm_str_{col}.fits')
        sigma_str = fits.getdata(f'str_arms_sigma/arm_str_{col}.fits')

        rho = np.arange(len(arm_str[:, 0])) - (len(arm_str[:, 0]) // 2)

        r = r * 0.75
        rho = rho * 0.75

        r_g, rho_g = np.meshgrid(r, rho)
        phi_g, rho_g = np.meshgrid(phi_span, rho)

        y = np.ravel(arm_str)
        y_err = np.ravel(sigma_str)
        rho_rav = np.ravel(rho_g)
        r_rav = np.ravel(r_g)
        phi_rav = np.ravel(phi_g)

        nona = ~np.isnan(y)

        X = np.array((phi_rav, r_rav, rho_rav))

        p0 = [np.nanmax(y), 0.01, 1, 0.2]
        bs = ([0, 0, 0, 0], [np.inf, 1000, np.max(rho) / 2, 1])

        fig, axs = plt.subplots(figsize=[32,12], nrows = 3, ncols = 4)
        axs[0, 0].imshow(arm_str, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
        axs[0, 1].set_visible(False)

        try:
            popt, pcov = curve_fit(o1_r_o1_r, X, y, p0 = p0, sigma = y_err[nona], bounds = bs, nan_policy = "omit", maxfev = 5000)
            perr = np.sqrt(np.diag(pcov))
            y_model = o1_r_o1_r(X, *popt).reshape(np.shape(arm_str))
            axs[1, 0].imshow(y_model, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
            axs[1, 0].set_title(f"{o1_r_o1_r.__name__}; coeffs {np.round(popt, 5)}")

            axs[1, 1].imshow((arm_str - y_model) / sigma_str, vmin = -5, vmax=5, cmap="PuOr", origin="lower")
            chisq = np.nansum(((arm_str - y_model) / sigma_str) ** 2) / (np.count_nonzero(~np.isnan(arm_str)) - len(p0))
            axs[1, 1].set_title(f"Deviation in sigma; total $\\chi^2$: {np.round(chisq, 3)}")
        except RuntimeError:
            continue

        ######

        p0_r1r = [popt[0], popt[1], np.max(phi_span) * 0.1, np.max(phi_span) * 0.1, 0, np.max(phi_span), popt[2], popt[3]]
        bs_r1r = ([0, -np.inf, 0, 0, -np.inf, -np.inf, 0, 0],
                [np.inf, np.inf, np.max(phi_span) * 0.5, np.max(phi_span) * 0.5, np.inf, np.inf, np.max(rho) / 2, 1])
        try:
            popt_r1r, pcov_r1r = curve_fit(fin_r_o1_r, X, y, p0 = p0_r1r, sigma = y_err[nona], bounds = bs_r1r, nan_policy = "omit", maxfev = 5000)
            perr_r1r = np.sqrt(np.diag(pcov_r1r))
            y_model_r1r = fin_r_o1_r(X, *popt_r1r).reshape(np.shape(arm_str))
            axs[2, 0].imshow(y_model_r1r, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
            axs[2, 0].set_title(f"{fin_r_o1_r.__name__}; coeffs {np.round(popt_r1r, 5)}")

            axs[2, 1].imshow((arm_str - y_model_r1r) / sigma_str, vmin = -5, vmax=5, cmap="PuOr", origin="lower")
            chisq_r1r = np.nansum(((arm_str - y_model_r1r) / sigma_str) ** 2) / (np.count_nonzero(~np.isnan(arm_str)) - len(p0_r1r))
            axs[2, 1].set_title(f"Deviation in sigma; total $\\chi^2$: {np.round(chisq_r1r, 3)}")

            popt_str = np.array2string(popt_r1r).replace('\n', '')
            pcov_str = np.array2string(pcov_r1r).replace('\n', '')
            arm_profiles.append(f"{col} {fin_r_o1_r.__name__} {popt_str} {pcov_str}")
        except RuntimeError:
            pass

        ######

        p0_r2r = [popt[0], popt[1], np.max(phi_span) * 0.1, np.max(phi_span) * 0.1, 0, np.max(phi_span), popt[2], popt[3], 0]
        bs_r2r = ([0, -np.inf, 0, 0, -np.inf, -np.inf, 0, 0, -1],
                [np.inf, np.inf, np.max(phi_span) * 0.5, np.max(phi_span) * 0.5, np.inf, np.inf, np.max(rho) / 2, 1, 1])
        try:
            popt_r2r, pcov_r2r = curve_fit(fin_r_o2_r, X, y, p0 = p0_r2r, sigma = y_err[nona], bounds = bs_r2r, nan_policy = "omit", maxfev = 5000)
            perr_r2r = np.sqrt(np.diag(pcov_r2r))
            y_model_r2r = fin_r_o2_r(X, *popt_r2r).reshape(np.shape(arm_str))
            axs[0, 2].imshow(y_model_r2r, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
            axs[0, 2].set_title(f"{fin_r_o2_r.__name__}; coeffs {np.round(popt_r2r, 5)}")

            axs[0, 3].imshow((arm_str - y_model_r2r) / sigma_str, vmin = -5, vmax=5, cmap="PuOr", origin="lower")
            chisq_r2r = np.nansum(((arm_str - y_model_r2r) / sigma_str) ** 2) / (np.count_nonzero(~np.isnan(arm_str)) - len(p0_r2r))
            axs[0, 3].set_title(f"Deviation in sigma; total $\\chi^2$: {np.round(chisq_r2r, 3)}")

            popt_str = np.array2string(popt_r2r).replace('\n', '')
            pcov_str = np.array2string(pcov_r2r).replace('\n', '')
            arm_profiles.append(f"{col} {fin_r_o2_r.__name__} {popt_str} {pcov_str}")
        except RuntimeError:
            pass

        ######

        arm_check = f"{this_gal}_{col}"
        if arm_check in d_cat:
            d_locs = d_cat[arm_check]
            if len(d_locs) == 1:
                func_d = fin_r_o1_r_d1
                p0_r2rd = [popt[0], popt[1], np.max(phi_span) * 0.1, np.max(phi_span) * 0.1, 0, np.max(phi_span), popt[2], popt[3],
                            2, d_locs[0], np.max(phi_span) * 0.1]
                bs_r2rd = ([0, -np.inf, 0, 0, -np.inf, -np.inf, 0, 0,
                            0, 0, 0],
                            [np.inf, np.inf, np.max(phi_span) * 0.5, np.max(phi_span) * 0.5, np.inf, np.inf, np.max(rho) / 2, 1,
                            1000, np.max(phi_span), np.max(phi_span) * 0.5])
            else:
                func_d = fin_r_o1_r_d2
                p0_r2rd = [popt[0], popt[1], np.max(phi_span) * 0.1, np.max(phi_span) * 0.1, 0, np.max(phi_span), popt[2], popt[3],
                            2, d_locs[0], np.max(phi_span) * 0.1, 2, d_locs[1], np.max(phi_span) * 0.1]
                bs_r2rd = ([0, -np.inf, 0, 0, -np.inf, -np.inf, 0, 0,
                            0, 0, 0, 0, 0, 0],
                            [np.inf, np.inf, np.max(phi_span) * 0.5, np.max(phi_span) * 0.5, np.inf, np.inf, np.max(rho) / 2, 1,
                            1000, np.max(phi_span), np.max(phi_span) * 0.5, 1000, np.max(phi_span), np.max(phi_span) * 0.5])

            try:
                popt_r2rd, pcov_r2rd = curve_fit(func_d, X, y, p0 = p0_r2rd, sigma = y_err[nona], bounds = bs_r2rd, nan_policy = "omit", maxfev = 5000)
                perr_r2rd = np.sqrt(np.diag(pcov_r2rd))
                y_model_r2rd = func_d(X, *popt_r2rd).reshape(np.shape(arm_str))
                axs[1, 2].imshow(y_model_r2rd, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
                axs[1, 2].set_title(f"{func_d.__name__}; coeffs {np.round(popt_r2rd, 5)}")

                axs[1, 3].imshow((arm_str - y_model_r2rd) / sigma_str, vmin = -5, vmax=5, cmap="PuOr", origin="lower")
                chisq_r2rd = np.nansum(((arm_str - y_model_r2rd) / sigma_str) ** 2) / (np.count_nonzero(~np.isnan(arm_str)) - len(p0_r2rd))
                axs[1, 3].set_title(f"Deviation in sigma; total $\\chi^2$: {np.round(chisq_r2rd, 3)}")

                popt_str = np.array2string(popt_r2rd).replace('\n', '')
                pcov_str = np.array2string(pcov_r2rd).replace('\n', '')
                arm_profiles.append(f"{col} {func_d.__name__} {popt_str} {pcov_str}")
            except RuntimeError:
                pass
        else:
            axs[1, 2].set_visible(False)
            axs[1, 3].set_visible(False)

        ######

        p0_f2 = [popt[0], 1, np.max(phi_span) * 0.1, np.max(phi_span) * 0.1, 0, np.max(phi_span), popt[2], popt[3], 0]
        bs_f2 = ([0, -np.inf, 0, 0, -np.inf, -np.inf, 0, 0, -1],
                [np.inf, np.inf, np.max(phi_span) * 0.5, np.max(phi_span) * 0.5, np.inf, np.inf, np.max(rho) / 2, 1, 1])
        try:
            popt_f2, pcov_f2 = curve_fit(fin_phi_o2_r, X, y, p0 = p0_f2, sigma = y_err[nona], bounds = bs_f2, nan_policy = "omit", maxfev = 5000)
            perr_f2 = np.sqrt(np.diag(pcov_f2))
            y_model_f2 = fin_phi_o2_r(X, *popt_f2).reshape(np.shape(arm_str))
            axs[2, 2].imshow(y_model_f2, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
            axs[2, 2].set_title(f"{fin_phi_o2_r.__name__}; coeffs {np.round(popt_f2, 5)}")

            axs[2, 3].imshow((arm_str - y_model_f2) / sigma_str, vmin = -5, vmax=5, cmap="PuOr", origin="lower")
            chisq_f2 = np.nansum(((arm_str - y_model_f2) / sigma_str) ** 2) / (np.count_nonzero(~np.isnan(arm_str)) - len(p0_f2))
            axs[2, 3].set_title(f"Deviation in sigma; total $\\chi^2$: {np.round(chisq_f2, 3)}")
        except RuntimeError:
            continue

        fig.suptitle(f"{this_gal} {col} arm")
        fig.tight_layout()
        fig.savefig(f"../../images/grids/2d_fits/{this_gal}_{col}.png", dpi = 300)
        plt.close(fig)

        print(f"{col} done")

    if os.path.exists("spiral_params/profiles_general.dat"):
        os.remove("spiral_params/profiles_general.dat")
    with open("spiral_params/profiles_general.dat", "w") as file:
        for line in arm_profiles:
            file.write(f"{line}\n")

gals = np.sort(glob.glob("*"))
os.chdir(gals[0])
for gal in gals:
    #if gal not in ["NGC0628", "NGC1232", "NGC4535", "NGC5247"]:
    #    continue
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

    #general_fit_str_arms(col_arr, phi_arr, r_arr, flip, d_cat)
    fit_profiles_1d(col_arr, phi_arr, r_arr, flip, d_cat)
    #fit_profile_2d(col_arr, phi_arr, r_arr, flip, d_cat)
    print(f"{gal} done")