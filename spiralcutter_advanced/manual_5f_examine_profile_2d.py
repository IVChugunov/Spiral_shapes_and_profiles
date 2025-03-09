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
from libs.twod_funcs_mod import *
from scipy.optimize import curve_fit

def fit_profile_2d(col_arr, phi_arr, r_arr, flip, d_cat):

    this_gal = os.getcwd().split("/")[-1]

    for i in range(len(col_arr)):
        col = col_arr[i]
        if col != "green":
            continue
        phi_span = phi_arr[i]
        r = r_arr[i]

        if flip:
            phi_span = np.abs(phi_span - np.max(phi_span))
        else:
            phi_span = np.abs(phi_span - np.min(phi_span))

        max_hw = int(np.round(np.max(r) * 0.2))
        max_phi = np.nanmax(phi_span)

        arm_str = fits.getdata(f'str_arms_azavg/arm_str_{col}.fits')
        sigma_str = fits.getdata(f'str_arms_sigma/arm_str_{col}.fits')

        rho = np.arange(len(arm_str[:, 0])) - (len(arm_str[:, 0]) // 2)

        if len(arm_str[:, 0]) > 300:
            continue

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
        N_pix = len(y[nona])

        X = np.array((phi_rav, r_rav, rho_rav))

        p0 = [np.nanmax(y), 0.01, 1, 0.2]
        bs = ([0, -10, -np.max(rho), -1], [np.inf, 10, np.max(rho) / 2, 1])

        fig, axs = plt.subplots(figsize=[16,14], nrows = 10, ncols = 2, layout = "constrained")
        axs[0, 0].imshow(arm_str, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
        axs[0, 0].set_title("Straightened arm image", fontsize = 14)
        axs[0, 0].xaxis.set_visible(False)
        axs[0, 0].yaxis.set_visible(False)
        axs[0, 1].set_visible(False)

        try:
            popt, pcov = curve_fit(o1_r_o1_r, X, y, p0 = p0, sigma = y_err[nona], bounds = bs, nan_policy = "omit", maxfev = 5000)
            perr = np.sqrt(np.diag(pcov))
            y_model = o1_r_o1_r(X, *popt).reshape(np.shape(arm_str))
        except RuntimeError:
            continue

        ######

        funcs = [main_func_cw, main_func_nozpw, main_func_cbr, main_func_sharp, main_func, main_func_w_asymm, main_func_w_n, main_func_add]
        func_names = ["Constant width ($w_1 = 0$)",
                      "Width proportional to radius ($w_0 = 0$)",
                      "No exponential decrease ($h_{inv} = 0$)",
                      "No growth / cutoff parts ($\\psi_{growth} = \\psi_{cutoff} = 0$)",
                      "Baseline function",
                      "Non-zero skewness ($S_0, S_1 \\neq 0$)",
                      "Non-zero skewness and Sersic profile ($S_0, S_1, n \\neq 0$)",
                      "Extra parameters added ($w_2, h_{inv2} \\neq 0$)"]
        ind_main = 4
        p0s =   [[0, 0, *popt[:2], max_phi * 0.1, max_phi * 0.1, max_phi * 0.1, max_phi * 0.9, popt[-2]],
                 [0, 0, *popt[:2], max_phi * 0.1, max_phi * 0.1, max_phi * 0.1, max_phi * 0.9, popt[-1]],
                 [0, 0, popt[0], max_phi * 0.1, max_phi * 0.1, max_phi * 0.1, max_phi * 0.9, *popt[-2:]],
                 [0, 0, *popt[:2], max_phi * 0.1, max_phi * 0.9, *popt[-2:]],
                 [0, 0, *popt[:2], max_phi * 0.1, max_phi * 0.1, max_phi * 0.1, max_phi * 0.9, *popt[-2:]],
                 [0, 0, *popt[:2], max_phi * 0.1, max_phi * 0.1, max_phi * 0.1, max_phi * 0.9, *popt[-2:], 0, 0],
                 [0, 0, *popt[:2], max_phi * 0.1, max_phi * 0.1, max_phi * 0.1, max_phi * 0.9, *popt[-2:], 0, 0, 0.5],
                 [0, 0, *popt[:2], 0, max_phi * 0.1, max_phi * 0.1, max_phi * 0.1, max_phi * 0.9, *popt[-2:], 0]
                ]

        bsl =   [[-np.inf, -np.inf, 0, -10, 0, 0, -np.inf, -np.inf, -np.max(rho)],
                 [-np.inf, -np.inf, 0, -10, 0, 0, -np.inf, -np.inf, -1],
                 [-np.inf, -np.inf, 0, 0, 0, -np.inf, -np.inf, -np.max(rho), -1],
                 [-np.inf, -np.inf, 0, -10, 0, 0, -np.max(rho), -1],
                 [-np.inf, -np.inf, 0, -10, 0, 0, -np.inf, -np.inf, -np.max(rho), -1],
                 [-np.inf, -np.inf, 0, -10, 0, 0, -np.inf, -np.inf, -np.max(rho), -1, -0.5, -0.5],
                 [-np.inf, -np.inf, 0, -10, 0, 0, -np.inf, -np.inf, -np.max(rho), -1, -0.5, -0.5, 0.01],
                 [-np.inf, -np.inf, 0, -10, -10, 0, 0, -np.inf, -np.inf, -np.max(rho), -2, -2],
                ]

        bsh =   [[np.inf, np.inf, np.inf, 10, np.inf, np.inf, np.inf, np.inf, np.max(rho)],
                 [np.inf, np.inf, np.inf, 10, np.inf, np.inf, np.inf, np.inf, 1],
                 [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.max(rho), 1],
                 [np.inf, np.inf, np.inf, 10, max_phi, max_phi, np.max(rho), 1],
                 [np.inf, np.inf, np.inf, 10, np.inf, np.inf, np.inf, np.inf, np.max(rho), 1],
                 [np.inf, np.inf, np.inf, 10, np.inf, np.inf, np.inf, np.inf, np.max(rho), 1, 0.5, 0.5],
                 [np.inf, np.inf, np.inf, 10, np.inf, np.inf, np.inf, np.inf, np.max(rho), 1, 0.5, 0.5, 2],
                 [np.inf, np.inf, np.inf, 10, 10, np.inf, np.inf, np.inf, np.inf, np.max(rho), 2, 2],
                ]
        
        popt_sel = p0s[ind_main]
        for j in range(len(funcs)):
            func = funcs[j]
            fname = func_names[j]
            p0 = p0s[j]
            bs = [bsl[j], bsh[j]]
            row = j + 1

            try:
                print(func.__name__)
                popt, pcov = curve_fit(func, X, y, p0 = p0, sigma = y_err[nona], bounds = bs, nan_policy = "omit", maxfev = 1000)
                if j == ind_main:
                    popt_sel = popt
                perr = np.sqrt(np.diag(pcov))
                y_model = func(X, *popt).reshape(np.shape(arm_str))
                axs[row, 0].imshow(y_model, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
                axs[row, 0].set_title(fname, fontsize = 14)
                axs[row, 0].xaxis.set_visible(False)
                axs[row, 0].yaxis.set_visible(False)

                axs[row, 1].imshow((arm_str - y_model) / sigma_str, vmin = -5, vmax=5, cmap="PuOr", origin="lower")
                chisq = np.nansum(((arm_str - y_model) / sigma_str) ** 2) / (np.count_nonzero(~np.isnan(arm_str)) - len(p0))
                axs[row, 1].set_title(f"Normalised residual; total $\\chi^2$: {np.round(chisq, 3)}", fontsize = 14)
                axs[row, 1].xaxis.set_visible(False)
                axs[row, 1].yaxis.set_visible(False)

            except RuntimeError:
                pass


        arm_check = f"{this_gal}_{col}"
        if (arm_check in d_cat):
            d_locs = d_cat[arm_check]
            if len(d_locs) == 1:
                func = main_func_d1
                fname = "One Gaussian dip added"
                p0 = [*popt_sel, 2, d_locs[0], max_phi * 0.1]
                if gal == "NGC4535" and col == "red":
                    p0 = [*popt_sel, 2, d_locs[0], max_phi * 0.03]
                bs = [[*bsl[ind_main], 0, 0, 0], [*bsh[ind_main], 1000, max_phi, max_phi * 0.5]]
            else:
                func = main_func_d2
                fname = "Two Gaussian dips added"
                p0 = [*popt_sel, 2, d_locs[0], max_phi * 0.1, 2, d_locs[1], max_phi * 0.1]
                bs = [[*bsl[ind_main], 0, 0, 0, 0, 0, 0], [*bsh[ind_main], 1000, max_phi, max_phi * 0.5, 1000, max_phi, max_phi * 0.5]]

            try:
                popt, pcov = curve_fit(func, X, y, p0 = p0, sigma = y_err[nona], bounds = bs, nan_policy = "omit", maxfev = 5000)
                perr = np.sqrt(np.diag(pcov))
                y_model = func(X, *popt).reshape(np.shape(arm_str))
                axs[9, 0].imshow(y_model, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
                axs[9, 0].set_title(fname, fontsize = 14)
                axs[9, 0].xaxis.set_visible(False)
                axs[9, 0].yaxis.set_visible(False)

                axs[9, 1].imshow((arm_str - y_model) / sigma_str, vmin = -5, vmax=5, cmap="PuOr", origin="lower")
                chisq = np.nansum(((arm_str - y_model) / sigma_str) ** 2) / (np.count_nonzero(~np.isnan(arm_str)) - len(p0))
                axs[9, 1].set_title(f"Normalised residual; total $\\chi^2$: {np.round(chisq, 3)}", fontsize = 14)
                axs[9, 1].xaxis.set_visible(False)
                axs[9, 1].yaxis.set_visible(False)

            except RuntimeError:
                pass

        else:
            axs[9, 0].set_visible(False)
            axs[9, 1].set_visible(False)
        
        fig.suptitle(f'{this_gal}: "{col}" spiral arm and models', fontsize = 20, y = 0.96, x = 0.72)
        scm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-5, vmax=5, clip=False), cmap="PuOr")
        cbar = fig.colorbar(scm, ax=axs, pad = 0.01, aspect=50, ticks = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label("Residual normalised to noise, $\\sigma$", size = 16, labelpad = -40)
        fig.savefig(f"../../images/selected/NGC1300_green_multiple_fits.png", dpi = 100)
        plt.close(fig)

        print(f"{col} done")

gals = ["NGC1300"]
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
        "NGC4535_green": [2, 4.5],
        "NGC4535_red": [3, 4.75],
        "NGC5247_blue": [3, 6]
    }

    fit_profile_2d(col_arr, phi_arr, r_arr, flip, d_cat)
    print(f"{gal} done")