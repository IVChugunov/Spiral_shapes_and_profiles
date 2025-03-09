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
    arm_profiles = []

    for i in range(len(col_arr)):
        col = col_arr[i]
        fn = f"../../images/grids/2d_fits_final/{this_gal}_{col}.png"
        if os.path.exists(fn):
            os.remove(fn)
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

        fig, axs = plt.subplots(figsize=[32,20], nrows = 5, ncols = 4)
        axs[0, 0].imshow(arm_str, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
        axs[0, 1].set_visible(False)

        try:
            popt, pcov = curve_fit(o1_r_o1_r, X, y, p0 = p0, sigma = y_err[nona], bounds = bs, nan_policy = "omit", maxfev = 5000)
            perr = np.sqrt(np.diag(pcov))
            y_model = o1_r_o1_r(X, *popt).reshape(np.shape(arm_str))
        except RuntimeError:
            continue

        ######

        funcs = [main_func_cw, main_func_nozpw, main_func_cbr, main_func_sharp, main_func, main_func_w_asymm, main_func_w_n, main_func_add]
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
            p0 = p0s[j]
            bs = [bsl[j], bsh[j]]
            column = ((j + 1) // 5) * 2
            row = ((j + 1) % 5)

            try:
                print(func.__name__)
                popt, pcov = curve_fit(func, X, y, p0 = p0, sigma = y_err[nona], bounds = bs, nan_policy = "omit", maxfev = 1000)
                if j == ind_main:
                    popt_sel = popt
                perr = np.sqrt(np.diag(pcov))
                y_model = func(X, *popt).reshape(np.shape(arm_str))
                axs[row, column].imshow(y_model, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
                axs[row, column].set_title(f"{func.__name__}")

                axs[row, column + 1].imshow((arm_str - y_model) / sigma_str, vmin = -5, vmax=5, cmap="PuOr", origin="lower")
                chisq = np.nansum(((arm_str - y_model) / sigma_str) ** 2) / (np.count_nonzero(~np.isnan(arm_str)) - len(p0))
                axs[row, column + 1].set_title(f"Deviation in sigma; total $\\chi^2$: {np.round(chisq, 3)}")

                popt_str = np.array2string(popt).replace('\n', '')
                pcov_str = np.array2string(pcov).replace('\n', '')
                arm_profiles.append(f"{col} {func.__name__} {chisq} {N_pix} {popt_str} {pcov_str}")
            except RuntimeError:
                arm_profiles.append(f"{col} {func.__name__} [nan] {N_pix} [nan] [nan]")


        arm_check = f"{this_gal}_{col}"
        if (arm_check in d_cat):
            d_locs = d_cat[arm_check]
            if len(d_locs) == 1:
                func = main_func_d1
                p0 = [*popt_sel, 2, d_locs[0], max_phi * 0.1]
                if gal == "NGC4535" and col == "red":
                    p0 = [*popt_sel, 2, d_locs[0], max_phi * 0.03]
                bs = [[*bsl[ind_main], 0, 0, 0], [*bsh[ind_main], 1000, max_phi, max_phi * 0.5]]
            else:
                func = main_func_d2
                p0 = [*popt_sel, 2, d_locs[0], max_phi * 0.1, 2, d_locs[1], max_phi * 0.1]
                bs = [[*bsl[ind_main], 0, 0, 0, 0, 0, 0], [*bsh[ind_main], 1000, max_phi, max_phi * 0.5, 1000, max_phi, max_phi * 0.5]]

            try:
                popt, pcov = curve_fit(func, X, y, p0 = p0, sigma = y_err[nona], bounds = bs, nan_policy = "omit", maxfev = 5000)
                perr = np.sqrt(np.diag(pcov))
                y_model = func(X, *popt).reshape(np.shape(arm_str))
                axs[4, 2].imshow(y_model, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
                axs[4, 2].set_title(f"{func.__name__}")

                axs[4, 3].imshow((arm_str - y_model) / sigma_str, vmin = -5, vmax=5, cmap="PuOr", origin="lower")
                chisq = np.nansum(((arm_str - y_model) / sigma_str) ** 2) / (np.count_nonzero(~np.isnan(arm_str)) - len(p0))
                axs[4, 3].set_title(f"Deviation in sigma; total $\\chi^2$: {np.round(chisq, 3)}")

                popt_str = np.array2string(popt).replace('\n', '')
                pcov_str = np.array2string(pcov).replace('\n', '')
                arm_profiles.append(f"{col} {func.__name__} {chisq} {N_pix} {popt_str} {pcov_str}")
            except RuntimeError:
                arm_profiles.append(f"{col} {func.__name__} [nan] {N_pix} [nan] [nan]")

        else:
            axs[4, 2].set_visible(False)
            axs[4, 3].set_visible(False)

        fig.suptitle(f"{this_gal} {col} arm")
        fig.tight_layout()
        fig.savefig(f"../../images/grids/2d_fits_final/{this_gal}_{col}.png", dpi = 300)
        plt.close(fig)

        print(f"{col} done")

    if os.path.exists("spiral_params/profiles_2d_final.dat"):
        os.remove("spiral_params/profiles_2d_final.dat")
    with open("spiral_params/profiles_2d_final.dat", "w") as file:
        for line in arm_profiles:
            file.write(f"{line}\n")

gals = np.sort(glob.glob("*"))
os.chdir(gals[0])
for gal in gals:
    if not gal in ["NGC4535"]:
        continue
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