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
from scipy.optimize import curve_fit, minimize

def model_base(params, phi, r, rho):
    I0, ih, ih2, ih3, gr, co, st, endcor, w0, w1, w2, S0, S1, ncor = params
    end = 1 - endcor
    ninv = 1 / (ncor + 0.5)

    g_gr = np.exp(-((phi - st - gr) ** 2) / (2 * (gr / 3) ** 2))
    g_e = np.exp(-((phi - end + co) ** 2) / (2 * (co / 3) ** 2))
    hvs = np.where(phi < st + gr, g_gr, 1)
    hvs = hvs * np.where(phi > end - co, g_e, 1)
    I_parallel = I0 * np.exp(-(ih * r + ih2 * r ** 2 + ih3 * r ** 3)) * hvs

    w_loc = w0 + w1 * r + w2 * r ** 2
    S_loc = S0 + S1 * r
    #offset = rho + o0 + (o1 * r)
    offset = rho
    w_in = w_loc * (1 - S_loc) / 2
    w_out = w_loc * (1 + S_loc) / 2
    offset = rho + (w_out - w_in) / 2

    I_perp_in = np.exp(-np.log(2) * np.power(np.abs(offset / w_in), ninv))
    I_perp_out = np.exp(-np.log(2) * np.power(np.abs(offset / w_out), ninv))
    I_perp = np.where(offset > 0, I_perp_out, I_perp_in)

    I = I_parallel * I_perp
    return I

def regularization(params, lambda_reg):
    return lambda_reg * np.sum(np.abs(np.array(params)))

def objective(params, phi, r, rho, image_data, sigma_data, lambda_reg = 1e-1):
    model_data = model_base(params, phi, r, rho)
    residual = (image_data - model_data) / sigma_data
    return np.nanmean(residual ** 2) + regularization(params, lambda_reg)

def fit_profile_2d(col_arr, phi_arr, r_arr, flip, d_cat):
    this_gal = os.getcwd().split("/")[-1]

    param_str = np.array(["lambda", "chisq", "I0", "ih", "ih2", "ih3", "growth", "cutoff", "start", "endcor", "w0", "w1", "w2", "S0", "S1", "ncor"])
    out_table = np.zeros((len(param_str), 5))
    #out_table[:, 0] = param_str

    for i in range(len(col_arr)):
        col = col_arr[i]
        phi_span = phi_arr[i]
        r = r_arr[i]

        if flip:
            phi_span = np.abs(phi_span - np.max(phi_span))
        else:
            phi_span = np.abs(phi_span - np.min(phi_span))

        arm_str = fits.getdata(f'str_arms_azavg/arm_str_{col}.fits')
        sigma_str = fits.getdata(f'str_arms_sigma/arm_str_{col}.fits')

        f_norm_c = np.nanquantile(arm_str * 100, 0.99)
        arm_str = arm_str / f_norm_c
        sigma_str = sigma_str / f_norm_c

        rho = np.arange(len(arm_str[:, 0])) - (len(arm_str[:, 0]) // 2)

        r = r / np.nanmax(r) #norm to 1
        rho = rho / np.nanmax(rho)
        rho = rho * 0.2
        phi_span = phi_span / np.nanmax(phi_span)
        #r = r * 2
        #phi_span = phi_span * 2

        r_g, rho_g = np.meshgrid(r, rho)
        phi_g, rho_g = np.meshgrid(phi_span, rho)

        y = np.ravel(arm_str)
        y_err = np.ravel(sigma_str)
        rho_rav = np.ravel(rho_g)
        r_rav = np.ravel(r_g)
        phi_rav = np.ravel(phi_g)

        nona = ~np.isnan(y)

        #X = np.array((phi_rav, r_rav, rho_rav))

        p0 = [np.nanmax(y), 1, 0, 0, 0.1, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0]
        b_lower = [0, -np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -0.5, -0.5, -0.5]
        b_upper = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.5, 0.5, np.inf, np.inf, np.inf, 0.5, 0.5, 3.5]
        bs = np.array((b_lower, b_upper)).T

        fig, axs = plt.subplots(figsize=[30,9], nrows = 3, ncols = 5)
        axs[0, 0].set_visible(False)
        axs[0, 1].set_visible(False)
        axs[0, 2].imshow(arm_str, norm=LogNorm(vmin=0.001, vmax=0.1), cmap="inferno", origin="lower")
        axs[0, 3].set_visible(False)
        axs[0, 4].set_visible(False)

        lams = [0, 0.01, 0.1, 1, 10]
        for j in range(len(lams)):
            lam = lams[j]

            try:
                result = minimize(objective, p0, args=(phi_rav[nona], r_rav[nona], rho_rav[nona], y[nona], y_err[nona], lam), bounds = bs)
                popt = result.x

                y_model = model_base(popt, phi_rav, r_rav, rho_rav).reshape(np.shape(arm_str))
                axs[1, j].imshow(y_model, norm=LogNorm(vmin=0.0001, vmax=0.1), cmap="inferno", origin="lower")
                axs[1, j].set_title(f"lambda = {lam}; coeffs {np.round(popt, 3)}")

                axs[2, j].imshow((arm_str - y_model) / sigma_str, vmin = -5, vmax=5, cmap="PuOr", origin="lower")
                chisq = np.nansum(((arm_str - y_model) / sigma_str) ** 2) / (np.count_nonzero(~np.isnan(arm_str)) - len(p0))
                axs[2, j].set_title(f"Deviation in sigma; total $\\chi^2$: {np.round(chisq, 3)}")
            except RuntimeError:
                popt = np.zeros_like(p0) * np.nan
                chisq = np.nan
            out_table[0, j] = lam
            out_table[1, j] = chisq
            out_table[2:, j] = popt

        fig.suptitle(f"{this_gal} {col} arm")
        fig.tight_layout()
        fig.savefig(f"../../images/grids/regularization/{this_gal}_{col}.png", dpi = 300)
        plt.close(fig)

        np.savetxt(f"../../tables/{this_gal}_{col}.dat", out_table.T, header = np.array2string(param_str).replace('\n', ''), fmt = "%8.3f")
        print(f"{col} done")

gals = np.sort(glob.glob("*"))
os.chdir(gals[0])
for gal in gals:
    #if not (gal.startswith("NGC0") or gal.startswith("NGC0") or gal.startswith("NGC0")):
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

    fit_profile_2d(col_arr, phi_arr, r_arr, flip, d_cat)
    print(f"{gal} done")