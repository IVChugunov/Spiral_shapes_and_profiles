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
from scipy.optimize import curve_fit

def fit_profiles_1d(col_arr, phi_arr, r_arr, flip):
    I_all = []
    I_e_all = []
    r_all = []

    this_gal = os.getcwd().split("/")[-1]
    arm_profiles = []

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
        arm_str, sigma_str, phi_span, r = squeeze(arm_str_o, sigma_str_o, phi_span_o, r_o, 20)

        I = np.zeros_like(arm_str[0, :]) * np.nan
        I_e = np.zeros_like(arm_str[0, :]) * np.nan
        w = np.zeros_like(arm_str[0, :]) * np.nan
        w_e = np.zeros_like(arm_str[0, :]) * np.nan
        sk = np.zeros_like(arm_str[0, :]) * np.nan
        sk_e = np.zeros_like(arm_str[0, :]) * np.nan
        n = np.zeros_like(arm_str[0, :]) * np.nan
        n_e = np.zeros_like(arm_str[0, :]) * np.nan

        for j in range(len(arm_str[0, :])):
            y = arm_str[:, j]
            sigma = sigma_str[:, j]
            x = np.arange(len(y)) - (len(y) // 2)
            nona = ~np.isnan(y)
            try:
                p0 = [np.nanmax(y), (np.max(x[nona]) - np.min(x[nona])) / 2, (np.max(x[nona]) - np.min(x[nona])) / 2, 0]
                bs = ([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
                popt_g, pcov_g = curve_fit(gaussian_asymm_offs, x[nona], y[nona], p0 = p0, bounds = bs, sigma = sigma[nona])

                p0 = [np.nanmax(y), popt_g[1], popt_g[2], 0.5, popt_g[3]]
                bs = ([0, 0, 0, 0.01, popt_g[3] - 0.001], [np.inf, np.inf, np.inf, 2, popt_g[3] + 0.001])
                popt, pcov = curve_fit(sersic_asymm_offs, x[nona], y[nona], p0 = p0, bounds = bs, sigma = sigma[nona])

                perr = np.sqrt(np.diag(pcov))
                if (popt[0] == 0) or (np.abs(popt[1]) > np.max(r)) or (np.abs(popt[2]) > np.max(r)):
                    pass
                else:
                    I[j] = popt[0]
                    I_e[j] = perr[0]

                    w[j] = np.abs(popt[1]) + np.abs(popt[2])
                    w_e[j] = perr[1] + perr[2]

                    sk[j] = (np.abs(popt[2]) - np.abs(popt[1])) / (np.abs(popt[2]) + np.abs(popt[1]))
                    sk_e[j] = (perr[1] / np.abs(popt[1])) + (perr[2] / np.abs(popt[2]))

                    n[j] = popt[3]
                    n_e[j] = perr[3]
            except RuntimeError:
                pass
            except ValueError:
                pass
            except TypeError:
                pass

        r = r * 0.75
        w = w * 0.75
        w_e = w_e * 0.75

        fig, axs = plt.subplots(figsize=[16,20], nrows = 5, ncols = 2)

        axs[1, 0].errorbar(r, w, yerr = w_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[1, 1].errorbar(phi_span, w, yerr = w_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[2, 0].errorbar(r, I, yerr = I_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[2, 1].errorbar(phi_span, I, yerr = I_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[3, 0].errorbar(r, sk, yerr = sk_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[3, 1].errorbar(phi_span, sk, yerr = sk_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[4, 0].errorbar(r, n, yerr = n_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)
        axs[4, 1].errorbar(phi_span, n, yerr = n_e, c=col, marker = ".", markersize = 10, ls = "", elinewidth=2)

        n_wts = 1 / (n_e ** 2)
        nona = np.isfinite(n_wts)
        med_n = np.median(n[nona])
        avg_n = np.average(n[nona], weights = n_wts[nona])
        std_n = np.sqrt(np.average((n[nona] - avg_n)**2, weights=n_wts[nona]))

        r_str = np.array2string(r[nona]).replace('\n', '')
        sk_str = np.array2string(sk[nona]).replace('\n', '')
        ske_str = np.array2string(sk_e[nona]).replace('\n', '')
        arm_profiles.append(f"{col} {np.median(n[nona])} {r_str} {sk_str} {ske_str}")

        axs[4, 0].plot([], [], c = "w", label = f"Median n: {np.round(med_n, 2)}")

        axs[0, 0].imshow(arm_str_o, norm=LogNorm(vmin=0.01, vmax=3), cmap="inferno", origin="lower")
        axs[0, 0].set_title("Linearized arm")

        axs[0, 1].set_visible(False)

        axs[1, 0].set_title("width vs r")
        axs[1, 0].set_ylabel("width, arcsec")
        axs[1, 0].set_ylim(np.nanmin(w) * 0.8, np.nanmax(w) * 1.1)

        axs[1, 1].set_title("width vs phi")
        axs[1, 1].set_ylim(np.nanmin(w) * 0.8, np.nanmax(w) * 1.1)
        
        nz = I > 0
        axs[2, 0].set_title("I vs r")
        axs[2, 0].set_xlabel("r, arcsec")
        axs[2, 0].set_ylabel("I, MJy/sterad")
        axs[2, 0].set_ylim(np.nanmin(I[nz]) * 0.8, np.nanmax(I[nz]) * 1.2)
        axs[2, 0].set_yscale("log")
        axs[2, 1].set_title("I vs phi")
        axs[2, 1].set_xlabel("phi, radians")
        axs[2, 1].set_ylim(np.nanmin(I[nz]) * 0.8, np.nanmax(I[nz]) * 1.2)
        axs[2, 1].set_yscale("log")

        axs[3, 0].set_title("skewness vs r")
        axs[3, 0].set_ylabel("S")
        axs[3, 0].set_ylim(np.nanmin(sk) - 0.1, np.nanmax(sk) + 0.1)
        axs[3, 1].set_title("skewness vs phi")
        axs[3, 1].set_ylim(np.nanmin(sk) - 0.1, np.nanmax(sk) + 0.1)

        axs[4, 0].set_title("Sersic n vs r")
        axs[4, 0].set_ylabel("n")
        axs[4, 0].set_ylim(0, np.nanmax(n) + 0.1)
        axs[4, 0].legend()
        axs[4, 1].set_title("Sersic n vs phi")
        axs[4, 1].set_ylim(0, np.nanmax(n) + 0.1)

        gal = os.getcwd().split("/")[-1]
        fig.suptitle(f"{this_gal} {col} arm profiles")
        fig.tight_layout()
        fig.savefig(f"../../images/grids/precise_1d_fits/{this_gal}_{col}.png", dpi = 300)
        plt.close(fig)

    if os.path.exists("spiral_params/profiles_precise.dat"):
        os.remove("spiral_params/profiles_precise.dat")
    with open("spiral_params/profiles_precise.dat", "w") as file:
        for line in arm_profiles:
            file.write(f"{line}\n")


gals = np.sort(glob.glob("*"))
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

    fit_profiles_1d(col_arr, phi_arr, r_arr, flip)
    print(f"{gal} done")