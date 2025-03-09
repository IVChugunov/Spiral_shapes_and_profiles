#! /usr/bin/env python3

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pyregion
import glob
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel
from libs.ImfitModelFork import *
from libs.spiral_funcs import *
from matplotlib.colors import LogNorm
from scipy import interpolate
from scipy.ndimage import gaussian_filter, rotate
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def sersic_asymm(x, I_0, r_max, r_i, r_o, n_i, n_o):
    I_i = I_0 * np.exp(-((np.abs((r_max - x) / r_i)) ** (1 / n_i)))
    I_o = I_0 * np.exp(-((np.abs((x - r_max) / r_o)) ** (1 / n_o)))
    I = np.where(x > r_max, I_o, I_i)
    return I

def exponential(x, I0, h):
    return I0 * np.exp(-(x / h))

def linear(x, a, b):
    return a * x + b

def deproject(phi_p, r_p, w_p, re_p, PA, ell):
    x = np.cos(phi_p - PA) * r_p
    y = np.sin(phi_p - PA) * r_p
    y_corr = y / (1 - ell)
    rotations = ((np.abs(phi_p - PA) + np.pi) // (2 * np.pi)) * np.sign(phi_p - PA)
    phi_d = np.arctan2(y_corr, x) + PA + (rotations * 2 * np.pi)
    r_d = np.sqrt(x ** 2 + y_corr ** 2)
    w_d = w_p * (r_d / r_p)
    re_d = re_p * (r_d / r_p)
    return (phi_d, r_d, w_d, re_d)

def read_lines(lines_good, cen):
    with open(lines_good, "r") as file:
        lines = []
        colors = []
        for line in file:
            if "line(" in line:
                c = line.split("(")[1].split(")")[0].split(",")
                lines.append([float(c[0]), float(c[1]), float(c[2]), float(c[3])])
                if "color=" in line:
                    colors.append(line.split("=")[-1].split("\n")[0])
                else:
                    colors.append("green")
        lines = np.array(lines)
        colors = np.array(colors)
    return lines, colors

def read_cuts(cuts_fitted, cuts_all):
    col_arr = []
    params_arr = []
    cuts_arr = []

    with open(cuts_fitted, "r") as file:
        for line in file:
            params = line.split(", ")
            color = params[0]
            phi = float(params[1])
            fit = params[2][1:-2].split()

            if len(fit) == 1:
                continue
            else:
                I, r, w, _, r_err, _ = fit
                col_arr.append(color)
                params_arr.append([phi, I, r, w, r_err])
    file.close()

    col_arr = np.array(col_arr)
    params_arr = np.array(params_arr).astype(float)

    with open(cuts_all, "r") as file:
        for line in file:
            params = line.split(", ")
            color = params[0]
            phi = float(params[1])
            rmin = float(params[2])
            rmax = float(params[3])
            cut = np.array(params[5][1:-2].split(), dtype = float)
            cut_coord = np.linspace(rmin, rmax, len(cut))

            is_nona = (col_arr == color) * (params_arr[:, 0] == phi)
            if np.sum(is_nona) > 0:
                cuts_arr.append([cut_coord, cut])
    file.close()

    cuts_arr = np.array(cuts_arr, dtype = object)
    return (col_arr, params_arr, cuts_arr)

def lines_to_polar(cen, lines):
    xc, yc = cen
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]
    phi = np.arctan2(y1 - yc, x1 - xc)
    r1 = np.sqrt((x1 - xc) ** 2 + (y1 - yc) ** 2)
    r2 = np.sqrt((x2 - xc) ** 2 + (y2 - yc) ** 2)
    return (phi, r1, r2)

def find_good_cuts(cuts_fitted, cuts_all, lines_good, cen):
    good_cuts = []
    lines, col_line = read_lines(lines_good, cen)
    phi_line, r1_line, r2_line = lines_to_polar(cen, lines)
    r_line = (r1_line + r2_line) / 2
    w_line = r2_line - r_line

    col_fit, params_fit, cuts_data = read_cuts(cuts_fitted, cuts_all)

    phi_fit = params_fit[:, 0]
    r_fit = params_fit[:, 2]

    for i in range(len(r_line)):
        fit_by_color = (col_fit == col_line[i])
        phi_diff = np.degrees(phi_fit - phi_line[i])
        fit_by_phi = (phi_diff % 360 <= 1) + (phi_diff % 360 >= 359)
        fit_by_r = (np.abs(r_fit - r_line[i]) < 1)
        try:
            good_cuts.append(np.argwhere(fit_by_color * fit_by_phi * fit_by_r)[0][0])
        except IndexError:
            pass
    col_fit = col_fit[good_cuts]
    params_fit = params_fit[good_cuts, :]
    cuts_data = cuts_data[good_cuts]
    return (col_fit, params_fit, cuts_data)

def find_disc_break(imfit_file):
    # find the biggest disc, then its center and position angle
    imfit_model = ImfitModel(imfit_file)
    disc = imfit_model.get_disc()
    try:
        r_d_break = disc.get_par_by_name("r_break").value * 0.75
    except Exception:
        r_d_break = np.nan
    return r_d_break

def draw_grids(gal, col_fit, arms, phi_all, I_all, r_all, w_all, re_all, image, lines_good):
    o2b1_cat = {
        "NGC0613": ["cyan", "red"],
        "NGC0628": ["green", "red"],
        "NGC1300": ["green"],
        "NGC3184": ["green", "red"],
        "NGC4321": ["red"],
        "NGC4535": ["red"],
        "NGC5236": ["green"],
        "NGC7412": ["red"]
    }

    o3b1_cat = {
        "NGC1042": ["green", "red"],
        "NGC1232": ["red"],
        "NGC1300": ["red"],
        "NGC1672": ["green", "red"],
        "NGC4303": ["red"]
    }
    o4_cat = {
        "NGC0613": ["magenta"],
        "NGC1232": ["green"],
        "NGC4303": ["green"],
        "NGC5085": ["green", "red"],
        "NGC5236": ["red"]
    }

    o5_cat = {

    }

    o2b2_cat = {
        "NGC1566": ["green", "red"],
        "NGC4321": ["green"],
        "NGC5247": ["blue"]
    }

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(figsize=[12,16], ncols = 2, nrows = 4, gridspec_kw={'width_ratios': [1, 2]})

    arm_shapes = []
    wdevs_arr = []
    bics_arr = []

    arms_real = []
    r_breaks = []

    reg = pyregion.open(lines_good)
    patch_list, artist_list = reg.get_mpl_patches_texts()

    phi_gr = phi_all[col_fit == 'green']
    r_gr = r_all[col_fit == 'green']
    p0 = [r_gr[0], 0]
    popt, pcov = curve_fit(sp_o1, np.radians(phi_gr - np.min(phi_gr)), r_gr, p0 = p0, sigma = 0.1 * r_gr)
    if popt[1] < 0:
        direction = -1
    else:
        direction = 1

    popt, pcov = curve_fit(sp_o2, np.radians(phi_gr - np.min(phi_gr)), r_gr, p0 = [*popt, 0], sigma = 0.1 * r_gr)
    phi_unif = np.arange(np.min(phi_gr), np.max(phi_gr), 3)
    r_unif = np.interp(phi_unif, phi_gr, r_gr)
    r_model = sp_o2(np.radians(phi_unif - phi_unif[0]), *popt)
    for j in range(len(r_unif)):
        if np.nanmin(np.abs(phi_unif[j] - phi_gr)) > 3:
            r_unif[j] = np.nan
    r_unif_sm = convolve(r_unif - r_model, Gaussian1DKernel(stddev=10), boundary = "extend")
    r_dev = (r_unif_sm + r_model) - r_unif
    rel_err = np.sqrt(np.nanmean((r_dev ** 2) / (r_unif ** 2)))

    for i in range(len(arms)):
        this_arm = (col_fit == arms[i])
        if np.sum(this_arm) == 0:
            continue
        arms_real.append(arms[i])
        phi = phi_all[this_arm]
        I = I_all[this_arm]
        r = r_all[this_arm]
        w = w_all[this_arm]
        wf = (w_all[this_arm] * 2 * np.sqrt(2 * np.log(2)))
        
        r_err = r * rel_err

        err_frac = np.nanmean(r_err / wf)

        #estimate the number of correlated points, needed to modify BIC then
        fwhm = 1.66 #arcsec
        if direction > 0:
            phi_bs = [phi[0]]
        else:
            phi_bs = [phi[-1]]
        while True:
            r_here = np.interp(phi_bs[-1], phi_gr, r_gr)
            dphi = np.nanmax((np.degrees(1.66 / r_here), np.degrees(1/12))) # 1/12 is from cut width
            phi_next = phi_bs[-1] + (direction * dphi)
            phi_bs.append(phi_next)
            if ((phi_next > np.max(phi)) or (phi_next < np.min(phi))):
                phi_bs = np.array(phi_bs)
                break
        binned_phi, _ = np.histogram(phi, np.sort(phi_bs))
        N_indep = np.sum(binned_phi > 0)
        eff_psf_factor = len(phi) / N_indep

        #ax2.scatter(phi, r, c=arms[i], s = 3, alpha = 0.5)
        ax2.errorbar(phi, r, yerr=wf / 5, c=arms[i], marker = ".", markersize = 3, ls = "", elinewidth=0.5, alpha = 0.5)
        ax8.errorbar(phi, r, yerr=wf / 5, c=arms[i], marker = ".", markersize = 3, ls = "", elinewidth=0.5, alpha = 0.5)

        if len(r) <= 5:
            continue
        
        phi_arr = np.linspace(phi[0], phi[-1], 100)
        p0 = [r[0], 0]
        popt_i, pcov_i = curve_fit(sp_o1, np.radians(phi - phi[0]), r, p0 = p0, sigma = np.sqrt(r))

        phi_diff = np.radians(np.max(phi) - np.min(phi))

        fs = [sp_o1, sp_o2, sp_o3, sp_o4, sp_o5, sp_o1_br1, sp_o2_br1, sp_o3_br1, sp_o1_br2, sp_o2_br2, arch_sp, arch_sp_o3, hyp_sp, hyp_sp_o3, sp_wave]
        orders = np.array([1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1, 3, 1, 3, 4])
        brs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0])
        types = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, -1, -1, -2, -2, -3])
        nparss = orders * (brs + 1) + brs
        wdevs = np.zeros(len(nparss))
        bics = np.zeros(len(nparss))
        lss = ["", ":", "-", "", "--", "", "-.", "", "", "", "", "--", "", "-.", "-"]
        als = np.ones(len(nparss))
        lws = np.ones(len(nparss))
        cs = [arms[i]] * len(nparss)
        cs[-1] = "k"

        break_f = np.nan
        r_break = np.nan
        func_id = 2 #by default
        if phi_diff < np.pi * 0.5:
            func_id = 1
        if phi_diff >= np.pi:
            func_id = 3
        if gal in o4_cat:
            if arms[i] in o4_cat[gal]:
                func_id = 3
        if gal in o5_cat:
            if arms[i] in o5_cat[gal]:
                func_id = 4
        if gal in o2b1_cat:
            if arms[i] in o2b1_cat[gal]:
                func_id = 6
                break_f = 6
        if gal in o3b1_cat:
            if arms[i] in o3b1_cat[gal]:
                func_id = 7
                break_f = 7
        if gal in o2b2_cat:
            if arms[i] in o2b2_cat[gal]:
                func_id = 9
                break_f = 9
        if phi_diff < np.pi * 0.5:
            func_id = 1

        lss[func_id] = "-"
        lws[func_id] = 3
        als[func_id] = 0.5
        cs[func_id] = "k"

        for j in range(len(fs)):
            br = brs[j]
            order = orders[j]

            if br == 0:
                p = [r[0], popt_i[1], *[0] * (order - 1)]
                b = ([0, *[-np.inf] * order], [np.inf, *[np.inf] * order])
            elif br == 1:
                p = [r[0], phi_diff * 0.5, popt_i[1] * 1.1, *[0] * (order - 1), popt_i[1] * 0.9, *[0] * (order - 1)]
                if ((gal == "NGC1300") and (arms[i] == "red")):
                    p = [r[0], phi_diff * 0.2, popt_i[1] * 1.1, *[0] * (order - 1), popt_i[1] * 0.9, *[0] * (order - 1)]
                if ((gal == "NGC5247") and (arms[i] == "blue")):
                    p = [r[0], phi_diff * 0.8, popt_i[1] * 1.1, *[0] * (order - 1), popt_i[1] * 0.9, *[0] * (order - 1)]
                b = ([0, 0.1 * phi_diff, *[-np.inf] * order * 2], [np.inf, 0.9 * phi_diff, *[np.inf] * order * 2])
            else:
                p = [r[0], phi_diff * 0.3, phi_diff * 0.7, popt_i[1] * 1.1, *[0] * (order - 1), popt_i[1], *[0] * (order - 1), popt_i[1] * 0.9, *[0] * (order - 1)]
                b = ([0, 0.1 * phi_diff, 0.1 * phi_diff, *[-np.inf] * order * 3], [np.inf, 0.9 * phi_diff, 0.9 * phi_diff, *[np.inf] * order * 3])

            if types[j] == -3:
                p = [r[0], popt_i[1], 0.1, phi_diff, 0]
                b = ([0, -np.inf, -np.inf, phi_diff / 4, -np.inf], [np.inf, *[np.inf] * order])

            try:
                popt, pcov = curve_fit(fs[j], np.radians(phi - phi[0]), r, p0 = p, bounds = b, sigma = r_err)
                if j == break_f:
                    r_break = float(fs[j](popt[1], *popt))
                    r_breaks.append(r_break)
                    if j == 9:
                        r_break = float(fs[j](popt[2], *popt))
                        r_breaks.append(r_break)
                if j == func_id:
                    name_sel = fs[j].__name__
                    popt_sel = popt
                    pcov_sel = pcov
                r_arr = fs[j](np.radians(phi_arr - phi_arr[0]), *popt)

                #if types[j] == -3:
                #    print(arms[i])
                #    print(f"period: {np.round(np.degrees(popt[3]), 3)} pm {np.round(np.degrees(np.sqrt(pcov[3, 3])), 3)}")
                #    print()

                #resid = (((fs[j](np.radians(phi - phi[0]), *popt) - r) / wf) ** 2) - ((r_err / wf) ** 2)
                #w_dev = np.sqrt(np.sum(resid) / (len(r) - len(p)))
                resid = np.sqrt((fs[j](np.radians(phi - phi[0]), *popt) - r) ** 2)
                #w_dev = np.nansum(resid / wf) / (len(r) - len(p))
                w_dev = np.nanmean(resid / wf)
                bic = np.nansum(((fs[j](np.radians(phi - phi[0]), *popt) - r) ** 2) / (r_err ** 2)) + nparss[j] * np.log(len(r))
                bic = bic / len(r)
                if w_dev < 0.01:
                    wdevs[j] = np.nan
                else:
                    wdevs[j] = w_dev
                bics[j] = bic
                if phi_diff > np.pi * 0.5:
                    if j in [2, 11, 13]:
                        ax8.plot(phi_arr, r_arr, c=cs[j], lw = lws[j], ls = lss[j], alpha = als[j])
                    if j not in [11, 13]:
                        ax2.plot(phi_arr, r_arr, c=cs[j], lw = lws[j], ls = lss[j], alpha = als[j])
                else:
                    if j == 1:
                        ax2.plot(phi_arr, r_arr, c=cs[j], lw = lws[j], ls = lss[j], alpha = als[j])
            except RuntimeError:
                wdevs[j] = np.nan
                bics[j] = np.nan

        if phi_diff > np.pi * 0.5:
            ax2.plot([], [], c=arms[i], lw = 1, label = f"dev_w: o3 {np.round(wdevs[2], 3)}, o5 {np.round(wdevs[4], 3)},\no2b1 {np.round(wdevs[6], 3)}, wave {np.round(wdevs[-1], 3)}")
            ax8.plot([], [], c=arms[i], lw = 1, label = f"dev_w: o3 {np.round(wdevs[2], 3)}, arch_o3 {np.round(wdevs[11], 3)}, hyp_o3 {np.round(wdevs[13], 3)}")
        else:
            ax2.plot(phi_arr, r_arr, c=arms[i], ls = "-", lw = 3, alpha = 0.3, label = f"dev_w: o2 {np.round(wdevs[1], 3)}")#, label = f"$\\mu$ = {mu_avg}, $\\Delta \\mu$ = {mu_delta}")

        ax4.plot(nparss[types == 0], wdevs[types == 0], c=arms[i], marker = "o", ls = "-", ms = 3, lw = 0.5)
        ax4.plot(nparss[types == 1], wdevs[types == 1], c=arms[i], marker = "s", ls = "--", ms = 3, lw = 0.5)
        ax4.plot(nparss[(types == 2) * (wdevs < 100)], wdevs[(types == 2) * (wdevs < 100)], c=arms[i], marker = "^", ls = ":", ms = 3, lw = 0.5)
        ax4.plot(nparss[types == -1], wdevs[types == -1], c=arms[i], mec = "k", mew = 0.5, marker = "*", ls = "", ms = 5)
        ax4.plot(nparss[(types == -2) * (wdevs < 100)], wdevs[(types == -2) * (wdevs < 100)], c=arms[i], mec = "k", mew = 0.5, marker = "P", ls = "", ms = 5)
        ax4.plot(nparss[(types == -3) * (wdevs < 100)], wdevs[(types == -3) * (wdevs < 100)], c=arms[i], mec = "k", mew = 0.5, marker = "v", ls = "", ms = 5)
        if np.sum(wdevs) != 0:
            ax4.plot([], [], c=arms[i], ls = "-", lw = 1, label = f"length: {np.round(np.degrees(phi_diff))}$^\\circ$")

        ax6.plot(nparss[types == 0], bics[types == 0], c=arms[i], marker = "o", ls = "-", ms = 3, lw = 0.5)
        ax6.plot(nparss[types == 1], bics[types == 1], c=arms[i], marker = "s", ls = "--", ms = 3, lw = 0.5)
        ax6.plot(nparss[(types == 2) * (bics < 100)], bics[(types == 2) * (bics < 100)], c=arms[i], marker = "^", ls = ":", ms = 3, lw = 0.5)
        ax6.plot(nparss[types == -1], bics[types == -1], c=arms[i], mec = "k", mew = 0.5, marker = "*", ls = "", ms = 5)
        ax6.plot(nparss[(types == -2) * (bics < 100)], bics[(types == -2) * (bics < 100)], c=arms[i], mec = "k", mew = 0.5, marker = "P", ls = "", ms = 5)
        ax6.plot(nparss[(types == -3) * (bics < 100)], bics[(types == -3) * (bics < 100)], c=arms[i], mec = "k", mew = 0.5, marker = "v", ls = "", ms = 5)

        popt_str = np.array2string(popt_sel).replace('\n', '')
        pcov_str = np.array2string(pcov_sel).replace('\n', '')
        wdevs_str = np.array2string(wdevs).replace('\n', '')
        bics_str = np.array2string(bics).replace('\n', '')
        arm_shapes.append(f"{arms[i]} {name_sel} {phi[0]} {phi[-1]} {popt_str} {pcov_str}")
        wdevs_arr.append(f"{arms[i]} {name_sel} {phi[0]} {phi[-1]} {wdevs_str}")
        bics_arr.append(f"{arms[i]} {name_sel} {phi[0]} {phi[-1]} {bics_str}")

    with open("spiral_params/shapes_masked.dat", "w") as file:
        for line in arm_shapes:
            file.write(f"{line}\n")

    with open("spiral_params/wdevs_masked.dat", "w") as file:
        for line in wdevs_arr:
            file.write(f"{line}\n")

    with open("spiral_params/BICs_masked.dat", "w") as file:
        for line in bics_arr:
            file.write(f"{line}\n")

    cmap = matplotlib.cm.Greys_r
    cmap.set_bad("k",1.)
    ax1.imshow(image, norm=LogNorm(vmin=0.01, vmax=20), cmap=cmap, origin="lower")
    ys, xs = np.shape(image)
    ys = np.round(ys * 0.75, 2)
    xs = np.round(xs * 0.75, 2)
    ax1.set_xlabel(f"{xs} arcsec")
    ax1.set_ylabel(f"{ys} arcsec")
    for p in patch_list:
        p.set_linewidth(0.2)
        ax1.add_patch(p)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(gal)

    ax2.plot([], [], c="b", ls = "-", lw = 3, alpha = 0.3, label = "o2")
    ax2.plot([], [], c="b", ls = "-", lw = 1, label = "o3")
    ax2.plot([], [], c="b", ls = "--", lw = 1, label = "o5")
    ax2.plot([], [], c="b", ls = "-.", lw = 1, label = "o2b1")
    ax2.plot([], [], c="b", ls = ":", lw = 1, label = "o2b2")
    ax2.plot([], [], c="black", ls = "-", lw = 1, label = "Wave")
    ax2.legend(fontsize=6)
    ax2.set_ylim(np.min(r_all) * 0.8, np.max(r_all) * 1.25)
    ax2.set_yscale('log')
    ax2.set_title('r vs phi')

    ax3.set_visible(False)

    ax4.plot([], [], c="k", marker = "o", ls = "-", ms = 3, lw = 0.5, label = "No breaks")
    ax4.plot([], [], c="k", marker = "s", ls = "--", ms = 3, lw = 0.5, label = "One break")
    ax4.plot([], [], c="k", marker = "^", ls = ":", ms = 3, lw = 0.5, label = "Two breaks")
    ax4.plot([], [], c="k", mec = "k", mew = 0.5, marker = "*", ls = "", ms = 5, lw = 0.5, label = "Archimedean")
    ax4.plot([], [], c="k", mec = "k", mew = 0.5, marker = "P", ls = "", ms = 5, lw = 0.5, label = "Hyperbolic")
    ax4.plot([], [], c="k", mec = "k", mew = 0.5, marker = "v", ls = "", ms = 5, lw = 0.5, label = "Wave")
    ax4.legend(fontsize=8)
    ax4.grid(True, which="both")
    ax4.set_yscale('log')
    ax4.set_title('fit quality vs model complexity')
    #ax4.set_ylabel("$\\chi^2$")
    ax4.set_ylabel("Avg. deviation in w")
    ax4.set_xlabel("N params")

    ax5.set_visible(False)

    ax6.grid(True, which="both")
    ax6.set_yscale('log')
    ax6.set_title('fit quality vs model complexity')
    ax6.set_ylabel("BIC reduced")
    ax6.set_xlabel("N params")

    ax7.set_visible(False)

    ax8.plot([], [], c="k", ls = "-", lw = 1, label = "o3")
    ax8.plot([], [], c="k", ls = "--", lw = 1, label = "arch_o3")
    ax8.plot([], [], c="k", ls = "-.", lw = 1, label = "hyp_o3")
    ax8.legend(fontsize=8)
    ax8.set_ylim(np.min(r_all) * 0.8, np.max(r_all) * 1.25)
    ax8.set_yscale('log')
    ax8.set_title('r vs phi (to show log/arch/hyp spirals)')

    fig.tight_layout()
    fig.savefig(f"../../images/grids/shapes_masked/{gal}.png", dpi = 300)
    return arms_real, r_breaks

def analyze(col_fit, params_fit, cuts_data, cen, PA, ell, image, lines_good):
    gal = os.getcwd().split("/")[-1]
    phi_all_p = params_fit[:, 0]
    I_all = params_fit[:, 1]
    r_all_p = params_fit[:, 2]
    w_all_p = params_fit[:, 3]
    re_all_p = params_fit[:, 4]
    
    phi_all, r_all, w_all, re_all = deproject(phi_all_p, r_all_p, w_all_p, re_all_p, PA + (np.pi / 2), ell)
    phi_all = np.degrees(phi_all)
    r_all = r_all * 0.75
    w_all = w_all * 0.75
    re_all = re_all * 0.75
    arms = np.unique(col_fit)
    arms_real, r_breaks = draw_grids(gal, col_fit, arms, phi_all, I_all, r_all, w_all, re_all, image, lines_good)
    return r_breaks
    
def spiralcutter(img_name, fit, cuts_all, cuts_fitted, lines_good):    
    # find disc
    image = fits.getdata(img_name)
    model = ImfitModel(fit)
    disc = model.get_disc()
    if disc is None:
        raise Exception(f"No Exponential-like or Sersic function found in model {fit}!")
    xcen = disc.get_par_by_name("X0").value
    ycen = disc.get_par_by_name("Y0").value
    PA = np.radians(disc.get_par_by_name("PA").value)
    ell = disc.get_par_by_name("ell").value
    cen = np.array([xcen, ycen])

    col_fit, params_fit, cuts_data = find_good_cuts(cuts_fitted, cuts_all, lines_good, cen)
    r_breaks = analyze(col_fit, params_fit, cuts_data, cen, PA, ell, image, lines_good)
    return r_breaks

gals = np.sort(glob.glob("*"))
os.chdir(gals[0])

for gal in gals:
    #if not gal in ["NGC1232", "NGC1300"]:
    #    continue
    os.chdir(f"../{gal}")
    image = "image.fits"
    fit = "masked/fit_masked.imfit"
    cuts_all = "cuts_all_masked.dat"
    cuts_fitted = "cuts_fits_masked.dat"
    lines_good = "lines_masked_good.reg"
    r_breaks = spiralcutter(image, fit, cuts_all, cuts_fitted, lines_good)
    print(f"{gal} done")

"""
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=[8,8], ncols = 2, nrows = 2)
ax1.hist(chisqs_o3, np.arange(0, 36, 1), density=False, ec="black", fc="b")
ax2.hist(chisqs_arch_o3, np.arange(0, 36, 1), density=False, ec="black", fc="b")
ax3.hist(chisqs_arch_o3 - chisqs_o3, np.arange(-8, 8.5, 0.5), density=False, ec="black", fc="b",
    label = f"mean: {np.round(np.nanmean(chisqs_arch_o3 - chisqs_o3), 2)}")
ax4.hist((chisqs_arch_o3 - chisqs_o3) / (chisqs_arch_o3 + chisqs_o3), np.arange(-0.5, 0.55, 0.05), density=False, ec="black", fc="b",
    label = f"mean: {np.round(np.nanmean((chisqs_arch_o3 - chisqs_o3) / (chisqs_arch_o3 + chisqs_o3)), 3)}")


ax1.set_xlabel('$\\chi^2$')
ax1.set_ylabel('N')
ax1.set_title('log o3')
ax1.set_xlim(-1, 36)

ax2.set_xlabel('$\\chi^2$')
ax2.set_title('arch o3')
ax2.set_xlim(-1, 36)

ax3.set_xlabel('$\\chi^2$ diff.')
ax3.set_ylabel('N')
ax3.set_title('arch o3 - log o3')
ax3.set_xlim(-8, 8)
ax3.legend(fontsize=8)

ax4.set_xlabel('$\\chi^2$ rel. diff.')
ax4.set_title('rel.diff\n(arch o3 - log o3) / (arch o3 + log o3)')
ax4.set_xlim(-0.5, 0.5)
ax4.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"../../images/grids/shapes/log_vs_arch.png", dpi = 300)
"""