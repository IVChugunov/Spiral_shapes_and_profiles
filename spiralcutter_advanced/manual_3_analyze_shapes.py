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

    fig, (ax1, ax2) = plt.subplots(figsize=[12,4], ncols = 2, nrows = 1, gridspec_kw={'width_ratios': [1, 2]})

    arm_shapes = []
    arms_real = []
    r_breaks = []

    chisq_o3 = []
    chisq_arch_o3 = []

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
        w = w_all[this_arm]
        wf = (w_all[this_arm] * 2 * np.sqrt(2 * np.log(2)))
        #r_err = re_all[this_arm]

        r_err = r * rel_err

        #ax2.scatter(phi, r, c=arms[i], s = 3, alpha = 0.5)
        ax2.errorbar(phi, r, yerr=r_err, c=arms[i], marker = ".", markersize = 5, ls = "", elinewidth=1, zorder = -5)

        if len(r) <= 5:
            continue
        
        phi_arr = np.linspace(phi[0], phi[-1], 100)
        p0 = [r[0], 0]
        popt_i, pcov_i = curve_fit(sp_o1, np.radians(phi - phi[0]), r, p0 = p0, sigma = np.sqrt(r))
        #mu = np.round(np.degrees(np.arctan(popt[1])) * direction, 1)

        phi_diff = np.radians(np.max(phi) - np.min(phi))

        fs = [sp_o1, sp_o2, sp_o3, sp_o4, sp_o5, sp_o1_br1, sp_o2_br1, sp_o3_br1, sp_o1_br2, sp_o2_br2, arch_sp, arch_sp_o3, hyp_sp, hyp_sp_o3]
        orders = np.array([1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1, 3, 1, 3])
        brs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 0])
        types = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, -1, -1, -2, -2])
        nparss = orders * (brs + 1) + brs
        chisqs = np.zeros(len(nparss))
        lss = [":", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        als = np.ones(len(nparss))
        lws = np.ones(len(nparss))
        cs = ["k"] * len(nparss)

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

        if func_id == 9:
            lss[func_id] = "-"
            lws[func_id] = 3
            als[func_id] = 0.5
            cs[func_id] = "k"
        if func_id == 2:
            lss[func_id] = "-."
        if func_id == 3:
            lss[func_id] = "-"

        for j in range(len(fs)):
            br = brs[j]
            order = orders[j]

            if br == 0:
                p = [r[0], popt_i[1], *[0] * (order - 1)]
                b = ([0, *[-np.inf] * order], [np.inf, *[np.inf] * order])
            elif br == 1:
                p = [r[0], phi_diff * 0.5, popt_i[1] * 1.1, *[0] * (order - 1), popt_i[1] * 0.9, *[0] * (order - 1)]
                b = ([0, 0.1 * phi_diff, *[-np.inf] * order * 2], [np.inf, 0.9 * phi_diff, *[np.inf] * order * 2])
            else:
                p = [r[0], phi_diff * 0.3, phi_diff * 0.8, popt_i[1] * 1.1, *[0] * (order - 1), popt_i[1], *[0] * (order - 1), popt_i[1] * 0.9, *[0] * (order - 1)]
                b = ([0, 0.1 * phi_diff, 0.1 * phi_diff, *[-np.inf] * order * 3], [np.inf, 0.9 * phi_diff, 0.9 * phi_diff, *[np.inf] * order * 3])

            if phi_diff > np.pi * 0.5:
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
                    chisq = np.sum(((fs[j](np.radians(phi - phi[0]), *popt) - r) / r_err) ** 2) / (len(r) - len(p))
                    chisqs[j] = chisq
                    if j not in [11, 13]:
                        ax2.plot(phi_arr, r_arr, c=cs[j], lw = lws[j], ls = lss[j], alpha = als[j])
                except RuntimeError:
                    chisqs[j] = np.nan
            else:
                p0 = [r[0], popt_i[1], 0]
                popt, pcov = curve_fit(sp_o2, np.radians(phi - phi[0]), r, p0 = p0, sigma = r_err)
                name_sel = "sp_o2"
                popt_sel = popt
                pcov_sel = pcov
                r_arr = sp_o2(np.radians(phi_arr - phi_arr[0]), *popt)
                chisq = np.sum(((sp_o2(np.radians(phi - phi[0]), *popt) - r) / r_err) ** 2) / (len(r) - len(p0))
        if phi_diff > np.pi * 0.5:
            ax2.plot([], [], c=arms[i], lw = 1)#, label = f"$\\chi^2$: o3 {np.round(chisqs[2], 1)}, o5 {np.round(chisqs[4], 1)}, o2b1 {np.round(chisqs[6], 1)}")
            chisq_o3.append(chisqs[2])
            chisq_arch_o3.append(chisqs[11])
        else:
            ax2.plot(phi_arr, r_arr, c="k", ls = "--", lw = 1)#, label = f"$\\chi^2$: o2 {np.round(chisq, 2)}")#, label = f"$\\mu$ = {mu_avg}, $\\Delta \\mu$ = {mu_delta}")

        popt_str = np.array2string(popt_sel).replace('\n', '')
        pcov_str = np.array2string(pcov_sel).replace('\n', '')
        arm_shapes.append(f"{arms[i]} {name_sel} {phi[0]} {phi[-1]} {popt_str} {pcov_str}")

    cmap = matplotlib.cm.Greys_r
    cmap.set_bad("k",1.)
    ys, xs = np.shape(image)
    y1 = ys // 4
    y2 = 3 * ys // 4
    x1 = xs // 4
    x2 = 3 * xs // 4
    ax1.plot([xs * 0.27, xs * 0.27 + 80], [ys * 0.27, ys * 0.27], c = "w", lw = 2)
    ax1.text(xs * 0.27 + 40, ys * 0.29, "1'", c = "w", ha = "center", va = "center", size = 12)
    ax1.imshow(image, norm=LogNorm(vmin=0.01, vmax=20), cmap=cmap, origin="lower")
    ax1.set_xlim(x1, x2)
    ax1.set_ylim(y1, y2)
    ys = np.round(ys * 0.75, 2)
    xs = np.round(xs * 0.75, 2)
    #ax1.set_xlabel(f"{xs} arcsec")
    #ax1.set_ylabel(f"{ys} arcsec")
    for p in patch_list:
        p.set_linewidth(0.2)
        ax1.add_patch(p)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(gal)

    ax2.plot([], [], c="k", ls = ":", lw = 1, label = "N = 1")
    ax2.plot([], [], c="k", ls = "--", lw = 1, label = "N = 2")
    ax2.plot([], [], c="k", ls = "-.", lw = 1, label = "N = 3")
    ax2.plot([], [], c="k", ls = "-", lw = 1, label = "N = 4")
    ax2.plot([], [], c="k", ls = "-", lw = 3, alpha = 0.3, label = "breaks = 2, N = 2")
    ax2.legend(fontsize=8)
    ax2.set_ylim(np.min(r_all) * 0.8, np.max(r_all) * 1.25)
    ax2.set_yscale('log')
    ax2.set_xlabel('$\\varphi$, degrees')
    ax2.set_ylabel('$r$, arcsec')
    ax2.set_title('Arm shapes in log-polar coordinates')

    fig.tight_layout()
    fig.savefig(f"../../images/selected/NGC5247_shapes.png", dpi = 300)
    return arms_real, r_breaks, chisq_o3, chisq_arch_o3

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
    arms_real, r_breaks, chisq_o3, chisq_arch_o3 = draw_grids(gal, col_fit, arms, phi_all, I_all, r_all, w_all, re_all, image, lines_good)
    return r_breaks, chisq_o3, chisq_arch_o3

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
    r_breaks, chisq_o3, chisq_arch_o3 = analyze(col_fit, params_fit, cuts_data, cen, PA, ell, image, lines_good)
    return r_breaks, chisq_o3, chisq_arch_o3

gals = np.sort(glob.glob("*"))
os.chdir(gals[0])


gal = "NGC5247"
os.chdir(f"../{gal}")
image = "image.fits"
fit = "fit_nosp.imfit"
cuts_all = "cuts_all.dat"
cuts_fitted = "cuts_fits.dat"
lines_good = "lines_good.reg"
r_breaks, chisq_o3, chisq_arch_o3 = spiralcutter(image, fit, cuts_all, cuts_fitted, lines_good)
print(f"{gal} done")