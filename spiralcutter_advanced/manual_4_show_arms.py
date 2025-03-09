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
from matplotlib import patches
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy import interpolate
from scipy.ndimage import gaussian_filter, rotate
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

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

def reproject(phi_d, r_d, PA, ell):
    x = np.cos(phi_d - PA) * r_d
    y = np.sin(phi_d - PA) * r_d
    y_corr = y * (1 - ell)
    rotations = ((np.abs(phi_d - PA) + np.pi) // (2 * np.pi)) * np.sign(phi_d - PA)
    phi_p = np.arctan2(y_corr, x) + PA + (rotations * 2 * np.pi)
    r_p = np.sqrt(x ** 2 + y_corr ** 2)
    return (phi_p, r_p)

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

def from_polar(cen, r, phi):
    xy = np.transpose(np.array((r * np.cos(phi), r * np.sin(phi))))
    return xy + cen

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

def draw_grids(gal, col_fit, arms, phi_all, I_all, r_all, w_all, re_all, image, spirals, lines_good, dots, cen, PA, ell):
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

    fig = plt.figure(figsize=[12,12])
    gs = GridSpec(3, 3, left=0, right=1, wspace=0, height_ratios = [2, 2, 1.1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0], projection = "polar")
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, :])
    #ax8 = fig.add_subplot(gs[3, :])
    #ax9 = fig.add_subplot(gs[4, :])
    #fig, axs = plt.subplots(figsize=[12,12], ncols = 3, nrows = 4, gridspec_kw={'height_ratios': [2, 2, 1, 1]})

    arm_shapes = []
    arms_real = []

    reg = pyregion.open(lines_good)
    patch_list, artist_list = reg.get_mpl_patches_texts()

    reg_dots = pyregion.open(dots)
    dot_colors = []
    dot_coords = []
    for dot in reg_dots:
        dot_coords.append(dot.__dict__["coord_list"])
        dot_colors.append(dot.__dict__["attr"][1]["color"])
    dot_coords = np.array(dot_coords)

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
        if gal in o3b1_cat:
            if arms[i] in o3b1_cat[gal]:
                func_id = 7
        if gal in o2b2_cat:
            if arms[i] in o2b2_cat[gal]:
                func_id = 9

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

            try:
                popt, pcov = curve_fit(fs[j], np.radians(phi - phi[0]), r, p0 = p, bounds = b, sigma = r_err)
                if j == func_id:
                    name_sel = fs[j].__name__
                    popt_sel = popt
                    pcov_sel = pcov
                    r_arr = fs[j](np.radians(phi_arr - phi_arr[0]), *popt)

                    phi_p, r_p = reproject(np.radians(phi_arr), r_arr, PA + (np.pi / 2), ell)

                    xy = from_polar(cen, r_p / 0.75, phi_p)
                    x = xy[:, 0]
                    y = xy[:, 1]
                    ax6.plot(x, y, c=arms[i], lw=1, zorder = 5)
            except RuntimeError:
                pass

        popt_str = np.array2string(popt_sel).replace('\n', '')
        pcov_str = np.array2string(pcov_sel).replace('\n', '')
        arm_shapes.append(f"{arms[i]} {name_sel} {phi[0]} {phi[-1]} {popt_str} {pcov_str}")

    cmap = matplotlib.cm.Greys_r
    cmap.set_bad("k",1.)
    ys, xs = np.shape(image)
    x1 = xs / 6
    x2 = 5 * xs / 6
    y1 = 0
    y2 = ys

    ax1.imshow(image - spirals, norm=LogNorm(vmin=0.01, vmax=20), cmap=cmap, origin="lower")
    ax1.set_xlim(x1, x2)
    ax1.set_ylim(y1, y2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Underlying disc + central comps\n(subtracted)")

    ax2.plot([x1 + xs * 0.02, x1 + xs * 0.02 + 80], [y1 + ys * 0.02, y1 + ys * 0.02], c = "w", lw = 2)
    ax2.text(x1 + xs * 0.02 + 40, y1 + ys * 0.04, "1'", c = "w", ha = "center", va = "center", size = 12)
    ax2.imshow(image, norm=LogNorm(vmin=0.01, vmax=20), cmap=cmap, origin="lower")
    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    ax2.scatter(dot_coords[:, 0], dot_coords[:, 1], c = dot_colors, marker = "x", s = 10, lw = 0.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Image with manual spiral tracing")

    ax3.imshow(spirals, norm=LogNorm(vmin=0.001, vmax=4), cmap=cmap, origin="lower")
    ax3.set_xlim(x1, x2)
    ax3.set_ylim(y1, y2)
    for p in patch_list:
        p.set_linewidth(0.2)
        ax3.add_patch(p)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("Spirals image (with fitted slices)")

    ax6.imshow(spirals, norm=LogNorm(vmin=0.001, vmax=4), cmap=cmap, origin="lower")
    ax6.set_xlim(x1, x2)
    ax6.set_ylim(y1, y2)
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_title("Analytical shape of spirals")

    #ax_spec = plt.subplot(4, 3, 4, projection='polar')
    phi_base = np.linspace(0, np.pi * 1.5, 100)
    r_base = 5 * np.exp(phi_base * 0.3)
    ax4.plot(phi_base, r_base, lw=2, color = "k", ls = "--")
    for i in [-3, -2, -1, 1, 2, 3]:
        ax4.plot(phi_base, r_base + i, lw=0.5, color = "b")
    for i in np.linspace(0, np.pi * 1.5, 13):
        r_i = 5 * np.exp(i * 0.3)
        ax4.plot([i, i], [r_i - 3, r_i + 3], lw=2, color = "r")
    ax4.set_ylim(0, 25)
    ax4.text(np.radians(45), 15, "$r$", rotation = 45, fontsize = 12)
    ax4.set_xlabel("$\\varphi$", fontsize = 12)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.grid(True)
    ax4.set_title("Arm in polar coordinates")
    #axs[1, 0].set_visible(False)

    ax5.plot(np.degrees(phi_base), r_base - r_base, lw=4, color = "k", ls = "--")
    for i in [-3, -2, -1, 1, 2, 3]:
        ax5.plot(np.degrees(phi_base), r_base - r_base + i, lw=0.5, color = "b")
    for i in np.linspace(0, np.pi * 1.5, 13):
        ax5.plot([np.degrees(i), np.degrees(i)], [-3, 3], lw=2, color = "r")
    ax5.set_ylim(-20, 20)
    ax5.set_xlabel("$\\psi$", fontsize = 12)
    ax5.set_ylabel("$r - r(\\psi)$", fontsize = 12)
    ax5.set_xticklabels([])
    ax5.set_yticklabels([])
    ax5.grid(True)
    ax5.set_title("Straightened arm")

    str_arm_green = fits.getdata("str_arms_nomask/arm_str_green.fits")[:, 60:-60]
    str_arm_green = np.where(str_arm_green < 0, np.nanmin(abs(str_arm_green)), str_arm_green)
    #str_arm_red = fits.getdata("str_arms_nomask/arm_str_red.fits")[:, 60:-60]
    #str_arm_red = np.where(str_arm_red < 0, np.nanmin(abs(str_arm_red)), str_arm_red)

    ax7.imshow(str_arm_green, norm=LogNorm(vmin=0.01, vmax=1), cmap="gray", origin="lower")
    ax7.axhline(np.shape(str_arm_green)[0] // 2, c = "green", lw=2, zorder = 5)
    ax7.set_title("Example straightened arm")
    ax7.set_xticks([])
    ax7.set_yticks([])

    #ax8.imshow(str_arm_red, norm=LogNorm(vmin=0.01, vmax=1), cmap="gray", origin="lower")
    #ax8.set_xticks([])
    #ax8.set_yticks([])

    transFigure = fig.transFigure.inverted()

    arrow = patches.FancyArrowPatch(
        [0.38, 0.75],  # posA
        [0.27, 0.75],  # posB
        #shrinkA=0,  # so tail is exactly on posA (default shrink is 2)
        #shrinkB=0,
        transform=fig.transFigure,
        color="b",
        arrowstyle="-|>",  # "normal" arrow
        mutation_scale=30,  # controls arrow head size
        linewidth=3,
    )
    fig.patches.append(arrow)

    arrow = patches.FancyArrowPatch(
        [0.62, 0.75],  # posA
        [0.73, 0.75],  # posB
        #shrinkA=0,  # so tail is exactly on posA (default shrink is 2)
        #shrinkB=0,
        transform=fig.transFigure,
        color="b",
        arrowstyle="-|>",  # "normal" arrow
        mutation_scale=30,  # controls arrow head size
        linewidth=3,
    )
    fig.patches.append(arrow)

    arrow = patches.FancyArrowPatch(
        [0.83, 0.65],  # posA
        [0.83, 0.58],  # posB
        #shrinkA=0,  # so tail is exactly on posA (default shrink is 2)
        #shrinkB=0,
        transform=fig.transFigure,
        color="b",
        arrowstyle="-|>",  # "normal" arrow
        mutation_scale=30,  # controls arrow head size
        linewidth=3,
    )
    fig.patches.append(arrow)

    arrow = patches.FancyArrowPatch(
        [0.24, 0.36],  # posA
        [0.36, 0.36],  # posB
        #shrinkA=0,  # so tail is exactly on posA (default shrink is 2)
        #shrinkB=0,
        transform=fig.transFigure,
        color="b",
        arrowstyle="-|>",  # "normal" arrow
        mutation_scale=30,  # controls arrow head size
        linewidth=3,
    )
    fig.patches.append(arrow)

    arrow = patches.FancyArrowPatch(
        [0.83, 0.32],  # posA
        [0.64, 0.24],  # posB
        #shrinkA=0,  # so tail is exactly on posA (default shrink is 2)
        #shrinkB=0,
        transform=fig.transFigure,
        color="b",
        arrowstyle="-|>",  # "normal" arrow
        mutation_scale=30,  # controls arrow head size
        linewidth=3,
    )
    fig.patches.append(arrow)

    fig.suptitle(f"Straightening the spiral arms: {gal} example")

    fig.tight_layout()
    fig.savefig(f"../../images/selected/str_arms_example.png")
    plt.close(fig)

def analyze(col_fit, params_fit, cuts_data, cen, PA, ell, image, spirals, lines_good, dots):
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
    draw_grids(gal, col_fit, arms, phi_all, I_all, r_all, w_all, re_all, image, spirals, lines_good, dots, cen, PA, ell)

def spiralcutter(img_name, spirals_name, fit, cuts_all, cuts_fitted, lines_good, dots):    
    # find disc
    image = fits.getdata(img_name)
    spirals = fits.getdata(spirals_name)
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
    analyze(col_fit, params_fit, cuts_data, cen, PA, ell, image, spirals, lines_good, dots)

gals = ["NGC1300"]
os.chdir(gals[0])
for gal in gals:
    os.chdir(f"../{gal}")
    image = "image.fits"
    spirals = "masked/residuals_masked.fits"
    fit = "fit_nosp.imfit"
    cuts_all = "cuts_all.dat"
    cuts_fitted = "cuts_fits.dat"
    lines_good = "lines_good.reg"
    dots = "dots.reg"
    spiralcutter(image, spirals, fit, cuts_all, cuts_fitted, lines_good, dots)
    print(f"{gal} done")