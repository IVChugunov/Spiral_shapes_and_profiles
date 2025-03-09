#! /usr/bin/env python3

import os
import argparse
import numpy as np
from astropy.io import fits
from scipy import interpolate
from scipy.ndimage import rotate
from libs.ImfitModelFork import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob

def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def pa(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (np.arctan2(y2 - y1, x2 - x1))

def from_polar(cen, r, phi):
    xy = np.transpose(np.array((r * np.cos(phi), r * np.sin(phi))))
    return xy + cen

def rotate_around_origin(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    point_r = np.array((qx, qy))
    return point_r

def read_reg(name_reg, cen):
    with open(name_reg, "r") as file:
        dots = []
        colors = []
        markers = []
        for line in file:
            if "point" in line:
                c = line.split("(")[1].split(")")[0].split(",")
                dots.append([float(c[0]), float(c[1])])
                if "color=" in line:
                    colors.append(line.split("=")[-1].split("\n")[0])
                else:
                    colors.append("green")
                markers.append(line.split("point=")[-1].split()[0])
        dists = np.zeros(len(dots))
        dots = np.array(dots)
        markers = np.array(markers)
        for i in range(len(dists)):
            dists[i] = dist(cen, dots[i])
        colors = np.array(colors)
        return dots, colors, dists, markers

def rotate_image_and_point(image, mask, angle, p_mark):
    y, x = image.shape
    origin = np.array((x / 2, y / 2))

    image_r = rotate(image, np.degrees(angle), reshape = False)
    mask_r = rotate(mask, np.degrees(angle), reshape = False)
    masked_r = np.where(mask_r > 0.5, True, False)

    image_r_m = np.ma.masked_array(image_r, masked_r)

    #cen_r = rotate_around_origin(origin, cen, angle)
    p_mark_r = rotate_around_origin(origin, p_mark, -angle)
    #y coord is constant

    return image_r_m, p_mark_r

def patch_mask(mask, cen, dist_min):
    ys, xs = np.shape(mask)
    x = np.arange(xs)
    y = np.arange(ys)
    xs, ys = np.meshgrid(x, y, sparse=True)
    rr = np.sqrt((ys - cen[0])**2 + (xs - cen[1])**2)
    patch = np.where(rr < dist_min, 1, 0)
    mask = mask + patch
    return(mask)


def pa_setting(dots, colors, cen):
    arms = np.unique(colors)
    posang = np.zeros(len(dots))
    for i in range(len(arms)):
        dots_c = dots[colors == arms[i]]
        pa_c = np.zeros(len(dots_c))
        for j in range(len(dots_c)):
            pa_c[j] = pa(cen, dots_c[j])
        #позиционные углы идут так, чтобы срез был по направлению изнутри наружу
        posang[colors == arms[i]] = pa_c
    return posang

def calc_rphi(dots, dists, posang, colors):
    arms = np.unique(colors)
    arm_funcs = []
    phi_lims = []
    for i in range(len(arms)):
        dots_c = dots[colors == arms[i]]
        dists_c = dists[colors == arms[i]]
        posang_c = posang[colors == arms[i]]
        steps_c = np.diff(posang_c)
        steps_c = np.where(steps_c > np.pi, steps_c - (2 * np.pi), steps_c)
        steps_c = np.where(steps_c < -np.pi, steps_c + (2 * np.pi), steps_c)
        phi_c = np.cumsum(steps_c) + posang_c[0]
        phi_c = np.insert(phi_c, 0, posang_c[0])
        func = interpolate.interp1d(phi_c, dists_c, fill_value="extrapolate")
        arm_funcs.append(func)
        phi_lims.append([np.min(phi_c), np.max(phi_c)])
    return arm_funcs, phi_lims

def calc_r_of_phi(phi, arm_funcs, phi_lims):
    r_arr = []
    for i in range(len(arm_funcs)):
        this_lims = phi_lims[i]
        this_func = arm_funcs[i]
        up_turns = int(np.floor((this_lims[1] - phi) / (2 * np.pi)))
        down_turns = int(np.floor((phi - this_lims[0]) / (2 * np.pi)))
        all_phi = phi + 2 * np.pi * np.arange(-down_turns, up_turns + 1)
        try:
            r_arr.append(*this_func(all_phi))
        except TypeError:
            pass
    r_arr = np.array(r_arr)
    return r_arr

def cuts_creation(image, mask, dots, dists, posang, colors, cen, PA, ell):
    arms = np.unique(colors)
    cuts = []
    arm_funcs, phi_lims = calc_rphi(dots, dists, posang, colors)
    plt.figure(figsize=[12,8])
    file = open("cuts_all_masked.dat", "w")
    for i in range(len(arms)):
        phi_dense = np.arange(*phi_lims[i], np.radians(3))
        r_dense = arm_funcs[i](phi_dense)
        dots_dense = from_polar(cen, r_dense, phi_dense)
        seq_w = []
        #вызываем создание срезов
        for j in range(len(phi_dense)):
            r_arr = calc_r_of_phi(phi_dense[j], arm_funcs, phi_lims)

            r_inner = r_arr[r_arr < r_dense[j]]
            r_outer = r_arr[r_arr > r_dense[j]]

            if len(r_inner) > 0:
                w_in = np.min(((r_dense[j] - np.max(r_inner)) / 3, r_dense[j] / 3))
            else:
                w_in = r_dense[j] / 3

            if len(r_outer) > 0:
                w_out = np.min(((np.min(r_outer) - r_dense[j]) / 3, r_dense[j] / 2))
            else:
                w_out = r_dense[j] / 2

            w_in = int(np.round(w_in))
            w_out = int(np.round(w_out))

            s_th = int(np.max((np.round((w_in + w_out) / 20), 1)))

            image_r, dot = rotate_image_and_point(image, mask, phi_dense[j], dots_dense[j])

            x_mark = int(np.round(dot[0]))
            y_mark = int(np.round(dot[1]))

            cut = np.mean(image_r[y_mark - s_th:y_mark + s_th + 1, x_mark - w_in:x_mark + w_out + 1], axis = 0)

            cut_coord = np.arange(len(cut)) - w_in
            cut_str = np.array2string(cut).replace('\n', '')
            file.write(f"{arms[i]}, {phi_dense[j]}, {np.min(cut_coord) + r_dense[j]}, {np.max(cut_coord) + r_dense[j]}, {r_dense[j]}, {cut_str}\n")
            print(np.degrees(phi_dense[j]))
        print(f"{arms[i]} arm done")
    file.close()

def make_segment(p, pa, w_in, w_out):
    x, y = p
    x_in = x - np.cos(pa) * w_in
    x_out = x + np.cos(pa) * w_out
    y_in = y - np.sin(pa) * w_in
    y_out = y + np.sin(pa) * w_out
    p_in = np.array([x_in, y_in])
    p_out = np.array([x_out, y_out])
    return p_in, p_out

def spiralcutter(name_img, fit, dots_reg, mask, psf):    
    # find disc
    model = ImfitModel(fit)
    disc = model.get_disc()
    if disc is None:
        raise Exception(f"No Exponential-like or Sersic function found in model {fit}!")
    xcen = disc.get_par_by_name("X0").value
    ycen = disc.get_par_by_name("Y0").value
    PA = np.radians(disc.get_par_by_name("PA").value)
    ell = disc.get_par_by_name("ell").value
    cen = np.array([xcen, ycen])

    dots, colors, dists, markers = read_reg(dots_reg, cen)
    dist_min = np.nanmin(dists) / 2
    image = fits.getdata(name_img)
    mask = patch_mask(fits.getdata(mask), (ycen, xcen), dist_min)
    pa_arr = pa_setting(dots, colors, cen)
    cuts_creation(image, mask, dots, dists, pa_arr, colors, cen, PA, ell)
    print("Done!")
    print()


gals = np.sort(glob.glob("*"))
#gals = ["NGC4535"]
os.chdir(gals[0])
for gal in gals:
    os.chdir(f"../{gal}")
    spiralcutter("masked/residuals_masked.fits", "masked/fit_masked.imfit", "dots.reg", "mask.fits", "psf.fits")
    print(f"{gal} done")