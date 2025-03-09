#! /usr/bin/env python3

import argparse
import os
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.ndimage import rotate
from libs.ImfitModelFork import *
from libs.spiral_funcs import *
from libs.helper_funcs import *
from scipy.optimize import curve_fit

def rotate_around_origin(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def from_polar(cen, r_prop, phi_prop, PA, ell):
    x = np.cos(phi_prop) * r_prop
    y_deproj = np.sin(phi_prop) * r_prop
    y = y_deproj * (1 - ell)
    x_pic = x * np.cos(PA) - y * np.sin(PA)
    y_pic = x * np.sin(PA) + y * np.cos(PA)
    xy = np.transpose(np.array((x_pic, y_pic)))
    return xy + cen

def unfold_to_polar(image, mask_img, folder, xc, yc, pa, ell, col_arr, phi_arr, r_arr):
    #r_arr = r_arr / 0.75 #to px
    ys, xs = np.shape(image)
    good_inds = np.isfinite(image)
    if folder == "str_arms_sigma_vertical":
        image = np.where(good_inds, image, np.max(image[good_inds]))
    else:
        image = np.where(good_inds, image, 0)
    image_interp = RectBivariateSpline(np.arange(ys), np.arange(xs), image)
    mask_interp = RectBivariateSpline(np.arange(ys), np.arange(xs), np.clip(mask_img, 0, 1))

    ysize, xsize = image.shape
    y = np.arange(ysize)
    x = np.arange(xsize)
    xv, yv = np.meshgrid(x, y)
    xp, yp = rotate_around_origin([xc, yc], [xv, yv], -np.radians(pa))
    r_map = np.sqrt((yp - yc) ** 2 + ((xp - xc) / (1 - ell)) ** 2)

    r_max = int(np.floor(np.max(r_map)))

    phi_range = np.linspace(0, 2 * np.pi, 720, endpoint = False)
    r_range = np.linspace(0, r_max, r_max, endpoint = False)
    polar_r, polar_phi = np.meshgrid(r_range, phi_range)

    xy = from_polar(np.array((xc, yc)), polar_r, polar_phi, np.radians(pa) + np.pi / 2, ell)
    x_set = xy[:, :, 0]
    y_set = xy[:, :, 1]

    polar_img = image_interp(y_set, x_set, grid = False) # 0.5 deg azimuthally, 1 px radially resolution
    mask_img = mask_interp(y_set, x_set, grid = False)
    if folder == "str_arms_sigma_vertical":
        polar_img = np.where(polar_img > 0, polar_img, np.max(polar_img))
    else:
        polar_img = np.where(mask_img < 0.5, polar_img, np.nan)
    os.makedirs(folder, exist_ok = True)
    #fits.PrimaryHDU(polar_img).writeto(f'{folder}/polar.fits', overwrite = True)

    for i in range(len(col_arr)):
        col = col_arr[i]
        phi = phi_arr[i]
        r = r_arr[i]
        #max_hw = int(np.round(np.max(r) * 0.2))
        #rho_span = np.arange(-max_hw, max_hw + 1)
        dphi_span = np.radians(np.arange(-60, 61))
        #rho_2d, phi_2d = np.meshgrid(rho_span, phi_span)
        #r_2d = rho_2d + np.transpose([r])
        r_2d, dphi_2d = np.meshgrid(r, dphi_span)
        #r_2d = np.transpose([r])
        phi_2d = dphi_2d + phi
        xy_sp = from_polar(np.array((xc, yc)), r_2d, phi_2d, np.radians(pa) + np.pi / 2, ell)
        x_sp = xy_sp[:, :, 0]
        y_sp = xy_sp[:, :, 1]

        arm_str = image_interp(y_sp, x_sp, grid = False) # 0.5 deg azimuthally, 1 px radially resolution
        mask_str = mask_interp(y_sp, x_sp, grid = False)
        if folder == "str_arms_sigma_vertical":
            arm_str = np.where(arm_str > 0, arm_str, np.max(polar_img))
        #arm_str = np.where(np.transpose(np.abs(rho_2d)) < 0.3 * r, arm_str, np.nan)
        arm_str = np.where(mask_str < 0.5, arm_str, np.nan)

        mask_other = np.zeros_like(arm_str)
        for j in range(len(col_arr)):
            if j == i:
                continue
            phi_o = phi_arr[j]
            r_o = r_arr[j]
            fr_o = interp1d(phi_o, r_o, bounds_error = False)
            for k in np.arange(-3,4):
                mask_other_r = np.transpose((r_2d > (fr_o(phi_2d + (k * 2 * np.pi)) * 0.85)) * (r_2d < (fr_o(phi_2d + (k * 2 * np.pi)) * 1.15)))
                mask_other = mask_other + mask_other_r
        arm_str = np.where(mask_other < 0.5, arm_str, np.nan)
        fits.PrimaryHDU(arm_str).writeto(f'{folder}/arm_str_{col}.fits', overwrite = True)

gals = np.sort(glob.glob("*"))
#gals = ["NGC4535"]
os.chdir(gals[0])
for gal in gals:
    os.chdir(f"../{gal}")
    imfit_path = f"../../galaxies_images/{gal}/fit_nosp.imfit"
    xc, yc, pa, ell = find_fit_params(imfit_path)
    col_arr, phi_arr, r_arr = read_shapes_file(pa, uv = True)

    image = fits.getdata("spirals.fits")

    mask = np.where(np.isnan(image), 1, 0)
    yc, xc = np.array(np.shape(image)) / 2

    folder = "str_arms_azavg_vertical"
    unfold_to_polar(image, mask, folder, xc, yc, pa, ell, col_arr, phi_arr, r_arr)

    image = fits.getdata("sigma.fits")
    image[:, :] = np.nanmin(image)
    folder = "str_arms_sigma_vertical"
    unfold_to_polar(image, mask, folder, xc, yc, pa, ell, col_arr, phi_arr, r_arr)

    print(f"{gal} done")