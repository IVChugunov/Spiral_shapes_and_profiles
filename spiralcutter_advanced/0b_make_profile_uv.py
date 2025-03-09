from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import rotate, gaussian_filter
from scipy.interpolate import CubicSpline
from libs.ImfitModelFork import ImfitModel
import warnings
import glob

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

def expdisc(r, I_0, h):
    I = I_0 * np.exp(-r / h)
    return I

def br_expdisc(r, I_0, h1, h2, r_break):
    I_rb = I_0 * np.exp(-r_break / h1)
    I_in = I_0 * np.exp(-r / h1)
    I_out = I_rb * np.exp(-(r - r_break) / h2)
    I = np.where(r > r_break, I_out, I_in)
    return I

def rotate_around_origin(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def find_az_avg(img, mask_img, xc, yc, pa, ell, q = 0.1, avg = False):
    r_arr = []
    f_arr = []
    ysize, xsize = img.shape
    y = np.arange(ysize)
    x = np.arange(xsize)
    xv, yv = np.meshgrid(x, y)
    xp, yp = rotate_around_origin([xc, yc], [xv, yv], -np.radians(pa))
    r_map = np.sqrt((yp - yc) ** 2 + ((xp - xc) / (1 - ell)) ** 2)
    
    n = int(np.ceil(np.max(r_map) + 1))
    bbs = np.arange(0, n) + 0.5 #bin borders
    rmns = np.arange(0, n)
    
    r_bins = np.digitize(r_map, bbs)
    flux = np.zeros(n)
    flux_map = np.zeros_like(img)

    for i in range(n):
        inds = (r_bins == i)
        nomask = (mask_img == 0)
        if np.sum(inds * nomask) != 0:
            if avg:
                flux[i] = np.mean(img[inds * nomask])
            else:
                flux[i] = np.quantile(img[inds * nomask], q)
        else:
            flux[i] = 0
    flux = gaussian_filter(flux, 2)
    Ir = CubicSpline(rmns, flux)
    flux = np.where(flux > 0, flux, 0)
    disc = Ir(r_map)
    disc = np.where(disc > 0, disc, 0)
    return disc, flux
    
def find_fit_params(imfit_file):
    # find the biggest disc, then its center and position angle
    imfit_model = ImfitModel(imfit_file)
    disc = imfit_model.get_disc()
    x_cen, y_cen = disc.get_center()
    posang = disc.get_par_by_name("PA").value
    ell = disc.get_par_by_name("ell").value
    I_0 = disc.get_par_by_name("I_0").value
    try:
        h = [disc.get_par_by_name("h").value]
    except Exception:
        h1 = disc.get_par_by_name("h1").value
        h2 = disc.get_par_by_name("h2").value
        r_break = disc.get_par_by_name("r_break").value
        h = [h1, h2, r_break]
    return x_cen, y_cen, posang, ell, I_0, h

gals = np.sort(glob.glob("*"))
os.chdir(gals[0])
for gal in gals:
    os.chdir(f"../{gal}")

    imfit_path = f"../../fit_results/{gal}/fit_nosp.imfit"
    x_cen, y_cen, pa, ell, I_0, h = find_fit_params(imfit_path)

    image_orig = fits.getdata(f"image.fits")
    y_cen, x_cen = np.array(np.shape(image_orig)) / 2
    mask = np.where(np.isnan(image_orig), 1, 0)

    disc_sub, flux_sub = find_az_avg(image_orig, mask, x_cen - 1, y_cen - 1, pa, ell)

    r = np.arange(len(flux_sub))

    image_sub = image_orig - disc_sub
    fits.PrimaryHDU(image_sub).writeto("spirals.fits", overwrite = True)

    tab = np.array([r, flux_sub])#, flux_model_disc])
    np.savetxt("az_avg_profiles.txt", tab)