#! /usr/bin/env python3

import glob
import os
import numpy as np
from scipy.optimize import curve_fit
from libs.ImfitModelFork import *

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def from_polar(cen, r, phi):
    xy = np.transpose(np.array((r * np.cos(phi), r * np.sin(phi))))
    return xy + cen

def process_cuts(fit, cuts_all, cuts_fitted, lines_reg):
    model = ImfitModel(fit)
    disc = model.get_disc()
    if disc is None:
        raise Exception(f"No Exponential-like or Sersic function found in model {fit}!")
    xcen = disc.get_par_by_name("X0").value
    ycen = disc.get_par_by_name("Y0").value
    cen = np.array([xcen, ycen])

    with open(cuts_all, "r") as file:
        dists = []
        for line in file:
            rc = float(line.split(", ")[4])
            dists.append(rc)
    file.close()
    
    dists = np.array(dists)
    dmax = np.nanmax(dists)

    cuts_fits = []
    cuts_ds9 = []
    with open(cuts_all, "r") as file:
        for line in file:
            params = line.split(", ")
            color = params[0]
            phi = float(params[1])
            rmin = float(params[2])
            rmax = float(params[3])
            rc = float(params[4])
            cut = np.array(params[5][1:-2].split()).astype(float)
            cut_coord = np.linspace(rmin, rmax, len(cut))
            
            nona = ~np.isnan(cut) * ~np.isinf(cut)
            p0 = (np.nanmax(cut[nona]), rc, 10)
            fit = cut_fitting(cut_coord, cut, p0, dmax)
            try:
                popt, perr = fit
                fs = str(np.concatenate((popt, perr), axis = None)).replace("\n", "")
            except TypeError:
                popt = np.nan
                perr = np.nan
                fs = np.nan
            fit_str = f"{color}, {phi}, {fs}\n"
            cuts_fits.append(fit_str)
            try:
                r1 = popt[1] - popt[2]
                r2 = popt[1] + popt[2]

                x1, y1 = from_polar(cen, r1, phi)
                x2, y2 = from_polar(cen, r2, phi)

                cuts_ds9.append(f"line({x1},{y1},{x2},{y2}) # color={color}\n")
            except TypeError:
                pass

    file.close()
    with open(cuts_fitted, "w") as file:
        for fit_str in cuts_fits:
            file.write(fit_str)
    file.close()

    write_to_reg(lines_reg, cuts_ds9)

def cut_fitting(cut_coord, cut, p0, dmax):
    try:
        nona = ~np.isnan(cut) * ~np.isinf(cut)
        popt, pcov = curve_fit(gaussian, cut_coord[nona], cut[nona], p0=p0)
        perr = np.sqrt(np.diag(pcov))
        popt[2] = np.abs(popt[2])
        if (popt[1] > dmax * 2) or (popt[1] - popt[2] < 0) or (popt[0] < 0) or (popt[2] < 1.5):
            return np.nan
        else:
            return popt, perr
            #return str(np.concatenate((popt, perr), axis = None)).replace("\n", "")
    except Exception:
        return np.nan

def write_to_reg(out_reg, cuts_ds9):
    with open(out_reg, "w") as file:
        file.write('# Region file format: DS9 version 4.1\n')
        file.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        file.write('image\n')
        for cut in cuts_ds9:
            file.write(cut)

gals = np.sort(glob.glob("*"))
os.chdir(gals[0])
for gal in gals:
    os.chdir(f"../{gal}")
    process_cuts("fit_nosp.imfit", "cuts_all.dat", "cuts_fits.dat", "lines.reg")
    print(f"{gal} done")