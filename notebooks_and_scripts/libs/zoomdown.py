#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt


def main():
    pass


def zoomdown(in_array, zoom_ratio, nan_lim=0.5, fix_nan_flux=True, pad=True):
    # Determine the shape of the output array
    in_y_size, in_x_size = in_array.shape
    out_x_size = int(math.ceil(in_x_size * zoom_ratio))
    out_y_size = int(math.ceil(in_y_size * zoom_ratio))

    # Pad input array
    pad_width = ((0, math.ceil(1/zoom_ratio)),
                 (0, math.ceil(1/zoom_ratio)))
    in_array = np.pad(in_array, pad_width=pad_width, mode="edge")

    out_array = np.zeros((out_y_size, out_x_size))
    # Iterate over indices of the resulting (i.e. smaller) array
    for x in np.arange(out_x_size, dtype=float):
        for y in np.arange(out_y_size, dtype=float):
            # Compute the coordinates of this out array pixel inside of the
            # input array
            x_in_lb = x / zoom_ratio  # Left bottom corner
            y_in_lb = y / zoom_ratio  #
            x_in_rb = (x+1) / zoom_ratio  # Right bottom corner
            y_in_rb = y_in_lb             #
            x_in_lt = x_in_lb             # Left top corner
            y_in_lt = (y+1) / zoom_ratio  #
            x_in_rt = x_in_rb  # Right top corner
            y_in_rt = y_in_lt  #
            # Calculate the total intensiy in this square
            intensities = []
            weights = []
            # Add left bottom corner
            weights.append((1 - x_in_lb % 1) * (1 - y_in_lb % 1))
            intensities.append(in_array[int(y_in_lb), int(x_in_lb)])
            # Add right bottom corner
            weights.append((x_in_rb % 1) * (1 - y_in_rb % 1))
            intensities.append(in_array[int(y_in_rb), int(x_in_rb)])
            # Add left top corner
            weights.append((1 - x_in_lt % 1) * (y_in_lt % 1))
            intensities.append(in_array[int(y_in_lt), int(x_in_lt)])
            # Add right top corner
            weights.append((x_in_rt % 1) * (y_in_rt % 1))
            intensities.append(in_array[int(y_in_rt), int(x_in_rt)])
            # Add left border (without corners)
            left_border_pixel_area = 1 - x_in_lb % 1
            if left_border_pixel_area != 0:
                weights.extend([left_border_pixel_area] * (int(y_in_lt) - int(y_in_lb+1)))
                intensities.extend(in_array[int(y_in_lb+1): int(y_in_lt), int(x_in_lb)])
            # Add top border (without corners)
            top_border_pixel_area = y_in_lt % 1
            if top_border_pixel_area != 0:
                weights.extend([top_border_pixel_area] * (int(x_in_rt) - int(x_in_lt+1)))
                intensities.extend(in_array[int(y_in_lt), int(x_in_lt+1): int(x_in_rt)])
            # Add right border (without corners)
            right_border_pixel_area = x_in_rt % 1
            if right_border_pixel_area != 0:
                weights.extend([right_border_pixel_area] * (int(y_in_rt) - int(y_in_rb+1)))
                intensities.extend(in_array[int(y_in_rb+1): int(y_in_rt), int(x_in_rb)])
            # Add bottom border (without cirners)
            bottom_border_pixel_area = 1 - y_in_lb % 1
            if bottom_border_pixel_area != 0:
                weights.extend([bottom_border_pixel_area] * (int(x_in_rb) - int(x_in_lb+1)))
                intensities.extend(in_array[int(y_in_lb), int(x_in_lb+1): int(x_in_rb)])
            # Add central region (the whole pixels)
            central = in_array[int(y_in_lb+1): int(y_in_rt), int(x_in_lb+1): int(x_in_rt)].ravel()
            weights.extend(np.ones_like(central))
            intensities.extend(central)
            # Write the intensity into the output array
            weights = np.array(weights)
            intensities = np.array(intensities)
            nan_fraction = np.sum(weights[np.isnan(intensities)]) / np.sum(weights)
            if nan_fraction > nan_lim:
                # The total area of nan pixels is greater than the given limit,
                # so we write nan in the output array
                out_array[int(y), int(x)] = np.nan
            else:
                weights = np.array(weights)
                intensities = np.array(intensities)
                out_array[int(y), int(x)] = np.nansum(intensities * weights)
                if (fix_nan_flux is True) and (nan_fraction > 0.0):
                    out_array[int(y), int(x)] /= (1-nan_fraction)
    if pad is False:
        if in_x_size * zoom_ratio > int(in_x_size*zoom_ratio):
            out_array = out_array[:, :-1]
        if in_y_size * zoom_ratio > int(in_y_size*zoom_ratio):
            out_array = out_array[:-1, :]
    return out_array


if __name__ == '__main__':
    orig = np.random.random((101, 101))
    orig[20:35, 39:59] = np.nan
    zoomed = zoomdown(orig, 0.1, pad=False)
    print(zoomed.shape)
    print(np.sum(orig), np.sum(zoomed))
    plt.subplot(121)
    plt.imshow(orig)
    plt.subplot(122)
    plt.imshow(zoomed)
    plt.show()
