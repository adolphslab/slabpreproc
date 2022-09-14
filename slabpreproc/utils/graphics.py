#!/usr/bin/env python3
"""
Graph plotting and image figure functions

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2019-05-31 JMT From scratch

MIT License

Copyright (c) 2022 Mike Tyszka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import periodogram
from skimage.util import montage
from skimage.exposure import rescale_intensity


def plot_motion_timeseries(motion_df, plot_fname):
    """
    Plots x, y, z displacement and rotation timeseries from MCFLIRT registrations

    :param motion_df: dataframe
        Motion correction parameters
    :param plot_fname: str
        Output plot filename
    :return:
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))

    # Plot axis displacements in mm
    motion_df.plot(
        x='Time_s',
        y=['Dx_mm', 'Dy_mm', 'Dz_mm'],
        kind='line',
        ax=axs[0]
    )
    axs[0].grid(color='gray', linestyle=':', linewidth=1)

    # Plot axis rotations in radians
    motion_df.plot(
        x='Time_s',
        y=['Rx_rad', 'Ry_rad', 'Rz_rad'],
        kind='line',
        ax=axs[1]
    )
    axs[1].grid(color='gray', linestyle=':', linewidth=1)

    # Space subplots without title overlap
    plt.tight_layout()

    # Save plot to file
    plt.savefig(plot_fname, dpi=300)

    # Close plot
    plt.close()


def plot_motion_powerspec(motion_df, plot_fname):
    """
    Plot total motion power spectrum

    :param motion_df: dataframe

    :param plot_fname:
    :return:
    """

    # Extract vectors from dataframe
    t = motion_df['Time_s'].values

    # Extract moco par values as numpy arrays
    Dxyz = motion_df[['Dx_mm', 'Dy_mm', 'Dz_mm']].values
    Rxyz = motion_df[['Rx_rad', 'Ry_rad', 'Rz_rad']].values

    # Sampling frequency (Hz)
    fs = 1.0 / (t[1] - t[0])

    # Framewise displacement timeseries
    fd = motion_df['FD_mm']

    # Power spectra framewise displacement
    f, pspec = periodogram(fd, fs, scaling='spectrum')

    # Drop first point (zero)
    f = f[1:]
    pspec = pspec[1:]

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    # Power dB relative to row max
    p_max = np.max(pspec)
    if p_max < 1e-10:
        p_db = np.zeros_like(pspec)
    else:
        p_db = 10.0 * np.log10(pspec / np.max(pspec))

    axs.plot(f, p_db)
    axs.set_title('Framewise Displacement (mm)', loc='left')
    axs.grid(color='gray', linestyle=':', linewidth=1)
    axs.set_xlabel('Frequency (Hz)')

    # Space subplots without title overlap
    plt.tight_layout()

    # Save plot to file
    plt.savefig(plot_fname, dpi=300)

    # Close plot
    plt.close()


def orthoslices(img_nii, ortho_fname, cmap='viridis', irng='default'):
    img3d = img_nii.get_data()

    # Intensity scaling
    if 'robust' in irng:
        vmin, vmax = np.percentile(img3d, (1, 99))
    elif 'noscale' in irng:
        nc = plt.get_cmap(cmap).N
        vmin, vmax = 0, nc
    else:
        vmin, vmax = np.min(img3d), np.max(img3d)

    fig, axs = plt.subplots(1, 3, figsize=(7, 2.4), constrained_layout=True)

    nx, ny, nz = img3d.shape
    hx, hy, hz = int(nx / 2), int(ny / 2), int(nz / 2)

    # Extract central section for each orientation
    # Assumes RAS orientation
    m_sag = img3d[hx, :, :].transpose()
    m_cor = img3d[:, hy, :].transpose()
    m_tra = img3d[:, :, hz].transpose()

    # Use transverse image for colorbar reference
    trafig = axs[0].imshow(
        m_tra,
        cmap=plt.get_cmap(cmap),
        vmin=vmin, vmax=vmax,
        aspect='equal',
        origin='lower'
    )
    axs[0].set_title('Transverse')
    axs[0].axis('off')

    axs[1].imshow(
        m_cor,
        cmap=plt.get_cmap(cmap),
        vmin=vmin, vmax=vmax,
        aspect='equal',
        origin='lower'
    )
    axs[1].set_title('Coronal')
    axs[1].axis('off')

    axs[2].imshow(
        m_sag,
        cmap=plt.get_cmap(cmap),
        vmin=vmin, vmax=vmax,
        aspect='equal',
        origin='lower'
    )
    axs[2].set_title('Sagittal')
    axs[2].axis('off')

    plt.colorbar(trafig, ax=axs, location='right', shrink=0.75)

    # Save plot to file
    plt.savefig(ortho_fname, dpi=300)

    # Close plot
    plt.close()

    return ortho_fname


def orthoslice_montage(img_nii, montage_fname, cmap='viridis', irng='default'):
    orient_name = ['Axial', 'Coronal', 'Sagittal']

    img3d = img_nii.get_data()

    plt.subplots(1, 3, figsize=(7, 2.4))

    for ax in [0, 1, 2]:

        # Transpose dimensions for given orientation
        ax_order = np.roll([2, 0, 1], ax)
        s = np.transpose(img3d, ax_order)

        # Downsample to 9 images in first dimension
        nx = s.shape[0]
        xx = np.linspace(0, nx - 1, 9).astype(int)
        s = s[xx, :, :]

        # Construct 3x3 montage of slices
        m2d = montage(s, fill='mean', grid_shape=(3, 3))

        # Intensity scaling
        if 'default' in irng:
            m2d = rescale_intensity(m2d, in_range='image', out_range=(0, 1))
        elif 'robust' in irng:
            pmin, pmax = np.percentile(m2d, (1, 99))
            m2d = rescale_intensity(m2d, in_range=(pmin, pmax), out_range=(0, 1))
        else:
            # Do nothing
            pass

        plt.subplot(1, 3, ax + 1)
        plt.imshow(m2d,
                   cmap=plt.get_cmap(cmap),
                   aspect='equal',
                   origin='lower')
        plt.title(orient_name[ax])

        plt.axis('off')
        plt.subplots_adjust(bottom=0.0, top=0.9, left=0.0, right=1.0)

    # Remove excess space
    plt.tight_layout()

    # Save plot to file
    plt.savefig(montage_fname, dpi=300)

    # Close plot
    plt.close()


def image_montage(img3d, montage_fname, dims=(4, 6), cmap='viridis', irng='default', axis=2):

    # Montage dimensions
    n_rows, n_cols = dims

    # Downsample to 4x6 = 24 images in specified axis
    nn = img3d.shape[axis]
    inds = np.linspace(0, nn - 1, n_rows * n_cols).astype(int)

    # Downsample and reorder axis to place downsampled axis first
    if axis == 0:
        img3d = img3d[inds, ...]
    elif axis == 1:
        img3d = img3d[:, inds, :].transpose([1, 0, 2])
    else:
        img3d = img3d[..., inds].transpose([2, 0, 1])

    # Construct 3x3 montage of slices
    m2d = montage(img3d, fill='mean', grid_shape=(n_rows, n_cols))

    # Intensity scaling
    if 'default' in irng:
        m2d = rescale_intensity(m2d, in_range='image', out_range=(0, 1))
    elif 'robust' in irng:
        pmin, pmax = np.percentile(m2d, (1, 99))
        m2d = rescale_intensity(m2d, in_range=(pmin, pmax), out_range=(0, 1))
    else:
        # Do nothing
        pass

    fig, axs = plt.subplots(1, 1, figsize=(18, 12))
    axs.imshow(
        m2d,
        cmap=plt.get_cmap(cmap),
        aspect='equal',
        origin='lower'
    )

    # Tidy up axes
    plt.axis('off')
    plt.tight_layout()

    # Save plot to file
    plt.savefig(montage_fname, dpi=300)

    # Close plot
    plt.close()
