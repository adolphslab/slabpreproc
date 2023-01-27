#!/usr/bin/env python
"""
Slab fMRI quality control workflow
Expects unwarped, motion corrected BOLD data in the individual's anatomic space
Deterministic labels should be provided in the same individual anatomic space.

AUTHOR : Mike Tyszka
PLACE  : Caltech
"""

import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.interfaces.utility as util
import nipype.algorithms.confounds as confounds
import nipype.pipeline.engine as pe

from ..interfaces.motion import Motion
from ..interfaces.dropout import Dropout


def build_qc_wf(nthreads=2):
    """
    :param nthreads: int
        Maximum number of threads allowed
    :return:
    """

    # QC inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'tpl_bold_mag_preproc',
                'tpl_bold_sbref',
                'tpl_mean_seepi',
                'tpl_bmask',
                'tpl_b0_rads',
                'tpl_dseg',
                'moco_pars',
                'bold_mag_meta'
            ]
        ),
        name='inputs'
    )

    # Voxel-wise tSFNR map
    bold_tsfnr = pe.Node(
        confounds.TSNR(
            regress_poly=2,  # Quadratic detrending
        ),
        name='bold_tsfnr'
    )

    # FD and LPF FD from FSL motion parameters
    calc_fd = pe.Node(
        confounds.FramewiseDisplacement(
            parameter_source='FSL',
            save_plot=False,
        ),
        name='calc_fd'
    )

    build_motion_table = pe.Node(
        Motion(),
        name='build_motion_table'
    )

    # Inputs: in_file, mask
    # Outputs : out_file
    bold_tsfnr_roistats = pe.Node(
        afni.ROIStats(
            stat=['mean', 'sigma', 'voxels'],
            nomeanout=True,
            format1DR=True,
            nobriklab=True
        ),
        name='bold_tsfnr_roistats'
    )

    # Estimate EPI dropout from template aligned SBRef and mean SE-EPI image
    est_dropout = pe.Node(
        Dropout(),
        name='est_dropout'
    )

    # Define outputs from QC workflow
    outputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'tpl_bold_tmean',
                'tpl_bold_tsd',
                'tpl_bold_tsfnr',
                'tpl_bold_tsfnr_roistats',
                'tpl_dropout',
                'motion_csv'
            ]
        ),
        name='outputs'
    )

    # QC workflow setup
    qc_wf = pe.Workflow(name='qc_wf')

    qc_wf.connect([
        (inputs, bold_tsfnr, [('tpl_bold_mag_preproc', 'in_file')]),

        # Pass tSFNR and labels to ROI stats
        (bold_tsfnr, bold_tsfnr_roistats, [('tsnr_file', 'in_file')]),
        (inputs, bold_tsfnr_roistats, [('tpl_dseg', 'mask')]),

        # Estimated region dropout from SBRef and mean SE-EPI
        (inputs, est_dropout, [
            ('tpl_bold_sbref', 'sbref'),
            ('tpl_mean_seepi', 'mseepi'),
            ('tpl_bmask', 'bmask')
        ]),

        # Calculate FD and LPF FD from FSL motion parameters
        (inputs, calc_fd, [('moco_pars', 'in_file')]),
        (inputs, build_motion_table, [('moco_pars', 'moco_pars')]),
        (calc_fd, build_motion_table, [('out_file', 'fd_pars')]),
        (inputs, build_motion_table, [('bold_mag_meta', 'bold_mag_meta')]),

        # Return all stats images
        (bold_tsfnr, outputs, [('mean_file', 'tpl_bold_tmean')]),
        (bold_tsfnr, outputs, [('stddev_file', 'tpl_bold_tsd')]),
        (bold_tsfnr, outputs, [('tsnr_file', 'tpl_bold_tsfnr')]),
        (bold_tsfnr_roistats, outputs, [('out_file', 'tpl_bold_tsfnr_roistats')]),
        (est_dropout, outputs, [('dropout', 'tpl_dropout')]),
        (build_motion_table, outputs, [('motion_csv', 'motion_csv')])
    ])

    return qc_wf


def estimate_noise_sigma():

    sigma = 1.0

    return sigma