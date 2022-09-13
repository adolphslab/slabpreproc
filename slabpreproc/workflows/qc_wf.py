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


def build_qc_wf():

    # QC inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'tpl_bold_preproc',
                'tpl_dseg',
                'tpl_b0_rads',
                'moco_pars',
                'bold_meta'
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
            save_plot=True,
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

    # Estimate EPI dropout from template aligned TOPUP B0 map
    est_sigloss = pe.Node(
        fsl.SigLoss(
            echo_time=0.03,
            slice_direction='z'
        ),
        name='est_sigloss'
    )

    # Define outputs from QC workflow
    outputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'tpl_bold_tmean',
                'tpl_bold_tsd',
                'tpl_bold_tsfnr',
                'tpl_bold_tsfnr_roistats',
                'tpl_sigloss',
                'motion_csv'
            ]
        ),
        name='outputs'
    )

    # QC workflow setup
    qc_wf = pe.Workflow(name='qc_wf')

    qc_wf.connect([
        (inputs, bold_tsfnr, [('tpl_bold_preproc', 'in_file')]),

        # Pass tSFNR and labels to ROI stats
        (bold_tsfnr, bold_tsfnr_roistats, [('tsnr_file', 'in_file')]),
        (inputs, bold_tsfnr_roistats, [('tpl_dseg', 'mask')]),

        # Sigloss from TOPUP B0 field in rad/s
        (inputs, est_sigloss, [('tpl_b0_rads', 'in_file')]),

        # Calculate FD and LPF FD from FSL motion parameters
        (inputs, calc_fd, [('moco_pars', 'in_file')]),
        (inputs, build_motion_table, [('moco_pars', 'moco_pars')]),
        (calc_fd, build_motion_table, [('out_file', 'fd_pars')]),
        (inputs, build_motion_table, [('bold_meta', 'bold_meta')]),


        # Return all stats images
        (bold_tsfnr, outputs, [('mean_file', 'tpl_bold_tmean')]),
        (bold_tsfnr, outputs, [('stddev_file', 'tpl_bold_tsd')]),
        (bold_tsfnr, outputs, [('tsnr_file', 'tpl_bold_tsfnr')]),
        (bold_tsfnr_roistats, outputs, [('out_file', 'tpl_bold_tsfnr_roistats')]),
        (est_sigloss, outputs, [('out_file', 'tpl_sigloss')]),
        (build_motion_table, outputs, [('motion_csv', 'motion_csv')])
    ])

    return qc_wf


def estimate_noise_sigma():

    sigma = 1.0

    return sigma