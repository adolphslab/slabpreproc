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


def build_qc_wf():
    """
    Build quality control workflow
    """

    # Workflow input node
    in_node = pe.Node(
        util.IdentityInterface(
            fields=[
                'tpl_bold_mag_preproc',
                'tpl_sbref_mag_preproc',
                'tpl_seepi_mag_preproc',
                'tpl_bmask',
                'tpl_b0_rads',
                'tpl_dseg',
                'moco_pars',
                'bold_meta'
            ]
        ),
        name='in_node'
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

    # Workflow output node
    out_node = pe.Node(
        util.IdentityInterface(
            fields=[
                'tpl_bold_mag_tmean',
                'tpl_bold_mag_tsd',
                'tpl_bold_mag_tsfnr',
                'tpl_bold_mag_tsfnr_roistats',
                'tpl_dropout',
                'motion_csv'
            ]
        ),
        name='out_node'
    )

    # QC workflow setup
    qc_wf = pe.Workflow(name='qc_wf')

    qc_wf.connect([
        (in_node, bold_tsfnr, [('tpl_bold_mag_preproc', 'in_file')]),

        # Pass tSFNR and labels to ROI stats
        (bold_tsfnr, bold_tsfnr_roistats, [('tsnr_file', 'in_file')]),
        (in_node, bold_tsfnr_roistats, [('tpl_dseg', 'mask')]),

        # Estimated region dropout from SBRef and mean SE-EPI
        (in_node, est_dropout, [
            ('tpl_sbref_mag_preproc', 'sbref'),
            ('tpl_seepi_mag_preproc', 'mseepi'),
            ('tpl_bmask', 'bmask')
        ]),

        # Calculate FD and LPF FD from FSL motion parameters
        (in_node, calc_fd, [('moco_pars', 'in_file')]),
        (in_node, build_motion_table, [('moco_pars', 'moco_pars')]),
        (calc_fd, build_motion_table, [('out_file', 'fd_pars')]),
        (in_node, build_motion_table, [('bold_meta', 'bold_meta')]),

        # Return all stats images
        (bold_tsfnr, out_node, [('mean_file', 'tpl_bold_mag_tmean')]),
        (bold_tsfnr, out_node, [('stddev_file', 'tpl_bold_mag_tsd')]),
        (bold_tsfnr, out_node, [('tsnr_file', 'tpl_bold_mag_tsfnr')]),
        (bold_tsfnr_roistats, out_node, [('out_file', 'tpl_bold_mag_tsfnr_roistats')]),
        (est_dropout, out_node, [('dropout', 'tpl_dropout')]),
        (build_motion_table, out_node, [('motion_csv', 'motion_csv')])
    ])

    return qc_wf


def estimate_noise_sigma():

    sigma = 1.0

    return sigma