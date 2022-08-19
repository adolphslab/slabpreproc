#!/usr/bin/env python
"""
Slab fMRI quality control workflow

AUTHOR : Mike Tyszka
PLACE  : Caltech
"""

import nipype.interfaces.afni as afni
import nipype.interfaces.utility as util
import nipype.algorithms.confounds as confounds
import nipype.pipeline.engine as pe


def build_qc_wf():

    # QC inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=['bold', 'anat', 'labels']
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

    # ROI tSFNR stats (median, sd, min, max)
    # Inputs: in_file, mask
    # Outputs : out_file
    bold_tsfnr_roistats = pe.Node(
        afni.ROIStats(
            args='-median -sigma -minmax'
        ),
        name='bold_tsfnr_roistats'
    )

    # Define outputs from preproc workflow
    outputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold_tmean',
                'bold_tsd',
                'bold_detrended',
                'bold_tsfnr',
                'bold_tsfnr_roistats'
            ]
        ),
        name='outputs'
    )

    # QC workflow setup
    qc_wf = pe.Workflow(name='qc_wf')

    qc_wf.connect([
        (inputs, bold_tsfnr, [('bold', 'in_file')]),

        # Pass tSFNR and labels to ROI stats
        (bold_tsfnr, bold_tsfnr_roistats, [('tsnr_file', 'in_file')]),
        (inputs, bold_tsfnr_roistats, [('labels', 'mask')]),

        # Return all stats images
        (bold_tsfnr, outputs, [('mean_file', 'bold_tmean')]),
        (bold_tsfnr, outputs, [('stddev_file', 'bold_tsd')]),
        (bold_tsfnr, outputs, [('detrended_file', 'bold_detrended')]),
        (bold_tsfnr, outputs, [('tsnr_file', 'bold_tsfnr')]),
        (bold_tsfnr_roistats, outputs, [('out_file', 'bold_tsfnr_roistats')])
    ])

    return qc_wf


def estimate_noise_sigma():

    sigma = 1.0

    return sigma