#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.algorithms.confounds as confounds
import nipype.pipeline.engine as pe


def build_wf_qc():

    # QC inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=['bold', 'anat']
        ),
        name='inputs'
    )

    # Calc HPF sigma in volumes
    # Return an operator string usable by fslmaths
    # hpf_sigma = pe.Node(
    #     util.Function(
    #         input_names=['bold'],
    #         output_names=['op_string'],
    #         function=calc_hpf_sigma
    #     ),
    #     name='hpf_sigma'
    # )
    #
    # # High pass filter (> 0.01 Hz) to remove low frequency baseline for tSD calc
    # # fslmaths HPF uses robust non-linear baseline estimation, superior to linear filtering
    # bold_thpf = pe.Node(
    #     fsl.ImageMaths(
    #         out_file='bold_thpf.nii.gz'
    #     ),
    #     name='bold_hpf'
    # )

    bold_tsfnr = pe.Node(
        confounds.TSNR(
            regress_poly=2,  # Quadratic detrending
        ),
        name='bold_tsfnr'
    )

    # Define outputs from preproc workflow
    outputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold_tmean',
                'bold_tsd',
                'bold_detrended',
                'bold_tsfnr'
            ]
        ),
        name='outputs'
    )

    # QC workflow setup
    wf_qc = pe.Workflow(name='wf_qc')

    wf_qc.connect([
        (inputs, bold_tsfnr, [('bold', 'in_file')]),

        # Return all stats images
        (bold_tsfnr, outputs, [('mean_file', 'bold_tmean')]),
        (bold_tsfnr, outputs, [('stddev_file', 'bold_tsd')]),
        (bold_tsfnr, outputs, [('detrended_file', 'bold_detrended')]),
        (bold_tsfnr, outputs, [('tsnr_file', 'bold_tsfnr')])
    ])

    return wf_qc


def calc_hpf_sigma():

    import numpy as np

    # Get BOLD TR
    # TODO: Get this from BOLD metadata
    tr_s = 0.8

    # Convert 100 s to volumes
    n_vol_100 = np.round(100.0 / tr_s).astype(int)

    print(f'calc_hpf_sigma: 100 s = {n_vol_100:d} volumes @ TR {tr_s} s')

    # Construct fslmaths operator string
    op_string = f'-bptf {n_vol_100} -1'

    return op_string
