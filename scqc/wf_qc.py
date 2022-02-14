#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.pipeline.engine as pe


def build_wf_qc():

    # QC inputs
    qc_inputs = pe.Node(
        util.IdentityInterface(
            fields=['bold', 'anat']
        ),
        name='qc_inputs'
    )

    # Calc HPF sigma in volumes
    # Return an operator string usable by fslmaths
    hpf_sigma = pe.Node(
        util.Function(
            input_names=['bold'],
            output_names=['op_string'],
            function=calc_hpf_sigma
        ),
        name='hpf_sigma'
    )

    # High pass filter to remove slow trends and baseline
    bold_thpf = pe.Node(
        fsl.ImageMaths(
            out_file='bold_thpf.nii.gz'
        ),
        name='bold_hpf'
    )

    # Temporal mean
    bold_tmean = pe.Node(
        afni.TStat(
            args='-mean',
            out_file='bold_tmean.nii.gz'
        ),
        name='bold_tmean'
    )

    # Temporal SD without linear detrending
    bold_tsd = pe.Node(
        afni.TStat(
            args='-stdevNOD',
            out_file='bold_tsd.nii.gz'
        ),
        name='bold_tsd'
    )

    bold_tsfnr = pe.Node(
        fsl.ImageMaths(
            op_string='-div',
            out_file='bold_tsfnr.nii.gz'
        ),
        name='bold_tsfnr'
    )

    # Define outputs from preproc workflow
    qc_outputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold_tmean',
                'bold_tsd',
                'bold_thpf',
                'bold_tsfnr'
            ]
        ),
        name='qc_outputs'
    )

    # QC workflow setup
    qc_wf = pe.Workflow(name='qc_wf')

    qc_wf.connect([
        (qc_inputs, bold_tmean, [('bold', 'in_file')]),
        (qc_inputs, bold_thpf, [('bold', 'in_file')]),
        (hpf_sigma, bold_thpf, [('op_string', 'op_string')]),
        (bold_thpf, bold_tsd, [('out_file', 'in_file')]),
        (bold_tmean, bold_tsfnr, [('out_file', 'in_file')]),
        (bold_tsd, bold_tsfnr, [('out_file', 'in_file2')]),

        # Return all stats images
        (bold_tmean, qc_outputs, [('out_file', 'bold_tmean')]),
        (bold_tsd, qc_outputs, [('out_file', 'bold_tsd')]),
        (bold_thpf, qc_outputs, [('out_file', 'bold_thpf')]),
        (bold_tsfnr, qc_outputs, [('out_file', 'bold_tsfnr')])
    ])

    return qc_wf


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
