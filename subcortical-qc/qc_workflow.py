#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.io as io
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe

def build_qc_wf():

    # QC inputs
    qc_inputs = pe.Node(
        util.IdentityInterface(
            fields=['bold', 'anat']
        ),
        name='qc_inputs'
    )

    # Temporal SD
    bold_tmean = pe.Node(
        afni.TStat(
            args='-mean',
            out_file='bold_tmean.nii.gz'
        ),
        name='bold_tmean'
    )

    # Temporal SD (AFNI demeans and linear detrends prior to calculated tSD)
    bold_tsd = pe.Node(
        afni.TStat(
            args='-stdev',
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
            fields=['tsfnr']
        ),
        name='qc_outputs'
    )

    # QC workflow setup
    qc_wf = pe.Workflow(name='qc_wf')

    qc_wf.connect([
        (qc_inputs, bold_tmean, [('bold', 'in_file')]),
        (qc_inputs, bold_tsd, [('bold', 'in_file')]),
        (bold_tmean, bold_tsfnr, [('out_file', 'in_file')]),
        (bold_tsd, bold_tsfnr, [('out_file', 'in_file2')]),
        (bold_tsfnr, qc_outputs, [('out_file', 'tsfnr')])
    ])

    return qc_wf