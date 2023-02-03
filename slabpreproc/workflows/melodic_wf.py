#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe


def build_melodic_wf(tr_s=1.0):

    melodic_wf = pe.Workflow(name='melodic_wf')

    inputs = pe.Node(
        pe.utils.IdentityInterface(
            fields=[
                'tpl_bold_mag_preproc',
                'tpl_bold_tmean',
                'tpl_t1_head',
                'tpl_bmask',
                'bold_mag_meta'
            ]
        ),
        name='inputs'
    )

    # Melodic calculation mask
    # Combine robust highpass threshold with template brain mask
    threshold = pe.Node(fsl.maths.Threshold(thresh=10, use_robust_range=True), name='threshold')
    binarize = pe.Node(fsl.maths.UnaryMaths(operation='bin'), name='binarize')
    mask_merge = pe.Node(fsl.maths.BinaryMaths(operation='mul'), name='mask_merge')

    # Melodic node
    melodic = pe.Node(
        fsl.MELODIC(
            approach='symm',
            no_bet=True,
            tr_sec=tr_s,
            mm_thresh=0.5,
            out_stats=True,
            report=True,
        ),
        name='melodic'
    )

    outputs = pe.Node(
        pe.utils.IdentityInterface(
            fields=['out_dir']
        ),
        name='outputs'
    )

    melodic_wf.connect([
        (inputs, threshold, [('tpl_bold_tmean', 'in_file')]),
        (threshold, binarize, [('out_file', 'in_file')]),
        (binarize, mask_merge, [('out_file', 'in_file')]),
        (inputs, mask_merge, [('tpl_bmask', 'operand_file')]),
        (inputs, melodic, [
            ('tpl_t1_head', 'bg_image'),
            ('tpl_bold_mag_preproc', 'in_files'),
        ]),
        (mask_merge, melodic, [('out_file', 'mask')]),
        (melodic, outputs, [('out_dir', 'out_dir')])
    ])

    return melodic_wf
