#!/usr/bin/env python
# coding: utf-8

import os.path as op
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe


def build_melodic_wf():

    melodic_wf = pe.Workflow(name='melodic_wf')

    melodic_inputs = pe.Node(
        pe.utils.IdentityInterface(
            fields=['bold', 'mask']
        ),
        name='melodic_inputs'
    )

    # Smoothing node
    smooth = pe.Node(
        fsl.utils.Smooth(
            sigma=2.0
        ),
        name='smooth'
    )

    # Melodic node
    melodic = pe.Node(
        fsl.MELODIC(
            approach='symm',
            no_bet=True,
            bg_threshold=10,
            tr_sec=1.06,
            mm_thresh=0.5,
            out_stats=True,
            report=True,
            out_dir=op.realpath('output/melodic')
        ),
        name='melodic'
    )

    melodic_outputs = pe.Node(
        pe.utils.IdentityInterface(
            fields=['melodic_output']
        ),
        name='melodic_outputs'
    )

    melodic_wf.connect([
        (melodic_inputs, smooth, [('bold', 'in_file')]),
        (melodic_inputs, melodic, [('mask', 'mask')]),
        (smooth, melodic, [('smoothed_file', 'in_files')]),
        (melodic, melodic_outputs, [('out_dir', 'melodic_output')])
    ])

    return melodic_wf