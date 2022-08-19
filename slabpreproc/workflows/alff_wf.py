#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.afni as afni
import nipype.pipeline.engine as pe


def build_alff_wf():

    wf_alff = pe.Workflow(name='wf_alff')

    # Temporal bandpass filter [0.01, 0.1]
    bandpass = pe.Node(
        afni.Bandpass(
            highpass=0.01,
            lowpass=0.1
        ),
        name='bandpass'
    )

    # Compute voxelwise ALFF
    alff = pe.Node(
        afni.TStat(
            args='-stdev'
        ),
        name='alff'
    )

    wf_alff.connect([
        (bandpass, alff, [('out_file', 'in_file')])
    ])

    return wf_alff
