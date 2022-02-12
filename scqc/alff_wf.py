#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.io as io
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe

def build_alff_wf():

    alff_wf = pe.Workflow(name='alff_wf')

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

    return alff_wf