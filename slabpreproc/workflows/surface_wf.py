#!/usr/bin/env python
# coding: utf-8
"""
Freesurfer surface-based resampling of BOLD data

AUTHOR : Mike Tyszka
PLACE  : Caltech
"""

# Borrow template to fsnative transform approach from smriprep/workflows/surfaces.py
from niworkflows.interfaces.freesurfer import (
    PatchedLTAConvert as LTAConvert,
    PatchedRobustRegister as RobustRegister,
)

import nipype.pipeline.engine as pe
from .fmriprep.resampling import init_bold_surf_wf


def build_surface_wf():

    surface_wf = pe.Workflow(name='surface_wf')

    inputnode = pe.Node(
        pe.utils.IdentityInterface(
            fields=[
                'fs_subjects_dir',
                'subject_id',
                'tpl_bold'
                'tpl_t1w_head',
                'fs_t1w_head'
            ]
        ),
        name='inputnode'
    )

    # Calculate robust transform from individual template space to fsnative space
    tpl2fsnative_xfm = pe.Node(
        RobustRegister(auto_sens=True, est_int_scale=True), name="tpl2fsnative_xfm"
    )

    # fMRIPrep adapted BOLD to surface resampling
    bold_surf_wf = init_bold_surf_wf(
        mem_gb=0.1,
        surface_spaces=['fsnative'],
    )

    outputnode = pe.Node(
        pe.utils.IdentityInterface(
            fields=['surfaces']
        ),
        name='outputnode'
    )

    surface_wf.connect([

        # Construct transform from FreeSurfer conformed image to
        (inputnode, tpl2fsnative_xfm, [('tpl_t1w', 'source_file')]),
        (inputnode, tpl2fsnative_xfm, [('fs_t1w', 'target_file')]),
        (inputnode, bold_surf_wf, [
            ('fs_subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
            ('tpl_bold', 'inputnode.source_file')
        ]),
        (tpl2fsnative_xfm, bold_surf_wf, [('out_reg_file', 't1w2fsnative_xfm')]),
        (bold_surf_wf, outputnode, [('outputnode.surfaces', 'surfaces')]),
    ])

    return surface_wf
