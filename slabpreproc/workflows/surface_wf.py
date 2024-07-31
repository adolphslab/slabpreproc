#!/usr/bin/env python
# coding: utf-8
"""
Freesurfer surface-based resampling of BOLD data

AUTHOR : Mike Tyszka
PLACE  : Caltech
"""

# Borrow template to fsnative transform approach from smriprep/workflows/surfaces.py
import nipype.pipeline.engine as pe
from nipype.interfaces import freesurfer as fs
from niworkflows.interfaces.surf import GiftiSetAnatomicalStructure


def build_surface_wf():

    surface_wf = pe.Workflow(name='surface_wf')

    inputs = pe.Node(
        pe.utils.IdentityInterface(
            fields=[
                'fs_subjects_dir',
                'subject_id',
                'tpl_bold',
                'tpl_t1w_head',
                'fs_t1w_head'
            ]
        ),
        name='inputs'
    )

    # Calculate robust transform from individual template space to fsnative space
    # Use FS CLI to generate LTA-format transform
    tpl2fsnative_xfm = pe.Node(
        fs.RobustRegister(auto_sens=True, est_int_scale=True), name="tpl2fsnative_xfm"
    )

    # Derived from fmriprep 23.0.2 bold/workflows/resampling.py
    bold_fsnative = pe.MapNode(
        fs.SampleToSurface(
            interp_method="trilinear",
            out_type="gii",
            override_reg_subj=True,
            sampling_method="average",
            sampling_range=(0, 1, 0.2),
            sampling_units="frac",
        ),
        iterfield=["hemi"],
        name="bold_fsnative",
    )
    bold_fsnative.inputs.hemi = ["lh", "rh"]

    update_gifti = pe.MapNode(
        GiftiSetAnatomicalStructure(),
        iterfield=["in_file"],
        name="update_gifti",
    )

    joinnode = pe.JoinNode(
        pe.utils.IdentityInterface(fields=["surfaces", "target"]),
        joinsource="itersource",
        name="joinnode",
    )

    outputnode = pe.Node(
        pe.utils.IdentityInterface(
            fields=['fs_bold']
        ),
        name='outputnode'
    )

    # Connect everything together
    surface_wf.connect([

        # Construct transform from FreeSurfer conformed image to
        (inputs, tpl2fsnative_xfm, [('tpl_t1w_head', 'source_file')]),
        (inputs, tpl2fsnative_xfm, [('fs_t1w_head', 'target_file')]),
        (inputs, bold_fsnative, [
            ('fs_subjects_dir', 'subjects_dir'),
            ('subject_id', 'subject_id'),
            ('subject_id', 'target_subject'),
            ('tpl_bold', 'source_file')
        ]),
        (tpl2fsnative_xfm, bold_fsnative, [('out_reg_file', 'reg_file')]),

        (bold_fsnative, update_gifti, [("out_file", "in_file")]),
        (update_gifti, joinnode, [("out_file", "surfaces")]),
        (joinnode, outputnode, [('surfaces', 'surfaces')]),


    ])

    return surface_wf
