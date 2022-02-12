#!/usr/bin/env python
# coding: utf-8
"""
ANTs SyN register FSL Harvard-Oxford subcortical atlas to individual T1
"""

import numpy as np
import nipype.interfaces.utility as util
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe

def build_atlas_wf():

    # Atlas inputs
    atlas_inputs = pe.Node(
        util.IdentityInterface(
            fields=['ind_epi', 'ind_t1', 'atlas_t1', 'atlas_labels']
        ),
        name='atlas_inputs'
    )

    # Fast SyN register atlas T1 to individual T1
    t1_atlas_to_ind = pe.Node(
        ants.RegistrationSynQuick(
            transform_type='s',
            num_threads=4
        ),
        name='t1_atlas_to_ind',
        terminal_output=None,
    )

    # Rigid body register individual T1 to unwarped EPI
    t1_ind_to_epi = pe.Node(
        ants.RegistrationSynQuick(
            transform_type='r',
            num_threads=4
        ),
        name='t1_ind_to_epi',
        terminal_output=None,
    )

    # Concatenate atlas to T1 warp and T1 to EPI affine
    atlas_to_epi_tx = pe.Node(
        ants.ComposeMultiTransform(),
        name='atlas_to_epi_tx'
    )

    # Transform atlas labels to EPI space
    labels_atlas_to_epi = pe.Node(
        ants.ApplyTransforms(),
        name='labels_atlas_to_epi'
    )

    # Define workflow outputs
    atlas_outputs = pe.Node(
        util.IdentityInterface(
            fields=[
                't1_atlas_ind'
                'labels_atlas_ind',
                't1_atlas_epi',
                'labels_atlas_epi',
                't1_ind_epi'
            ]
        ),
        name='atlas_outputs'
    )

    # Atlas workflow setup
    atlas_wf = pe.Workflow(name='atlas_wf')

    atlas_wf.connect([
        (atlas_inputs, t1_atlas_to_ind, [
            ('ind_t1', 'fixed_image'),
            ('atlas_t1', 'moving_image')
        ]),
        (atlas_inputs, t1_ind_to_epi, [
            ('ind_epi', 'fixed_image'),
            ('ind_t1', 'moving_image')
        ]),
        (atlas_inputs, labels_atlas_to_epi, [('atlas_labels', 'in_file')]),
        (atlas_to_epi_tx, labels_atlas_to_epi, [('composite_tx', 'composite_tx')]),
        (labels_atlas_to_epi, atlas_outputs, [('out_file', 'labels_atlas_epi')]),
        (t1_ind_to_epi, atlas_outputs, [('warped_image', 't1_ind_epi')])
    ])

    return atlas_wf