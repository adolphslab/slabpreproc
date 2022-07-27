#!/usr/bin/env python
# coding: utf-8
"""
ANTs SyN register FSL Harvard-Oxford subcortical atlas to individual T1
"""

import nipype.interfaces.utility as util
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe


def build_wf_template():

    # Atlas inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'epi_ind',
                't1_ind',
                't1_atlas',
                'labels_atlas'
            ]
        ),
        name='inputs'
    )

    # Fast SyN register atlas T1 to individual T1
    reg_t1_atlas2ind = pe.Node(
        ants.RegistrationSynQuick(
            transform_type='s',
            num_threads=4
        ),
        name='reg_t1_atlas2ind',
        terminal_output=None,
        needed_outputs=[
            'forward_warp_field',
            'inverse_warp_field',
            'out_matrix',
            'warped_image'
        ]
    )

    # Rigid body register individual T1 to unwarped EPI
    reg_t1_ind2epi = pe.Node(
        ants.RegistrationSynQuick(
            transform_type='r',
            num_threads=4
        ),
        name='reg_t1_ind2epi',
        terminal_output=None,
    )

    # Create multitransform list of ANTs warp and affine files
    # [atlas-t1 warp, atlas-t1 affine, t1-epi affine]
    merge_tx_files = pe.Node(
        util.Merge(3),
        name='merge_tx_files'
    )

    # Resample atlas T1w to EPI space
    resamp_t1_atlas2epi = pe.Node(
        ants.WarpImageMultiTransform(
            num_threads=4
        ),
        name='resamp_t1_atlas2epi'
    )

    # Resample atlas labels to EPI space
    resamp_labels_atlas2epi = pe.Node(
        ants.WarpImageMultiTransform(
            use_nearest=True,
            num_threads=4
        ),
        name='resamp_labels_atlas2epi'
    )

    # Define workflow outputs
    outputs = pe.Node(
        util.IdentityInterface(
            fields=[
                't1_atlas2ind',
                't1_ind2epi',
                't1_atlas2epi',
                'labels_atlas2ind',
                'labels_atlas2epi',
            ]
        ),
        name='outputs'
    )

    # Atlas workflow setup
    wf_atlas = pe.Workflow(name='wf_atlas')

    wf_atlas.connect([

        # Feed inputs to registration nodes
        (inputs, reg_t1_atlas2ind, [
            ('t1_ind', 'fixed_image'),
            ('t1_atlas', 'moving_image')
        ]),
        (inputs, reg_t1_ind2epi, [
            ('epi_ind', 'fixed_image'),
            ('t1_ind', 'moving_image')
        ]),

        # Composite atlas-indT1 and indT1-indEPI transforms
        (reg_t1_atlas2ind, merge_tx_files, [
            ('forward_warp_field', 'in2'),
            ('out_matrix', 'in3')
        ]),
        (reg_t1_ind2epi, merge_tx_files, [('out_matrix', 'in1')]),

        # Warp atlas T1 into indEPI space
        (inputs, resamp_t1_atlas2epi, [('t1_atlas', 'input_image')]),
        (inputs, resamp_t1_atlas2epi, [('epi_ind', 'reference_image')]),
        (merge_tx_files, resamp_t1_atlas2epi, [('out', 'transformation_series')]),

        # Warp atlas labels into indEPI space
        (inputs, resamp_labels_atlas2epi, [('labels_atlas', 'input_image')]),
        (inputs, resamp_labels_atlas2epi, [('epi_ind', 'reference_image')]),
        (merge_tx_files, resamp_labels_atlas2epi, [('out', 'transformation_series')]),

        # Output results
        (reg_t1_atlas2ind, outputs, [('warped_image', 't1_atlas2ind')]),
        (reg_t1_ind2epi, outputs, [('warped_image', 't1_ind2epi')]),
        (resamp_t1_atlas2epi, outputs, [('output_image', 't1_atlas2epi')]),
        (resamp_labels_atlas2epi, outputs, [('output_image', 'labels_atlas2epi')]),
    ])

    return wf_atlas
