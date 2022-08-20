#!/usr/bin/env python
# coding: utf-8
"""
Register unwarped BOLD EPI space to individual T2w template via the unwarped T2w mean SE-EPI
- Final step of functional preprocessing following BOLD motion and distortion corrections
"""

import nipype.interfaces.utility as util
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe


def build_template_wf():

    # Template inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold_preproc',
                'sbref_preproc',
                'seepi_ref',
                'tpl_t2_head'
            ]
        ),
        name='inputs'
    )

    # Rigid body register average unwarped SE-EPI (from TOPUP) to individual T2w template
    reg_seepi_tpl = pe.Node(
        ants.RegistrationSynQuick(
            transform_type='r',
            num_threads=4
        ),
        name='reg_seepi_tpl',
        terminal_output=None,
    )

    # Resample 4D BOLD to individual template space
    resamp_bold_tpl = pe.Node(
        ants.WarpTimeSeriesImageMultiTransform(num_threads=4),
        name='resamp_bold_tpl'
    )

    # Resample 3D SBRef to individual template space
    resamp_sbref_tpl = pe.Node(
        ants.WarpImageMultiTransform(num_threads=4),
        name='resamp_sbref_tpl'
    )

    # Define workflow outputs
    outputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'tpl_bold_preproc',
                'tpl_sbref_preproc',
                'tpl_seepi_ref',
            ]
        ),
        name='outputs'
    )

    # Template registration workflow setup
    template_wf = pe.Workflow(name='template_wf')

    template_wf.connect([

        # Register SE-EPI unwarped midspace to individual template space
        (inputs, reg_seepi_tpl, [
            ('tpl_t2_head', 'fixed_image'),
            ('seepi_ref', 'moving_image')
        ]),

        # Resample unwarped BOLD to template space
        (inputs, resamp_bold_tpl, [('bold_preproc', 'input_image')]),
        (reg_seepi_tpl, resamp_bold_tpl, [('out_matrix', 'transformation_series')]),
        (inputs, resamp_bold_tpl, [('tpl_t2_head', 'reference_image')]),

        # Resample unwarped SBref to template space
        (inputs, resamp_sbref_tpl, [('sbref_preproc', 'input_image')]),
        (reg_seepi_tpl, resamp_sbref_tpl, [('out_matrix', 'transformation_series')]),
        (inputs, resamp_sbref_tpl, [('tpl_t2_head', 'reference_image')]),

        # Output results
        (resamp_bold_tpl, outputs, [('output_image', 'tpl_bold_preproc')]),
        (resamp_sbref_tpl, outputs, [('output_image', 'tpl_sbref_preproc')]),
        (reg_seepi_tpl, outputs, [('warped_image', 'tpl_seepi_ref')]),

    ])

    return template_wf
