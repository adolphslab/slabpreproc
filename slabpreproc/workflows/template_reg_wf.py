#!/usr/bin/env python
# coding: utf-8
"""
Register unwarped BOLD EPI space to individual T2w template via the unwarped T2w mean SE-EPI
- Final step of functional preprocessing following BOLD motion and distortion corrections
"""

import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe


def build_template_reg_wf():

    # Template inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold_preproc',
                'sbref_preproc',
                'seepi_unwarp_mean',
                'tpl_t2_head'
            ]
        ),
        name='inputs'
    )

    # Estimate rigid body transform from unwarped SE-EPI midspace to individual T2w
    # template space
    reg_seepi_tpl = pe.Node(
        fsl.FLIRT(
            dof=6,
            cost='corratio',
            interp='spline',
            searchr_x=[-45, 45],
            searchr_y=[-15, 15],
            searchr_z=[-15, 15],
            output_type='NIFTI_GZ'
        ),
        name='reg_seepi_tpl',
        terminal_output=None,
    )

    # Resample 4D BOLD to individual template space
    resamp_bold_tpl = pe.Node(
        fsl.FLIRT(
            apply_xfm=True,
            interp='spline',
            output_type='NIFTI_GZ'
        ),
        name='resamp_bold_tpl'
    )

    # Resample 3D SBRef to individual template space
    resamp_sbref_tpl = pe.Node(
        fsl.FLIRT(
            apply_xfm=True,
            interp='trilinear',
            output_type='NIFTI_GZ'
        ),
        name='resamp_sbref_tpl'
    )

    # Define workflow outputs
    outputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'tpl_bold_preproc',
                'tpl_sbref_preproc',
                'tpl_seepi_unwarp_mean',
            ]
        ),
        name='outputs'
    )

    # Template registration workflow setup
    template_reg_wf = pe.Workflow(name='template_reg_wf')

    template_reg_wf.connect([

        # Register SE-EPI unwarped midspace to individual template space
        (inputs, reg_seepi_tpl, [
            ('seepi_unwarp_mean', 'in_file'),
            ('tpl_t2_head', 'reference'),
        ]),

        # Resample unwarped BOLD to template space
        (inputs, resamp_bold_tpl, [('bold_preproc', 'in_file')]),
        (inputs, resamp_bold_tpl, [('tpl_t2_head', 'reference')]),
        (reg_seepi_tpl, resamp_bold_tpl, [('out_matrix_file', 'in_matrix_file')]),

        # Resample unwarped SBref to template space
        (inputs, resamp_sbref_tpl, [('sbref_preproc', 'in_file')]),
        (inputs, resamp_sbref_tpl, [('tpl_t2_head', 'reference')]),
        (reg_seepi_tpl, resamp_sbref_tpl, [('out_matrix_file', 'in_matrix_file')]),

        # Output results
        (reg_seepi_tpl, outputs, [('out_file', 'tpl_seepi_unwarp_mean')]),
        (resamp_bold_tpl, outputs, [('out_file', 'tpl_bold_preproc')]),
        (resamp_sbref_tpl, outputs, [('out_file', 'tpl_sbref_preproc')]),


    ])

    return template_reg_wf
