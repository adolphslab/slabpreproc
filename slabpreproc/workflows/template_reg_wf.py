#!/usr/bin/env python
# coding: utf-8
"""
Register unwarped BOLD EPI space to individual T2w template via the unwarped T2w mean SE-EPI
- Final step of functional preprocessing following BOLD motion and distortion corrections
"""

import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import nipype.interfaces.c3 as c3
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

    # FLIRT rigid body registration preferred over antsAI

    flirt_seepi_tpl = pe.Node(
        fsl.FLIRT(
            dof=6,
            cost='corratio',
            interp='spline',
            searchr_x=[-45, 45],
            searchr_y=[-15, 15],
            searchr_z=[-15, 15],
            output_type='NIFTI_GZ',
            terminal_output='none'
        ),
        name='flirt_seepi_tpl',
    )

    # Convert FLIRT affine matrix to ITK transform for ANTs resampling
    # Prefer ITK Lanczos sinc over FSL spline or sinc
    fsl2itk = pe.Node(
        c3.C3dAffineTool(
            fsl2ras=True,
            itk_transform=True
        ),
        name='fsl2itk'
    )

    resamp_seepi_tpl = pe.Node(
        ants.ApplyTransforms(
            interpolation='LanczosWindowedSinc',
            input_image_type = 0
        ),
        name='resamp_seepi_tpl'
    )

    resamp_sbref_tpl = pe.Node(
        ants.ApplyTransforms(
            interpolation='LanczosWindowedSinc',
            input_image_type = 0
        ),
        name='resamp_sbref_tpl'
    )

    resamp_bold_tpl = pe.Node(
        ants.ApplyTransforms(
            interpolation='LanczosWindowedSinc',
            input_image_type=3
        ),
        name='resamp_bold_tpl'
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

        # FLIRT register unwarped SE-EPI midspace to individual template space
        (inputs, flirt_seepi_tpl, [
            ('seepi_unwarp_mean', 'in_file'),
            ('tpl_t2_head', 'reference'),
        ]),

        # Convert FLIRT transform to ITK
        (flirt_seepi_tpl, fsl2itk, [('out_matrix_file', 'transform_file')]),
        (inputs, fsl2itk, [
            ('seepi_unwarp_mean', 'source_file'),
            ('tpl_t2_head', 'reference_file'),
        ]),

        # Resample unwarped SE-EPI midspace to template space
        (inputs, resamp_seepi_tpl, [('seepi_unwarp_mean', 'input_image')]),
        (inputs, resamp_seepi_tpl, [('tpl_t2_head', 'reference_image')]),
        (fsl2itk, resamp_seepi_tpl, [('itk_transform', 'transforms')]),

        # Resample unwarped SBref to template space
        (inputs, resamp_sbref_tpl, [('sbref_preproc', 'input_image')]),
        (inputs, resamp_sbref_tpl, [('tpl_t2_head', 'reference_image')]),
        (fsl2itk, resamp_sbref_tpl, [('itk_transform', 'transforms')]),

        # Resample unwarped BOLD to template space
        (inputs, resamp_bold_tpl, [('bold_preproc', 'input_image')]),
        (inputs, resamp_bold_tpl, [('tpl_t2_head', 'reference_image')]),
        (fsl2itk, resamp_bold_tpl, [('itk_transform', 'transforms')]),

        # Output results
        (resamp_seepi_tpl, outputs, [('output_image', 'tpl_seepi_unwarp_mean')]),
        (resamp_bold_tpl, outputs, [('output_image', 'tpl_bold_preproc')]),
        (resamp_sbref_tpl, outputs, [('output_image', 'tpl_sbref_preproc')]),


    ])

    return template_reg_wf
