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

from ..interfaces import Cart2Pol


def build_resamp_epi2tpl_wf(antsthreads=2):
    """
    :param antsthreads: int
        Maximum number of threads allowed for ANTs/ITK modules
    :return:
    """

    # Workflow input node
    in_node = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold_re_preproc',
                'bold_im_preproc',
                'sbref_mag_preproc',
                'seepi_mag_preproc',
                'ses_t2w_head',
                'tpl_t2w_head',
                'topup_b0_rads',
            ]
        ),
        name='in_node'
    )

    # Estimate rigid body transform from unwarped SE-EPI to session T2w

    # FLIRT angular search parameters
    # Restrict search to +/- 6 degrees. Libera; limits for inter-session head rotations
    alpha_max = 6
    dalpha_coarse = (2 * alpha_max) // 3 + 1
    dalpha_fine = (2 * alpha_max) // 12 + 1

    # FLIRT rigid body registration preferred over antsAI
    flirt_epi2anat = pe.Node(
        fsl.FLIRT(
            dof=6,
            cost='corratio',
            interp='spline',
            uses_qform=True,
            searchr_x=[-alpha_max, alpha_max],
            searchr_y=[-alpha_max, alpha_max],
            searchr_z=[-alpha_max, alpha_max],
            coarse_search=dalpha_coarse,
            fine_search=dalpha_fine,
            out_matrix_file='tx_epi2anat.mat',
            output_type='NIFTI_GZ',
            terminal_output='none'
        ),
        name='flirt_epi2anat',
    )

    # Estimate rigid body transform from session T2w to template T2w
    flirt_anat2tpl = pe.Node(
        fsl.FLIRT(
            dof=6,
            cost='corratio',
            out_matrix_file='tx_anat2tpl.mat',
            output_type='NIFTI_GZ',
            terminal_output='none'
        ),
        name='flirt_anat2tpl',
    )

    # Concatenate EPI to anat and anat to tpl rigid transforms
    concat_tx = pe.Node(fsl.ConvertXFM(out_file='tx_epi2tpl.mat', concat_xfm=True), name='concat_tx')

    # Convert FLIRT affine matrix to ITK transform for ANTs resampling
    # Prefer ITK Lanczos sinc (reduced resampling blur) over FSL spline or sinc
    fsl2itk = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True), name='fsl2itk')

    resamp_seepi_tpl = pe.Node(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', input_image_type=0, num_threads=antsthreads),
        name='resamp_seepi_tpl'
    )

    resamp_sbref_tpl = pe.Node(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', input_image_type=0, num_threads=antsthreads),
        name='resamp_sbref_tpl'
    )

    resamp_b0_tpl = pe.Node(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', input_image_type=0, num_threads=antsthreads),
        name='resamp_b0_tpl'
    )

    resamp_bold_re_tpl = pe.Node(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', input_image_type=3, num_threads=antsthreads,),
        name='resamp_bold_re_tpl'
    )

    resamp_bold_im_tpl = pe.Node(
        ants.ApplyTransforms(interpolation='LanczosWindowedSinc', input_image_type=3, num_threads=antsthreads,),
        name='resamp_bold_im_tpl'
    )

    bold_cart2pol = pe.Node(Cart2Pol(), name='bold_cart2pol')

    # Workflow output node
    out_node = pe.Node(
        util.IdentityInterface(
            fields=[
                'tpl_bold_mag_preproc',
                'tpl_bold_phs_preproc',
                'tpl_sbref_mag_preproc',
                'tpl_seepi_mag_preproc',
                'tpl_b0_rads'
            ]
        ),
        name='out_node'
    )

    # Template registration workflow setup
    resamp_epi2tpl_wf = pe.Workflow(name='resamp_epi2tpl_wf')

    resamp_epi2tpl_wf.connect([

        # Register SE-EPI to session T2w
        (in_node, flirt_epi2anat, [
            ('seepi_mag_preproc', 'in_file'),
            ('ses_t2w_head', 'reference'),
        ]),

        # Register session T2w to T2w template
        (in_node, flirt_anat2tpl, [
            ('ses_t2w_head', 'in_file'),
            ('tpl_t2w_head', 'reference'),
        ]),

        # Concatenate epi > sess T2 > tpl T2 transforms
        # Note ConvertXFM interface maps in_file2 to first arg after -concat
        (flirt_anat2tpl, concat_tx, [('out_matrix_file', 'in_file2')]),
        (flirt_epi2anat, concat_tx, [('out_matrix_file', 'in_file')]),

        # Convert EPI to template transform from FSL to ITK format
        (concat_tx, fsl2itk, [('out_file', 'transform_file')]),
        (in_node, fsl2itk, [
            ('seepi_mag_preproc', 'source_file'),
            ('tpl_t2w_head', 'reference_file'),
        ]),

        # Resample unwarped SE-EPI midspace to template space
        (in_node, resamp_seepi_tpl, [('seepi_mag_preproc', 'input_image')]),
        (in_node, resamp_seepi_tpl, [('tpl_t2w_head', 'reference_image')]),
        (fsl2itk, resamp_seepi_tpl, [('itk_transform', 'transforms')]),

        # Resample unwarped SBref mag to template space
        (in_node, resamp_sbref_tpl, [('sbref_mag_preproc', 'input_image')]),
        (in_node, resamp_sbref_tpl, [('tpl_t2w_head', 'reference_image')]),
        (fsl2itk, resamp_sbref_tpl, [('itk_transform', 'transforms')]),

        # Resample TOPUP B0 estimate to template space
        (in_node, resamp_b0_tpl, [('topup_b0_rads', 'input_image')]),
        (in_node, resamp_b0_tpl, [('tpl_t2w_head', 'reference_image')]),
        (fsl2itk, resamp_b0_tpl, [('itk_transform', 'transforms')]),

        # Resample unwarped BOLD real to template space
        (in_node, resamp_bold_re_tpl, [('bold_re_preproc', 'input_image')]),
        (in_node, resamp_bold_re_tpl, [('tpl_t2w_head', 'reference_image')]),
        (fsl2itk, resamp_bold_re_tpl, [('itk_transform', 'transforms')]),

        # Resample unwarped BOLD imag to template space
        (in_node, resamp_bold_im_tpl, [('bold_im_preproc', 'input_image')]),
        (in_node, resamp_bold_im_tpl, [('tpl_t2w_head', 'reference_image')]),
        (fsl2itk, resamp_bold_im_tpl, [('itk_transform', 'transforms')]),

        # Convert BOLD re/im to mag/phs
        (resamp_bold_re_tpl, bold_cart2pol, [('output_image', 'bold_re')]),
        (resamp_bold_im_tpl, bold_cart2pol, [('output_image', 'bold_im')]),

        # Output results
        (bold_cart2pol, out_node, [('bold_mag', 'tpl_bold_mag_preproc')]),
        (bold_cart2pol, out_node, [('bold_phs_rad', 'tpl_bold_phs_preproc')]),
        (resamp_seepi_tpl, out_node, [('output_image', 'tpl_seepi_mag_preproc')]),
        (resamp_sbref_tpl, out_node, [('output_image', 'tpl_sbref_mag_preproc')]),
        (resamp_b0_tpl, out_node, [('output_image', 'tpl_b0_rads')]),

    ])

    return resamp_epi2tpl_wf
