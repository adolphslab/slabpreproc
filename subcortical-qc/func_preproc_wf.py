#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.io as io
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe

def build_preproc_wf():

    # Preproc inputs
    preproc_inputs = pe.Node(
        util.IdentityInterface(
            fields=('bold', 'seepi', 'sbref', 'mocoref', 'anat')
        ),
        name='preproc_inputs'
    )

    # Concatenate SE-EPI images
    concat = pe.Node(
        fsl.Merge(
            dimension='t',
            output_type='NIFTI_GZ'
        ),
        name='concat')

    # TOPUP estimation
    topup = pe.Node(
        fsl.TOPUP(
            encoding_file=seepi_enc_fname,
            output_type='NIFTI_GZ'
        ),
        name='topup')

    # Apply TOPUP correction to SBRef
    applytopup_sbref = pe.Node(
        fsl.ApplyTOPUP(
            method='jac',
            encoding_file=sbref_enc_fname,
            output_type="NIFTI_GZ",
        ),
        name='apply_topup_sbref'
    )

    # Motion correction : Align warped AP BOLD series to warped AP SBRef
    mcflirt = pe.Node(
        fsl.MCFLIRT(
            cost='mutualinfo'
        ),
        name='mcflirt'
    )

    # Apply TOPUP correction to moco BOLD run
    applytopup_bold = pe.Node(
        fsl.ApplyTOPUP(
            method='jac',
            encoding_file=sbref_enc_fname,
            output_type="NIFTI_GZ"
        ),
        name='apply_topup_bold')

    # Skull strip unwarped sbref
    bet_sbref = pe.Node(
        fsl.BET(
            mask=True,
            robust=True,
            output_type="NIFTI_GZ"
        ),
        name='bet_sbref'
    )

    # Skull strip unwarped bold - apply mask from sbref
    bet_bold = pe.Node(
        fsl.ImageMaths(),
        name='bet_bold'
    )

    # Rigid body register anat to sbref
    anat_to_sbref = pe.Node(
        ants.RegistrationSynQuick(
            transform_type='r',
            num_threads=4
        ),
        name='anat_to_sbref',
        terminal_output=None,
    )

    # Define outputs from preproc workflow
    preproc_outputs = pe.Node(
        util.IdentityInterface(
            fields=('bold', 'anat', 'sbref'),
        ),
        name='preproc_outputs'
    )

    # Rig workflow
    preproc_wf = pe.Workflow(name='preproc_wf')

    preproc_wf.connect([
        (preproc_inputs, concat, [('seepi', 'in_files')]),
        (preproc_inputs, applytopup_sbref, [('sbref', 'in_files')]),
        (preproc_inputs, mcflirt, [
            ('bold', 'in_file'),
            ('mocoref', 'ref_file')
        ]),
        (concat, topup, [('merged_file', 'in_file')]),
        (topup, applytopup_sbref, [
            ('out_fieldcoef', 'in_topup_fieldcoef'),
            ('out_movpar', 'in_topup_movpar')
        ]),
        (mcflirt, applytopup_bold, [('out_file', 'in_files')]),
        (topup, applytopup_bold, [
            ('out_fieldcoef', 'in_topup_fieldcoef'),
            ('out_movpar', 'in_topup_movpar')
        ]),
        (applytopup_sbref, anat_to_sbref, [('out_corrected', 'fixed_image')]),
        (preproc_inputs, anat_to_sbref, [('anat', 'moving_image')]),
        (applytopup_sbref, preproc_outputs, [('out_corrected', 'sbref')]),
        (applytopup_bold, preproc_outputs, [('out_corrected', 'bold')]),
        (anat_to_sbref, preproc_outputs, [('warped_image', 'anat')]),
    ])

    return preproc_wf