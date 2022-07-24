#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

from ..interfaces import TOPUPEncFile


def build_wf_func_preproc():

    # Preproc inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=(
                'bold',
                'sbref',
                'fmaps',
                'anat')
        ),
        name='inputs'
    )

    # Create an encoding file for SE-EPI TOPUP fitting
    seepi_enc_file = pe.Node(
        TOPUPEncFile(),
        name='seepi_enc_file'
    )

    # Create an encoding file for SBRef and BOLD correction with TOPUP
    sbref_enc_file = pe.Node(
        TOPUPEncFile(),
        name='sbref_enc_file'
    )

    # Concatenate SE-EPI images
    concat = pe.Node(
        fsl.Merge(
            dimension='t',
            output_type='NIFTI_GZ'
        ),
        name='concat'
    )

    # TOPUP estimation
    topup = pe.Node(
        fsl.TOPUP(
            output_type='NIFTI_GZ'
        ),
        name='topup')

    # Apply TOPUP correction to SBRef
    applytopup_sbref = pe.Node(
        fsl.ApplyTOPUP(
            method='jac',
            output_type="NIFTI_GZ",
        ),
        name='apply_topup_sbref'
    )

    # Motion correction : Align warped AP BOLD series to temporal mean volume
    # See FMRIB tech report for cost function performance
    # https://www.fmrib.ox.ac.uk/datasets/techrep/tr02mj1/tr02mj1/node27.html

    mcflirt = pe.Node(
        fsl.MCFLIRT(
            cost='normcorr',
            dof=6,
            mean_vol=True,
            save_plots=True
        ),
        name='mcflirt'
    )

    # Apply TOPUP correction to moco BOLD run
    applytopup_bold = pe.Node(
        fsl.ApplyTOPUP(
            method='jac',
            output_type="NIFTI_GZ"
        ),
        name='apply_topup_bold')

    # Define outputs from preproc workflow
    outputs = pe.Node(
        util.IdentityInterface(
            fields=('bold', 'sbref', 'moco_pars'),
        ),
        name='outputs'
    )

    # Build workflow
    wf_func_preproc = pe.Workflow(name='wf_func_preproc')

    wf_func_preproc.connect([

        # Create TOPUP encoding files
        (inputs, sbref_enc_file, [('sbref', 'epis')]),
        (inputs, seepi_enc_file, [('fmaps', 'epis')]),

        # Motion correct BOLD series
        (inputs, mcflirt, [('bold', 'in_file')]),

        # Fit TOPUP model to SE-EPIs
        (inputs, concat, [('fmaps', 'in_files')]),
        (concat, topup, [('merged_file', 'in_file')]),
        (seepi_enc_file, topup, [('encoding_file', 'encoding_file')]),

        # TOPUP correct SBRef
        (inputs, applytopup_sbref, [('sbref', 'in_files')]),
        (sbref_enc_file, applytopup_sbref, [('encoding_file', 'encoding_file')]),
        (topup, applytopup_sbref, [
            ('out_fieldcoef', 'in_topup_fieldcoef'),
            ('out_movpar', 'in_topup_movpar')
        ]),

        # TOPUP correct BOLD series
        (mcflirt, applytopup_bold, [('out_file', 'in_files')]),
        (sbref_enc_file, applytopup_bold, [('encoding_file', 'encoding_file')]),
        (topup, applytopup_bold, [
            ('out_fieldcoef', 'in_topup_fieldcoef'),
            ('out_movpar', 'in_topup_movpar')
        ]),

        # Output results
        (mcflirt, outputs, [('par_file', 'moco_pars')]),
        (applytopup_sbref, outputs, [('out_corrected', 'sbref')]),
        (applytopup_bold, outputs, [('out_corrected', 'bold')]),
    ])

    return wf_func_preproc
