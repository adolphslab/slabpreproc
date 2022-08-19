#!/usr/bin/env python
"""
Functional MRI preprocessing workflow
"""

import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

from ..interfaces import TOPUPEncFile


def build_func_preproc_wf():

    # Preproc inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=(
                'bold', 'bold_meta',
                'sbref', 'sbref_meta',
                'fmaps', 'fmaps_meta')
        ),
        name='inputs'
    )

    # Motion correction of warped BOLD series using the warped SBRef as reference
    # See FMRIB tech report for relative cost function performance
    # https://www.fmrib.ox.ac.uk/datasets/techrep/tr02mj1/tr02mj1/node27.html

    mcflirt = pe.Node(
        fsl.MCFLIRT(
            cost='normcorr',
            dof=6,
            save_plots=True
        ),
        name='mcflirt'
    )

    # Create an encoding file for SE-EPI correction with TOPUP
    # Use the corrected output from this node as a T2w intermediate reference
    # for registering the unwarped BOLD EPI slab space to the individual T2w template
    seepi_enc_file = pe.Node(
        TOPUPEncFile(),
        name='seepi_enc_file'
    )

    # Create an encoding file for SBRef and BOLD correction with TOPUP
    sbref_enc_file = pe.Node(
        TOPUPEncFile(),
        name='sbref_enc_file'
    )

    # Make the first fmap in the list the registration reference
    # Assumed to have the same PE direction as the BOLD series
    # TODO: check that the first fmap PE direction matches that of the BOLD series
    fmap_ref = pe.Node(
        util.Select(index=[0]),
        name='get_fmap_ref'
    )

    # Concatenate SE-EPI images into a single 4D image
    # Required for FSL TOPUP implementation
    concat = pe.Node(
        fsl.Merge(
            dimension='t',
            output_type='NIFTI_GZ'
        ),
        name='concat'
    )

    # FSL TOPUP correction estimation
    # This node also returns the corrected SE-EPI images used later for
    # registration of EPI to individual T2w space
    topup_est = pe.Node(
        fsl.TOPUP(
            output_type='NIFTI_GZ'
        ),
        name='topup_est'
    )

    # Apply TOPUP correction to BOLD and SBRef
    unwarp_bold = pe.Node(
        fsl.ApplyTOPUP(
            output_type='NIFTI_GZ'
        ),
        name='unwarp_bold'
    )
    unwarp_sbref = pe.Node(
        fsl.ApplyTOPUP(
            output_type='NIFTI_GZ'
        ),
        name='unwarp_sbref'
    )

    # Define outputs for the fMRI preproc workflow
    outputs = pe.Node(
        util.IdentityInterface(
            fields=(
                'bold_preproc',
                'sbref_preproc',
                'seepi_ref',
                'moco_pars'),
        ),
        name='outputs'
    )

    #
    # Build the workflow
    #

    func_preproc_wf = pe.Workflow(name='func_preproc_wf')

    func_preproc_wf.connect([

        # Motion correct BOLD series to the SBRef image
        (inputs, mcflirt, [
            ('bold', 'in_file'),
            ('sbref', 'ref_file')
        ]),

        # Create TOPUP encoding files
        # Requires both the image to be unwarped and the metadata for that image
        (inputs, sbref_enc_file, [('sbref', 'epi_list')]),
        (inputs, sbref_enc_file, [('sbref_meta', 'meta_list')]),
        (inputs, seepi_enc_file, [('fmaps', 'epi_list')]),
        (inputs, seepi_enc_file, [('fmaps_meta', 'meta_list')]),

        # Extract the first fmap in the list to use as a BOLD to fmap registration reference
        (inputs, fmap_ref, [('fmaps', 'inlist')]),

        # Estimate TOPUP corrections from fmap images
        (inputs, concat, [('fmaps', 'in_files')]),
        (concat, topup_est, [('merged_file', 'in_file')]),
        (seepi_enc_file, topup_est, [('encoding_file', 'encoding_file')]),

        # Apply TOPUP correction to motion corrected BOLD and SBRef
        (sbref_enc_file, unwarp_bold, [('encoding_file', 'encoding_file')]),
        (sbref_enc_file, unwarp_sbref, [('encoding_file', 'encoding_file')]),
        (mcflirt, unwarp_bold, [('out_file', 'in_files')]),
        (inputs, unwarp_sbref, [('sbref', 'in_files')]),

        # Output results
        (unwarp_bold, outputs, [('out_corrected', 'bold_preproc')]),
        (unwarp_sbref, outputs, [('out_corrected', 'sbref_preproc')]),
        (topup_est, outputs, [('out_corrected', 'seepi_ref')]),
        (mcflirt, outputs, [('par_file', 'moco_pars')]),
    ])

    return func_preproc_wf
