#!/usr/bin/env python
"""
Functional MRI preprocessing workflow
"""

import numpy as np
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe

from ..interfaces import TOPUPEncFile
from ..interfaces import SEEPIRef

# For later single warp registration of BOLD volumes to individual structural space
# from niworkflows.interfaces.itk import MCFLIRT2ITK


def build_func_preproc_wf(n_threads=2):

    # Preproc inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=(
                'bold', 'bold_meta',
                'sbref', 'sbref_meta',
                'seepis', 'seepis_meta')
        ),
        name='inputs'
    )

    # Identify SE-EPI fieldmap with same PE direction as BOLD SBRef
    get_seepi_ref = pe.Node(SEEPIRef(), name='seepi_ref')

    # Rigid-body pre-align SBRef to SEEPIRef prior to motion correction of BOLD
    # timeseries to SBRef.
    sbref2seepi = pe.Node(
        ants.RegistrationSynQuick(
            transform_type='r',
            num_threads=n_threads
        ),
        name='sbref2seepi',
        terminal_output=None
    )

    # Motion correction of warped BOLD series using the warped, fmap-aligned SBRef as reference
    # See FMRIB tech report for relative cost function performance
    # https://www.fmrib.ox.ac.uk/datasets/techrep/tr02mj1/tr02mj1/node27.html

    mcflirt = pe.Node(
        fsl.MCFLIRT(
            cost='normcorr',
            dof=6,
            save_mats=True,  # Save rigid transform matrices for single-shot, per-volume resampling
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

    seepi_unwarp_mean = pe.Node(
        fsl.maths.MeanImage(
            dimension='T',
            output_type='NIFTI_GZ'
        ),
        name='seepi_unwarp_mean'
    )

    # Apply TOPUP correction to BOLD and SBRef
    unwarp_bold = pe.Node(
        fsl.ApplyTOPUP(
            method='jac',
            output_type='NIFTI_GZ'
        ),
        name='unwarp_bold'
    )
    unwarp_sbref = pe.Node(
        fsl.ApplyTOPUP(
            method='jac',
            output_type='NIFTI_GZ'
        ),
        name='unwarp_sbref'
    )


    # Derive signal dropout map from TOPUP B0 field estimate
    # JMT Maybe replace this with template MEMPRAGE B0 map

    hz2rads = pe.Node(
        fsl.ImageMaths(
            op_string=f'-mul {2.0 * np.pi}'
        ),
        name='hz2rads'
    )

    # Define outputs for the fMRI preproc workflow
    outputs = pe.Node(
        util.IdentityInterface(
            fields=(
                'bold_preproc',
                'sbref_preproc',
                'seepi_unwarp_mean',
                'moco_pars',
                'topup_b0_rads'
            ),
        ),
        name='outputs'
    )

    #
    # Build the workflow
    #

    func_preproc_wf = pe.Workflow(name='func_preproc_wf')

    func_preproc_wf.connect([

        # Extract the first fmap in the list to use as a BOLD to fmap registration reference
        (inputs, get_seepi_ref, [
            ('seepis', 'seepis'),
            ('seepis_meta', 'seepis_meta'),
            ('sbref_meta', 'sbref_meta')
        ]),

        # Register the SBRef to the SE-EPI with the same PE direction
        (inputs, sbref2seepi, [('sbref', 'moving_image')]),
        (get_seepi_ref, sbref2seepi, [('seepi_ref', 'fixed_image')]),

        # Motion correct BOLD series to the fmap-aligned SBRef image
        (inputs, mcflirt, [('bold', 'in_file')]),
        (sbref2seepi, mcflirt, [('warped_image', 'ref_file')]),

        # Create TOPUP encoding files
        # Requires both the image to be unwarped and the metadata for that image
        (inputs, sbref_enc_file, [('sbref', 'epi_list')]),
        (inputs, sbref_enc_file, [('sbref_meta', 'meta_list')]),
        (inputs, seepi_enc_file, [('seepis', 'epi_list')]),
        (inputs, seepi_enc_file, [('seepis_meta', 'meta_list')]),

        # Estimate TOPUP corrections from fmap images
        (inputs, concat, [('seepis', 'in_files')]),
        (concat, topup_est, [('merged_file', 'in_file')]),
        (seepi_enc_file, topup_est, [('encoding_file', 'encoding_file')]),

        # Create SE-EPI reference (temporal mean of unwarped seepis)
        (topup_est, seepi_unwarp_mean, [('out_corrected', 'in_file')]),

        # Apply TOPUP correction to motion corrected BOLD
        (topup_est, unwarp_bold, [
            ('out_fieldcoef', 'in_topup_fieldcoef'),
            ('out_movpar', 'in_topup_movpar')
        ]),
        (sbref_enc_file, unwarp_bold, [('encoding_file', 'encoding_file')]),
        (mcflirt, unwarp_bold, [('out_file', 'in_files')]),

        # Apply TOPUP correction to SBRef
        (topup_est, unwarp_sbref, [
            ('out_fieldcoef', 'in_topup_fieldcoef'),
            ('out_movpar', 'in_topup_movpar')
        ]),
        (sbref_enc_file, unwarp_sbref, [('encoding_file', 'encoding_file')]),
        (inputs, unwarp_sbref, [('sbref', 'in_files')]),

        # Rescale TOPUP B0 map from Hz to rad/s
        (topup_est, hz2rads, [('out_field', 'in_file')]),

        # Output results
        (unwarp_bold, outputs, [('out_corrected', 'bold_preproc')]),
        (unwarp_sbref, outputs, [('out_corrected', 'sbref_preproc')]),
        (seepi_unwarp_mean, outputs, [('out_file', 'seepi_unwarp_mean')]),
        (hz2rads, outputs, [('out_file', 'topup_b0_rads')]),
        (mcflirt, outputs, [('par_file', 'moco_pars')]),
    ])

    return func_preproc_wf
