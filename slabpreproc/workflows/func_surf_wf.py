#!/usr/bin/env python
"""
Functional MRI resampling to surface spaces
- output vertex mesh, vertex BOLD timeseries and atlas labels in GIFTI format

AUTHOR : Mike Tyszka
PLACE  : Caltech Brain Imaging Center
DATES  : 2023-03-24 JMT From scratch
LICENSE : MIT Copyright 2023
"""

import numpy as np
import nipype.interfaces.utility as util
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

from ..interfaces import TOPUPEncFile
from ..interfaces import SEEPIRef


def build_func_surf_wf():
    """
    :param nthreads: int
        Maximum number of threads allowed
    :return:
    """

    # Preproc inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=(
                'bold_mag',
                'bold_mag_meta',
                'sbref',
                'sbref_meta',
                'seepis',
                'seepis_meta')
        ),
        name='inputs'
    )

    # Identify SE-EPI fieldmap with same PE direction as BOLD SBRef
    get_seepi_ref = pe.Node(SEEPIRef(), name='seepi_ref')


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
    # JMT Consider deriving from high resolution MEMPRAGE B0 map

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
                'bold_mag_preproc',
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

     ])

    return func_preproc_wf
