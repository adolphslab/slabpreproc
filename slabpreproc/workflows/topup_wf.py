#!/usr/bin/env python
"""
slab appropriate TOPUP SDC workflow for slabpreproc
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


def build_topup_wf(antsthreads=2):
    """
    Estimate unwarping fieldmap from SE-EPI images using FSL TOPUP
    
    Workflow returns TOPUP warpfield and fieldmap estimates with unwarped SEEPI reference image

    :param antsthreads: int
        Maximum number of threads allowed
    :return:
    """

    # Workflow input node
    in_node = pe.Node(
        util.IdentityInterface(
            fields=(
                'sbref_mag_list', 'sbref_meta_list',
                'seepi_mag_list', 'seepi_meta_list')
        ),
        name='in_node'
    )

    # Create an encoding file for SE-EPI fmap correction with TOPUP
    # Use the corrected output from this node as a T2w intermediate reference
    # for registering the unwarped BOLD EPI slab space to the individual T2w template
    seepi_enc_file = pe.Node(TOPUPEncFile(), name='seepi_enc_file')

    # Concatenate SE-EPI fieldmap mag images into a single 4D image
    # Required for FSL TOPUP implementation
    concat = pe.Node(fsl.Merge(dimension='t', output_type='NIFTI_GZ'), name='concat')

    # FSL TOPUP correction estimation
    # This node also returns the corrected SE-EPI images used later for
    # registration of SE-EPI to SBRef space
    # Defaults to b02b0.cnf TOPUP config file
    topup_est = pe.Node(fsl.TOPUP(output_type='NIFTI_GZ'), name='topup_est')

    # Average TOPUP unwarped AP/PA mag SE-EPIs
    seepi_uw_avg = pe.Node(
        fsl.maths.MeanImage(dimension='T', output_type='NIFTI_GZ'),
        name='seepi_uw_ref'
    )

    # Workflow output node
    out_node = pe.Node(
        util.IdentityInterface(
            fields=(
                'seepi_uw_ref',
                'topup_b0_hz'
            ),
        ),
        name='out_node'
    )

    #
    # Build the workflow
    #

    topup_wf = pe.Workflow(name='topup_wf')

    topup_wf.connect([

        # Create TOPUP encoding files
        # Requires both the image to be unwarped and the metadata for that image
        (in_node, seepi_enc_file, [('seepi_mag_list', 'epi_list')]),
        (in_node, seepi_enc_file, [('seepi_meta_list', 'meta_list')]),

        # Estimate TOPUP corrections from SE-EPI mag images
        # FUTURE: upgrade to complex TOPUP when implemented
        (in_node, concat, [('seepi_mag_list', 'in_files')]),
        (concat, topup_est, [('merged_file', 'in_file')]),
        (seepi_enc_file, topup_est, [('encoding_file', 'encoding_file')]),

        # Create SE-EPI mag reference (mean of unwarped SE-EPI mag images)
        (topup_est, seepi_uw_avg, [('out_corrected', 'in_file')]),

        # Output results
        (seepi_uw_avg, out_node, [('out_file', 'seepi_uw_ref')]),
        (topup_est, out_node, [('out_field', 'topup_b0_hz')]),
    ])

    return topup_wf
