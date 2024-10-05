#!/usr/bin/env python
"""
Complex-valued functional MRI preprocessing workflow
"""

import numpy as np
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe

from ..interfaces import TOPUPEncFile
from ..interfaces import SEEPIRef
from ..interfaces import Pol2Cart

# For later single warp registration of BOLD volumes to individual structural space
# from niworkflows.interfaces.itk import MCFLIRT2ITK


def build_func_preproc_wf(antsthreads=2):
    """
    :param antsthreads: int
        Maximum number of threads allowed
    :return:
    """

    # Workflow input node
    in_node = pe.Node(
        util.IdentityInterface(
            fields=(
                'bold_mag', 'bold_phs', 'bold_meta',
                'sbref_mag', 'sbref_phs', 'sbref_meta',
                'seepi_mag', 'seepi_phs', 'seepi_meta')
        ),
        name='in_node'
    )

    # Convert BOLD phase image from Siemens units [-4096, 4096] to radians [-pi, pi]
    # 4096/pi = 1303.7972938088
    siemens2rads = pe.Node(
        fsl.BinaryMaths(
            operation='div',
            operand_value=1303.7972938088),
        name='siemens2rads'
    )

    # Convert 4D BOLD mag/phs to re/im (polar to cartesian)
    pol2cart = pe.Node(Pol2Cart(), name='pol2cart', terminal_output=None)

    # Identify SE-EPI fieldmap with same PE direction as BOLD SBRef
    get_seepi_ref = pe.Node(SEEPIRef(), name='get_seepi_ref')

    # Rigid-body pre-align SBRef to SEEPIRef prior to motion correction of BOLD
    # timeseries to SBRef.
    reg_sbref2seepi = pe.Node(
        ants.RegistrationSynQuick(transform_type='r', num_threads=antsthreads),
        name='reg_sbref2seepi',
        terminal_output=None
    )

    # Calculate head motion correction from warped BOLD mag image series
    # See FMRIB tech report for relative cost function performance
    # https://www.fmrib.ox.ac.uk/datasets/techrep/tr02mj1/tr02mj1/node27.html
    # Save rigid transform matrices for motion correction of re, im components
    hmc_est = pe.Node(fsl.MCFLIRT(cost='normcorr', dof=6, save_mats=True, save_plots=True), name='hmc_est')

    # Create an encoding file for SE-EPI fmap correction with TOPUP
    # Use the corrected output from this node as a T2w intermediate reference
    # for registering the unwarped BOLD EPI slab space to the individual T2w template
    seepi_enc_file = pe.Node(TOPUPEncFile(), name='seepi_enc_file')

    # Create an encoding file for SBRef and BOLD correction with TOPUP
    sbref_enc_file = pe.Node(TOPUPEncFile(), name='sbref_enc_file')

    # Concatenate SE-EPI fieldmap mag images into a single 4D image
    # Required for FSL TOPUP implementation
    concat = pe.Node(fsl.Merge(dimension='t', output_type='NIFTI_GZ'), name='concat')

    # FSL TOPUP correction estimation
    # This node also returns the corrected SE-EPI images used later for
    # registration of SE-EPI to SBRef space
    # Defaults to b02b0.cnf TOPUP config file
    topup_est = pe.Node(fsl.TOPUP(output_type='NIFTI_GZ'), name='topup_est')

    # Average unwarped AP and PA mag SE-EPIs
    # Use this to register SE-EPI to SBRef space
    seepi_mag_avg = pe.Node(
        fsl.maths.MeanImage(dimension='T', output_type='NIFTI_GZ'),
        name='seepi_mag_avg'
    )

    # Split 4D images into sequence of 3D volumes
    prehmc_4dto3d_re = pe.Node(fsl.Split(dimension='t'), name='prehmc_4dto3d_re')
    prehmc_4dto3d_im = pe.Node(fsl.Split(dimension='t'), name='prehmc_4dto3d_im')

    # Apply per-volume HMC to real and imag BOLD images
    apply_hmc_re = pe.MapNode(
        fsl.ApplyXFM(apply_xfm=True, interp='trilinear'),
        iterfield=['in_file', 'in_matrix_file'], name='apply_hmc_re',
    )
    apply_hmc_im = pe.MapNode(
        fsl.ApplyXFM(apply_xfm=True, interp='trilinear'),
        iterfield=['in_file', 'in_matrix_file'], name='apply_hmc_im',
    )

    # Merge 3D real and imag volumes back to 4D
    posthmc_3dto4d_re = pe.Node(fsl.Merge(dimension='t'), name='posthmc_3dto4d_re')
    posthmc_3dto4d_im = pe.Node(fsl.Merge(dimension='t'), name='posthmc_3dto4d_im')

    # Apply TOPUP unwarp to 4D BOLD real and imag images
    unwarp_bold_re = pe.Node(fsl.ApplyTOPUP(method='jac', output_type='NIFTI_GZ'), name='unwarp_bold_re')
    unwarp_bold_im = pe.Node(fsl.ApplyTOPUP(method='jac', output_type='NIFTI_GZ'), name='unwarp_bold_im')

    # Apply TOPUP correction to SBRef real and imag volumes
    unwarp_sbref = pe.Node(fsl.ApplyTOPUP(method='jac', output_type='NIFTI_GZ'), name='unwarp_sbref')

    # Derive signal dropout map from TOPUP B0 field estimate
    # JMT Consider deriving from high resolution MEMPRAGE B0 map
    hz2rads = pe.Node(fsl.ImageMaths(op_string=f'-mul {2.0 * np.pi}'), name='hz2rads')

    # Workflow output node
    out_node = pe.Node(
        util.IdentityInterface(
            fields=(
                'bold_re_preproc',
                'bold_im_preproc',
                'sbref_mag_preproc',
                'seepi_mag_preproc',
                'moco_pars',
                'topup_b0_rads'
            ),
        ),
        name='out_node'
    )

    #
    # Build the workflow
    #

    func_preproc_wf = pe.Workflow(name='func_preproc_wf')

    func_preproc_wf.connect([

        # Extract the first fmap in the list to use as a BOLD to fmap registration reference
        (in_node, get_seepi_ref, [
            ('seepi_mag', 'seepi_mag'),
            ('seepi_meta', 'seepi_meta'),
            ('sbref_meta', 'sbref_meta')
        ]),

        # Convert BOLD phase from Siemens units to radians
        (in_node, siemens2rads, [('bold_phs', 'in_file')]),

        # Convert BOLD from mag/phs to re/im
        (in_node, pol2cart, [('bold_mag', 'bold_mag')]),
        (siemens2rads, pol2cart, [('out_file', 'bold_phs_rad')]),

        # Register the SBRef to the SE-EPI with the same PE direction
        (in_node, reg_sbref2seepi, [('sbref_mag', 'moving_image')]),
        (get_seepi_ref, reg_sbref2seepi, [('seepi_mag_ref', 'fixed_image')]),

        # Estimate HMC of BOLD mag volumes relative to mean SE-EPI mag
        (in_node, hmc_est, [('bold_mag', 'in_file')]),
        (reg_sbref2seepi, hmc_est, [('warped_image', 'ref_file')]),

        # Split BOLD re/im series into 3D volumes
        (pol2cart, prehmc_4dto3d_re, [('bold_re', 'in_file')]),
        (pol2cart, prehmc_4dto3d_im, [('bold_im', 'in_file')]),

        # Apply per-volume HMC to BOLD re/im series
        (prehmc_4dto3d_re, apply_hmc_re, [('out_files', 'in_file')]),
        (prehmc_4dto3d_im, apply_hmc_im, [('out_files', 'in_file')]),
        (reg_sbref2seepi, apply_hmc_re, [('warped_image', 'reference')]),
        (reg_sbref2seepi, apply_hmc_im, [('warped_image', 'reference')]),
        (hmc_est, apply_hmc_re, [('mat_file', 'in_matrix_file')]),
        (hmc_est, apply_hmc_im, [('mat_file', 'in_matrix_file')]),

        # Merge BOLD re/im series back into 4D
        (apply_hmc_re, posthmc_3dto4d_re, [('out_file', 'in_files')]),
        (apply_hmc_im, posthmc_3dto4d_im, [('out_file', 'in_files')]),

        # Create TOPUP encoding files
        # Requires both the image to be unwarped and the metadata for that image
        (in_node, sbref_enc_file, [('sbref_mag', 'epi_list')]),
        (in_node, sbref_enc_file, [('sbref_meta', 'meta_list')]),
        (in_node, seepi_enc_file, [('seepi_mag', 'epi_list')]),
        (in_node, seepi_enc_file, [('seepi_meta', 'meta_list')]),

        # Estimate TOPUP corrections from SE-EPI mag images
        # FUTURE: upgrade to complex TOPUP when implemented
        (in_node, concat, [('seepi_mag', 'in_files')]),
        (concat, topup_est, [('merged_file', 'in_file')]),
        (seepi_enc_file, topup_est, [('encoding_file', 'encoding_file')]),

        # Create SE-EPI mag reference (mean of unwarped SE-EPI mag images)
        (topup_est, seepi_mag_avg, [('out_corrected', 'in_file')]),

        # Apply TOPUP correction to real and imag HMC BOLD
        (topup_est, unwarp_bold_re, [
            ('out_fieldcoef', 'in_topup_fieldcoef'),
            ('out_movpar', 'in_topup_movpar')
        ]),
        (topup_est, unwarp_bold_im, [
            ('out_fieldcoef', 'in_topup_fieldcoef'),
            ('out_movpar', 'in_topup_movpar')
        ]),
        (sbref_enc_file, unwarp_bold_re, [('encoding_file', 'encoding_file')]),
        (sbref_enc_file, unwarp_bold_im, [('encoding_file', 'encoding_file')]),
        (posthmc_3dto4d_re, unwarp_bold_re, [('merged_file', 'in_files')]),
        (posthmc_3dto4d_im, unwarp_bold_im, [('merged_file', 'in_files')]),

        # Apply TOPUP correction to SBRef
        (topup_est, unwarp_sbref, [
            ('out_fieldcoef', 'in_topup_fieldcoef'),
            ('out_movpar', 'in_topup_movpar')
        ]),
        (sbref_enc_file, unwarp_sbref, [('encoding_file', 'encoding_file')]),
        (in_node, unwarp_sbref, [('sbref_mag', 'in_files')]),

        # Rescale TOPUP B0 map from Hz to rad/s
        (topup_est, hz2rads, [('out_field', 'in_file')]),

        # Output results
        (unwarp_bold_re, out_node, [('out_corrected', 'bold_re_preproc')]),
        (unwarp_bold_im, out_node, [('out_corrected', 'bold_im_preproc')]),
        (unwarp_sbref, out_node, [('out_corrected', 'sbref_mag_preproc')]),
        (seepi_mag_avg, out_node, [('out_file', 'seepi_mag_preproc')]),
        (hz2rads, out_node, [('out_file', 'topup_b0_rads')]),
        (hmc_est, out_node, [('par_file', 'moco_pars')]),
    ])

    return func_preproc_wf
