#!/usr/bin/env python
"""
Complex-valued functional MRI preprocessing workflow
"""

import nipype.interfaces.ants as ants
import nipype.interfaces.c3 as c3
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import numpy as np
# fMRIprep (24.2.0) interfaces for one-shot resampling
from fmriprep.interfaces.resampling import (ResampleSeries, DistortionParameters)
# niworkflows MCFLIRT to ITK conversion
from niworkflows.interfaces.itk import MCFLIRT2ITK
# sdcflows TOPUP workflow
from sdcflows.workflows.fit.pepolar import init_topup_wf

# Slabpreproc interfaces
from ..interfaces import (TOPUPEncFile, LapUnwrap, TempUnwrap, SEEPIRef)


def build_func_preproc_wf(antsthreads=2):
    """
    :param antsthreads: int
        Maximum number of threads allowed
    :return:
    """

    # Workflow input node
    inputnode = pe.Node(
        util.IdentityInterface(
            fields=(
                'bold_mag',
                'bold_phs',
                'bold_meta',
                'sbref_mag',
                'sbref_phs',
                'sbref_meta',
                'seepi_mag_list',
                'seepi_phs_list',
                'seepi_meta_list',
                'ses_t2w_head',
                'tpl_t2w_head',
            )
        ),
        name='inputnode'
    )

    # Convert BOLD phase image from Siemens units [-4096, 4096] to radians [-pi, pi]
    # 4096/pi = 1303.7972938088
    siemens2rads = pe.Node(
        fsl.BinaryMaths(
            operation='div',
            operand_value=1303.7972938088),
        name='siemens2rads'
    )

    # Laplacian phase unwrap spatial dimensions prior to any spatial resampling
    lap_unwrap = pe.Node(LapUnwrap(), name='lap_unwrap', terminal_output=None)

    # Temporally phase unwrap and demean prior to spatial resampling
    temp_unwrap = pe.Node(TempUnwrap(k_t=100), name='temp_unwrap', terminal_output=None)

    # Identify SE-EPI fieldmap with same PE direction as BOLD SBRef
    get_seepi_ref = pe.Node(SEEPIRef(), name='get_seepi_ref')

    # Rigid-body pre-align SBRef to SEEPIRef prior to motion correction of BOLD
    # timeseries to SBRef.
    reg_sbref2seepi = pe.Node(
        ants.RegistrationSynQuick(transform_type='r', num_threads=antsthreads),
        name='reg_sbref2seepi',
        terminal_output=None
    )

    # Create an encoding file for SE-EPI fmap correction with TOPUP
    # Use the corrected output from this node as a T2w intermediate reference
    # for registering the unwarped BOLD EPI slab space to the individual T2w template
    seepi_enc_file = pe.Node(TOPUPEncFile(), name='seepi_enc_file')

    # Create an encoding file for SBRef and BOLD correction with TOPUP
    sbref_enc_file = pe.Node(TOPUPEncFile(), name='sbref_enc_file')

    # HMC of warped BOLD series using the warped, fmap-aligned SBRef as reference
    # See FMRIB tech report for relative cost function performance
    # https://www.fmrib.ox.ac.uk/datasets/techrep/tr02mj1/tr02mj1/node27.html

    hmc_est = pe.Node(
        fsl.MCFLIRT(
            cost='normcorr',
            dof=6,
            save_mats=True,  # Save rigid transform matrices for single-shot, per-volume resampling
            save_plots=True
        ),
        name='hmc_est'
    )

    # Convert mcflirt HMC affine matrices to single ITK text file
    itk_hmc = pe.Node(
        MCFLIRT2ITK(),
        name='itk_hmc'
    )

    # Get TOPUP distortion parameters from metadata
    dist_pars = pe.Node(
        DistortionParameters(),
        name='dist_pars'
    )

    # Setup TOPUP SDC workflow
    topup_wf = init_topup_wf(
        name='topup_wf',
        omp_nthreads=1
    )

    # Average TOPUP unwarped AP/PA mag SE-EPIs
    epi_uw_ref = pe.Node(
        fsl.maths.MeanImage(dimension='T', output_type='NIFTI_GZ'),
        name='epi_uw_ref'
    )

    # Estimate rigid body transform from unwarped SE-EPI to session T2w

    # FLIRT rigid angular search parameters
    # Restrict search to +/- 6 degrees. Liberal limits for inter-session head rotations
    alpha_max = 6
    dalpha_coarse = (2 * alpha_max) // 3 + 1
    dalpha_fine = (2 * alpha_max) // 12 + 1

    # Estimate rigid transform from T2w EPI to session T2w space
    # FLIRT rigid body registration preferred over antsAI for partial brain slab data
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
            out_matrix_file='flirt_anat2tpl.mat',
            output_type='NIFTI_GZ',
            terminal_output='none'
        ),
        name='flirt_anat2tpl',
    )

    # Combine EPI-anat and anat-template rigid transforms
    # Concatenate EPI to anat and anat to tpl rigid transforms
    flirt_epi2tpl = pe.Node(fsl.ConvertXFM(
        out_file='flirt_epi2tpl.mat',
        concat_xfm=True),
        name='flirt_epi2tpl'
    )

    # Convert EPI-to-template rigid transform to ITK format
    itk_epi2tpl = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True), name='itk_epi2tpl')

    # Make list from ITK HMC transforms and EPI-to-template transform
    # Pass this list to fmriprep ResampleSeries interface (transforms arg)
    itk_hmc_epi2tpl = pe.Node(util.Merge(2), name='itk_hmc_epi2tpl', run_without_submitting=True)

    # Resample B0 fieldmap and EPI reference to template space
    resample_topup_b0 = pe.Node(
        ants.ApplyTransforms(num_threads=antsthreads),
        name='resample_topup_b0'
    )
    resample_epi_ref = pe.Node(
        ants.ApplyTransforms(num_threads=antsthreads),
        name='resample_epi_ref'
    )

    # Rescale template-space B0 map from Hz to rad/s
    hz2rads = pe.Node(fsl.ImageMaths(op_string=f'-mul {2.0 * np.pi}'), name='hz2rads')

    # One-shot resample BOLD timeseries to template space (including HMC and SDC)
    resample_bold_mag = pe.Node(
        ResampleSeries(jacobian=True, num_threads=antsthreads),
        name='resample_bold_mag'
    )
    resample_bold_phs = pe.Node(
        ResampleSeries(jacobian=False, num_threads=antsthreads),
        name='resample_bold_phs'
    )

    # Workflow output node
    outputnode = pe.Node(
        util.IdentityInterface(
            fields=(
                'tpl_bold_mag_preproc',
                'tpl_bold_phs_preproc',
                'tpl_sbref_mag_preproc',
                'tpl_epi_ref_preproc',
                'tpl_topup_b0_rads',
                'moco_pars',
            ),
        ),
        name='outputnode'
    )

    #
    # Build the workflow
    #

    func_preproc_wf = pe.Workflow(name='func_preproc_wf')

    func_preproc_wf.connect([

        # Extract the first fmap in the list to use as a BOLD to fmap registration reference
        (inputnode, get_seepi_ref, [
            ('seepi_mag_list', 'seepi_mag_list'),
            ('seepi_meta_list', 'seepi_meta_list'),
            ('sbref_meta', 'sbref_meta')
        ]),

        # Convert BOLD phase from Siemens units to radians
        (inputnode, siemens2rads, [('bold_phs', 'in_file')]),

        # Laplacian unwrap spatial dimensions of BOLD phase timeseries
        (siemens2rads, lap_unwrap, [('out_file', 'phi_w')]),

        # Temporally unwrap and demean BOLD phase timeseries
        (siemens2rads, temp_unwrap, [('out_file', 'phi_w')]),

        # Register the SBRef to the SE-EPI with the same PE direction (typically AP)
        (inputnode, reg_sbref2seepi, [('sbref_mag', 'moving_image')]),
        (get_seepi_ref, reg_sbref2seepi, [('seepi_mag_ref', 'fixed_image')]),

        # Estimate HMC transforms of BOLD AP mag volumes to SBRef AP in SE-EPI AP space
        (inputnode, hmc_est, [('bold_mag', 'in_file')]),
        (reg_sbref2seepi, hmc_est, [('warped_image', 'ref_file')]),

        # Convert MCFLIRT matrix list to single ITK transform file
        (hmc_est, itk_hmc, [('mat_file', 'in_files')]),
        (reg_sbref2seepi, itk_hmc, [
            ('warped_image', 'in_reference'),
            ('warped_image', 'in_source'),
        ]),

        # Estimate TOPUP warp correction from SE-EPI pair
        (inputnode, topup_wf, [
            ('seepi_mag_list', 'inputnode.in_data'),
            ('seepi_meta_list', 'inputnode.metadata'),
        ]),

        # Create T2w EPI mag reference (mean of AP and PA SE-EPIs)
        (topup_wf, epi_uw_ref, [('outputnode.fmap_ref', 'in_file')]),

        # Register EPI T2w  to session T2w
        (epi_uw_ref, flirt_epi2anat, [('out_file', 'in_file')]),
        (inputnode, flirt_epi2anat, [('ses_t2w_head', 'reference')]),

        # Register session T2w to template T2w
        (inputnode, flirt_anat2tpl, [('ses_t2w_head', 'in_file'), ('tpl_t2w_head', 'reference'),]),

        # Merge EPI-anat and anat-template rigid transforms (FSL/FLIRT format)
        (flirt_epi2anat, flirt_epi2tpl, [('out_matrix_file', 'in_file')]),
        (flirt_anat2tpl, flirt_epi2tpl, [('out_matrix_file', 'in_file2')]),

        # Convert EPI to template transform from FSL to ITK format
        (flirt_epi2tpl, itk_epi2tpl, [('out_file', 'transform_file')]),
        (epi_uw_ref, itk_epi2tpl, [('out_file', 'source_file')]),
        (inputnode, itk_epi2tpl, [('tpl_t2w_head', 'reference_file')]),

        # Resample EPI reference to template space
        (epi_uw_ref, resample_epi_ref, [('out_file', 'input_image')]),
        (inputnode, resample_epi_ref, [('tpl_t2w_head', 'reference_image')]),
        (itk_epi2tpl, resample_epi_ref, [('itk_transform', 'transforms')]),

        # Resample B0 map (Hz) to template space (for one-shot BOLD resampling)
        (topup_wf, resample_topup_b0, [('outputnode.fmap', 'input_image')]),
        (inputnode, resample_topup_b0, [('tpl_t2w_head', 'reference_image')]),
        (itk_epi2tpl, resample_topup_b0, [('itk_transform', 'transforms')]),

        # Rescale template-space B0 map from Hz to rad/s
        (resample_topup_b0, hz2rads, [('output_image', 'in_file')]),

        # Make a two-element list of HMC and EPI-to-template ITK transforms
        (itk_hmc, itk_hmc_epi2tpl, [('out_file', 'in1')]),
        (itk_epi2tpl, itk_hmc_epi2tpl, [('itk_transform', 'in2')]),

        # Extract distortion parameters from BOLD metadata
        (inputnode, dist_pars, [('bold_meta', 'metadata')]),

        # One-shot resample BOLD mag timeseries to template space
        (inputnode, resample_bold_mag, [('bold_mag', 'in_file'), ('tpl_t2w_head', 'ref_file')]),
        (resample_topup_b0, resample_bold_mag, [('output_image', 'fieldmap')]),
        (itk_hmc_epi2tpl, resample_bold_mag, [('out', 'transforms')]),
        (dist_pars, resample_bold_mag, [('readout_time', 'ro_time'), ('pe_direction', 'pe_dir'),]),

        # One-shot resample BOLD phase timeseries to template space
        # (lap_unwrap, resample_bold_phs, [('phi_uw', 'in_file')]),
        (temp_unwrap, resample_bold_phs, [('phi_uw', 'in_file')]),
        (inputnode, resample_bold_phs, [('tpl_t2w_head', 'ref_file')]),
        (resample_topup_b0, resample_bold_phs, [('output_image', 'fieldmap')]),
        (itk_hmc_epi2tpl, resample_bold_phs, [('out', 'transforms')]),
        (dist_pars, resample_bold_phs, [('readout_time', 'ro_time'), ('pe_direction', 'pe_dir'),]),

        # Output results
        (resample_bold_mag, outputnode, [('out_file', 'tpl_bold_mag_preproc')]),
        (resample_bold_phs, outputnode, [('out_file', 'tpl_bold_phs_preproc')]),
        (resample_epi_ref, outputnode, [('output_image', 'tpl_epi_ref_preproc')]),
        (hz2rads, outputnode, [('out_file', 'tpl_topup_b0_rads')]),
        (hmc_est, outputnode, [('par_file', 'moco_pars')]),
    ])

    return func_preproc_wf
