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

# Slabpreproc workflows
from ..workflows.topup_wf import build_topup_wf

# Slabpreproc interfaces
from ..interfaces import (TOPUPEncFile, LapUnwrap, ComplexPhaseDifference, SEEPIRef)


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

    # Complex phase difference with first volume (radians)
    dphi = pe.Node(ComplexPhaseDifference(), name='dphi', terminal_output=None)

    # Rigid-body pre-align SBRef to SEEPIRef prior to motion correction of BOLD
    # timeseries to SBRef.
    itk_sbref2seepi = pe.Node(
        ants.RegistrationSynQuick(transform_type='r', num_threads=antsthreads),
        name='itk_sbref2seepi',
        terminal_output=None
    )

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

    # Identify warped SE-EPI fieldmap with same PE direction as warped SBRef
    # Used to calculate the rigid transform from SBRef to SE-EPI spaces for HMC and SDC
    get_seepi_ref = pe.Node(SEEPIRef(), name='get_seepi_ref')

    # Setup TOPUP SDC workflow
    topup_wf = build_topup_wf(antsthreads=2)

    # Estimate rigid body transform from unwarped SE-EPI to session T2w

    # FLIRT rigid angular search parameters
    # Restrict search to +/- 6 degrees. Liberal limits for inter-session head rotations
    alpha_max = 6
    dalpha_coarse = (2 * alpha_max) // 3 + 1
    dalpha_fine = (2 * alpha_max) // 12 + 1

    # Estimate rigid transform from T2w EPI to session T2w space
    # FLIRT rigid body registration preferred over antsAI for partial brain slab data
    flirt_seepi2anat = pe.Node(
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
            out_matrix_file='tx_seepi2anat.mat',
            output_type='NIFTI_GZ',
            terminal_output='none'
        ),
        name='flirt_seepi2anat',
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

    # Chain SEEPI2anat and anat2template rigid transforms (FSL/FLIRT)
    flirt_epi2tpl = pe.Node(fsl.ConvertXFM(
        out_file='flirt_epi2tpl.mat',
        concat_xfm=True),
        name='flirt_epi2tpl'
    )

    # Convert SEEPI2template rigid transform to ITK format
    itk_seepi2tpl = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True), name='itk_seepi2tpl')

    # Chain SBRef2SEEPI and SEEPI2template transforms (ITK)
    itk_sbref2tpl = pe.Node(util.Merge(2), name='itk_sbref2tpl', run_without_submitting=True)

    # Create list from ITK HMC transforms and EPI-to-template transform
    # Pass this list to fmriprep ResampleSeries interface (transforms arg)
    itk_hmc_seepi2tpl = pe.Node(util.Merge(2), name='itk_hmc_seepi2tpl', run_without_submitting=True)

    # Conventional multistep resampling of 3D EPI reference images to individual template space
    resample_topup_b0_hz = pe.Node(
        ants.ApplyTransforms(num_threads=antsthreads),
        name='resample_topup_b0_hz'
    )
    resample_seepiref = pe.Node(
        ants.ApplyTransforms(num_threads=antsthreads),
        name='resample_seepiref'
    )
    resample_sbref = pe.Node(
        ResampleSeries(jacobian=True, num_threads=antsthreads),
        name='resample_sbref'
    )

    # One-shot resample BOLD timeseries to template space (including HMC and SDC)
    # Only the 4D images are resampled this way using the fmriprep ResampleSeries interface
    resample_bold_mag = pe.Node(
        ResampleSeries(jacobian=True, num_threads=antsthreads),
        name='resample_bold_mag'
    )
    resample_bold_phs = pe.Node(
        ResampleSeries(jacobian=False, num_threads=antsthreads),
        name='resample_bold_phs'
    )
    resample_bold_dphi = pe.Node(
        ResampleSeries(jacobian=False, num_threads=antsthreads),
        name='resample_bold_dphi'
    )

    # Workflow output node
    outputnode = pe.Node(
        util.IdentityInterface(
            fields=(
                'tpl_bold_mag_preproc',
                'tpl_bold_phs_preproc',
                'tpl_bold_dphi_preproc',
                'tpl_sbref_preproc',
                'tpl_seepiref_preproc',
                'tpl_topup_b0_hz',
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

        # Convert BOLD phase from Siemens units to radians
        (inputnode, siemens2rads, [('bold_phs', 'in_file')]),

        # Laplacian unwrap spatial dimensions of BOLD phase timeseries
        (siemens2rads, lap_unwrap, [('out_file', 'phi_w')]),

        # Complex phase difference with first volume
        (inputnode, dphi, [('bold_mag', 'mag')]),
        (siemens2rads, dphi, [('out_file', 'phi_w')]),

        # Identify the SE-EPI fieldmap with the same PE direction as the BOLD series
        (inputnode, get_seepi_ref, [
            ('seepi_mag_list', 'seepi_mag_list'),
            ('seepi_meta_list', 'seepi_meta_list'),
            ('sbref_meta', 'sbref_meta')
        ]),

        # Register the SBRef to the SE-EPI with the same PE direction (typically AP)
        (inputnode, itk_sbref2seepi, [('sbref_mag', 'moving_image')]),
        (get_seepi_ref, itk_sbref2seepi, [('seepi_mag_ref', 'fixed_image')]),

        # Estimate HMC transforms of BOLD AP mag volumes to SBRef AP in SE-EPI AP space
        (inputnode, hmc_est, [('bold_mag', 'in_file')]),
        (itk_sbref2seepi, hmc_est, [('warped_image', 'ref_file')]),

        # Convert MCFLIRT matrix list to single ITK transform file
        (hmc_est, itk_hmc, [('mat_file', 'in_files')]),
        (itk_sbref2seepi, itk_hmc, [
            ('warped_image', 'in_reference'),
            ('warped_image', 'in_source'),
        ]),

        # Estimate TOPUP warp correction from SE-EPI pair
        (inputnode, topup_wf, [
            ('seepi_mag_list', 'in_node.seepi_mag_list'),
            ('seepi_meta_list', 'in_node.seepi_meta_list'),
        ]),

        # Register unwarped T2w SEEPI  to session T2w SPACE
        (topup_wf, flirt_seepi2anat, [('out_node.seepi_uw_ref', 'in_file')]),
        (inputnode, flirt_seepi2anat, [('ses_t2w_head', 'reference')]),

        # Register session T2w to template T2w
        (inputnode, flirt_anat2tpl, [('ses_t2w_head', 'in_file'), ('tpl_t2w_head', 'reference'),]),

        # Chain SEEPI2anat and anat2template rigid transforms (FSL/FLIRT format)
        (flirt_seepi2anat, flirt_epi2tpl, [('out_matrix_file', 'in_file')]),
        (flirt_anat2tpl, flirt_epi2tpl, [('out_matrix_file', 'in_file2')]),

        # Convert chained SEEPI2template transform from FSL to ITK format
        (flirt_epi2tpl, itk_seepi2tpl, [('out_file', 'transform_file')]),
        (topup_wf, itk_seepi2tpl, [('out_node.seepi_uw_ref', 'source_file')]),
        (inputnode, itk_seepi2tpl, [('tpl_t2w_head', 'reference_file')]),

        # Chain SBRef2SEEPI and SEEPI2template to yield SBRef2Template TX (ITK format)
        (itk_sbref2seepi, itk_sbref2tpl, [('out_matrix', 'in1')]),
        (itk_seepi2tpl, itk_sbref2tpl, [('itk_transform', 'in2')]),

        # Resample SE-EPI reference to template space
        (topup_wf, resample_seepiref, [('out_node.seepi_uw_ref', 'input_image')]),
        (inputnode, resample_seepiref, [('tpl_t2w_head', 'reference_image')]),
        (itk_seepi2tpl, resample_seepiref, [('itk_transform', 'transforms')]),

        # Resample SE-EPI B0 map (rad/s) to template space (for one-shot BOLD resampling)
        (topup_wf, resample_topup_b0_hz, [('out_node.topup_b0_hz', 'input_image')]),
        (inputnode, resample_topup_b0_hz, [('tpl_t2w_head', 'reference_image')]),
        (itk_seepi2tpl, resample_topup_b0_hz, [('itk_transform', 'transforms')]),

        # Make a two-element list of HMC and EPI-to-template ITK transforms
        (itk_hmc, itk_hmc_seepi2tpl, [('out_file', 'in1')]),
        (itk_seepi2tpl, itk_hmc_seepi2tpl, [('itk_transform', 'in2')]),

        # Extract distortion parameters from BOLD metadata
        (inputnode, dist_pars, [('bold_meta', 'metadata')]),

        # One-shot resample SBRef to template space
        (inputnode, resample_sbref, [('sbref_mag', 'in_file'), ('tpl_t2w_head', 'ref_file')]),
        (resample_topup_b0_hz, resample_sbref, [('output_image', 'fieldmap')]),
        (itk_sbref2tpl, resample_sbref, [('out', 'transforms')]),
        (dist_pars, resample_sbref, [('readout_time', 'ro_time'), ('pe_direction', 'pe_dir'),]),

        # One-shot resample BOLD mag timeseries to template space
        (inputnode, resample_bold_mag, [('bold_mag', 'in_file'), ('tpl_t2w_head', 'ref_file')]),
        (resample_topup_b0_hz, resample_bold_mag, [('output_image', 'fieldmap')]),
        (itk_hmc_seepi2tpl, resample_bold_mag, [('out', 'transforms')]),
        (dist_pars, resample_bold_mag, [('readout_time', 'ro_time'), ('pe_direction', 'pe_dir'),]),

        # One-shot resample BOLD phase timeseries to template space
        (lap_unwrap, resample_bold_phs, [('phi_uw', 'in_file')]),
        (inputnode, resample_bold_phs, [('tpl_t2w_head', 'ref_file')]),
        (resample_topup_b0_hz, resample_bold_phs, [('output_image', 'fieldmap')]),
        (itk_hmc_seepi2tpl, resample_bold_phs, [('out', 'transforms')]),
        (dist_pars, resample_bold_phs, [('readout_time', 'ro_time'), ('pe_direction', 'pe_dir')]),

        # One-shot resample BOLD phase difference timeseries to template space
        (dphi, resample_bold_dphi, [('dphi', 'in_file')]),
        (inputnode, resample_bold_dphi, [('tpl_t2w_head', 'ref_file')]),
        (resample_topup_b0_hz, resample_bold_dphi, [('output_image', 'fieldmap')]),
        (itk_hmc_seepi2tpl, resample_bold_dphi, [('out', 'transforms')]),
        (dist_pars, resample_bold_dphi, [('readout_time', 'ro_time'), ('pe_direction', 'pe_dir')]),

        # Output results
        (resample_bold_mag, outputnode, [('out_file', 'tpl_bold_mag_preproc')]),
        (resample_bold_phs, outputnode, [('out_file', 'tpl_bold_phs_preproc')]),
        (resample_bold_dphi, outputnode, [('out_file', 'tpl_bold_dphi_preproc')]),
        (resample_seepiref, outputnode, [('output_image', 'tpl_seepiref_preproc')]),
        (resample_sbref, outputnode, [('out_file', 'tpl_sbref_preproc')]),
        (resample_topup_b0_hz, outputnode, [('output_image', 'tpl_topup_b0_hz')]),
        (hmc_est, outputnode, [('par_file', 'moco_pars')]),
    ])

    return func_preproc_wf
