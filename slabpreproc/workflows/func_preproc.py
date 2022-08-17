#!/usr/bin/env python
"""
Functional MRI preprocessing workflow

"""

import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

from sdcflows.workflows.apply.registration import init_coeff2epi_wf
from sdcflows.workflows.apply.correction import init_unwarp_wf

from niworkflows.func.util import init_bold_reference_wf

from ..interfaces import TOPUPEncFile


def build_wf_func_preproc():

    # Set to a reasonable default value for now
    omp_nthreads = 4

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

    # Initial BOLD reference image
    initial_boldref_wf = init_bold_reference_wf(
        name="initial_boldref_wf",
        omp_nthreads=omp_nthreads,
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

    # Select the first fmap from the list of fmaps.
    # Assumed to have the same PE direction as the BOLD series
    # TODO: check that the first fmap PE direction matches that of the BOLD series
    fmap_ref = pe.Node(
        util.Select(index=[0]),
        name='get_fmap_ref'
    )

    # Create a signal mask for the fmap reference
    # TODO: Use a more principled threshold (Otsu, etc)
    fmap_thresh = pe.Node(
        fsl.maths.Threshold(
            thresh=10,  # Threshold at 10% of robust range
            use_nonzero_voxels=True,
            use_robust_range=True
        ),
        name='fmap_thresh'
    )
    fmap_mask = pe.Node(
        fsl.maths.UnaryMaths(
            operation='bin',
            output_datatype='char'
        ),
        name='fmap_mask'
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
    topup_est = pe.Node(
        fsl.TOPUP(
            output_type='NIFTI_GZ'
        ),
        name='topup_est')

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

    # Use the sdcflows approach for combining motion correction transforms and
    # susceptibility distortion correction (SDC) into combined per-volume warps
    # Refs:
    # https://github.com/nipreps/fmriprep/blob/master/fmriprep/workflows/bold/base.py

    # Split the 4D BOLD series into individual 3D volume
    bold_split = pe.Node(
        fsl.Split(dimension="t"),
        name="bold_split"
    )

    # From fMRIPrep and SDCFlows
    # Register the TOPUP field coefficients to the warped EPI space
    coeff2epi_wf = init_coeff2epi_wf(omp_nthreads=4, write_coeff=True)

    # From fMRIPrep and SDCFlows
    # Init the BOLD unwarp workflow
    unwarp_wf = init_unwarp_wf(omp_nthreads=4)

    # Define outputs for the fMRI preproc workflow
    outputs = pe.Node(
        util.IdentityInterface(
            fields=(
                'bold_preproc',
                'sbref_preproc',
                'moco_pars'),
        ),
        name='outputs'
    )

    #
    # Build the workflow
    #

    wf_func_preproc = pe.Workflow(name='wf_func_preproc')

    wf_func_preproc.connect([

        # Create BOLD reference image (niworkflows)
        (inputs, initial_boldref_wf, [
            ('bold', 'inputs.bold_file'),
            ('sbref', 'inputs.sbref_file')
        ]),

        # Create TOPUP encoding files
        # Requires both the image to be unwarped and the metadata for that image
        (inputs, sbref_enc_file, [('sbref', 'epi_list')]),
        (inputs, sbref_enc_file, [('sbref_meta', 'meta_list')]),
        (inputs, seepi_enc_file, [('fmaps', 'epi_list')]),
        (inputs, seepi_enc_file, [('fmaps_meta', 'meta_list')]),

        # Extract the first fmap in the list to use as a reference image
        # This image is still warped
        (inputs, fmap_ref, [('fmaps', 'inlist')]),

        # Create a binary signal mask for the fmap ref
        (fmap_ref, fmap_thresh, [('out', 'in_file')]),
        (fmap_thresh, fmap_mask, [('out_file', 'in_file')]),

        # Motion correct BOLD series to the middle image of the series
        (inputs, mcflirt, [('bold', 'in_file')]),

        # Fit TOPUP model to SE-EPIs
        (inputs, concat, [('fmaps', 'in_files')]),
        (concat, topup_est, [('merged_file', 'in_file')]),
        (seepi_enc_file, topup_est, [('encoding_file', 'encoding_file')]),

        # Register the TOPUP coefficients to warped BOLD EPI space
        (topup_est, coeff2epi_wf, [("out_fieldcoef", "inputnode.fmap_coeff")]),
        (fmap_ref, coeff2epi_wf, [("out", "inputnode.fmap_ref")]),
        (fmap_mask, coeff2epi_wf, [("fmap_mask", "inputnode.fmap_mask")]),

        (initial_boldref_wf, coeff2epi_wf, [
            ("outputnode.ref_image", "inputnode.target_ref"),
            ("outputnode.bold_mask", "inputnode.target_mask")
        ]),

        (coeff2epi_wf, unwarp_wf, [
            ("outputnode.fmap_coeff", "inputnode.fmap_coeff")]),

        (mcflirt, unwarp_wf, [
            ("xforms", "inputnode.hmc_xforms")]),

        (bold_split, unwarp_wf, [
            ("out_files", "inputnode.distorted")]),

        # Output results
        (mcflirt, outputs, [('par_file', 'moco_pars')]),
        (unwarp_wf, outputs, [
            ('out_corrected', 'sbref_preproc')
        ])
    ])

    return wf_func_preproc
