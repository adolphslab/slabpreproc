#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe


def build_wf_func_preproc():

    # Preproc inputs
    inputs = pe.Node(
        util.IdentityInterface(
            fields=('bold', 'anat')
        ),
        name='inputs'
    )

    # Find SBRef, MOCORef and TOPUP SE-EPIs
    aux_files = pe.Node(
        util.Function(
            input_names=['bold'],
            output_names=['sbref', 'mocoref', 'seepi_list'],
            function=find_aux_files
        ),
        name='aux_files'
    )

    # Grab info needed by TOPUP from SE-EPI files
    # Returns a list of readout times and PE directions
    seepi_topup_info = pe.Node(
        util.Function(
            input_names=['epis'],
            output_names=['readout_times', 'encoding_direction'],
            function=get_topup_info
        ),
        name='seepi_topup_info'
    )

    sbref_topup_info = pe.Node(
        util.Function(
            input_names=['epis'],
            output_names=['readout_times', 'encoding_direction'],
            function=get_topup_info
        ),
        name='sbref_topup_info'
    )

    # Create an encoding file for SE-EPI TOPUP fitting
    seepi_enc_file = pe.Node(
        util.Function(
            input_names=['readout_times', 'encoding_direction'],
            output_names=['encoding_file'],
            function=make_encoding_file
        ),
        name='seepi_enc_file'
    )

    # Create an encoding file for SBRef and BOLD correction with TOPUP
    sbref_enc_file = pe.Node(
        util.Function(
            input_names=['readout_times', 'encoding_direction'],
            output_names=['encoding_file'],
            function=make_encoding_file
        ),
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
            output_type="NIFTI_GZ"
        ),
        name='apply_topup_bold')

    # Define outputs from preproc workflow
    outputs = pe.Node(
        util.IdentityInterface(
            fields=('bold', 'sbref'),
        ),
        name='outputs'
    )

    # Build workflow
    wf_func_preproc = pe.Workflow(name='wf_func_preproc')

    wf_func_preproc.connect([
        (inputs, aux_files, [('bold', 'bold')]),

        (aux_files, seepi_topup_info, [('seepi_list', 'epis')]),
        (aux_files, sbref_topup_info, [('sbref', 'epis')]),

        (sbref_topup_info, sbref_enc_file, [
            ('readout_times', 'readout_times'),
            ('encoding_direction', 'encoding_direction')
        ]),
        (seepi_topup_info, seepi_enc_file, [
            ('readout_times', 'readout_times'),
            ('encoding_direction', 'encoding_direction')
        ]),

        (aux_files, concat, [('seepi_list', 'in_files')]),
        (inputs, mcflirt, [('bold', 'in_file')]),
        (aux_files, mcflirt, [('mocoref', 'ref_file')]),

        # Fit TOPUP model to SE-EPIs
        (concat, topup, [('merged_file', 'in_file')]),
        (seepi_enc_file, topup, [('encoding_file', 'encoding_file')]),

        # TOPUP correct SBRef
        (aux_files, applytopup_sbref, [('sbref', 'in_files')]),
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
        (applytopup_sbref, outputs, [('out_corrected', 'sbref')]),
        (applytopup_bold, outputs, [('out_corrected', 'bold')]),
    ])

    return wf_func_preproc


def find_aux_files(bold):

    import os.path as op
    import bids
    from nipype import logging

    # Create a logger for this function
    logger = logging.getLogger("nipype.interface")

    # BOLD series basename
    bold_bname = op.basename(bold)
    logger.info(f'Locating auxiliary files for {bold_bname}')

    # SBRef by filename substitution
    sbref = bold.replace('_bold', '_sbref')

    # Build BIDS layout
    bids_dir = op.dirname(op.dirname(op.dirname(op.dirname(bold))))

    logger.info(f'Indexing {bids_dir}')

    layout = bids.BIDSLayout(bids_dir, validate=False)

    # Find EPI fieldmaps intended for this BOLD series (depends heavily on bidskit --bind-fmaps)
    epi_fmaps = layout.get(suffix='epi', extension='nii', regex_search=True)

    logger.info(f'Found {len(epi_fmaps)} EPI fieldmaps')

    # Find all SE-EPI fieldmaps with an IntendedFor field
    seepi_list = []
    for fmap in epi_fmaps:
        meta = fmap.get_metadata()
        if 'IntendedFor' in meta:
            for target in meta['IntendedFor']:
                if bold_bname in target:
                    # Need to use .path not .filename
                    seepi_list.append(fmap.path)

    assert len(seepi_list) > 0, "No SE-EPI fieldmaps with an IntendedFor field found"

    # Use AP SE-EPI as MOCO reference
    # TODO: Match SEEPI phase enc dir to BOLD
    mocoref = seepi_list[0]

    return sbref, mocoref, seepi_list


def get_topup_info(epis):
    """
    Get TOPUP metadata from BIDS tree for each EPI image
    Function must be standalone - don't use utility functions within this package

    :param epis:
    :return: readout_times
    :return: encoding_direction
    """

    from niworkflows.utils.bids import get_metadata_for_nifti

    # Convert single strings into list of length 1
    if type(epis) == str:
        epis = [epis]

    readout_times = []
    encoding_direction = []

    # Extract readout time and PE direction for each EPI image provided
    for epi_fname in epis:

        t_ro = get_metadata_for_nifti(epi_fname, validate=False)['TotalReadoutTime']
        readout_times.append(t_ro)

        # Convert BIDS PE direction (i, j, k) to FSL direction (x, y, z)
        bids_pe_dir = get_metadata_for_nifti(epi_fname, validate=False)['PhaseEncodingDirection']
        fsl_pe_dir = bids_pe_dir.replace('i', 'x').replace('j', 'y').replace('k', 'z')
        encoding_direction.append(fsl_pe_dir)

    return readout_times, encoding_direction


def make_encoding_file(readout_times, encoding_direction):

    import tempfile
    from pathlib import Path
    import os.path as op
    import numpy as np
    from nipype.utils.logger import logging

    logger = logging.getLogger('nipype.interface')

    enc_mat = []

    for ic, t_ro in enumerate(readout_times):

        d_pe = encoding_direction[ic]

        logger.info(f'> make_encoding_file : {t_ro} {d_pe}')

        if d_pe == 'x':
            v_enc = [1, 0, 0, t_ro]
        elif d_pe == 'x-':
            v_enc = [-1, 0, 0, t_ro]
        elif d_pe == 'y':
            v_enc = [0, 1, 0, t_ro]
        elif d_pe == 'y-':
            v_enc = [0, -1, 0, t_ro]
        elif d_pe == 'z':
            v_enc = [0, 0, 1, t_ro]
        elif d_pe == 'z-':
            v_enc = [0, 0, -1, t_ro]
        else:
            print(f'* Unknown PE direction {d_pe}')
            v_enc = [0, 1, 0, t_ro]

        # Add encoding row to encoding matrix
        enc_mat.append(v_enc)

    # Store the encoding file in a temp folder
    tmp_dir = Path(tempfile.mkdtemp())
    encoding_file = tmp_dir / 'topup_encoding_file.txt'

    if not op.isfile(encoding_file):
        try:
            np.savetxt(fname=str(encoding_file), X=enc_mat, fmt="%2d %2d %2d %9.6f")
        except IOError:
            print(f'* Could not save encoding matrix to {str(encoding_file)}')

    logger.info(f'Saved encoding matrix to {encoding_file}')

    return encoding_file
