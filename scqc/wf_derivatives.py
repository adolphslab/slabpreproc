#!/usr/bin/env python
# coding: utf-8

"""
Build workflow to name and place pipeline results appropriately in the BIDS derivatives folder
Based on code fragments from https://github.com/nipreps/fmriprep/blob/master/fmriprep/workflows/bold/outputs.py
"""

import os

import bids.layout
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe


def build_wf_derivatives(deriv_dir):
    """
    :param deriv_dir: Path object
        Absolute path to derivatives subfolder for this workflow
    :return: None
    """

    wf_derivatives = pe.Workflow(name='wf_derivatives')

    # Create input node for all expected subcortical QC results
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'source_file',
                'bold_preproc',
                'sbref_preproc',
                'bold_tmean',
                'bold_tsd',
                'bold_detrended',
                'bold_tsfnr',
                't1_atlas2ind',
                't1_ind2epi',
                't1_atlas2epi',
                'labels_atlas2epi',
            ]),
        name='inputs'
    )

    # Build individual data sinks for each input
    ds_bold_preproc = pe.Node(util.Function(
        input_names=['source_file', 'in_file', 'deriv_dir', 'new_suffix', 'datatype'], output_names=[],
        function=bids_sink), name='ds_bold_preproc')
    ds_bold_preproc.inputs.deriv_dir = deriv_dir
    ds_bold_preproc.inputs.new_suffix = 'recon-preproc_bold'
    ds_bold_preproc.inputs.datatype = 'preproc'

    ds_sbref_preproc = pe.Node(util.Function(
        input_names=['source_file', 'in_file', 'deriv_dir', 'new_suffix', 'datatype'], output_names=[],
        function=bids_sink), name='ds_sbref_preproc')
    ds_sbref_preproc.inputs.deriv_dir = deriv_dir
    ds_sbref_preproc.inputs.new_suffix = 'recon-preproc_sbref'
    ds_sbref_preproc.inputs.datatype = 'preproc'

    ds_bold_tmean = pe.Node(util.Function(
        input_names=['source_file', 'in_file', 'deriv_dir', 'new_suffix', 'datatype'], output_names=[],
        function=bids_sink), name='ds_bold_tmean')
    ds_bold_tmean.inputs.deriv_dir = deriv_dir
    ds_bold_tmean.inputs.new_suffix = 'recon-tmean_bold'
    ds_bold_tmean.inputs.datatype = 'qc'

    ds_bold_tsd = pe.Node(util.Function(
        input_names=['source_file', 'in_file', 'deriv_dir', 'new_suffix', 'datatype'], output_names=[],
        function=bids_sink), name='ds_bold_tsd')
    ds_bold_tsd.inputs.deriv_dir = deriv_dir
    ds_bold_tsd.inputs.new_suffix = 'recon-tsd_bold'
    ds_bold_tsd.inputs.datatype = 'qc'

    ds_bold_detrended = pe.Node(util.Function(
        input_names=['source_file', 'in_file', 'deriv_dir', 'new_suffix', 'datatype'], output_names=[],
        function=bids_sink), name='ds_bold_detrended')
    ds_bold_detrended.inputs.deriv_dir = deriv_dir
    ds_bold_detrended.inputs.new_suffix = 'recon-detrended_bold'
    ds_bold_detrended.inputs.datatype = 'qc'

    ds_bold_tsfnr= pe.Node(util.Function(
        input_names=['source_file', 'in_file', 'deriv_dir', 'new_suffix', 'datatype'], output_names=[],
        function=bids_sink), name='ds_bold_tsfnr')
    ds_bold_tsfnr.inputs.deriv_dir = deriv_dir
    ds_bold_tsfnr.inputs.new_suffix = 'recon-tsfnr_bold'
    ds_bold_tsfnr.inputs.datatype = 'qc'

    ds_t1_atlas2ind = pe.Node(util.Function(
        input_names=['source_file', 'in_file', 'deriv_dir', 'new_suffix', 'datatype'], output_names=[],
        function=bids_sink), name='ds_t1_atlas2ind')
    ds_t1_atlas2ind.inputs.deriv_dir = deriv_dir
    ds_t1_atlas2ind.inputs.new_suffix = 'desc-atlas2ind_T1w'
    ds_t1_atlas2ind.inputs.datatype = 'atlas'

    ds_t1_ind2epi = pe.Node(util.Function(
        input_names=['source_file', 'in_file', 'deriv_dir', 'new_suffix', 'datatype'], output_names=[],
        function=bids_sink), name='ds_t1_ind2epi')
    ds_t1_ind2epi.inputs.deriv_dir = deriv_dir
    ds_t1_ind2epi.inputs.new_suffix = 'desc-ind2epi_T1w'
    ds_t1_ind2epi.inputs.datatype = 'atlas'

    ds_t1_atlas2epi= pe.Node(util.Function(
        input_names=['source_file', 'in_file', 'deriv_dir', 'new_suffix', 'datatype'], output_names=[],
        function=bids_sink), name='ds_t1_atlas2epi')
    ds_t1_atlas2epi.inputs.deriv_dir = deriv_dir
    ds_t1_atlas2epi.inputs.new_suffix = 'desc-atlas2epi_T1w'
    ds_t1_atlas2epi.inputs.datatype = 'atlas'

    ds_labels_atlas2epi= pe.Node(util.Function(
        input_names=['source_file', 'in_file', 'deriv_dir', 'new_suffix', 'datatype'], output_names=[],
        function=bids_sink), name='ds_labels_atlas2epi')
    ds_labels_atlas2epi.inputs.deriv_dir = deriv_dir
    ds_labels_atlas2epi.inputs.new_suffix = 'desc-atlas2epi_dseg'
    ds_labels_atlas2epi.inputs.datatype = 'atlas'

    # Connect workflow
    wf_derivatives.connect([
        (inputs, ds_bold_preproc, [('source_file', 'source_file'), ('bold_preproc', 'in_file')]),
        (inputs, ds_sbref_preproc, [('source_file', 'source_file'), ('sbref_preproc', 'in_file')]),
        (inputs, ds_bold_tmean, [('source_file', 'source_file'), ('bold_tmean', 'in_file')]),
        (inputs, ds_bold_tsd, [('source_file', 'source_file'), ('bold_tsd', 'in_file')]),
        (inputs, ds_bold_detrended, [('source_file', 'source_file'), ('bold_detrended', 'in_file')]),
        (inputs, ds_bold_tsfnr, [('source_file', 'source_file'), ('bold_tsfnr', 'in_file')]),
        (inputs, ds_t1_atlas2ind, [('source_file', 'source_file'), ('t1_atlas2ind', 'in_file')]),
        (inputs, ds_t1_ind2epi, [('source_file', 'source_file'), ('t1_ind2epi', 'in_file')]),
        (inputs, ds_t1_atlas2epi, [('source_file', 'source_file'), ('t1_atlas2epi', 'in_file')]),
        (inputs, ds_labels_atlas2epi, [('source_file', 'source_file'), ('labels_atlas2epi', 'in_file')]),

    ])

    return wf_derivatives


def bids_sink(source_file, in_file, deriv_dir, new_suffix, datatype):
    """

    :param source_file: str, pathlike

    :param in_file: str, pathlike
        File to write to derivatives output subfolder
    :param deriv_dir: Path
        Derivatives output subfolder path
    :param new_suffix: str
        Suffix to replace existing source_file suffix in derivatives output
    :param datatype: str
        Data type for subfolder creation
    :return:
    """

    from os import makedirs
    import os.path as op
    import bids
    import shutil
    from nipype.utils.logger import logging

    logger = logging.getLogger('nipype.interface')

    # Source file basename
    source_bname = op.basename(source_file)

    # Get entities from source_file
    keys = bids.layout.parse_file_entities(source_file)

    subj_id = keys['subject']
    sess_id = keys['session']
    old_suffix = keys['suffix']

    # Create subject/session/datatype output folder
    out_dir = op.join(deriv_dir, 'sub-' + subj_id, 'ses-' + sess_id, datatype)
    makedirs(out_dir, exist_ok=True)

    # Output file path. Replace current suffix with new suffix
    out_file = op.join(out_dir, source_bname.replace(old_suffix, new_suffix))

    # Copy in_file to deriv_dir/subj_dir/sess_dir/out_file
    logger.info(f'Copying {in_file} to {out_file}')
    shutil.copyfile(in_file, out_file)
