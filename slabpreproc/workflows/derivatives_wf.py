#!/usr/bin/env python
# coding: utf-8

"""
Build workflow to name and place pipeline results appropriately in the BIDS derivatives folder
"""

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

from ..interfaces.derivatives import BIDSDerivSink


def build_derivatives_wf(deriv_dir):
    """
    :param deriv_dir: Path object
        Absolute path to derivatives subfolder for this workflow
    :return: None
    """

    derivatives_wf = pe.Workflow(name='derivatives_wf')

    # Create input node for all expected subcortical QC results
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'source_file',
                'tpl_bold_preproc',
                'tpl_sbref_preproc',
                'tpl_seepi_ref',
                'tpl_bold_tmean',
                'tpl_bold_tsd',
                'tpl_bold_detrended',
                'tpl_bold_tsfnr',
                'tpl_bold_tsfnr_roistats',
                'moco_pars',
            ]),
        name='inputs'
    )

    # Build individual data sinks for each input
    # Allows for renaming and sorting into directories

    ds_bold_preproc = pe.Node(BIDSDerivSink(
        deriv_dir=deriv_dir,
        data_type='preproc',
        new_suffix='recon-preproc_bold'
    ), name='ds_bold_preproc')

    ds_sbref_preproc = pe.Node(BIDSDerivSink(
        deriv_dir=deriv_dir,
        data_type='preproc',
        new_suffix='recon-preproc_sbref'
    ), name='ds_sbref_preproc')

    ds_seepi_ref = pe.Node(BIDSDerivSink(
        deriv_dir=deriv_dir,
        data_type='preproc',
        new_suffix='recon-preproc_seepi'
    ), name='ds_seepi_ref')

    ds_bold_tmean = pe.Node(BIDSDerivSink(
        deriv_dir=deriv_dir,
        data_type='qc',
        new_suffix='recon-tmean_bold'
    ), name='ds_bold_tmean')

    ds_bold_tsd = pe.Node(BIDSDerivSink(
        deriv_dir=deriv_dir,
        data_type='qc',
        new_suffix='recon-tsd_bold'
    ), name='ds_bold_tsd')

    ds_bold_detrended = pe.Node(BIDSDerivSink(
        deriv_dir=deriv_dir,
        data_type='qc',
        new_suffix='recon-detrended_bold'
    ), name='ds_bold_detrended')

    ds_bold_tsfnr = pe.Node(BIDSDerivSink(
        deriv_dir=deriv_dir,
        data_type='qc',
        new_suffix='recon-tsfnr_bold'
    ), name='ds_bold_tsfnr')

    ds_bold_tsfnr_roistats = pe.Node(BIDSDerivSink(
        deriv_dir=deriv_dir,
        data_type='qc',
        new_suffix='recon-tsfnr_bold_roistats'
    ), name='ds_bold_tsfnr_roistats')

    ds_moco_pars = pe.Node(BIDSDerivSink(
        deriv_dir=deriv_dir,
        data_type='qc',
        new_suffix='recon-moco_bold_pars'
    ), name='ds_moco_pars')

    # Connect workflow
    # source_file passed to all data sinks for use as a filename template
    derivatives_wf.connect([

        # Slab preproc results
        (inputs, ds_bold_preproc, [('source_file', 'source_file'), ('tpl_bold_preproc', 'in_file')]),
        (inputs, ds_sbref_preproc, [('source_file', 'source_file'), ('tpl_sbref_preproc', 'in_file')]),
        (inputs, ds_seepi_ref, [('source_file', 'source_file'), ('tpl_seepi_ref', 'in_file')]),

        # QC results
        (inputs, ds_bold_tmean, [('source_file', 'source_file'), ('tpl_bold_tmean', 'in_file')]),
        (inputs, ds_bold_tsd, [('source_file', 'source_file'), ('tpl_bold_tsd', 'in_file')]),
        (inputs, ds_bold_detrended, [('source_file', 'source_file'), ('tpl_bold_detrended', 'in_file')]),
        (inputs, ds_bold_tsfnr, [('source_file', 'source_file'), ('tpl_bold_tsfnr', 'in_file')]),
        (inputs, ds_bold_tsfnr_roistats, [('source_file', 'source_file'), ('tpl_bold_tsfnr_roistats', 'in_file')]),
        (inputs, ds_moco_pars, [('source_file', 'source_file'), ('moco_pars', 'in_file')]),
    ])

    return derivatives_wf