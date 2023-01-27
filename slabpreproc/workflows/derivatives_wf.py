#!/usr/bin/env python
# coding: utf-8

"""
Build workflow to name and place pipeline results appropriately in the BIDS derivatives folder
"""

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

from ..interfaces.derivatives import DerivativesSorter


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
                'tpl_bold_mag_preproc',
                'tpl_sbref_preproc',
                'tpl_seepi_unwarp_mean',
                'tpl_bold_tmean',
                'tpl_bold_tsd',
                'tpl_bold_tsfnr',
                'tpl_bold_tsfnr_roistats',
                'tpl_b0_rads',
                'tpl_dropout',
                'motion_csv',
                'tpl_t1_head',
                'tpl_t2_head',
                'tpl_pseg',
                'tpl_dseg',
                'tpl_bmask'
            ]
        ),
        name='inputs'
    )

    # Sorting info
    sort_dict_list = [
        {'DataType': 'preproc', 'NewSuffix': 'recon-preproc_bold', 'FileType': 'Image'},
        {'DataType': 'preproc', 'NewSuffix': 'recon-preproc_sbref', 'FileType': 'Image'},
        {'DataType': 'preproc', 'NewSuffix': 'recon-preproc_seepi_unwarp_mean', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tmean_bold', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tsd_bold', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tsfnr_bold', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tsfnr_bold_roistats', 'FileType': 'Text'},
        {'DataType': 'qc', 'NewSuffix': 'recon-topup_fieldmap', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-topup_dropout', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-motion_pars', 'FileType': 'CSV'},
        {'DataType': 'atlas', 'NewSuffix': 'atlas-cit168_T1w', 'FileType': 'Image'},
        {'DataType': 'atlas', 'NewSuffix': 'atlas-cit168_T2w', 'FileType': 'Image'},
        {'DataType': 'atlas', 'NewSuffix': 'atlas-cit168_pseg', 'FileType': 'Image'},
        {'DataType': 'atlas', 'NewSuffix': 'atlas-cit168_dseg', 'FileType': 'Image'},
        {'DataType': 'atlas', 'NewSuffix': 'atlas-cit168_desc-brain_mask', 'FileType': 'Image'},
    ]

    # Create a list of all inputs
    deriv_list = pe.Node(
        util.Merge(numinputs=15),
        name='deriv_list'
    )

    # Build multi-input derivatives output sorter
    # Renames and sorts inputs into correct derivatives heirarchy
    deriv_sorter = pe.Node(
        DerivativesSorter(
            deriv_dir=deriv_dir,
            sort_dict_list=sort_dict_list
        ),
        name='deriv_sorter'
    )

    # Connect workflow
    # source_file passed to all data sinks for use as a filename template
    derivatives_wf.connect([

        # Slab preproc and QC results to BIDS derivatives sorter
        (inputs, deriv_sorter, [('source_file', 'source_file')]),

        # Create file list
        (inputs, deriv_list, [
            ('tpl_bold_mag_preproc', 'in1'),
            ('tpl_sbref_preproc', 'in2'),
            ('tpl_seepi_unwarp_mean', 'in3'),
            ('tpl_bold_tmean', 'in4'),
            ('tpl_bold_tsd', 'in5'),
            ('tpl_bold_tsfnr', 'in6'),
            ('tpl_bold_tsfnr_roistats', 'in7'),
            ('tpl_b0_rads', 'in8'),
            ('tpl_dropout', 'in9'),
            ('motion_csv', 'in10'),
            ('tpl_t1_head', 'in11'),
            ('tpl_t2_head', 'in12'),
            ('tpl_pseg', 'in13'),
            ('tpl_dseg', 'in14'),
            ('tpl_bmask', 'in15'),
        ]),

        # Pass file list to sorter
        (deriv_list, deriv_sorter, [('out', 'file_list')])

    ])

    return derivatives_wf
