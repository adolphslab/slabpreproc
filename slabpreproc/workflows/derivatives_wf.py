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

    # Workflow input node
    in_node = pe.Node(
        util.IdentityInterface(
            fields=[
                'source_file',
                'tpl_bold_mag_preproc',
                'tpl_sbref_preproc',
                'tpl_seepi_preproc',
                'tpl_bold_tmean',
                'tpl_bold_tsd',
                'tpl_bold_tsfnr',
                'tpl_bold_tsfnr_roistats',
                'tpl_b0_rads',
                'tpl_dropout',
                'motion_csv',
                'tpl_t1w_head',
                'tpl_t2w_head',
                'tpl_t1w_brain',
                'tpl_t2w_brain',
                'tpl_pseg',
                'tpl_dseg',
                'tpl_bmask',
                'melodic_out_dir'  # Folder - separate handling
            ]
        ),
        name='in_node'
    )

    # File sorting dictionary list
    file_sort_dicts = [
        {'DataType': 'preproc', 'NewSuffix': 'recon-preproc_bold', 'FileType': 'Image'},
        {'DataType': 'preproc', 'NewSuffix': 'recon-preproc_sbref', 'FileType': 'Image'},
        {'DataType': 'preproc', 'NewSuffix': 'recon-preproc_seepi_preproc', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tmean_bold', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tsd_bold', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tsfnr_bold', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tsfnr_bold_roistats', 'FileType': 'Text'},
        {'DataType': 'qc', 'NewSuffix': 'recon-topup_fieldmap', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-topup_dropout', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-motion_pars', 'FileType': 'CSV'},
        {'DataType': 'atlas', 'NewSuffix': '', 'FileType': 'Image'},
        {'DataType': 'atlas', 'NewSuffix': '', 'FileType': 'Image'},
        {'DataType': 'atlas', 'NewSuffix': '', 'FileType': 'Image'},
        {'DataType': 'atlas', 'NewSuffix': '', 'FileType': 'Image'},
        {'DataType': 'atlas', 'NewSuffix': '', 'FileType': 'Image'},
        {'DataType': 'atlas', 'NewSuffix': '', 'FileType': 'Image'},
        {'DataType': 'atlas', 'NewSuffix': '', 'FileType': 'Image'},
    ]

    # Folder sorting dictionary list - needs separate Traits handling
    folder_sort_dicts = [
        {'DataType': 'melodic', 'NewSuffix': 'melodic.ica', 'FileType': 'Folder'}
    ]

    # Create a list of all file in_node
    deriv_file_list = pe.Node(
        util.Merge(numinputs=len(file_sort_dicts)),
        name='deriv_file_list'
    )

    # Create a list of all folder in_node
    deriv_folder_list = pe.Node(
        util.Merge(numinputs=len(folder_sort_dicts)),
        name='deriv_folder_list'
    )

    # Build multi-input derivatives output sorter
    # Renames and sorts in_node into correct derivatives hierarchy
    deriv_sorter = pe.Node(
        DerivativesSorter(
            deriv_dir=deriv_dir,
            file_sort_dicts=file_sort_dicts,
            folder_sort_dicts=folder_sort_dicts
        ),
        name='deriv_sorter'
    )

    # Connect workflow
    derivatives_wf.connect([

        # Pass source BOLD filename to derivatives sorter as a filename template
        (in_node, deriv_sorter, [('source_file', 'source_file')]),

        # Create file list and pass to sorter
        (in_node, deriv_file_list, [
            ('tpl_bold_mag_preproc', 'in1'),
            ('tpl_sbref_preproc', 'in2'),
            ('tpl_seepi_preproc', 'in3'),
            ('tpl_bold_tmean', 'in4'),
            ('tpl_bold_tsd', 'in5'),
            ('tpl_bold_tsfnr', 'in6'),
            ('tpl_bold_tsfnr_roistats', 'in7'),
            ('tpl_b0_rads', 'in8'),
            ('tpl_dropout', 'in9'),
            ('motion_csv', 'in10'),
            ('tpl_t1w_head', 'in11'),
            ('tpl_t2w_head', 'in12'),
            ('tpl_t1w_brain', 'in13'),
            ('tpl_t2w_brain', 'in14'),
            ('tpl_pseg', 'in15'),
            ('tpl_dseg', 'in16'),
            ('tpl_bmask', 'in17')
        ]),

        (deriv_file_list, deriv_sorter, [('out', 'file_list')]),

        # Create folder list and pass to sorter
        (in_node, deriv_folder_list, [
            ('melodic_out_dir', 'in1')
        ]),

        (deriv_folder_list, deriv_sorter, [('out', 'folder_list')]),

    ])

    return derivatives_wf
