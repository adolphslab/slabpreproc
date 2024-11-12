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
    inputnode = pe.Node(
        util.IdentityInterface(
            fields=[
                'source_file',
                'tpl_bold_mag_preproc',
                'tpl_bold_phs_preproc',
                'tpl_bold_dphi_preproc',
                'tpl_seepiref_preproc',
                'tpl_sbref_preproc',
                'tpl_bold_mag_tmean',
                'tpl_bold_mag_tsd',
                'tpl_bold_mag_tsfnr',
                'tpl_bold_mag_tsfnr_roistats',
                'tpl_topup_b0_rads',
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
        name='inputnode'
    )

    # File sorting dictionary list
    file_sort_dicts = [
        {'DataType': 'preproc', 'NewSuffix': 'recon-preproc_part-mag_bold', 'FileType': 'Image'},
        {'DataType': 'preproc', 'NewSuffix': 'recon-preproc_part-phase_bold', 'FileType': 'Image'},
        {'DataType': 'preproc', 'NewSuffix': 'recon-preproc_part-phasediff_bold', 'FileType': 'Image'},
        {'DataType': 'preproc', 'NewSuffix': 'recon-preproc_part-mag_seepi', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tmean_part-mag_bold', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tsd_part-mag_bold', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tsfnr_part-mag_bold', 'FileType': 'Image'},
        {'DataType': 'qc', 'NewSuffix': 'recon-tsfnr_part-mag_bold_roistats', 'FileType': 'Text'},
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

    # Create a list of all file inputnode
    deriv_file_list = pe.Node(
        util.Merge(numinputs=len(file_sort_dicts)),
        name='deriv_file_list'
    )

    # Create a list of all folder inputnode
    deriv_folder_list = pe.Node(
        util.Merge(numinputs=len(folder_sort_dicts)),
        name='deriv_folder_list'
    )

    # Build multi-input derivatives output sorter
    # Renames and sorts inputnode into correct derivatives hierarchy
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
        (inputnode, deriv_sorter, [('source_file', 'source_file')]),

        # Create file list and pass to sorter
        (inputnode, deriv_file_list, [
            ('tpl_bold_mag_preproc', 'in1'),
            ('tpl_bold_phs_preproc', 'in2'),
            ('tpl_bold_dphi_preproc', 'in3'),
            ('tpl_seepiref_preproc', 'in4'),
            ('tpl_sbref_preproc', 'in5'),
            ('tpl_bold_mag_tmean', 'in6'),
            ('tpl_bold_mag_tsd', 'in7'),
            ('tpl_bold_mag_tsfnr', 'in8'),
            ('tpl_bold_mag_tsfnr_roistats', 'in9'),
            ('tpl_topup_b0_rads', 'in10'),
            ('tpl_dropout', 'in11'),
            ('motion_csv', 'in12'),
            ('tpl_t1w_head', 'in13'),
            ('tpl_t2w_head', 'in14'),
            ('tpl_t1w_brain', 'in15'),
            ('tpl_t2w_brain', 'in16'),
            ('tpl_pseg', 'in17'),
            ('tpl_dseg', 'in18'),
            ('tpl_bmask', 'in19')
        ]),

        (deriv_file_list, deriv_sorter, [('out', 'file_list')]),

        # Create folder list and pass to sorter
        (inputnode, deriv_folder_list, [
            ('melodic_out_dir', 'in1')
        ]),

        (deriv_folder_list, deriv_sorter, [('out', 'folder_list')]),

    ])

    return derivatives_wf
