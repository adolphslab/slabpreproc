#!/usr/bin/env python
# coding: utf-8

"""
Build the top level slab preprocessing workflow

AUTHOR : Mike Tyszka
PLACE  : Caltech Brain Imaging Center

MIT License

Copyright (c) 2022 Mike Tyszka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os

from .qc_wf import build_qc_wf
from .func_preproc_wf import build_func_preproc_wf
from .derivatives_wf import build_derivatives_wf
from .melodic_wf import build_melodic_wf

# WIP
# from .surface_wf import build_surface_wf

from ..interfaces.summaryreport import SummaryReport

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe


def build_func_wf(bold_work_dir, deriv_dir, bold_meta, melodic=False, antsthreads=2):
    """
    Build main subcortical QC workflow

    :param bold_work_dir: str
        Path to working directory for this BOLD series
    :param deriv_dir: str
        Path to derivatives directory
    :param bold_meta: dict
        BOLD magnitude image metadata
    :param melodic: bool
        Melodic ICA run flag
    :param antsthreads: int
        Maximum number of threads allowed
    :return:
    """

    # Workflow input node
    inputnode = pe.Node(
        util.IdentityInterface(
            fields=[
                'subject_id',
                'fs_subjects_dir',
                'bold_mag',
                'bold_phs',
                'bold_meta',
                'sbref_mag',
                'sbref_phs',
                'sbref_meta',
                'seepi_mag_list',
                'seepi_phs_list',
                'seepi_meta_list',
                'tpl_t1w_head',
                'tpl_t2w_head',
                'tpl_t1w_brain',
                'tpl_t2w_brain',
                'tpl_pseg',
                'tpl_dseg',
                'tpl_bmask',
                'fs_t1w_head',
                'ses_t2w_head',
            ]
        ),
        name='inputnode'
    )

    # Extract TR in seconds from metadata for melodic
    tr_s = bold_meta['RepetitionTime']

    # Sub-workflows setup
    func_preproc_wf = build_func_preproc_wf(antsthreads=antsthreads)
    qc_wf = build_qc_wf()
    derivatives_wf = build_derivatives_wf(deriv_dir)

    # Optional MELODIC ICA workflow
    melodic_wf = build_melodic_wf(tr_s=tr_s)

    # Summary report node
    summary_report = pe.Node(
        SummaryReport(deriv_dir=deriv_dir),
        overwrite=True,  # Always regenerate report
        name='summary_report'
    )

    # Workflow
    func_wf = pe.Workflow(
        base_dir=str(bold_work_dir),
        name='func_wf'
    )

    func_wf.connect([

        # Func workflow inputnode
        (inputnode, func_preproc_wf, [
            ('bold_mag',   'inputnode.bold_mag'),
            ('bold_phs',   'inputnode.bold_phs'),
            ('bold_meta',  'inputnode.bold_meta'),
            ('sbref_mag',  'inputnode.sbref_mag'),
            ('sbref_phs',  'inputnode.sbref_phs'),
            ('sbref_meta', 'inputnode.sbref_meta'),
            ('seepi_mag_list',   'inputnode.seepi_mag_list'),
            ('seepi_phs_list',   'inputnode.seepi_phs_list'),
            ('seepi_meta_list',  'inputnode.seepi_meta_list')
        ]),

        # Pass session to template T2 registration info
        (inputnode, func_preproc_wf, [
            ('ses_t2w_head', 'inputnode.ses_t2w_head'),
            ('tpl_t2w_head', 'inputnode.tpl_t2w_head'),
        ]),

        # Connect QC workflow
        (inputnode, qc_wf, [
            ('bold_meta', 'inputnode.bold_meta'),
            ('tpl_dseg', 'inputnode.tpl_dseg'),
            ('tpl_bmask', 'inputnode.tpl_bmask')
        ]),
        (func_preproc_wf, qc_wf, [
            ('outputnode.moco_pars', 'inputnode.moco_pars')
        ]),
        (func_preproc_wf, qc_wf, [
            ('outputnode.tpl_bold_mag_preproc', 'inputnode.tpl_bold_mag_preproc'),
            ('outputnode.tpl_epi_ref_preproc', 'inputnode.tpl_epi_ref_preproc'),
            ('outputnode.tpl_topup_b0_rads', 'inputnode.tpl_topup_b0_rads')
        ]),

        # Connect derivatives outputs
        (inputnode, derivatives_wf, [('bold_mag', 'inputnode.source_file')]),

        # Write individual template-space results to derivatives folder
        (func_preproc_wf, derivatives_wf, [
            ('outputnode.tpl_bold_mag_preproc', 'inputnode.tpl_bold_mag_preproc'),
            ('outputnode.tpl_bold_phs_preproc', 'inputnode.tpl_bold_phs_preproc'),
            ('outputnode.tpl_epi_ref_preproc', 'inputnode.tpl_epi_ref_preproc'),
            ('outputnode.tpl_topup_b0_rads', 'inputnode.tpl_topup_b0_rads')
        ]),

        # Write fsnative surface resampled BOLD to derivatives
        # (surface_wf, derivatives_wf, ['outputnode.surfaces', 'inputnode.']),

        # Write QC results to derivatives folder
        (qc_wf, derivatives_wf, [
            ('outputnode.tpl_bold_mag_tmean', 'inputnode.tpl_bold_mag_tmean'),
            ('outputnode.tpl_bold_mag_tsd', 'inputnode.tpl_bold_mag_tsd'),
            ('outputnode.tpl_bold_mag_tsfnr', 'inputnode.tpl_bold_mag_tsfnr'),
            ('outputnode.tpl_bold_mag_tsfnr_roistats', 'inputnode.tpl_bold_mag_tsfnr_roistats'),
            ('outputnode.tpl_dropout', 'inputnode.tpl_dropout'),
            ('outputnode.motion_csv', 'inputnode.motion_csv')
        ]),

        # Write atlas images and templates to derivatives folder
        (inputnode, derivatives_wf, [
            ('tpl_t1w_head', 'inputnode.tpl_t1w_head'),
            ('tpl_t2w_head', 'inputnode.tpl_t2w_head'),
            ('tpl_t1w_brain', 'inputnode.tpl_t1w_brain'),
            ('tpl_t2w_brain', 'inputnode.tpl_t2w_brain'),
            ('tpl_pseg', 'inputnode.tpl_pseg'),
            ('tpl_dseg', 'inputnode.tpl_dseg'),
            ('tpl_bmask', 'inputnode.tpl_bmask')
        ]),

        # Summary report
        (inputnode, summary_report, [
            ('bold_mag', 'source_bold'),
            ('bold_meta', 'source_bold_meta'),
            ('tpl_t1w_head', 't1w_head'),
            ('tpl_t2w_head', 't2w_head'),
            ('tpl_dseg', 'labels'),
        ]),
        (func_preproc_wf, summary_report, [
            ('outputnode.tpl_epi_ref_preproc', 'mseepi'),
            ('outputnode.tpl_topup_b0_rads', 'topup_b0_rads'),
        ]),
        (qc_wf, summary_report, [
            ('outputnode.tpl_bold_mag_tmean', 'tmean'),
            ('outputnode.tpl_bold_mag_tsfnr', 'tsfnr'),
            ('outputnode.tpl_dropout', 'dropout'),
            ('outputnode.motion_csv', 'motion_csv')
        ])
    ])

    # Optional melodic ICA
    if melodic:

        func_wf.connect([

            # Connect melodic workflow
            (func_preproc_wf, melodic_wf, [('outputnode.tpl_bold_mag_preproc', 'inputnode.tpl_bold_mag_preproc')]),
            (inputnode, melodic_wf, [
                ('tpl_t1w_head', 'inputnode.tpl_t1w_head'),
                ('tpl_bmask', 'inputnode.tpl_bmask')
            ]),
            (qc_wf, melodic_wf, [('outputnode.tpl_bold_mag_tmean', 'inputnode.tpl_bold_mag_tmean')]),

            # Write MELODIC results to derivatives folder
            (melodic_wf, derivatives_wf, [
                ('outputnode.out_dir', 'inputnode.melodic_out_dir'),
            ])

        ])

    # Plot main workflows
    graph_dir = "slabpreproc_graphs"
    os.makedirs(graph_dir, exist_ok=True)

    func_preproc_wf.write_graph(
        graph2use='colored',
        dotfilename=os.path.join(graph_dir, 'func_preproc_wf.dot')
    )

    qc_wf.write_graph(
        graph2use='colored',
        dotfilename=os.path.join(graph_dir, 'qc_wf.dot')
    )

    return func_wf
