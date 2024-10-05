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
from .resamp_epi2tpl_wf import build_resamp_epi2tpl_wf
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
    in_node = pe.Node(
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
                'seepi_mag',
                'seepi_phs',
                'seepi_meta',
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
        name='in_node'
    )

    # Extract TR in seconds from metadata for melodic
    tr_s = bold_meta['RepetitionTime']

    # Sub-workflows setup
    func_preproc_wf = build_func_preproc_wf(antsthreads=antsthreads)
    resamp_epi2tpl_wf = build_resamp_epi2tpl_wf(antsthreads=antsthreads)
    qc_wf = build_qc_wf()
    derivatives_wf = build_derivatives_wf(deriv_dir)

    if melodic:
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

        # Func workflow in_node
        (in_node, func_preproc_wf, [
            ('bold_mag',   'in_node.bold_mag'),
            ('bold_phs',   'in_node.bold_phs'),
            ('bold_meta',  'in_node.bold_meta'),
            ('sbref_mag',  'in_node.sbref_mag'),
            ('sbref_phs',  'in_node.sbref_phs'),
            ('sbref_meta', 'in_node.sbref_meta'),
            ('seepi_mag',   'in_node.seepi_mag'),
            ('seepi_phs',   'in_node.seepi_phs'),
            ('seepi_meta',  'in_node.seepi_meta')
        ]),

        # Pass session to template T2 registration info
        (in_node, resamp_epi2tpl_wf, [
            ('ses_t2w_head', 'in_node.ses_t2w_head'),
            ('tpl_t2w_head', 'in_node.tpl_t2w_head'),
        ]),

        # Pass preprocessed (motion and distortion corrected) BOLD and SBRef images
        # to template registration workflow
        (func_preproc_wf, resamp_epi2tpl_wf, [
            ('out_node.bold_re_preproc', 'in_node.bold_re_preproc'),
            ('out_node.bold_im_preproc', 'in_node.bold_im_preproc'),
            ('out_node.sbref_mag_preproc', 'in_node.sbref_mag_preproc'),
            ('out_node.seepi_mag_preproc', 'in_node.seepi_mag_preproc'),
            ('out_node.topup_b0_rads', 'in_node.topup_b0_rads')
        ]),

        # Connect QC workflow
        (in_node, qc_wf, [
            ('bold_meta', 'in_node.bold_meta'),
            ('tpl_dseg', 'in_node.tpl_dseg'),
            ('tpl_bmask', 'in_node.tpl_bmask')
        ]),
        (func_preproc_wf, qc_wf, [
            ('out_node.moco_pars', 'in_node.moco_pars')
        ]),
        (resamp_epi2tpl_wf, qc_wf, [
            ('out_node.tpl_bold_mag_preproc', 'in_node.tpl_bold_mag_preproc'),
            ('out_node.tpl_sbref_mag_preproc', 'in_node.tpl_sbref_mag_preproc'),
            ('out_node.tpl_seepi_mag_preproc', 'in_node.tpl_seepi_mag_preproc'),
            ('out_node.tpl_b0_rads', 'in_node.tpl_b0_rads')
        ]),

        # Connect derivatives outputs
        (in_node, derivatives_wf, [('bold_mag', 'in_node.source_file')]),

        # Write individual template-space results to derivatives folder
        (resamp_epi2tpl_wf, derivatives_wf, [
            ('out_node.tpl_bold_mag_preproc', 'in_node.tpl_bold_mag_preproc'),
            ('out_node.tpl_bold_phs_preproc', 'in_node.tpl_bold_phs_preproc'),
            ('out_node.tpl_sbref_mag_preproc', 'in_node.tpl_sbref_mag_preproc'),
            ('out_node.tpl_seepi_mag_preproc', 'in_node.tpl_seepi_mag_preproc'),
            ('out_node.tpl_b0_rads', 'in_node.tpl_b0_rads')
        ]),

        # Write fsnative surface resampled BOLD to derivatives
        # (surface_wf, derivatives_wf, ['outputnode.surfaces', 'in_node.']),

        # Write QC results to derivatives folder
        (qc_wf, derivatives_wf, [
            ('out_node.tpl_bold_mag_tmean', 'in_node.tpl_bold_mag_tmean'),
            ('out_node.tpl_bold_mag_tsd', 'in_node.tpl_bold_mag_tsd'),
            ('out_node.tpl_bold_mag_tsfnr', 'in_node.tpl_bold_mag_tsfnr'),
            ('out_node.tpl_bold_mag_tsfnr_roistats', 'in_node.tpl_bold_mag_tsfnr_roistats'),
            ('out_node.tpl_dropout', 'in_node.tpl_dropout'),
            ('out_node.motion_csv', 'in_node.motion_csv')
        ]),

        # Write atlas images and templates to derivatives folder
        (in_node, derivatives_wf, [
            ('tpl_t1w_head', 'in_node.tpl_t1w_head'),
            ('tpl_t2w_head', 'in_node.tpl_t2w_head'),
            ('tpl_t1w_brain', 'in_node.tpl_t1w_brain'),
            ('tpl_t2w_brain', 'in_node.tpl_t2w_brain'),
            ('tpl_pseg', 'in_node.tpl_pseg'),
            ('tpl_dseg', 'in_node.tpl_dseg'),
            ('tpl_bmask', 'in_node.tpl_bmask')
        ]),

        # Summary report
        (in_node, summary_report, [
            ('bold_mag', 'source_bold'),
            ('bold_meta', 'source_bold_meta'),
            ('tpl_t1w_head', 't1w_head'),
            ('tpl_t2w_head', 't2w_head'),
            ('tpl_dseg', 'labels'),
        ]),
        (resamp_epi2tpl_wf, summary_report, [
            ('out_node.tpl_seepi_mag_preproc', 'mseepi'),
            ('out_node.tpl_b0_rads', 'b0_rads'),
        ]),
        (qc_wf, summary_report, [
            ('out_node.tpl_bold_mag_tmean', 'tmean'),
            ('out_node.tpl_bold_mag_tsfnr', 'tsfnr'),
            ('out_node.tpl_dropout', 'dropout'),
            ('out_node.motion_csv', 'motion_csv')
        ])
    ])

    # Optional melodic ICA
    if melodic:

        func_wf.connect([

            # Connect melodic workflow
            (resamp_epi2tpl_wf, melodic_wf, [('out_node.tpl_bold_mag_preproc', 'in_node.tpl_bold_mag_preproc')]),
            (in_node, melodic_wf, [
                ('tpl_t1w_head', 'in_node.tpl_t1w_head'),
                ('tpl_bmask', 'in_node.tpl_bmask')
            ]),
            (qc_wf, melodic_wf, [('out_node.tpl_bold_mag_tmean', 'in_node.tpl_bold_mag_tmean')]),

            # Write MELODIC results to derivatives folder
            (melodic_wf, derivatives_wf, [
                ('out_node.out_dir', 'in_node.melodic_out_dir'),
            ])

        ])

    # Plot main workflows
    graph_dir = "slabpreproc_graphs"
    os.makedirs(graph_dir, exist_ok=True)

    func_preproc_wf.write_graph(
        graph2use='colored',
        dotfilename=os.path.join(graph_dir, 'func_preproc_wf.dot')
    )

    resamp_epi2tpl_wf.write_graph(
        graph2use='colored',
        dotfilename=os.path.join(graph_dir, 'resamp_epi2tpl_wf.dot')
    )

    qc_wf.write_graph(
        graph2use='colored',
        dotfilename=os.path.join(graph_dir, 'qc_wf.dot')
    )

    return func_wf
