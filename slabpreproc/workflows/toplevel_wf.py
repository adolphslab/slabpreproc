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
from .func_surf_wf import build_func_surf_wf
from .template_reg_wf import build_template_reg_wf
from .derivatives_wf import build_derivatives_wf
from .melodic_wf import build_melodic_wf
from ..interfaces.summaryreport import SummaryReport

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

# from nipype import config
# config.enable_debug_mode()


def build_toplevel_wf(work_dir, deriv_dir, bold_mag_meta, nthreads=2):
    """
    Build main subcortical QC workflow

    :param work_dir: str
        Path to working directory
    :param deriv_dir: str
        Path to derivatives directory
    :param bold_mag_meta: dict
        BOLD magnitude image metadata
    :param nthreads: int
        Maximum number of threads allowed
    :return:
    """

    # Input node setup
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold_mag',
                'bold_mag_meta',
                'sbref',
                'sbref_meta',
                'seepis',
                'seepis_meta',
                'tpl_t1_head',
                'tpl_t2_head',
                'tpl_pseg',
                'tpl_dseg',
                'tpl_bmask'
            ]
        ),
        name='inputs'
    )

    # Extract TR in seconds from metadata for melodic
    tr_s = bold_mag_meta['RepetitionTime']

    # Sub-workflows setup
    func_preproc_wf = build_func_preproc_wf(nthreads=nthreads)
    func_surf_wf = build_func_surf_wf()
    template_reg_wf = build_template_reg_wf(nthreads=nthreads)
    qc_wf = build_qc_wf(nthreads=nthreads)
    melodic_wf = build_melodic_wf(tr_s=tr_s)
    derivatives_wf = build_derivatives_wf(deriv_dir)

    # Summary report node
    summary_report = pe.Node(
        SummaryReport(deriv_dir=deriv_dir),
        overwrite=True,  # Always regenerate report
        name='summary_report'
    )

    # Workflow
    toplevel_wf = pe.Workflow(
        base_dir=str(work_dir),
        name='toplevel_wf'
    )

    toplevel_wf.connect([

        # Func preproc inputs
        (inputs, func_preproc_wf, [
            ('bold_mag', 'inputs.bold_mag'),
            ('bold_mag_meta', 'inputs.bold_mag_meta'),
            ('sbref', 'inputs.sbref'),
            ('sbref_meta', 'inputs.sbref_meta'),
            ('seepis', 'inputs.seepis'),
            ('seepis_meta', 'inputs.seepis_meta')
        ]),

        # Pass T2w individual template to registration workflow
        (inputs, template_reg_wf, [('tpl_t2_head', 'inputs.tpl_t2_head')]),

        # Pass preprocessed (motion and distortion corrected) BOLD and SBRef images
        # to template registration workflow
        (func_preproc_wf, template_reg_wf, [
            ('outputs.sbref_preproc', 'inputs.sbref_preproc'),
            ('outputs.bold_mag_preproc', 'inputs.bold_mag_preproc'),
            ('outputs.seepi_unwarp_mean', 'inputs.seepi_unwarp_mean'),
            ('outputs.topup_b0_rads', 'inputs.topup_b0_rads')
        ]),

        # Connect QC workflow
        (inputs, qc_wf, [
            ('bold_mag_meta', 'inputs.bold_mag_meta'),
            ('tpl_dseg', 'inputs.tpl_dseg'),
            ('tpl_bmask', 'inputs.tpl_bmask')
        ]),
        (func_preproc_wf, qc_wf, [
            ('outputs.moco_pars', 'inputs.moco_pars')
        ]),
        (template_reg_wf, qc_wf, [
            ('outputs.tpl_bold_mag_preproc', 'inputs.tpl_bold_mag_preproc'),
            ('outputs.tpl_seepi_unwarp_mean', 'inputs.tpl_mean_seepi'),
            ('outputs.tpl_sbref_preproc', 'inputs.tpl_bold_sbref'),
            ('outputs.tpl_b0_rads', 'inputs.tpl_b0_rads')
        ]),

        # Connect melodic workflow
        (template_reg_wf, melodic_wf, [
            ('outputs.tpl_bold_mag_preproc', 'inputs.tpl_bold_mag_preproc'),
        ]),
        (inputs, melodic_wf, [
            ('tpl_t1_head', 'inputs.tpl_t1_head'),
            ('tpl_bmask', 'inputs.tpl_bmask')
        ]),
        (qc_wf, melodic_wf, [
            ('outputs.tpl_bold_tmean', 'inputs.tpl_bold_tmean'),
        ]),

        # Connect derivatives outputs
        (inputs, derivatives_wf, [
            ('bold_mag', 'inputs.source_file'),
        ]),

        # Write individual template-space results to derivatives folder
        (template_reg_wf, derivatives_wf, [
            ('outputs.tpl_bold_mag_preproc', 'inputs.tpl_bold_mag_preproc'),
            ('outputs.tpl_sbref_preproc', 'inputs.tpl_sbref_preproc'),
            ('outputs.tpl_seepi_unwarp_mean', 'inputs.tpl_seepi_unwarp_mean'),
            ('outputs.tpl_b0_rads', 'inputs.tpl_b0_rads')
        ]),

        # Write QC results to derivatives folder
        (qc_wf, derivatives_wf, [
            ('outputs.tpl_bold_tmean', 'inputs.tpl_bold_tmean'),
            ('outputs.tpl_bold_tsd', 'inputs.tpl_bold_tsd'),
            ('outputs.tpl_bold_tsfnr', 'inputs.tpl_bold_tsfnr'),
            ('outputs.tpl_bold_tsfnr_roistats', 'inputs.tpl_bold_tsfnr_roistats'),
            ('outputs.tpl_dropout', 'inputs.tpl_dropout'),
            ('outputs.motion_csv', 'inputs.motion_csv')
        ]),

        # Write MELODIC results to derivatives folder
        (melodic_wf, derivatives_wf, [
            ('outputs.out_dir', 'inputs.melodic_out_dir'),
        ]),

        # Write atlas images and templates to derivatives folder
        (inputs, derivatives_wf, [
            ('tpl_t1_head', 'inputs.tpl_t1_head'),
            ('tpl_t2_head', 'inputs.tpl_t2_head'),
            ('tpl_pseg', 'inputs.tpl_pseg'),
            ('tpl_dseg', 'inputs.tpl_dseg'),
            ('tpl_bmask', 'inputs.tpl_bmask')
        ]),

        # Summary report
        (inputs, summary_report, [
            ('bold_mag', 'source_bold'),
            ('bold_mag_meta', 'source_bold_meta'),
            ('tpl_t1_head', 't1head'),
            ('tpl_t2_head', 't2head'),
            ('tpl_dseg', 'labels'),
        ]),
        (template_reg_wf, summary_report, [
            ('outputs.tpl_seepi_unwarp_mean', 'mseepi'),
            ('outputs.tpl_b0_rads', 'b0_rads'),
        ]),
        (qc_wf, summary_report, [
            ('outputs.tpl_bold_tmean', 'tmean'),
            ('outputs.tpl_bold_tsfnr', 'tsfnr'),
            ('outputs.tpl_dropout', 'dropout'),
            ('outputs.motion_csv', 'motion_csv')
        ])
    ])

    # Optional: plot main workflows

    graph_dir = "slabpreproc_graphs"
    os.makedirs(graph_dir, exist_ok=True)

    func_preproc_wf.write_graph(
        graph2use='colored',
        dotfilename=os.path.join(graph_dir, 'func_preproc_wf.dot')
    )

    template_reg_wf.write_graph(
        graph2use='colored',
        dotfilename=os.path.join(graph_dir, 'template_reg_wf.dot')
    )

    qc_wf.write_graph(
        graph2use='colored',
        dotfilename=os.path.join(graph_dir, 'qc_wf.dot')
    )

    return toplevel_wf
