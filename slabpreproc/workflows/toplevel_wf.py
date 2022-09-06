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
from .template_reg_wf import build_template_reg_wf
from .derivatives_wf import build_derivatives_wf

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

# from nipype import config
# config.enable_debug_mode()


def build_toplevel_wf(work_dir, deriv_dir, layout):
    """
    Build main subcortical QC workflow

    :param work_dir: str
        Path to working directory
    :param deriv_dir: str
        Path to derivatives directory
    :param layout: BIDSLayout
        Prefilled layout for BIDS dataset
    :return:
    """

    # Input node setup
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold', 'bold_meta',
                'sbref', 'sbref_meta',
                'seepis', 'seepis_meta',
                'tpl_t1_head', 'tpl_t2_head',
                'tpl_pseg', 'tpl_dseg', 'tpl_bmask'
            ]
        ),
        name='inputs'
    )

    # Sub-workflows setup
    func_preproc_wf = build_func_preproc_wf()
    template_reg_wf = build_template_reg_wf()
    qc_wf = build_qc_wf()
    derivatives_wf = build_derivatives_wf(deriv_dir)

    # Workflow
    toplevel_wf = pe.Workflow(
        base_dir=str(work_dir),
        name='toplevel_wf'
    )

    toplevel_wf.connect([

        # Func preproc inputs
        (inputs, func_preproc_wf, [
            ('bold', 'inputs.bold'),
            ('sbref', 'inputs.sbref'),
            ('seepis', 'inputs.seepis'),
            ('bold_meta', 'inputs.bold_meta'),
            ('sbref_meta', 'inputs.sbref_meta'),
            ('seepis_meta', 'inputs.seepis_meta')
        ]),

        # Pass T2w individual template to registration workflow
        (inputs, template_reg_wf, [('tpl_t2_head', 'inputs.tpl_t2_head')]),

        # Pass preprocessed (motion and distortion corrected) BOLD and SBRef
        # to template registration workflow
        (func_preproc_wf, template_reg_wf, [
            ('outputs.sbref_preproc', 'inputs.sbref_preproc'),
            ('outputs.bold_preproc', 'inputs.bold_preproc'),
            ('outputs.seepi_unwarp_mean', 'inputs.seepi_unwarp_mean')
        ]),

        # Connect QC workflow
        (template_reg_wf, qc_wf, [('outputs.tpl_bold_preproc', 'inputs.bold')]),
        (inputs, qc_wf, [('tpl_dseg', 'inputs.labels')]),

        # Connect derivatives outputs
        (inputs, derivatives_wf, [('bold', 'inputs.source_file')]),
        (func_preproc_wf, derivatives_wf, [('outputs.moco_pars', 'inputs.moco_pars')]),

        # Write individual template-space results to derivatives folder
        (template_reg_wf, derivatives_wf, [
            ('outputs.tpl_bold_preproc', 'inputs.tpl_bold_preproc'),
            ('outputs.tpl_sbref_preproc', 'inputs.tpl_sbref_preproc'),
            ('outputs.tpl_seepi_unwarp_mean', 'inputs.tpl_seepi_unwarp_mean'),
        ]),

        # Write QC results to derivatives folder
        (qc_wf, derivatives_wf, [
            ('outputs.bold_tmean', 'inputs.tpl_bold_tmean'),
            ('outputs.bold_tsd', 'inputs.tpl_bold_tsd'),
            ('outputs.bold_detrended', 'inputs.tpl_bold_detrended'),
            ('outputs.bold_tsfnr', 'inputs.tpl_bold_tsfnr'),
            ('outputs.bold_tsfnr_roistats', 'inputs.tpl_bold_tsfnr_roistats')
        ]),

        # Write atlas images and templates to derivatives folder
        (inputs, derivatives_wf, [
            ('tpl_t1_head', 'inputs.tpl_t1_head'),
            ('tpl_t2_head', 'inputs.tpl_t2_head'),
            ('tpl_pseg', 'inputs.tpl_pseg'),
            ('tpl_dseg', 'inputs.tpl_dseg'),
            ('tpl_bmask', 'inputs.tpl_bmask')
        ])
    ])

    # Optional: plot main workflows to sandbox

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
