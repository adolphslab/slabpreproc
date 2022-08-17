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

from .qc import build_wf_qc
from .func_preproc import build_wf_func_preproc
from .template import build_wf_template
from .derivatives import build_wf_derivatives

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

# from nipype import config
# config.enable_debug_mode()


def build_wf_toplevel(work_dir, deriv_dir, layout):
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

    # Subcortical QC workflow input node
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold', 'bold_meta',
                'sbref', 'sbref_meta',
                'fmaps', 'fmaps_meta',
                'ind_t1_brain', 'ind_t2_brain', 'ind_labels'
            ]
        ),
        name='inputs'
    )

    # Build sub-workflows
    wf_func_preproc = build_wf_func_preproc()
    wf_template = build_wf_template()
    wf_qc = build_wf_qc()
    wf_derivatives = build_wf_derivatives(deriv_dir)

    # Top level slab preproc workflow
    wf_toplevel = pe.Workflow(
        base_dir=str(work_dir),
        name='wf_toplevel'
    )

    wf_toplevel.connect([

        # Func preproc inputs
        (inputs, wf_func_preproc, [
            ('bold', 'inputs.bold'),
            ('sbref', 'inputs.sbref'),
            ('fmaps', 'inputs.fmaps'),
            ('bold_meta', 'inputs.bold_meta'),
            ('sbref_meta', 'inputs.sbref_meta'),
            ('fmaps_meta', 'inputs.fmaps_meta')
        ]),

        # Pass T2w individual template to registration workflow
        (inputs, wf_template, [
            ('ind_t2_brain', 'inputs.ind_t2_brain'),
        ]),

        # Pass preprocessed (motion and distortion corrected) BOLD and SBRef
        # to template registration workflow
        (wf_func_preproc, wf_template, [
            ('outputs.sbref_preproc', 'inputs.sbref_preproc'),
            ('outputs.bold_preproc', 'inputs.bold_preproc')
        ]),

        # Pass fMRI preproc results to QC workflow
        (wf_func_preproc, wf_qc, [
            ('outputs.bold', 'inputs.bold')
        ]),

        # Pass template labels to QC workflow
        (wf_template, wf_qc, [
            ('tpl_labels', 'inputs.tpl_labels')
        ]),

        # Pass original BOLD filename as source file for derivatives output filenaming
        (inputs, wf_derivatives, [
            ('bold', 'inputs.source_file')
        ]),

        # Write preproc correction results to derivatives folder
        (wf_func_preproc, wf_derivatives, [
            ('outputs.moco_pars', 'inputs.moco_pars')
        ]),

        # Write individual template space results to derivatives folder
        (wf_template, wf_derivatives, [
            ('outputs.tpl_bold', 'inputs.tpl_bold'),
            ('outputs.tpl_sbref', 'inputs.tpl_sbref'),
        ]),

        # Write QC results to derivatives folder
        (wf_qc, wf_derivatives, [
            ('outputs.tpl_bold_tmean', 'inputs.tpl_bold_tmean'),
            ('outputs.tpl_bold_tsd', 'inputs.tpl_bold_tsd'),
            ('outputs.tpl_bold_detrended', 'inputs.tpl_bold_detrended'),
            ('outputs.tpl_bold_tsfnr', 'inputs.tpl_bold_tsfnr'),
            ('outputs.tpl_bold_tsfnr_roistats', 'inputs.tpl_bold_tsfnr_roistats')
        ]),
    ])

    return wf_toplevel
