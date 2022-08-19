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

from .qc_wf import build_qc_wf
from .func_preproc_wf import build_func_preproc_wf
from .template_wf import build_template_wf
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
                'fmaps', 'fmaps_meta',
                'tpl_t1_brain',
                'tpl_t2_brain',
                'tpl_labels'
            ]
        ),
        name='inputs'
    )

    # Sub-workflows setup
    func_preproc_wf = build_func_preproc_wf()
    template_wf = build_template_wf()
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
            ('fmaps', 'inputs.fmaps'),
            ('bold_meta', 'inputs.bold_meta'),
            ('sbref_meta', 'inputs.sbref_meta'),
            ('fmaps_meta', 'inputs.fmaps_meta')
        ]),

        # Pass T2w individual template to registration workflow
        (inputs, template_wf, [
            ('tpl_t2_brain', 'inputs.tpl_t2_brain'),
        ]),

        # Pass preprocessed (motion and distortion corrected) BOLD and SBRef
        # to template registration workflow
        (func_preproc_wf, template_wf, [
            ('outputs.sbref_preproc', 'inputs.sbref_preproc'),
            ('outputs.bold_preproc', 'inputs.bold_preproc'),
            ('outputs.seepi_ref', 'inputs.seepi_ref')
        ]),

        # Pass fMRI preproc results to QC workflow
        (func_preproc_wf, qc_wf, [('outputs.bold_preproc', 'inputs.bold')]),

        # Pass template labels to QC workflow
        (inputs, qc_wf, [('tpl_labels', 'inputs.labels')]),

        # Pass original BOLD filename as source file for derivatives output filenaming
        (inputs, derivatives_wf, [
            ('bold', 'inputs.source_file')
        ]),

        # Write preproc correction results to derivatives folder
        (func_preproc_wf, derivatives_wf, [
            ('outputs.moco_pars', 'inputs.moco_pars')
        ]),

        # Write individual template space results to derivatives folder
        (template_wf, derivatives_wf, [
            ('outputs.tpl_bold_preproc', 'inputs.tpl_bold_preproc'),
            ('outputs.tpl_sbref_preproc', 'inputs.tpl_sbref_preproc'),
            ('outputs.tpl_seepi_ref', 'inputs.tpl_seepi_ref'),
        ]),

        # Write QC results to derivatives folder
        (qc_wf, derivatives_wf, [
            ('outputs.bold_tmean', 'inputs.tpl_bold_tmean'),
            ('outputs.bold_tsd', 'inputs.tpl_bold_tsd'),
            ('outputs.bold_detrended', 'inputs.tpl_bold_detrended'),
            ('outputs.bold_tsfnr', 'inputs.tpl_bold_tsfnr'),
            ('outputs.bold_tsfnr_roistats', 'inputs.tpl_bold_tsfnr_roistats')
        ]),
    ])

    return toplevel_wf
