#!/usr/bin/env python
# coding: utf-8

"""
Build the subcortical QC workflow

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

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
from wf_qc import build_wf_qc
from wf_func_preproc import build_wf_func_preproc
from wf_atlas import build_wf_atlas
from niworkflows.interfaces.bids import DerivativesDataSink

# Internal package imports
from wf_derivatives import build_wf_derivatives


def build_wf_scqc(work_dir, deriv_dir):
    """
    Build main subcortical QC workflow

    :param work_dir: str
        Path to working directory
    :param deriv_dir: str
        Path to derivatives directory
    :return:
    """

    # Subcortical QC workflow input node
    inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold',
                't1_ind',
                't1_atlas',
                'labels_atlas'
            ]),
        name='inputs'
    )

    # Build sub workflows
    wf_func_preproc = build_wf_func_preproc()
    wf_atlas = build_wf_atlas()
    wf_qc = build_wf_qc()
    wf_derivatives = build_wf_derivatives(deriv_dir)

    # Main subcortical QC workflow
    wf_scqc = pe.Workflow(
        base_dir=str(work_dir),
        name='wf_scqc'
    )

    wf_scqc.connect([

        # Pass images to preproc and atlas workflow
        (inputs, wf_func_preproc, [('bold', 'inputs.bold')]),
        (inputs, wf_atlas, [
            ('t1_ind', 'inputs.t1_ind'),
            ('t1_atlas', 'inputs.t1_atlas'),
            ('labels_atlas', 'inputs.labels_atlas')
        ]),

        # Pass fMRI preproc results to atlas workflow
        (wf_func_preproc, wf_atlas, [('outputs.sbref', 'inputs.epi_ind')]),

        # Pass fMRI preproc results to QC workflow
        (wf_func_preproc, wf_qc, [('outputs.bold', 'inputs.bold')]),

        # Pass original BOLD filename as source file for derivatives output filenaming
        (inputs, wf_derivatives, [('bold', 'inputs.source_file')]),

        # Write preproc results to derivatives folder
        (wf_func_preproc, wf_derivatives, [
            ('outputs.bold', 'inputs.bold_preproc'),
            ('outputs.sbref', 'inputs.sbref_preproc'),
        ]),

        # Write EPI-space atlas results to derivatives folder
        (wf_atlas, wf_derivatives, [
            ('outputs.t1_atlas2ind', 'inputs.t1_atlas2ind'),
            ('outputs.t1_ind2epi', 'inputs.t1_ind2epi'),
            ('outputs.t1_atlas2epi', 'inputs.t1_atlas2epi'),
            ('outputs.labels_atlas2epi', 'inputs.labels_atlas2epi'),
        ]),

        # Write QC results to derivatives folder
        (wf_qc, wf_derivatives, [
            ('outputs.bold_tmean', 'inputs.bold_tmean'),
            ('outputs.bold_tsd', 'inputs.bold_tsd'),
            ('outputs.bold_detrended', 'inputs.bold_detrended'),
            ('outputs.bold_tsfnr', 'inputs.bold_tsfnr'),
        ]),
    ])

    return wf_scqc
