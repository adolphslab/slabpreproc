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

# Wrapper class setting derivatives subfolder
from niworkflows.interfaces.bids import DerivativesDataSink as BIDSDerivatives
class DerivativesDataSink(BIDSDerivatives):
    out_path_base = 'scqc'

# Internal package imports
from qc_workflow import build_qc_wf
from func_preproc_wf import build_func_preproc_wf
from atlas_wf import build_atlas_wf



def build_scqc_wf(work_dir, deriv_dir):
    """
    Build main subcortical QC workflow

    :param work_dir: str
        Path to working directory
    :param deriv_dir: str
        Path to derivatives directory
    :return:
    """

    # Subcortical QC workflow input node
    scqc_inputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold',
                'ind_t1',
                'atlas_t1',
                'atlas_labels'
            ]),
        name='scqc_inputs'
    )

    # Subcortical QC outputs
    scqc_outputs = pe.Node(
        util.IdentityInterface(
            fields=[
                'bold_preproc',
                'sbref_preproc',
                'bold_tmean',
                'bold_tsd',
                'bold_thpf',
                'bold_tsnr',
                't1_atlas_epi',
                'labels_atlas_epi',
                't1_ind_epi'
            ]
        ),
        name='scqc_outputs'
    )

    # Create a BIDS derivatives sink for all outputs
    # bids_sink = pe.Node(
    #     DerivativesDataSink(),
    #     name='bids_sink'
    # )

    # Build sub workflows
    func_preproc_wf = build_func_preproc_wf()
    atlas_wf = build_atlas_wf()
    qc_wf = build_qc_wf()

    scqc_wf = pe.Workflow(
        base_dir=str(work_dir),
        name='scqc_wf'
    )

    scqc_wf.connect([

        # Pass images to preproc and atlas workflow
        (scqc_inputs, func_preproc_wf, [('bold', 'preproc_inputs.bold')]),
        (scqc_inputs, atlas_wf, [
            ('ind_t1', 'atlas_inputs.ind_t1'),
            ('atlas_t1', 'atlas_inputs.atlas_t1'),
            ('atlas_labels', 'atlas_inputs.atlas_labels')
        ]),

        # Pass fMRI preproc results to atlas workflow
        (func_preproc_wf, atlas_wf, [('preproc_outputs.sbref', 'atlas_inputs.ind_epi')]),

        # Pass fMRI preproc results to QC workflow
        (func_preproc_wf, qc_wf, [('preproc_outputs.bold', 'qc_inputs.bold')]),

        # Write preproc results to derivatives folder
        (func_preproc_wf, scqc_outputs, [
            ('preproc_outputs.bold', 'bold_preproc'),
            ('preproc_outputs.sbref', 'sbref_preproc'),
        ]),

        # Write EPI-space atlas results to derivatives folder
        (atlas_wf, scqc_outputs, [
            ('atlas_outputs.t1_atlas_epi', 't1_atlas_epi'),
            ('atlas_outputs.labels_atlas_epi', 'labels_atlas_epi'),
            ('atlas_outputs.t1_ind_epi', 't1_ind_epi'),
        ]),

        # Write QC results to derivatives folder
        (qc_wf, scqc_outputs, [
            ('qc_outputs.bold_tmean', 'bold_tmean'),
            ('qc_outputs.bold_tsd', 'bold_tsd'),
            ('qc_outputs.bold_thpf', 'bold_thpf'),
            ('qc_outputs.bold_tsfnr', 'bold_tsfnr'),
        ]),
    ])

    return scqc_wf
