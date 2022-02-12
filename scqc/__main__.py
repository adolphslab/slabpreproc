#!/usr/bin/env python
# coding: utf-8

"""
Fast subcortical fMRI quality control
- tSNR and tSFNR
- ALFF and fALFF
- Baseline drift map
- Rigid body motion correction with and without low pass filtering

Outputs QC results, motion and distortion corrected 4D BOLD series to derivatives

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
import os.path as op
import tempfile
import numpy as np
import pkg_resources
import bids

import nipype.interfaces.io as io
import nipype.interfaces.utility as util 
import nipype.pipeline.engine as pe

from sdcflows.utils import wrangler

from utils import (get_topup_pars, get_TR)
from qc_workflow import build_qc_wf
from func_preproc_wf import build_func_preproc_wf
from atlas_wf import build_atlas_wf

import argparse


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lightweight subcortical fMRI quality control')
    parser.add_argument('-d', '--bidsdir', default='.', help="BIDS dataset directory ['.']")
    parser.add_argument('-w', '--workdir', help='Work directory')
    parser.add_argument('-su', '--subjid', default=[], help='Subject ID (without sub- prefix)')
    parser.add_argument('-se', '--sessid', default=[], help='Sessin ID (without ses- prefix)')
    parser.add_argument('-ta', '--taskid', default=[], help='Task ID (without task- prefix)')

    # Parse command line arguments
    args = parser.parse_args()

    bids_dir = args.bidsdir

    if args.workdir:
        work_dir = args.workdir
    else:
        work_dir = tempfile.mkdtemp()

    # Create the work directory if necessary
    os.makedirs(work_dir, exist_ok=True)

    # Get atlas T1 template and labels filenames from package data
    atlas_t1_fname = pkg_resources.resource_filename(
        'scqc',
        'atlas/tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz'
    )
    atlas_labels_fname = pkg_resources.resource_filename(
        'scqc',
        'atlas/tpl-MNI152NLin2009cAsym_res-02_atlas-HOSPA_desc-th25_dseg.nii.gz'
    )
    atlas_brain_fname = pkg_resources.resource_filename(
        'scqc',
        'atlas/tpl-MNI152NLin2009cAsym_res-02_desc-brain_probseg.nii.gz'
    )

    print('Subcortical Quality Control')
    print(f'BIDS directory : {bids_dir}')
    print(f'Work directory : {work_dir}')

    # Init a pybids layout
    layout = bids.BIDSLayout(bids_dir)

    # Collect families of images for each BOLD series (BOLD, T1, SBRef, AP and PA SE-EPI)
    bold_families = build_bold_families(layout)

    # Build main workflow
    main_wf = build_main_wf(work_dir, out_dir, seepi_enc_fname, sbref_enc_fname, TR_s)

    # Loop over each BOLD series
    for fam in bold_families:

        bold_fname = fam['ap_fname']
        sbref_fname = fam['sbref_fname']
        ap_fname = fam['ap_fname']
        pa_fname = fam['pa_fname']
        t1_fname = fam['t1_fname']

        # Get TOPUP metadata from JSON sidecars and save to working directory
        seepi_etl, seepi_enc_mat = get_topup_pars(ap_fname)
        seepi_enc_fname = op.join(work_dir, 'seepi_enc.txt')
        if not op.isfile(seepi_enc_fname):
            np.savetxt(seepi_enc_fname, seepi_enc_mat, fmt="%2d %2d %2d %9.6f")

        sbref_etl, sbref_enc_mat = get_topup_pars(sbref_fname)
        sbref_enc_fname = op.join(work_dir, 'sbref_enc.txt')
        if not op.isfile(sbref_enc_fname):
            np.savetxt(sbref_enc_fname, sbref_enc_mat, fmt="%2d %2d %2d %9.6f")

        # Get TR from BOLD JSON sidecar
        TR_s = get_TR(bold_fname)

        # Pass image filenames to main workflow
        main_wf.inputs.main_inputs.bold = bold_fname
        main_wf.inputs.main_inputs.seepi = [ap_fname, pa_fname]
        main_wf.inputs.main_inputs.sbref = sbref_fname
        main_wf.inputs.main_inputs.mocoref = ap_fname
        main_wf.inputs.main_inputs.ind_t1 = t1_fname
        main_wf.inputs.main_inputs.atlas_labels = atlas_labels_fname
        main_wf.inputs.main_inputs.atlas_t1 = atlas_t1_fname

        # Run main workflow
        main_wf.run()


def build_main_wf(work_dir, out_dir, seepi_enc_fname, sbref_enc_fname, TR_s):

    main_wf = pe.Workflow(
        base_dir=op.realpath(work_dir),
        name='main_wf'
    )

    # Input nodes
    main_inputs = pe.Node(
        util.IdentityInterface(
            fields=['bids_dir', 'work_dir', 'subj_id', 'sess_id', 'task_id']),
        name='main_inputs'
    )

    # Output datasink
    main_outputs = pe.Node(
        io.DataSink(
            base_directory=op.realpath(out_dir)
        ),
        name='main_outputs'
    )

    # Build sub workflows
    func_preproc_wf = build_func_preproc_wf(seepi_enc_fname, sbref_enc_fname)
    atlas_wf = build_atlas_wf()
    qc_wf = build_qc_wf(TR_s)

    main_wf.connect([
        (main_inputs, func_preproc_wf, [
            ('bold', 'preproc_inputs.bold'),
            ('sbref', 'preproc_inputs.sbref'),
            ('seepi', 'preproc_inputs.seepi'),
            ('mocoref', 'preproc_inputs.mocoref'),
        ]),
        (func_preproc_wf, atlas_wf, [('preproc_outputs.sbref', 'atlas_inputs.ind_epi')]),
        (main_inputs, atlas_wf, [('', 'atlas_inputs.ind_epi')]),
        (func_preproc_wf, qc_wf, [
            ('preproc_outputs.bold', 'qc_inputs.bold'),
        ]),
        (func_preproc_wf, main_outputs, [
            ('preproc_outputs.bold', 'preproc.@bold'),
            ('preproc_outputs.sbref', 'preproc.@sbref'),
        ]),
        (qc_wf, main_outputs, [
            ('qc_outputs.tsfnr', 'qc.@tsfnr'),
        ]),
    ])

    return main_wf


if "__main__" in __name__:

    main()
