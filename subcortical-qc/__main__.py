#!/usr/bin/env python
# coding: utf-8

"""
Fast subcortical fMRI quality control
- tSNR and tSFNR
- ALFF and fALFF
- Baseline drift map

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
import numpy as np
import pkg_resources

import nipype.interfaces.io as io
import nipype.interfaces.utility as util 
import nipype.pipeline.engine as pe

from utils import (get_topup_pars, get_TR)
from qc_workflow import build_qc_wf
from func_preproc_wf import build_func_preproc_wf
from atlas_wf import build_atlas_wf

import argparse


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lightweight subcortical fMRI quality control')
    parser.add_argument('-b', '--bold', default='bold.nii.gz', help='4D BOLD Nifti filename')
    parser.add_argument('-sb', '--sbref', default='sbref.nii.gz', help='3D BOLD Nifti filename')
    parser.add_argument('-ap', '--apfmap', default='ap.nii.gz', help='3D AP SE-EPI Nifti filename')
    parser.add_argument('-pa', '--pafmap', default='pa.nii.gz', help='3D PA SE-EPI Nifti filename')
    parser.add_argument('-t1', '--t1w', default='t1w.nii.gz', help='3D T1w Nifti filename')

    # Parse command line arguments
    args = parser.parse_args()
    bold_fname = op.realpath(args.bold)
    sbref_fname = op.realpath(args.sbref)
    ap_fname = op.realpath(args.apfmap)
    pa_fname = op.realpath(args.pafmap)
    t1_fname = op.realpath(args.t1w)

    base_dir = op.dirname(bold_fname)
    out_dir = op.join(base_dir, 'output')
    work_dir = op.join(base_dir, 'work')

    if not op.isdir(work_dir):
        os.makedirs(work_dir)

    # Get atlas T1 template and labels filenames
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
    print(f'BOLD image  : {bold_fname}')
    print(f'SBRef image : {sbref_fname}')
    print(f'AP SE-EPI   : {ap_fname}')
    print(f'PA SE-EPI   : {pa_fname}')
    print(f'T1w image   : {t1_fname}')
    print(f'Output dir  : {out_dir}')
    print(f'Work dir    : {work_dir}')

    # Additional workflow arguments
    seepi_fname = [ap_fname, pa_fname]
    mocoref_fname = ap_fname

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

    # Build main workflow
    main_wf = build_main_wf(work_dir, out_dir, seepi_enc_fname, sbref_enc_fname, TR_s)

    # Pass image filenames to main workflow
    main_wf.inputs.main_inputs.bold = bold_fname
    main_wf.inputs.main_inputs.seepi = seepi_fname
    main_wf.inputs.main_inputs.sbref = sbref_fname
    main_wf.inputs.main_inputs.mocoref = mocoref_fname
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
            fields=['bold', 'seepi', 'mocoref', 'sbref', 'anat']),
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
        (func_preproc_wf, atlas_wf, [('preproc_outputs.sbref', 'atlas_inputs.ind_epi')])
        (main_inputs, atlas_wf, [('', 'atlas_inputs.ind_epi')])
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
