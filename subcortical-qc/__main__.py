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

import os.path as op
import numpy as np

from glob import glob

import nipype.interfaces.io as io
import nipype.interfaces.utility as util 
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe

import argparser

def main():

    # Processing setup

    # Acceleration parameters (slice grappa (M) x in-plane grappa (R))
    accel = 'm4r2'

    # BIDS metadata
    bids_subj = 'Damy001'
    bids_sess = '20220120'
    bids_task = 'rest'
    bids_acq = f'wb2p0{accel}'

    # Key directories
    bids_dir = op.realpath("/Users/jmt/Data/DenseAmygdala/Piloting/2022-01-20")

    # Get image filenames from BIDS tree
    func_source = io.BIDSDataGrabber(
        name='datasource',
        base_dir=bids_dir,
        subject=bids_subj,
        session=bids_sess,
        output_query={
            'bold': dict(task=bids_task, acquisition=bids_acq, part='mag', suffix='bold', extension='.nii.gz'),
            'sbref': dict(acquisition=bids_acq, part='mag', suffix='sbref', extension='.nii.gz'),
            'seepi': dict(acquisition=bids_acq, suffix='epi', extension='.nii.gz')
        }
    )
    images = func_source.run()

    # Fill input parameters for workflow
    bold_fname = images.outputs.bold[0]
    seepi_fname = images.outputs.seepi
    sbref_fname = images.outputs.sbref[0]
    mocoref_fname = seepi_fname[0]

    # Grab T1w structural from session 2021-12-21
    anat_source = io.BIDSDataGrabber(
        name='anat_source',
        base_dir='/Users/jmt/Data/DenseAmygdala/Piloting/2021-12-21',
        subject=bids_subj,
        session='20211221',
        output_query={
            't1': dict(acquisition='rms', run=2, suffix='T1w', extension='.nii.gz'),
        }
    )
    images = anat_source.run()

    # Fill input parameters for workflow
    anat_fname = images.outputs.t1[0]


    # Create TOPUP encoding files for SEEPI and SBREF
    seepi_enc_fname = op.realpath(f"seepi_{accel}_encoding.txt")
    seepi_etl = 0.0293706
    seepi_enc_mat = np.array([
        [0, 1, 0, seepi_etl],
        [0, -1, 0, seepi_etl]
    ])
    np.savetxt(seepi_enc_fname, seepi_enc_mat, fmt="%2d %2d %2d %9.6f")

    # Create TOPUP encoding file for SBRef images
    sbref_enc_fname = op.realpath(f"sbref_{accel}_encoding.txt")
    sbref_etl = 0.0293706
    sbref_enc_mat = np.array([
        [0, 1, 0, sbref_etl]
    ])
    np.savetxt(sbref_enc_fname, sbref_enc_mat, fmt="%2d %2d %2d %9.6f")

    # Pass image filenames to main workflow
    main_wf.inputs.main_inputs.bold = bold_fname
    main_wf.inputs.main_inputs.seepi = seepi_fname
    main_wf.inputs.main_inputs.sbref = sbref_fname
    main_wf.inputs.main_inputs.mocoref = mocoref_fname
    main_wf.inputs.main_inputs.anat = anat_fname

    main_wf = build_main_wf()

    # Run main workflow
    main_wf.run()


def build_main_wf():

    main_wf = pe.Workflow(
        base_dir=op.realpath('./work'),
        name='main_wf')

    # Input nodes
    main_inputs = pe.Node(
        util.IdentityInterface(
            fields=['bold', 'seepi', 'mocoref', 'sbref', 'anat']),
        name='main_inputs'
    )

    # Output datasink
    main_outputs = pe.Node(
        io.DataSink(
            base_directory = op.realpath('./output')
        ),
        name='main_outputs'
    )

    main_wf.connect([
        (main_inputs, preproc_wf, [
            ('bold', 'preproc_inputs.bold'),
            ('sbref', 'preproc_inputs.sbref'),
            ('seepi', 'preproc_inputs.seepi'),
            ('mocoref', 'preproc_inputs.mocoref'),
            ('anat', 'preproc_inputs.anat')
        ]),
        (preproc_wf, qc_wf, [
            ('preproc_outputs.bold', 'qc_inputs.bold'),
            ('preproc_outputs.anat', 'qc_inputs.anat'),
        ]),
        (preproc_wf, main_outputs, [
            ('preproc_outputs.bold', 'preproc.@bold'),
            ('preproc_outputs.anat', 'preproc.@anat'),
            ('preproc_outputs.sbref', 'preproc.@sbref'),
        ]),
        (qc_wf, main_outputs, [
            ('qc_outputs.tsfnr', 'qc.@tsfnr'),
        ]),
    ])



    # ## Melodic workflow

    # In[18]:


    melodic_wf = pe.Workflow(name='melodic_wf')

    melodic_inputs = pe.Node(
        pe.utils.IdentityInterface(
            fields=['bold', 'mask']
        ),
        name='melodic_inputs'
    )

    # Smoothing node
    smooth = pe.Node(
        fsl.utils.Smooth(
            sigma = 2.0
        ),
        name='smooth'
    )

    # Melodic node
    melodic = pe.Node(
        fsl.MELODIC(
            approach = 'symm',
            no_bet = True,
            bg_threshold = 10,
            tr_sec = 1.06,
            mm_thresh = 0.5,
            out_stats = True,
            report = True,
            out_dir = op.realpath('output/melodic')
        ),
        name='melodic'
    )

    melodic_outputs = pe.Node(
        pe.utils.IdentityInterface(
            fields=['melodic_output']
        ),
        name='melodic_outputs'
    )

    melodic_wf.connect([
        (melodic_inputs, smooth, [('bold', 'in_file')]),
        (melodic_inputs, melodic, [('mask', 'mask')]),
        (smooth, melodic, [('smoothed_file', 'in_files')]),
        (melodic, melodic_outputs, [('out_dir', 'melodic_output')])
    ])




