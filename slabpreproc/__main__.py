#!/usr/bin/env python
# coding: utf-8

"""
Fast subcortical fMRI quality control
- Grand reference to individual T2w template
- BOLD EPI and SE-EPI fieldmaps referenced to BOLD SBRef
- Distortion and head motion corrected BOLD series
- BOLD tSD, tMean, tHPF and tSFNR

Outputs QC results, motion and distortion corrected 4D BOLD series to derivatives

AUTHORS : Mike Tyszka and Yue Zhu
PLACE  : Caltech Brain Imaging Center

MIT License

Copyright (c) 2024 Mike Tyszka

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

from .workflows import build_func_wf

import os
import os.path as op
import sys
import re
import bids
import argparse
from templateflow import api as tflow

from nipype import (config, logging)


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Slab fMRI Preprocessing Pipeline')
    parser.add_argument('-d', '--bidsdir', default='.', help="BIDS dataset directory ['.']")
    parser.add_argument('-w', '--workdir', help='Work directory')
    parser.add_argument('--sub', required=True, help='Subject ID without sub- prefix')
    parser.add_argument('--ses', required=True, help='Session ID without ses- prefix')
    parser.add_argument('--antsthreads', required=False, type=int, default=2, choices=range(1, 9),
                        help="Max number of threads allowed for ANTs/ITK modules")
    parser.add_argument('--melodic', action='store_true', default=False, help="Run Melodic ICA")
    parser.add_argument('--debug', action='store_true', default=False, help="Debugging flag")

    # Parse command line arguments
    args = parser.parse_args()

    # BIDS dataset directory
    bids_dir = op.realpath(args.bidsdir)

    # Output derivatives folder
    der_dir = op.join(bids_dir, 'derivatives')

    # Slab preproc derivatives folder
    slab_der_dir = op.join(der_dir, 'slabpreproc')
    os.makedirs(slab_der_dir, exist_ok=True)

    # Freesurfer subjects folder
    fs_subjects_dir = op.join(der_dir, 'freesurfer')

    # Root nipype working directory
    # Session and series level workflows use explicit subfolders (below)
    if args.workdir:
        work_dir = op.realpath(args.workdir)
    else:
        work_dir = op.join(bids_dir, 'work')
    os.makedirs(work_dir, exist_ok=True)

    # Set nipype debug mode and logging to work/ folder
    if args.debug:
        config.enable_debug_mode()
        config.set('execution', 'stop_on_first_crash', 'true')
        config.set('execution', 'remove_unnecessary_outputs', 'false')
        config.set('logging', 'workflow_level', 'DEBUG')
        config.set('logging', 'interface_level', 'DEBUG')
        config.set('logging', 'node_level', 'DEBUG')
        config.update_config({
            'logging': {
                'log_directory': work_dir,
                'log_to_file': True
            }
        })
        logging.update_logging(config)

    # Subject and session IDs
    subj_id = args.sub
    sess_id = args.ses

    # Summary splash text
    print('Slab fMRI Preprocessing Pipeline')
    print(f'BIDS directory   : {bids_dir}')
    print(f'Work directory   : {work_dir}')
    print(f'Subject ID       : {subj_id}')
    print(f'Session ID       : {sess_id}')
    print(f'Max ANTs threads : {args.antsthreads}')
    print(f'Run Melodic ICA  : {args.melodic}')
    print(f'Debug mode       : {args.debug}')

    # Get T1 and T2 templates and subcortical labels from templateflow repo
    # Individual custom templates and labels must have been set up in
    # the TemplateFlow cache directory (typically $(HOME)/.cache/templateflow
    tpl_t1w_head_path = tflow.get(
        subj_id, desc=None, resolution=2,
        suffix='T1w', extension='nii.gz'
    )
    if not tpl_t1w_head_path:
        print(f'* Could not find T1w head template  - exiting')
        sys.exit(1)

    tpl_t2w_head_path = tflow.get(
        subj_id, desc=None, resolution=2,
        suffix='T2w', extension='nii.gz'
    )
    if not tpl_t2w_head_path:
        print(f'* Could not find T2w EPI head template - exiting')
        sys.exit(1)

    tpl_t1w_brain_path = tflow.get(
        subj_id, desc='brain', resolution=2,
        suffix='T1w', extension='nii.gz'
    )
    if not tpl_t1w_brain_path:
        print(f'* Could not find T1w brain template  - exiting')
        sys.exit(1)

    tpl_t2w_brain_path = tflow.get(
        subj_id, desc='brain', resolution=2,
        suffix='T2w', extension='nii.gz'
    )
    if not tpl_t2w_brain_path:
        print(f'* Could not find T2w brain template - exiting')
        sys.exit(1)

    tpl_pseg_path = tflow.get(
        subj_id, desc='subcort', resolution=2,
        suffix='pseg', extension='nii.gz'
    )
    if not tpl_pseg_path:
        print(f'* Could not find template pseg labels - exiting')
        sys.exit(1)

    tpl_dseg_path = tflow.get(
        subj_id, desc='subcort', resolution=2,
        suffix='dseg', extension='nii.gz'
    )
    if not tpl_dseg_path:
        print(f'* Could not find template dseg labels - exiting')
        sys.exit(1)

    tpl_bmask_path = tflow.get(
        subj_id, desc='brain', resolution=2,
        suffix='mask', extension='nii.gz'
    )
    if not tpl_bmask_path:
        print(f'* Could not find template brain mask - exiting')
        sys.exit(1)

    # Find the associated fsnative T1.mgz
    fs_t1w_head_path = op.join(fs_subjects_dir, subj_id, 'mri', 'T1.mgz')
    if not op.isfile(fs_t1w_head_path):
        print(f'* Could not find fsnative T1w head image - exiting')
        sys.exit(1)

    # Construct BIDS layout object for this dataset
    layout = gen_bids_layout(bids_dir)

    # Get list of available BOLD magnitude images for this subj/sess
    bold_mag_filter = {
        'datatype': 'func',
        'suffix': 'bold',
        'part': 'mag',
        'extension': ['.nii', '.nii.gz']
    }

    bold_mag_list = layout.get(subject=subj_id, session=sess_id, **bold_mag_filter)
    assert len(bold_mag_list) > 0, 'No BOLD EPI magnitude images found'

    # Get list of available bias corrected (norm) T2w structural images for this subj/sess
    t2w_filter = {
        'datatype': 'anat',
        'suffix': 'T2w',
        'extension': ['.nii', '.nii.gz']
    }

    t2w_list = layout.get(subject=subj_id, session=sess_id, invalid_filters='allow', **t2w_filter)
    assert len(t2w_list) > 0, 'No bias-corrected T2w images found'

    # Retain bias corrected T2w image as session anatomical reference
    ses_t2w_head_path = None
    for img in t2w_list:
        if "rec-norm" in img.filename:
            ses_t2w_head_path = op.join(img.dirname, img.filename)

    #
    # Within session BOLD series loop
    #

    for bold_mag in bold_mag_list:

        # Get BOLD series metadata
        bold_mag_path = bold_mag.path
        bold_meta = bold_mag.get_metadata()

        # Generate associated phase image pathname
        bold_phs_path = bold_mag.path.replace('mag', 'phase')
        if not op.isfile(bold_phs_path):
            print(f'* {bold_phs_path} does not exist - exiting')
            sys.exit(1)

        # Parse filename keys
        keys = bids.layout.parse_file_entities(bold_mag)

        # Save task ID
        task_id = keys['task']

        # Create BOLD series work folder inside session work folder
        bold_stub = op.basename(bold_mag).split(".nii")[0]
        bold_work_dir = op.join(work_dir, bold_stub)
        os.makedirs(bold_work_dir, exist_ok=True)

        # Find corresponding SBRef mag image
        bids_filter = {
            'datatype': 'func',
            'suffix': 'sbref',
            'part': 'mag',
            'extension': ['.nii', '.nii.gz'],
            'task': task_id
        }
        sbref_mag = layout.get(subject=subj_id, session=sess_id, **bids_filter)
        assert len(sbref_mag) > 0, print('No SBRef mag image found for this BOLD series')

        # Find corresponding SBRef phase image
        bids_filter['part'] = 'phase'
        sbref_phs = layout.get(subject=subj_id, session=sess_id, **bids_filter)
        assert len(sbref_phs) > 0, print('No SBRef phase image found for this BOLD series')

        # SBRef metadata (should only be one)
        # Use the mag image metadata
        sbref_mag_path = sbref_mag[0].path
        sbref_phs_path = sbref_phs[0].path
        sbref_meta = sbref_mag[0].get_metadata()

        # Find SE-EPI mag fieldmaps for this (subj, sess, task)
        bids_filter = {
            'datatype': 'fmap',
            'suffix': 'epi',
            'part': 'mag',
            'extension': ['.nii', '.nii.gz'],
            'acquisition': task_id
        }
        seepi_mag = layout.get(subject=subj_id, session=sess_id, **bids_filter)
        assert len(seepi_mag) >= 2, 'Fewer than 2 SE-EPI mag fieldmaps found'

        # Find associated SE-EPI phase fieldmaps
        bids_filter['part'] = 'phase'
        seepi_phs = layout.get(subject=subj_id, session=sess_id, **bids_filter)
        assert len(seepi_phs) >= 2, 'Fewer than 2 SE-EPI phase fieldmaps found'

        # Create SE-EPI fieldmap path and metadata lists
        seepi_mag_list = [fm.path for fm in seepi_mag]
        seepi_phs_list = [fp.path for fp in seepi_phs]
        seepi_meta_list = [fm.get_metadata() for fm in seepi_mag]

        # Build the subcortical QC workflow
        func_wf = build_func_wf(bold_work_dir, slab_der_dir, bold_meta, args.melodic, args.antsthreads)

        # Supply inputs to func_wf
        func_wf.inputs.inputnode.subject_id = subj_id
        func_wf.inputs.inputnode.fs_subjects_dir = fs_subjects_dir
        func_wf.inputs.inputnode.bold_mag = bold_mag_path
        func_wf.inputs.inputnode.bold_phs = bold_phs_path
        func_wf.inputs.inputnode.bold_meta = bold_meta
        func_wf.inputs.inputnode.sbref_mag = sbref_mag_path
        func_wf.inputs.inputnode.sbref_phs = sbref_phs_path
        func_wf.inputs.inputnode.sbref_meta = sbref_meta
        func_wf.inputs.inputnode.seepi_mag_list = seepi_mag_list
        func_wf.inputs.inputnode.seepi_phs_list = seepi_phs_list
        func_wf.inputs.inputnode.seepi_meta_list = seepi_meta_list
        func_wf.inputs.inputnode.tpl_t1w_head = tpl_t1w_head_path
        func_wf.inputs.inputnode.tpl_t2w_head = tpl_t2w_head_path
        func_wf.inputs.inputnode.tpl_t1w_brain = tpl_t1w_brain_path
        func_wf.inputs.inputnode.tpl_t2w_brain = tpl_t2w_brain_path
        func_wf.inputs.inputnode.tpl_pseg = tpl_pseg_path
        func_wf.inputs.inputnode.tpl_dseg = tpl_dseg_path
        func_wf.inputs.inputnode.tpl_bmask = tpl_bmask_path
        func_wf.inputs.inputnode.fs_t1w_head = fs_t1w_head_path
        func_wf.inputs.inputnode.ses_t2w_head = ses_t2w_head_path

        # Run workflow
        # Outputs are stored in the BIDS derivatives/slabpreproc folder tree
        func_wf.run()


def gen_bids_layout(bids_dir):
    """
    Create the BIDS layout object for this dataset

    :param bids_dir: str, pathlike
        Root directory of BIDS dataset
    :return: layout, BIDSLayout
        BIDS layout object
    """

    # Create BIDS layout indexer (highly recommend)
    # Borrowed from fmriprep config class
    bids_indexer = bids.BIDSLayoutIndexer(
        validate=False,
        ignore=(
            "code",
            "stimuli",
            "sourcedata",
            "models",
            "exclude",
            re.compile(r"^\."),
            re.compile(
                r"sub-[a-zA-Z\d]+(/ses-[a-zA-Z\d]+)?/(beh|dwi|eeg|ieeg|meg|perf)"
            ),
        ),
    )

    # Construct layout using indexer
    print(f'\nIndexing {bids_dir}')
    layout = bids.BIDSLayout(
        str(bids_dir),
        indexer=bids_indexer
    )
    print('Indexing Complete')

    return layout


if "__main__" in __name__:

    main()
