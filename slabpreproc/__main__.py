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

from .workflows import build_toplevel_wf

import os
import sys
import re
import os.path as op
from pathlib import Path
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
    parser.add_argument('--nthreads', required=False, type=int, default=2, choices=range(1, 8),
                        help="Max number of threads")
    parser.add_argument('--debug', action='store_true', default=False, help="Debugging flag")

    # Parse command line arguments
    args = parser.parse_args()

    # BIDS dataset directory
    bids_dir = Path(op.abspath(args.bidsdir))

    # Output derivatives folder
    deriv_dir = bids_dir / 'derivatives' / 'slabpreproc'
    os.makedirs(deriv_dir, exist_ok=True)

    # Working directory
    if args.workdir:
        work_dir = Path(args.workdir)
    else:
        work_dir = Path(op.join(bids_dir, 'work'))
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
    print(f'BIDS directory : {bids_dir}')
    print(f'Work directory : {work_dir}')
    print(f'Subject ID     : {subj_id}')
    print(f'Session ID     : {sess_id}')
    print(f'Max threads    : {args.nthreads}')
    print(f'Debug mode     : {args.debug}')

    # Get T1 and T2 templates and subcortical labels from templateflow repo
    # Individual custom templates and labels must have been set up in
    # the TemplateFlow cache directory (typically $(HOME)/.cache/templateflow
    tpl_t1_head_path = tflow.get(
        subj_id, desc=None, resolution=2,
        suffix='T1w', extension='nii.gz'
    )
    if not tpl_t1_head_path:
        print(f'* Could not find T1w head template  - exiting')
        sys.exit(1)

    tpl_t2_head_path = tflow.get(
        subj_id, desc=None, resolution=2,
        suffix='T2w', extension='nii.gz'
    )
    if not tpl_t2_head_path:
        print(f'* Could not find T2w head template - exiting')
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

    # Construct BIDS layout object for this dataset
    layout = gen_bids_layout(bids_dir)

    # Get available image lists for this subject and session
    mag_filter = {
        'datatype': 'func',
        'suffix': 'bold',
        'part': 'mag',
        'extension': ['.nii', '.nii.gz']
    }

    bold_mag_list = layout.get(subject=subj_id, session=sess_id, **mag_filter)
    assert len(bold_mag_list) > 0, 'No BOLD EPI magnitude images found'

    #
    # BOLD magnitude images loop
    #

    for bold_mag in bold_mag_list:

        # Get BOLD series metadata
        bold_mag_path = bold_mag.path
        bold_mag_meta = bold_mag.get_metadata()

        # Parse filename keys
        keys = bids.layout.parse_file_entities(bold_mag)

        # Separate work folder for each BOLD image
        bold_stub = op.basename(bold_mag).split(".nii")[0]
        this_work_dir = work_dir / bold_stub
        os.makedirs(this_work_dir, exist_ok=True)

        #
        # Find SBRef for this BOLD magnitude image
        #

        filter = {
            'datatype': 'func',
            'suffix': 'sbref',
            'part': 'mag',
            'extension': ['.nii', '.nii.gz'],
            'task': keys['task']
        }
        sbref = layout.get(subject=subj_id, session=sess_id, **filter)
        assert len(sbref) > 0, print('No SBRef found for this BOLD series')

        # SBRef metadata (should only be one)
        sbref_path = sbref[0].path
        sbref_meta = sbref[0].get_metadata()

        #
        # Find fieldmaps for this BOLD series
        #

        filter = {
            'datatype': 'fmap',
            'suffix': 'epi', 'part': 'mag', 'extension': ['.nii', '.nii.gz'],
            'acquisition': keys['task']
        }
        fmaps = layout.get(subject=subj_id, session=sess_id, **filter)
        assert len(fmaps) == 2, 'Fewer than 2 SE-EPI fieldmaps found'

        # Create fieldmap path and metadata lists
        fmap_paths = [fmap.path for fmap in fmaps]
        fmap_metas = [fmap.get_metadata() for fmap in fmaps]

        # Build the subcortical QC workflow
        toplevel_wf = build_toplevel_wf(this_work_dir, deriv_dir, bold_mag_meta, args.nthreads)

        # Supply input images
        toplevel_wf.inputs.inputs.bold_mag = bold_mag_path
        toplevel_wf.inputs.inputs.bold_mag_meta = bold_mag_meta
        toplevel_wf.inputs.inputs.sbref = sbref_path
        toplevel_wf.inputs.inputs.sbref_meta = sbref_meta
        toplevel_wf.inputs.inputs.seepis = fmap_paths
        toplevel_wf.inputs.inputs.seepis_meta = fmap_metas
        toplevel_wf.inputs.inputs.tpl_t1_head = tpl_t1_head_path
        toplevel_wf.inputs.inputs.tpl_t2_head = tpl_t2_head_path
        toplevel_wf.inputs.inputs.tpl_pseg = tpl_pseg_path
        toplevel_wf.inputs.inputs.tpl_dseg = tpl_dseg_path
        toplevel_wf.inputs.inputs.tpl_bmask = tpl_bmask_path

        # Run workflow
        # Workflow outputs are stored in a BIDS derivatives folder
        toplevel_wf.run()


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
