#!/usr/bin/env python
# coding: utf-8

"""
Fast subcortical fMRI quality control
- Distortion and head motion corrected BOLD series
- BOLD tSD, tMean, tHPF and tSFNR

TODO:
- Individual EPI space hard atlas labels and soft brain mask
- Baseline drift map
- FD with and without low pass filtering

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
import re
import os.path as op
from pathlib import Path
import bids
import argparse
import pkg_resources

# Internal imports
from .workflows import build_wf_toplevel


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Slab fMRI Preprocessing Pipeline')
    parser.add_argument('-d', '--bidsdir', default='.', help="BIDS dataset directory ['.']")
    parser.add_argument('-w', '--workdir', help='Work directory')
    parser.add_argument('--sub', required=True, help='Subject ID without sub- prefix')
    parser.add_argument('--ses', required=True, help='Session ID without ses- prefix')
    parser.add_argument('--wb', required=True, help='Whole brain SBRef task ID')

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

    # Subject and session IDs
    subj_id = args.sub
    sess_id = args.ses

    # Expand to all subjects
    # subj_list = collect_participants(bids_dir=str(bids_dir))
    # assert len(subj_list) > 0

    # Get atlas T1 template and labels filenames from package data
    t1_atlas = pkg_resources.resource_filename(
        'slabpreproc',
        'atlas/tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz'
    )
    labels_atlas = pkg_resources.resource_filename(
        'slabpreproc',
        'atlas/tpl-MNI152NLin2009cAsym_res-02_atlas-HOSPA_desc-th25_dseg.nii.gz'
    )

    print('Slab fMRI Preprocessing Pipeline')
    print(f'BIDS directory : {bids_dir}')
    print(f'Work directory : {work_dir}')
    print(f'Subject ID     : {subj_id}')
    print(f'Session ID     : {sess_id}')

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
                r"sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(beh|dwi|eeg|ieeg|meg|perf)"
            ),
        ),
    )

    # Construct layout using indexer
    print(f'\nIndexing {bids_dir}')
    layout = bids.BIDSLayout(
        str(bids_dir),
        indexer=bids_indexer
    )
    print('Indexing completed')

    # Get available image lists for this subject and session
    bold_list = layout.get(
        subject=subj_id, session=sess_id, datatype='func', suffix='bold', part='mag', extension='.nii'
    )
    assert len(bold_list) > 0, 'No BOLD EPI series found'

    t1_list = layout.get(
        subject=subj_id, session=sess_id, datatype='anat', suffix='T1w', part='mag', extension='.nii'
    )
    assert len(t1_list) > 0, 'No T1w structural images found'

    t2_list = layout.get(
        subject=subj_id, session=sess_id, datatype='anat', suffix='T2w', extension='.nii'
    )
    assert len(t2_list) > 0, 'No T2w structural images found'

    sbref_wb = layout.get(
        subject=subj_id, session=sess_id, datatype='func', task=args.wb, suffix='sbref', part='mag', extension='.nii'
    )

    # Use first T1w image returned as individual anatomical reference
    t1_ind = t1_list[0]

    # BOLD image loop
    for bold in bold_list:

        # Get BOLD series metadata
        bold_meta = bold.get_metadata()

        # Parse filename keys
        keys = bids.layout.parse_file_entities(bold)

        # Separate work folder for each BOLD image
        bold_stub = op.basename(bold).replace('_bold.nii', '')
        this_work_dir = work_dir / bold_stub
        os.makedirs(this_work_dir, exist_ok=True)

        # Get SBRef for this BOLD series
        sbref = layout.get(
            subject=subj_id, session=sess_id, datatype='func', suffix='sbref', part='mag', extension='.nii',
            task=keys['task']
        )
        assert len(sbref) > 0, print('No SBRef found for this BOLD series')

        # SBRef metadata (should only be one)
        sbref_meta = sbref[0].get_metadata()

        fmaps = layout.get(
            subject=subj_id, session=sess_id, datatype='fmap', suffix='epi', part='mag', extension='.nii',
            acquisition=keys['task']
        )
        assert len(fmaps) == 2, 'Fewer than 2 SE-EPI fieldmaps found'

        # Compile list of fmap metadata
        fmaps_meta = [fmap.get_metadata() for fmap in fmaps]

        # Build the subcortical QC workflow
        wf_main = build_wf_toplevel(this_work_dir, deriv_dir, layout)

        # Supply input images
        wf_main.inputs.inputs.bold = bold
        wf_main.inputs.inputs.bold_meta = bold_meta
        wf_main.inputs.inputs.sbref = sbref
        wf_main.inputs.inputs.sbref_meta = sbref_meta
        wf_main.inputs.inputs.sbref_wb = sbref_wb
        wf_main.inputs.inputs.fmaps = fmaps
        wf_main.inputs.inputs.fmaps_meta = fmaps_meta
        wf_main.inputs.inputs.t1_ind = t1_ind
        wf_main.inputs.inputs.t1_atlas = t1_atlas
        wf_main.inputs.inputs.labels_atlas = labels_atlas

        # Run workflow
        # Results are stored in BIDS derivatives folder
        wf_main.run()


if "__main__" in __name__:

    main()
