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
import os.path as op
from pathlib import Path
import argparse
import pkg_resources
from glob import glob

# Internal package imports
from .wf_main import build_wf_main


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Slab fMRI Preprocessing Pipeline')
    parser.add_argument('-d', '--bidsdir', default='.', help="BIDS dataset directory ['.']")
    parser.add_argument('-w', '--workdir', help='Work directory')
    parser.add_argument('--sub', required=True, help='Subject ID without sub- prefix')
    parser.add_argument('--ses', required=True, help='Session ID without ses- prefix')

    # Parse command line arguments
    args = parser.parse_args()

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
    probbrain_atlas = pkg_resources.resource_filename(
        'slabpreproc',
        'atlas/tpl-MNI152NLin2009cAsym_res-02_desc-brain_probseg.nii.gz'
    )

    print('Slab fMRI Preprocessing Pipeline')
    print(f'BIDS directory : {bids_dir}')
    print(f'Work directory : {work_dir}')
    print(f'Subject ID     : {subj_id}')
    print(f'Session ID     : {sess_id}')

    # Get list of all magnitude BOLD series for this subject
    bold_list = sorted(glob(str(bids_dir / f'sub-{subj_id}' / f'ses-{sess_id}' / 'func' / '*part-mag*_bold.nii')))
    assert len(bold_list) > 0

    # Get list of all bias corrected RMS MEMPRAGE images for this subjec
    t1_list = sorted(glob(str(bids_dir / f'sub-{subj_id}' / f'ses-{sess_id}' / 'anat' / '*rms*norm*T1w.nii')))
    assert len(t1_list) > 0

    t1_ind = t1_list[0]

    # BOLD image loop
    for bold in bold_list:

        # Separate work folder for each BOLD image
        bold_stub = op.basename(bold).replace('_bold.nii', '')
        this_work_dir = work_dir / bold_stub
        os.makedirs(this_work_dir, exist_ok=True)

        # Build the subcortical QC workflow
        wf_main = build_wf_main(this_work_dir, deriv_dir)

        # Supply input images
        wf_main.inputs.inputs.bold = bold
        wf_main.inputs.inputs.t1_ind = t1_ind
        wf_main.inputs.inputs.t1_atlas = t1_atlas
        wf_main.inputs.inputs.labels_atlas = labels_atlas

        # Run workflow
        # Results are stored in BIDS derivatives folder
        wf_main.run()


if "__main__" in __name__:

    main()
