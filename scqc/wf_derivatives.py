#!/usr/bin/env python
# coding: utf-8

"""
Build workflow to name and place pipeline results appropriately in the BIDS derivatives folder
Based on code fragments from https://github.com/nipreps/fmriprep/blob/master/fmriprep/workflows/bold/outputs.py
"""

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
from niworkflows.interfaces.bids import DerivativesDataSink


def build_wf_derivatives(deriv_dir):

    wf_derivatives = pe.Workflow(name='wf_derivatives')

    # Create input node for all expected subcortical QC results
    inputnode = pe.Node(
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
            ]),
        name='inputnode'
    )

    # Build individual data sinks for each input
    ds_bold_preproc = pe.Node(
        DerivativesDataSink(
            base_directory=str(deriv_dir),
            suffix='bold_preproc',
            compress=True
        ),
        name='ds_bold_preproc',
        run_without_submitting=True
    )

    return wf_derivatives