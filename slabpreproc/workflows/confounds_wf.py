#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.afni as afni
import nipype.pipeline.engine as pe


def build_confounds_wf():

    wf_confounds = pe.Workflow(name='wf_confounds')

    return wf_confounds
