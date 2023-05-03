#!/usr/bin/env python
# coding: utf-8

"""
Generate confound regressors for downstream modeling
- Unfiltered and low-pass filtered framewise displacement (FD)
- ICA-FIX non-neural IC temporal modes

aCompCor and mean tissue signal regressors are not exported currently

For thinking behind this decision see:
Pruim, R.H.R., Mennes, M., Buitelaar, J.K., Beckmann, C.F., 2015. Evaluation of ICA-AROMA and alternative
strategies for motion artifact removal in resting state fMRI. Neuroimage 112, 278–287.
https://doi.org/10.1016/j.neuroimage.2015.02.063

AUTHORS : Mike Tyszka and Yue Zhu
PLACE  : Caltech Brain Imaging Center

MIT License

Copyright (c) 2023 Mike Tyszka

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

import nipype.interfaces.afni as afni
import nipype.pipeline.engine as pe


def build_confounds_wf():

    wf_confounds = pe.Workflow(name='wf_confounds')

    return wf_confounds
