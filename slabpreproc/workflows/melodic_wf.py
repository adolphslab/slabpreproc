#!/usr/bin/env python
# coding: utf-8

import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

from ..interfaces import MelMask


def build_melodic_wf(tr_s=1.0):

    melodic_wf = pe.Workflow(name='melodic_wf')

    # Workflow input node
    in_node = pe.Node(
        pe.utils.IdentityInterface(
            fields=[
                'tpl_bold_mag_preproc',
                'tpl_bold_mag_tmean',
                'tpl_t1w_head',
                'tpl_bmask',
                'bold_meta'
            ]
        ),
        name='in_node'
    )

    # Melodic brain signal mask
    melmask = pe.Node(MelMask(), name='melmask')

    # Melodic node
    melodic = pe.Node(
        fsl.MELODIC(
            approach='symm',
            no_bet=True,
            tr_sec=tr_s,
            mm_thresh=0.5,
            out_stats=True,
            report=True,
        ),
        name='melodic'
    )

    # Workflow output node
    out_node = pe.Node(
        pe.utils.IdentityInterface(
            fields=['out_dir']
        ),
        name='out_node'
    )

    melodic_wf.connect([
        (in_node, melmask, [
            ('tpl_bold_mag_tmean', 'tmean'),
            ('tpl_bmask', 'bmask')
        ]),
        (in_node, melodic, [
            ('tpl_t1w_head', 'bg_image'),
            ('tpl_bold_mag_preproc', 'in_files'),
        ]),
        (melmask, melodic, [('melmask', 'mask')]),
        (melodic, out_node, [('out_dir', 'out_dir')])
    ])

    return melodic_wf
