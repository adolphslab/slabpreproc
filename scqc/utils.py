#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
import nipype.interfaces.io as io
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe

def get_topup_pars(nii_fname):

    json_fname = nii_fname.replace('.nii.gz', '.json')
    meta = read_json(json_fname)

    etl = meta['TotalReadoutTime']
    enc_mat = np.array([
        [0, 1, 0, etl],
        [0, -1, 0, etl]
    ])

    return etl, enc_mat


def get_TR(nii_fname):

    json_fname = nii_fname.replace('.nii.gz', '.json')
    meta = read_json(json_fname)

    return meta['RepetitionTime']


def read_json(fname):
    """
    Safely read JSON sidecar file into a dictionary
    :param fname: string
        JSON filename
    :return: dictionary structure
    """

    try:
        fd = open(fname, 'r')
        json_dict = json.load(fd)
        fd.close()
    except IOError:
        print('*** {}'.format(fname))
        print('*** JSON sidecar not found - returning empty dictionary')
        json_dict = dict()
    except json.decoder.JSONDecodeError:
        print('*** {}'.format(fname))
        print('*** JSON sidecar decoding error - returning empty dictionary')
        json_dict = dict()

    return json_dict