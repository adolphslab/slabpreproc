#!/usr/bin/env python
# coding: utf-8

from niworkflows.utils.bids import get_metadata_for_nifti

def get_readout_time(nii_fname):
    t_ro = get_metadata_for_nifti(nii_fname, validate=False)['TotalReadoutTime']
    return t_ro


def get_pe_dir(nii_fname):
    bids_pe_dir = get_metadata_for_nifti(nii_fname, validate=False)['PhaseEncodingDirection']
    fsl_pe_dir = bids_to_fsl_dirn(bids_pe_dir)
    return fsl_pe_dir


def get_TR(nii_fname):
    return get_metadata_for_nifti(nii_fname, validate=False)['RepetitionTime']


def bids_to_fsl_dirn(bids_dirn):
    return bids_dirn.replace('i', 'x').replace('j', 'y').replace('k', 'z')
