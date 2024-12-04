"""
Construct a brain signal mask for melodic ICA of slab BOLD

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2023-02-13 JMT Adapt from dropout.py
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
)


class MelMaskInputSpec(BaseInterfaceInputSpec):
    tmean = File(
        desc='tMean BOLD image in template space',
        exists=True,
        mandatory=True
    )

    bmask = File(
        desc="Probabilistic brain mask in template space",
        exists=True,
        mandatory=True
    )


class MelMaskOutputSpec(TraitedSpec):
    melmask = File(
        desc="Melodic ICA brain signal mask in template space",
    )


class MelMask(BaseInterface):
    input_spec = MelMaskInputSpec
    output_spec = MelMaskOutputSpec

    def _run_interface(self, runtime):

        # Load tMean BOLD image
        tmean_nii = nib.load(self.inputs.tmean)
        tmean_img = tmean_nii.get_fdata()

        # Slab mask (tMean BOLD signal > 0)
        # The template-space image is whole-brain, so many voxels may be outside the slab
        slab_mask = tmean_img > 0

        # Set signal threshold at 10% of 98th percentile
        thr = np.percentile(tmean_img[slab_mask], 98.0) * 0.1

        # tMean BOLD signal mask including non-brain tissue
        tmean_mask = tmean_img > thr

        # Load probabilistic brain mask image
        bmask_nii = nib.load(self.inputs.bmask)
        bmask_img = bmask_nii.get_fdata()
        brain_mask = bmask_img > 0.5

        # Construct melodic ICA brain signal mask from slab signal and brain mask
        melmask_img = np.logical_and(tmean_mask, brain_mask).astype(np.uint8)

        # Save melodic brain signal mask
        melmask_nii = nib.Nifti1Image(melmask_img, affine=tmean_nii.affine, header=tmean_nii.header)
        nib.save(melmask_nii, self._gen_outfile_name())

        return runtime

    def _list_outputs(self):
        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["melmask"] = self._gen_outfile_name()

        return outputs

    @staticmethod
    def _gen_outfile_name():
        return Path(os.getcwd()) / 'melmask.nii.gz'
