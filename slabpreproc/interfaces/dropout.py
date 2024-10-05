"""
Estimate regional dropout from coregistered SE-EPI and SBRef images
- Normalize to median signal intensity in both images
- SBRef/SE-EPI approximates dropout fraction if normalized
- Normalize median dropout to 1.0 and clamp to [0.0, 1.0]
- Moderate smoothing to eliminate veins, etc

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-09-07 JMT Adapt from motion.py
"""

import os
import nibabel as nib
from pathlib import Path

import numpy as np
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    TraitedSpec,
)


class DropoutInputSpec(BaseInterfaceInputSpec):
    mseepi = File(
        desc='Mean SE-EPI image in template space',
        exists=True,
        mandatory=True
    )

    sbref = File(
        desc='BOLD SBRef mag image in template space',
        mandatory=True,
        exists=True
    )

    bmask = File(
        desc="Brain mask in template space",
        exists=True,
        mandatory=True
    )


class DropoutOutputSpec(TraitedSpec):
    dropout = File(
        desc="Regional dropout estimate in template space",
    )


class Dropout(BaseInterface):
    input_spec = DropoutInputSpec
    output_spec = DropoutOutputSpec

    def _run_interface(self, runtime):

        # Load SBRef image
        sbref_nii = nib.load(self.inputs.sbref)
        sbref_img = sbref_nii.get_fdata()

        # Load mean SE-EPI image
        mseepi_nii = nib.load(self.inputs.mseepi)
        mseepi_img = mseepi_nii.get_fdata()

        # Load probabilistic brain mask image
        bmask_nii = nib.load(self.inputs.bmask)
        bmask_img = bmask_nii.get_fdata()
        brain_mask = bmask_img > 0.5
        not_brain_mask = np.logical_not(brain_mask)

        # Create conservative signal mask from SBRef and SE-EPI
        # These images may be signal slabs embedded in the larger
        # zero-filled template volume
        sig_mask = np.logical_and(sbref_img > 0, mseepi_img > 0)
        sig_mask = np.logical_and(sig_mask, brain_mask)

        # Calculate masked dropout image
        dropout_img = sbref_img / (mseepi_img + 1e-20)

        # Normalize dropout to median non-zero signal assuming
        # dropout regions account for less than half the brain volume
        median_sig = np.median(dropout_img[sig_mask])
        dropout_img = dropout_img / median_sig

        # Clamp dropout img to range [0, 1]
        dropout_img = np.clip(dropout_img, 0.0, 1.0)

        # Invert intensity to represent dropout (rather than retained signal)
        dropout_img = 1 - dropout_img

        # Set non-brain signal to 0.0
        dropout_img[not_brain_mask] = 0.0

        # Save dropout image
        dropout_nii = nib.Nifti1Image(dropout_img, affine=sbref_nii.affine)
        nib.save(dropout_nii, self._gen_outfile_name())

        return runtime

    def _list_outputs(self):
        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["dropout"] = self._gen_outfile_name()

        return outputs

    @staticmethod
    def _gen_outfile_name():
        return Path(os.getcwd()) / 'dropout.nii.gz'
