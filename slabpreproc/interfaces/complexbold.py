"""
Utility functions for converting complex BOLD between rect and polar representations

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-09-16 JMT Adapt from dropout.py
"""

import os
import nibabel as nib
from pathlib import Path

import numpy as np
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
)


"""
Complex polar (mag/phs) to rectangular (real/imag) conversion
"""


class Pol2RectInputSpec(BaseInterfaceInputSpec):

    bold_pol = File(
        desc='4D BOLD polar image (mag then polar concat in axis 3)',
        exists=True,
        mandatory=True
    )


class Pol2RectOutputSpec(TraitedSpec):
    bold_rect = File(
        desc="4D BOLD rectangular image (real then imag concat in axis 3)",
    )


class Pol2Rect(BaseInterface):
    input_spec = Pol2RectInputSpec
    output_spec = Pol2RectOutputSpec

    def _run_interface(self, runtime):

        # Load polar complex BOLD image
        bold_pol_nii = nib.load(self.inputs.bold_pol)
        bold_pol_img = bold_pol_nii.get_fdata()

        nt = bold_pol_img.shape[3]
        ht = int(nt/2)

        # Separate into mag and phase arrays
        bold_m = bold_pol_img[..., :(ht-1)]
        bold_p = bold_pol_img[..., ht:]

        # Calculate real and imaginary channels
        bold_z = bold_m + np.exp(1.0j * bold_p)
        bold_re, bold_im = np.real(bold_z), np.imag(bold_z)

        # Concatenate real and image channels into double-length 4D image
        bold_rect_img = np.concatenate([bold_re, bold_im], axis=3)

        # Save rectangular complex BOLD image
        bold_rect_nii = nib.Nifti1Image(bold_rect_img, affine=bold_pol_nii.affine)
        nib.save(bold_rect_nii, self._gen_outfile_name())

        return runtime

    def _list_outputs(self):
        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["bold_rect"] = self._gen_outfile_name()

        return outputs

    @staticmethod
    def _gen_outfile_name():
        return Path(os.getcwd()) / 'bold_z_rect.nii.gz'


"""
Complex rectangular (real/imag) to polar (mag/phs) conversion
"""


class Rect2PolInputSpec(BaseInterfaceInputSpec):
    bold_rect = File(
        desc="4D BOLD rectangular image (real then imag concat in axis 3)",
    )


class Rect2PolOutputSpec(TraitedSpec):
    bold_pol = File(
        desc='4D BOLD polar image (mag then polar concat in axis 3)',
        exists=True,
        mandatory=True
    )


class Rect2Pol(BaseInterface):
    input_spec = Rect2PolInputSpec
    output_spec = Rect2PolOutputSpec

    def _run_interface(self, runtime):

        # Load polar complex BOLD image
        bold_pol_nii = nib.load(self.inputs.bold_pol)
        bold_pol_img = bold_pol_nii.get_fdata()

        nt = bold_pol_img.shape[3]
        ht = int(nt/2)

        # Separate into mag and phase arrays
        bold_m = bold_pol_img[..., :(ht-1)]
        bold_p = bold_pol_img[..., ht:]

        # Calculate real and imaginary channels
        bold_z = bold_m + np.exp(1.0j * bold_p)
        bold_re, bold_im = np.real(bold_z), np.imag(bold_z)

        # Concatenate real and image channels into double-length 4D image
        bold_rect_img = np.concatenate([bold_re, bold_im], axis=3)

        # Save rectangular complex BOLD image
        bold_rect_nii = nib.Nifti1Image(bold_rect_img, affine=bold_pol_nii.affine)
        nib.save(bold_rect_nii, self._gen_outfile_name())

        return runtime

    def _list_outputs(self):
        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["bold_rect"] = self._gen_outfile_name()

        return outputs

    @staticmethod
    def _gen_outfile_name():
        return Path(os.getcwd()) / 'bold_z_rect.nii.gz'