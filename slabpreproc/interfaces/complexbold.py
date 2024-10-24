"""
Utility functions for converting complex BOLD between rect and polar representations

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-09-16 JMT Adapt from dropout.py
         2024-10-02 JMT Output separate real and imag component images
"""

import os
import nibabel as nib
from pathlib import Path

import numpy as np
from lapunwrap3d import LaplacianPhaseUnwrap3D
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
)


class Pol2CartInputSpec(BaseInterfaceInputSpec):

    bold_mag = File(
        desc='4D BOLD magnitude image',
        exists=True,
        mandatory=True
    )

    bold_phs_rad = File(
        desc='4D BOLD phase image (radians)',
        exists=True,
        mandatory=True
    )


class Pol2CartOutputSpec(TraitedSpec):

    bold_re = File(
        desc="4D BOLD real image",
    )

    bold_im = File(
        desc="4D BOLD imag image",
    )


class Pol2Cart(BaseInterface):

    input_spec = Pol2CartInputSpec
    output_spec = Pol2CartOutputSpec

    def _run_interface(self, runtime):

        # Load 4D mag and phase BOLD images
        bold_mag_nii = nib.load(self.inputs.bold_mag)
        bold_mag = bold_mag_nii.get_fdata()
        bold_phs_rad_nii = nib.load(self.inputs.bold_phs_rad)
        bold_phs_rad = bold_phs_rad_nii.get_fdata()

        # Calculate real and imaginary channels
        bold_z = bold_mag * np.exp(1.0j * bold_phs_rad)
        bold_re, bold_im = np.real(bold_z), np.imag(bold_z)

        # Save cartesian complex BOLD image
        bold_re_nii = nib.Nifti1Image(bold_re, affine=bold_mag_nii.affine)
        nib.save(bold_re_nii, self._gen_real_fname())
        bold_im_nii = nib.Nifti1Image(bold_im, affine=bold_mag_nii.affine)
        nib.save(bold_im_nii, self._gen_imag_fname())

        return runtime

    def _list_outputs(self):
        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["bold_re"] = self._gen_real_fname()
        outputs["bold_im"] = self._gen_imag_fname()

        return outputs

    @staticmethod
    def _gen_real_fname():
        return Path(os.getcwd()) / 'bold_re.nii.gz'

    @staticmethod
    def _gen_imag_fname():
        return Path(os.getcwd()) / 'bold_im.nii.gz'


class Cart2PolInputSpec(BaseInterfaceInputSpec):

    bold_re = File(
        desc='4D BOLD real image',
        exists=True,
        mandatory=True
    )

    bold_im = File(
        desc='4D BOLD imaginary image',
        exists=True,
        mandatory=True
    )


class Cart2PolOutputSpec(TraitedSpec):

    bold_mag = File(
        desc="4D BOLD magnitude image",
    )

    bold_phs_rad = File(
        desc="4D BOLD phase image (radians)",
    )


class Cart2Pol(BaseInterface):

    input_spec = Cart2PolInputSpec
    output_spec = Cart2PolOutputSpec

    def _run_interface(self, runtime):

        # Load 4D real and imag BOLD images
        bold_re_nii = nib.load(self.inputs.bold_re)
        bold_re = bold_re_nii.get_fdata()
        bold_im_nii = nib.load(self.inputs.bold_im)
        bold_im = bold_im_nii.get_fdata()

        # Calculate real and imaginary channels
        bold_z = bold_re + 1.0j * bold_im
        bold_mag, bold_phs_rad = np.abs(bold_z), np.angle(bold_z, deg=False)

        # Save polar complex BOLD image
        bold_mag_nii = nib.Nifti1Image(bold_mag, affine=bold_re_nii.affine)
        nib.save(bold_mag_nii, self._gen_mag_fname())
        bold_phs_rad_nii = nib.Nifti1Image(bold_phs_rad, affine=bold_re_nii.affine)
        nib.save(bold_phs_rad_nii, self._gen_phs_rad_fname())

        return runtime

    def _list_outputs(self):
        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["bold_mag"] = self._gen_mag_fname()
        outputs["bold_phs_rad"] = self._gen_phs_rad_fname()

        return outputs

    @staticmethod
    def _gen_mag_fname():
        return Path(os.getcwd()) / 'bold_mag.nii.gz'

    @staticmethod
    def _gen_phs_rad_fname():
        return Path(os.getcwd()) / 'bold_phs_rad.nii.gz'

class LapUnwrapInputSpec(BaseInterfaceInputSpec):

    phi_w = File(
        desc='Wrapped 3D or 3D x time phase image (radians)',
        exists=True,
        mandatory=True
    )


class LapUnwrapOutputSpec(TraitedSpec):

    phi_uw = File(
        desc="Laplacian unwrapped 3D or 3D x time phase image (radians))",
    )


class LapUnwrap(BaseInterface):

    input_spec = LapUnwrapInputSpec
    output_spec = LapUnwrapOutputSpec

    def _run_interface(self, runtime):

        # Load 3D or 3D x t wrapped phase image (radians)
        phi_w_nii = nib.load(self.inputs.phi_w)
        phi_w = phi_w_nii.get_fdata()

        # Laplacian phase unwrap spatial dimensions
        lapuw = LaplacianPhaseUnwrap3D(phi_w)
        phi_uw = lapuw.unwrap()

        # Save unwrapped phase image (radians)
        phi_uw_nii = nib.Nifti1Image(phi_uw, affine=phi_w_nii.affine)
        nib.save(phi_uw_nii, self._gen_phi_uw_fname())

        return runtime

    def _list_outputs(self):
        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["phi_uw"] = self._gen_phi_uw_fname()

        return outputs

    @staticmethod
    def _gen_phi_uw_fname():
        return Path(os.getcwd()) / 'phi_uw.nii.gz'
