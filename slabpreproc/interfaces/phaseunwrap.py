"""
Phase unwrapping
- Laplacian spatial unwrapping for 3D and 3D x time images
- Temporal phase unwrapping and post-HPF of 3D x time images

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2024-10-24 JMT Extract from complexbold.py
         2024-10-24 JMT Add temporal unwrapping with post-HPF
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy as sp
from lapunwrap3d import LaplacianPhaseUnwrap3D
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits
)


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
        return Path(os.getcwd()) / 'phi_lapuw.nii.gz'


class TempUnwrapInputSpec(BaseInterfaceInputSpec):
    phi_w = File(
        desc='Wrapped 3D x time phase image (radians)',
        exists=True,
        mandatory=True
    )

    k_t = traits.Int(
        desc='Savitzky-Golay kernel size',
        mandatory=True,
    )


class TempUnwrapOutputSpec(TraitedSpec):
    phi_uw = File(
        desc="Temporally unwrapped and high pass filtered 3D x time phase image (radians))",
    )


class TempUnwrap(BaseInterface):
    input_spec = TempUnwrapInputSpec
    output_spec = TempUnwrapOutputSpec

    def _run_interface(self, runtime):
        # Load 3D or 3D x t wrapped phase image (radians)
        phi_w_nii = nib.load(self.inputs.phi_w)
        phi_w = phi_w_nii.get_fdata()

        # Phase unwrap temporal dimension
        phi_uw = np.unwrap(phi_w, axis=3)

        # Savitzky-Golay baseline estimation
        bline = sp.signal.savgol_filter(phi_uw, window_length=self.inputs.k_t, polyorder=2, mode='mirror', axis=3)

        # HPF by baseline subtraction
        phi_uw = phi_uw - bline

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
        return Path(os.getcwd()) / 'phi_tempuw.nii.gz'
