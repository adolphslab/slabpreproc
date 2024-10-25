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
from lapunwrap3d import LaplacianPhaseUnwrap3D
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
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
