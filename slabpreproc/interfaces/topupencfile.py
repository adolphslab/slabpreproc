"""
Nipype interfaces to create and populate the EPI encoding file
required by TOPUP for distortion correction.

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-07-23 JMT From scratch
"""

import os
import os.path as op
import numpy as np
from pathlib import Path

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    InputMultiPath,
    TraitedSpec
)
from nipype.utils.filemanip import split_filename

"""
Collect encoding directions and EPI total effective readout times from SE-EPI fieldmaps
"""


class TOPUPEncFileInputSpec(BaseInterfaceInputSpec):
    epis = InputMultiPath(
        File(exists=True),
        copyfile=False,
        desc='EPI list (fieldmaps or SBRefs',
        mandatory=True)


class TOPUPEncFileOutputSpec(TraitedSpec):
    encoding_file = File(
        exists=True,
        desc="TOPUP encoding file"
    )


class TOPUPEncFile(BaseInterface):

    input_spec = TOPUPEncFileInputSpec
    output_spec = TOPUPEncFileOutputSpec

    def _run_interface(self, runtime):

        from bids.layout import parse_file_entities

        # Get readout time and phase encoding directions from fmap metadata
        for epi in epis:

            meta = epi.get_metadatat()
            t_ro = meta['TotalReadoutTime']

            # Convert BIDS PE direction (i, j, k) to FSL direction (x, y, z)
            bids_pe_dir = meta['PhaseEncodingDirection']
            fsl_pe_dir = bids_pe_dir.replace('i', 'x').replace('j', 'y').replace('k', 'z')

            enc_mat = []

            if fsl_pe_dir == 'x':
                v_enc = [1, 0, 0, t_ro]
            elif fsl_pe_dir == 'x-':
                v_enc = [-1, 0, 0, t_ro]
            elif fsl_pe_dir == 'y':
                v_enc = [0, 1, 0, t_ro]
            elif fsl_pe_dir == 'y-':
                v_enc = [0, -1, 0, t_ro]
            elif fsl_pe_dir == 'z':
                v_enc = [0, 0, 1, t_ro]
            elif fsl_pe_dir == 'z-':
                v_enc = [0, 0, -1, t_ro]
            else:
                print(f'* Unknown PE direction {fsl_pe_dir}')
                v_enc = [0, 1, 0, t_ro]

            # Add encoding row to encoding matrix
            enc_mat.append(v_enc)

        # Store the encoding file in a temp directory
        tmp_dir = Path(tempfile.mkdtemp())
        encoding_file = tmp_dir / 'topup_encoding_file.txt'

        if not op.isfile(encoding_file):
            try:
                np.savetxt(fname=str(encoding_file), X=enc_mat, fmt="%2d %2d %2d %9.6f")
            except IOError:
                print(f'* Could not save encoding matrix to {str(encoding_file)}')

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()
        fname = self.inputs.volume
        _, base, _ = split_filename(fname)
        outputs["encoding_file"] = os.path.abspath(base + '_thresholded.nii')
        return outputs

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        # Generate default output filename if non specified.
        if not isdefined(out_file) and isdefined(self.inputs.in_file):
            out_file = self._gen_fname(self.inputs.in_file, suffix="_brain")
            # Convert to relative path to prevent BET failure
            # with long paths.
            return op.relpath(out_file, start=os.getcwd())
        return out_file

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(self._gen_outfilename())

        basename = os.path.basename(outputs["out_file"])
        cwd = os.path.dirname(outputs["out_file"])
        kwargs = {"basename": basename, "cwd": cwd}

        if (isdefined(self.inputs.mesh) and self.inputs.mesh) or (
            isdefined(self.inputs.surfaces) and self.inputs.surfaces
        ):
            outputs["meshfile"] = self._gen_fname(
                suffix="_mesh.vtk", change_ext=False, **kwargs
            )
        if (isdefined(self.inputs.mask) and self.inputs.mask) or (
            isdefined(self.inputs.reduce_bias) and self.inputs.reduce_bias
        ):
            outputs["mask_file"] = self._gen_fname(suffix="_mask", **kwargs)
        if isdefined(self.inputs.outline) and self.inputs.outline:
            outputs["outline_file"] = self._gen_fname(suffix="_overlay", **kwargs)
        if isdefined(self.inputs.surfaces) and self.inputs.surfaces:
            outputs["inskull_mask_file"] = self._gen_fname(
                suffix="_inskull_mask", **kwargs
            )
            outputs["inskull_mesh_file"] = self._gen_fname(
                suffix="_inskull_mesh", **kwargs
            )
            outputs["outskull_mask_file"] = self._gen_fname(
                suffix="_outskull_mask", **kwargs
            )
            outputs["outskull_mesh_file"] = self._gen_fname(
                suffix="_outskull_mesh", **kwargs
            )
            outputs["outskin_mask_file"] = self._gen_fname(
                suffix="_outskin_mask", **kwargs
            )
            outputs["outskin_mesh_file"] = self._gen_fname(
                suffix="_outskin_mesh", **kwargs
            )
            outputs["skull_mask_file"] = self._gen_fname(suffix="_skull_mask", **kwargs)
        if isdefined(self.inputs.skull) and self.inputs.skull:
            outputs["skull_file"] = self._gen_fname(suffix="_skull", **kwargs)
        if isdefined(self.inputs.no_output) and self.inputs.no_output:
            outputs["out_file"] = Undefined
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._gen_outfilename()
        return None