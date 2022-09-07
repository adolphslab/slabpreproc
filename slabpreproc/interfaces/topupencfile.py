"""
Nipype interfaces to create and populate the EPI encoding file
required by TOPUP for distortion correction.

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-07-23 JMT From scratch
"""

import os
import numpy as np
from pathlib import Path

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    Undefined,
    InputMultiPath,
    InputMultiObject,
    TraitedSpec
)

from nipype.utils.filemanip import split_filename

"""
Collect encoding directions and EPI total effective readout times from SE-EPI fieldmaps
"""


class TOPUPEncFileInputSpec(BaseInterfaceInputSpec):

    epi_list = InputMultiPath(
        File(exists=True),
        copyfile=False,
        desc='List of EPI Nifti files',
        mandatory=True
    )

    meta_list = InputMultiObject(
        traits.DictStrAny,
        mandatory=True,
        desc="List of metadata dictionaries",
    )


class TOPUPEncFileOutputSpec(TraitedSpec):
    encoding_file = File(
        exists=True,
        desc="TOPUP encoding file"
    )


class TOPUPEncFile(BaseInterface):

    input_spec = TOPUPEncFileInputSpec
    output_spec = TOPUPEncFileOutputSpec

    def _run_interface(self, runtime):

        # Init encoding matrix
        enc_mat = []

        # Get readout time and phase encoding directions from fmap metadata
        for ec, epi_fname in enumerate(self.inputs.epi_list):

            epi_meta = self.inputs.meta_list[ec]
            t_ro = epi_meta['TotalReadoutTime']

            # Convert BIDS PE direction (i, j, k) to FSL direction (x, y, z)
            bids_pe_dir = epi_meta['PhaseEncodingDirection']
            fsl_pe_dir = bids_pe_dir.replace('i', 'x').replace('j', 'y').replace('k', 'z')

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

        # Store the encoding file in the runtime current working directory
        encoding_file = self._gen_encfile_name()
        np.savetxt(fname=str(encoding_file), X=enc_mat, fmt="%2d %2d %2d %9.6f")

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["encoding_file"] = self._gen_encfile_name()
        return outputs

    def _gen_encfile_name(self):
        return Path(os.getcwd()) / 'topup_encoding_file.txt'