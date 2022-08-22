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
    Undefined,
    InputMultiPath,
    InputMultiObject,
    TraitedSpec
)

from nipype.utils.filemanip import split_filename

"""
Collect encoding directions and EPI total effective readout times from SE-EPI fieldmaps
"""


class FMAPRefInputSpec(BaseInterfaceInputSpec):

    fmap_list = InputMultiPath(
        File(exists=True),
        copyfile=False,
        desc='List of SE-EPI fieldmap Nifti files',
        mandatory=True
    )

    meta_list = InputMultiObject(
        traits.DictStrAny,
        mandatory=True,
        desc="List of metadata dictionaries",
    )


class FMAPRefOutputSpec(TraitedSpec):
    encoding_file = File(
        exists=True,
        desc="TOPUP encoding file"
    )


class FMAPRef(BaseInterface):

    input_spec = FMAPRefInputSpec
    output_spec = FMAPRefOutputSpec

    def _run_interface(self, runtime):

        # Get phase encoding directions from fmap metadata
        for ec, epi_fname in enumerate(self.inputs.fmap_list):

            fmap_meta = self.inputs.meta_list[ec]

            # Convert BIDS PE direction (i, j, k) to FSL direction (x, y, z)
            bids_pe_dir = fmap_meta['PhaseEncodingDirection']

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["fmap_ref"] = self._gen_encfile_name()
        return outputs

    def _gen_encfile_name(self):
        return Path(os.getcwd()) / 'topup_encoding_file.txt'