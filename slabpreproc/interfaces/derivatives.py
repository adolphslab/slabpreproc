"""
Nipype interface to populate results into derivatives folders.

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-07-23 JMT From scratch
"""

import os
import os.path as op
import shutil

import numpy as np
from pathlib import Path

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    Directory,
    Str,
    Undefined,
    InputMultiPath,
    InputMultiObject,
    TraitedSpec
)

from nipype.utils.filemanip import split_filename

"""
Populate correct derivatives subfolder with input data file
"""


class BIDSDerivSinkInputSpec(BaseInterfaceInputSpec):

    in_file = File(
        desc="File to place in derivatives folder",
        exists=True
    )
    deriv_dir = Directory(
        desc="BIDS derivatives folder",
        exists=True
    )
    data_type = Str(desc="Type of output data (preproc, qc)")
    new_suffix = Str(desc="Replacement filename suffix")


class BIDSDerivSinkOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="Derivatives output file path"
    )


class BIDSDerivSink(BaseInterface):

    input_spec = BIDSDerivSinkInputSpec
    output_spec = BIDSDerivSinkOutputSpec

    def _run_interface(self, runtime):

        out_pname = self._gen_out_pname()
        out_dname = op.dirname(out_pname)

        # Safe create the derivatives output subfolder
        os.makedirs(out_dname, exist_ok=True)

        # Source file basename
        source_bname = op.basename(self.inputs.in_file)

        # Strip maximum of two extensions
        source_bname, _ = op.splitext(source_bname)
        source_bname, _ = op.splitext(source_bname)

        # Get entities from source_file
        keys = bids.layout.parse_file_entities(source_file)

        subj_id = keys['subject']
        sess_id = keys['session']
        old_suffix = keys['suffix']

        # Create subject/session/datatype output folder
        out_dir = op.join(deriv_dir, 'sub-' + subj_id, 'ses-' + sess_id, datatype)
        makedirs(out_dir, exist_ok=True)

        # Output file path. Replace current suffix with new suffix
        out_file = op.join(out_dir, source_bname.replace(old_suffix, new_suffix)) + '.txt'

        # Copy in_file to deriv_dir/subj_dir/sess_dir/out_file
        shutil.copyfile(in_file, out_file)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = self._gen_out_pname()
        return outputs

    def _gen_out_pname(self):
        """
        Construct the full path of the derivatives output
        file corresponding to the current image

        :return: out_file, str
            Derivatives output file path
        """

        out_pname = ""

        return out_pname