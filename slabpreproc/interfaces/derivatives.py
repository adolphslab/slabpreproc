"""
Nipype interface to populate results into derivatives folders.

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-07-23 JMT From scratch
"""

import os
import os.path as op
import shutil
import bids
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

    in_file = File(desc="File to place in derivatives folder", exists=True)
    source_file = File(desc="Source BOLD image for reference", exists=True)
    deriv_dir = Directory(desc="BIDS derivatives folder", exists=True)
    data_type = Str(desc="Type of output data (preproc, qc)")
    new_suffix = Str(desc="Replacement filename suffix")


class BIDSDerivSinkOutputSpec(TraitedSpec):

    out_file = File(desc="Derivatives output file path", exists=True)


class BIDSDerivSink(BaseInterface):

    input_spec = BIDSDerivSinkInputSpec
    output_spec = BIDSDerivSinkOutputSpec

    def _run_interface(self, runtime):

        # Copy in_file to deriv_dir/subj_dir/sess_dir/out_file
        shutil.copyfile(self.inputs.in_file, self._gen_out_pname())

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = self._gen_out_pname()
        return outputs

    def _gen_out_pname(self):
        """
        Construct the full path of the derivatives output
        file corresponding to the current image.
        Create output derivatives subfolder as needed

        :return: out_pname, str, pathlike
            Derivatives output file path
        """

        # Source BOLD image basename
        source_fname = self.inputs.source_file
        source_bname = op.basename(source_fname)

        # Strip maximum of two extensions
        source_bname, _ = op.splitext(source_bname)
        source_bname, _ = op.splitext(source_bname)

        # Get entities from source_file
        keys = bids.layout.parse_file_entities(source_fname)
        subj_id = keys['subject']
        sess_id = keys['session']
        old_suffix = keys['suffix']
        old_ext = keys['extension']

        # subject/session/datatype output subfolder
        deriv_dir = self.inputs.deriv_dir
        data_type = self.inputs.data_type
        out_dir = op.join(deriv_dir, 'sub-' + subj_id, 'ses-' + sess_id, data_type)

        # Save create derivatives subfolder
        os.makedirs(out_dir, exist_ok=True)

        # Output file path. Replace current suffix with new suffix
        new_suffix = self.inputs.new_suffix
        out_pname = op.join(out_dir, source_bname.replace(old_suffix, new_suffix)) + old_ext

        return out_pname