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
from glob import glob

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    Directory,
    InputMultiPath,
    InputMultiObject,
    TraitedSpec
)

from nipype.utils.filemanip import split_filename

"""
Populate correct derivatives subfolder with input data file
"""


class DerivativesSorterInputSpec(BaseInterfaceInputSpec):

    deriv_dir = Directory(
        desc="BIDS derivatives folder",
        exists=True)

    source_file = File(
        desc="Source BOLD image for reference",
        exists=True)

    file_list = InputMultiPath(
        File(exists=True),
        copyfile=False,
        desc='List of files to sort into derivatives folder',
        mandatory=True
    )

    file_sort_dicts = InputMultiObject(
        traits.DictStrAny,
        mandatory=True,
        desc="List of sorting info dictionaries corresponding to file_list",
    )

    folder_list = InputMultiPath(
        Directory(exists=True),
        copyfile=False,
        desc='List of folders to sort into derivatives folder',
        mandatory=True
    )

    folder_sort_dicts = InputMultiObject(
        traits.DictStrAny,
        mandatory=True,
        desc="List of sorting info dictionaries corresponding to folder_list",
    )


class DerivativesSorterOutputSpec(TraitedSpec):

    # Dummy output
    out_file = traits.Any(desc="Derivatives sorter dummy output")


class DerivativesSorter(BaseInterface):

    input_spec = DerivativesSorterInputSpec
    output_spec = DerivativesSorterOutputSpec

    def _run_interface(self, runtime):

        #
        # Parse the source BOLD image filename for useful keys
        #

        # Source BOLD image basename
        source_fname = self.inputs.source_file
        source_bname = op.basename(source_fname)

        # Get entities from source_file
        keys = bids.layout.parse_file_entities(source_fname)
        subj_id = keys['subject']
        sess_id = keys['session']
        old_suffix = keys['suffix']
        old_ext = keys['extension']

        # Strip maximum of two extensions
        source_bname, _ = op.splitext(source_bname)
        source_bname, _ = op.splitext(source_bname)

        # Output derivatives root folder
        deriv_dir = self.inputs.deriv_dir

        # Loop over all input files and associated sorting dicts
        for in_pname, sort_dict in zip(self.inputs.file_list, self.inputs.file_sort_dicts):

            # subject/session/datatype output subfolder

            data_type = sort_dict['DataType']
            out_dir = op.join(deriv_dir, 'sub-' + subj_id, 'ses-' + sess_id, data_type)

            # Safe create derivatives subfolder
            os.makedirs(out_dir, exist_ok=True)

            # Output file path. Replace current suffix with new suffix
            new_suffix = sort_dict['NewSuffix']
            out_pstub = op.join(out_dir, source_bname.replace(old_suffix, new_suffix))

            if 'Text' in sort_dict['FileType']:
                out_pname = out_pstub + '.txt'
            elif 'CSV' in sort_dict['FileType']:
                out_pname = out_pstub + '.csv'
            else:
                out_pname = out_pstub + old_ext

            # Copy input file to deriv_dir/subj_dir/sess_dir/out_file
            shutil.copyfile(in_pname, out_pname)

        # Loop over all input folders and associated sorting dicts
        for in_dname, sort_dict in zip(self.inputs.folder_list, self.inputs.folder_sort_dicts):

            # Subject/session output folder
            out_dir = op.join(deriv_dir, 'sub-' + subj_id, 'ses-' + sess_id, sort_dict['DerivativesFolder'])

            # Copy nipype folder to deriv_dir/subj_dir/sess_dir/out_dir
            shutil.copytree(in_dname, out_dir, dirs_exist_ok=True)

            # Remove all Nipype auxiliary files from output folder ('_*' and '*.pklz')
            fnames = glob(op.join(out_dir, '_*')) + glob(op.join(out_dir, '*.pklz'))
            for fname in fnames:
                if op.isfile(fname):
                    os.remove(fname)
                elif op.isdir(fname):
                    shutil.rmtree(fname)
                else:
                    pass

        return runtime

    def _list_outputs(self):

        # Dummy output file
        outputs = self._outputs().get()
        outputs["out_file"] = []
        return outputs
