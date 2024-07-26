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
        # Handles both .nii and nii.gz extensions
        source_bname, _ = op.splitext(source_bname)
        source_bname, _ = op.splitext(source_bname)

        # Output derivatives root folder
        deriv_dname = self.inputs.deriv_dir
        subjsess_deriv_dname = op.join(deriv_dname, 'sub-' + subj_id, 'ses-' + sess_id)

        # Loop over all input files and associated sorting dicts
        for in_pname, sort_dict in zip(self.inputs.file_list, self.inputs.file_sort_dicts):

            # Safe create data type subfolder (eg preproc)
            datatype_out_dname = op.join(subjsess_deriv_dname, sort_dict['DataType'])
            os.makedirs(datatype_out_dname, exist_ok=True)

            # Output file path
            # Construction depends on whether preproc output or atlas templates are being copied to derivatives

            if 'atlas' in sort_dict['DataType']:

                # Simple copy of template/label image to derivatives/.../atlas/ folder
                out_pname = op.join(datatype_out_dname, op.basename(in_pname))

            else:

                # Replace current suffix (eg _bold) with new suffix (eg _recon-preproc_sbref)
                new_suffix = sort_dict['NewSuffix']
                out_pstub = op.join(datatype_out_dname, source_bname.replace(old_suffix, new_suffix))

                # File type dependent extensions
                if 'Text' in sort_dict['FileType']:
                    out_pname = out_pstub + '.txt'
                elif 'CSV' in sort_dict['FileType']:
                    out_pname = out_pstub + '.csv'
                else:
                    out_pname = out_pstub + old_ext

            # Copy input file to deriv_dname/subj_dir/sess_dir/out_file
            shutil.copyfile(in_pname, out_pname)

        # Output folder handling
        # Copying nipype output folders (eg melodic) to derivatives

        # Loop over all input folders and associated sorting dicts
        for in_dname, sort_dict in zip(self.inputs.folder_list, self.inputs.folder_sort_dicts):

            # Safe create data type subfolder (eg melodic)
            datatype_out_dname = op.join(subjsess_deriv_dname, sort_dict['DataType'])
            os.makedirs(datatype_out_dname, exist_ok=True)

            # Output subfolder path. Replace current suffix (eg _bold) with new suffix (eg _melodic)
            new_suffix = sort_dict['NewSuffix']
            out_pname = op.join(datatype_out_dname, source_bname.replace(old_suffix, new_suffix))

            # Copy nipype folder to deriv_dname/subj_dir/sess_dir/task_out_dname
            print(f'  Copying {in_dname}')
            print(f'  to {out_pname}')
            shutil.copytree(in_dname, out_pname, dirs_exist_ok=True)

            # 2024-07-25 JMT Skip aux file removal - rmtree throwing errors for melodic _report tree
            # Remove all Nipype auxiliary files from output folder ('_*' and '*.pklz')
            # fnames = glob(op.join(out_pname, '_*')) + glob(op.join(out_pname, '*.pklz'))
            # for fname in fnames:
            #     if op.isfile(fname):
            #         os.remove(fname)
            #     elif op.isdir(fname):
            #         shutil.rmtree(fname)
            #     else:
            #         pass

        return runtime

    def _list_outputs(self):

        # Dummy output file
        outputs = self._outputs().get()
        outputs["out_file"] = []
        return outputs
