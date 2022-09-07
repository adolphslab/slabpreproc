"""
Nipype interface for additional motion analysis
- Low pass filtered FD

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-09-07 JMT Adapt from seepiref.py
"""

import os
from pathlib import Path
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    TraitedSpec,
)

"""
Calculate low pass filtered FD from FD
"""

class LPFFDInputSpec(BaseInterfaceInputSpec):

    fd = File(
        desc='FD filename',
        mandatory=True
    )

    cutoff = traits.Float(
        desc="Low pass filter cutoff frequency in Hz",
        mandatory=True,
    )

    bold_meta = traits.Dict(
        desc="BOLD timeseries metadata dictionary"
    )


class LPFFDOutputSpec(TraitedSpec):

    fd_lpf = File(
        desc="Low pass filtered FD filename",
        exists=True
    )


class LPFFD(BaseInterface):

    input_spec = LPFFDInputSpec
    output_spec = LPFFDOutputSpec

    def _run_interface(self, runtime):

        """
        Load unfiltered FD data from file
        Low pass filter
        Save LDF FD to new file
        """

        return runtime

    def _list_outputs(self):

        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["fd_lpf"] = self._gen_outfile_name()

        return outputs

    def _gen_outfile_name(self):
        return Path(os.getcwd()) / 'fd_lpf.txt'
