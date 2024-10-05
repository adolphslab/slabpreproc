"""
Nipype interface to identify the SE-EPI fieldmap with the same PE direction as the BOLD series

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-08-22 JMT Adapt from topupencfile.py
"""

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    InputMultiPath,
    InputMultiObject,
    TraitedSpec,
)

"""
Identify the SE-EPI fieldmap with the same PE direction as the BOLD series to be unwarped
Use the SBRef for the BOLD series for PE info
"""


class SEEPIRefInputSpec(BaseInterfaceInputSpec):

    seepi_mag = InputMultiPath(
        File(exists=True),
        copyfile=False,
        desc='List of SE-EPI fieldmap Nifti files',
        mandatory=True
    )

    seepi_meta = InputMultiObject(
        traits.DictStrAny,
        mandatory=True,
        desc="List of SE-EPI fieldmap metadata dictionaries",
    )

    sbref_meta = traits.Dict(
        desc="SBRef metadata dictionary"
    )


class SEEPIRefOutputSpec(TraitedSpec):

    seepi_mag_ref = File(
        exists=True,
        desc="Warped SE-EPI mag fmap with same PE as BOLD series"
    )


class SEEPIRef(BaseInterface):

    input_spec = SEEPIRefInputSpec
    output_spec = SEEPIRefOutputSpec

    def _run_interface(self, runtime):

        """
        All the work can be done in _list_outputs()
        """

        return runtime

    def _list_outputs(self):

        # Get the outputs dictionary
        outputs = self._outputs().get()

        # Get SBRef PE direction
        sbref_pe_dir = self.inputs.sbref_meta['PhaseEncodingDirection']

        # Loop over SE-EPI metadata extracting PE directions
        for fc, seepi_mag_fname in enumerate(self.inputs.seepi_mag):

            fmap_meta = self.inputs.seepi_meta[fc]
            fmap_pe_dir = fmap_meta['PhaseEncodingDirection']

            if fmap_pe_dir == sbref_pe_dir:
                outputs["seepi_mag_ref"] = seepi_mag_fname

        return outputs
