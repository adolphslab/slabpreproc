"""
Nipype interface to generate PDF report from preprocessing, including QC

AUTHOR : Mike Tyszka
PLACE  : Caltech
DATES  : 2022-08-22 JMT Adapt from topupencfile.py
"""

import os
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits
)
from ..utils import ReportPDF

from nipype.utils.filemanip import split_filename

"""
Identify the SE-EPI fieldmap with the same PE direction as the BOLD series to be unwarped
Use the SBRef for the BOLD series for PE info
"""


class ReportInputSpec(BaseInterfaceInputSpec):

    tpl_t1 = File(
        desc='Individual T1w head template',
        exists=True,
        mandatory=True,
    )

    tpl_t2 = File(
        desc='Individual T2w head template',
        exists=True,
        mandatory=True,
    )

    tpl_bold_mean = File(
        desc='Mean preprocessed BOLD in individual template space',
        exists=True,
        mandatory=True,
    )

    tpl_sbref = File(
        desc='Preprocessed BOLD SBRef in individual template space',
        exists=True,
        mandatory=True,
    )

    tpl_seepi_ref = File(
        desc='Preprocessed SE-EPI reference in individual template space',
        exists=True,
        mandatory=True,
    )

    tpl_tsfnr = File(
        desc='Preprocessed SE-EPI reference in individual template space',
        exists=True,
        mandatory=True,
    )

    metadata = traits.Dict(
        desc='BOLD metadata from BIDS JSON sidecar',
        exist=True,
        mandatory=True
    )


class ReportOutputSpec(TraitedSpec):

    report_pdf = File(
        exists=True,
        desc="Pipeline report PDF filename"
    )


class ReportRef(BaseInterface):

    input_spec = ReportInputSpec
    output_spec = ReportOutputSpec

    def _run_interface(self, runtime):

        # Collect image filenames in a dictionary
        img_dict = {
            'tpl_T1': self.inputs.tpl_T1,
            'tpl_T2': self.inputs.tpl_T1,
            'tpl_bold_mean': self.inputs.tpl_bold_mean,
            'tpl_sbref': self.inputs.tpl_sbref,
            'tpl_seepi_ref': self.inputs.tpl_seepi_ref,
            'tpl_tsfnr': self.inputs.tpl_tsfnr,
        }

        metadata = self.inputs.metadata

        metrics = {
            'fd': self.inputs.fd,
            'fd_lpf': self.inputs.fd_lpf
        }

        ReportPDF(img_dict, metadata, metrics)

        return runtime

    def _list_outputs(self):

        # Get the outputs dictionary
        outputs = self._outputs().get()
        outputs["report_pdf"] = self._gen_reportpdf_name()

        return outputs

    def _gen_reportpdf_name(self):
        return File(os.getcwd()) / 'report.pdf'