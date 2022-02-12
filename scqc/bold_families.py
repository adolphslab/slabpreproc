import bids

def build_bold_families(layout):

    bold_families = []

    # Get list of all BOLD series in tree
    bold_list = layout.get(
        suffix='bold',
        extension='.nii.gz',
    )

    bold_fam = {
        'subject': 'Damy001',
        'session': '20220203',
        'task': 'rest',
        'bold': bold_fname,
        'sbref': sbref_fname,
        'ap': ap_fname,
        'pa': pa_fname,
        'mocoref': ap_fname
    }

    return bold_families