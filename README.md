# OHBM-2016
Submission to OHBM 2016 on functional lateralization using the neurovault dataset.

### Installation

1. clone this repository
2. `pip install -r requirements.txt`

Note: if using `virtualenv`, run: `fix-osx-virtualenv env/` (if `env/` is your virtual env root)

### Analyses

Run `python main.py`. This calls:

* `generate_components.py` - Generate the components and per-component figures
* `compare_components.py` - Match up ICA components


### Outputs

* `ica_nii` - directory containing Nifti1 label maps for each of 20 ICA components when run on left-only, right-only, and both hemispheres.
* `ica_map` - Png images showing each component above (20 for each ICA run) when run on left-only, right-only, and both hemispheres.
