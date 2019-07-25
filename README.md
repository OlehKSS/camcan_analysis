# Cam-CAN Analysis

This package combines different tools created for the alalysis of the [Cambridge Center for Aging and Neuroscience (Cam-CAN) data set](https://www.cam-can.org/index.php?content=dataset). Structural MRI data should be preprocessed using [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/). To prepare fMRI files use [pypreprocess](https://github.com/neurospin/pypreprocess), a sample configuration can be found in *camcan_analysis/scripts/preprocessing*

## Dependencies

* FreeSurfer
* scipy>=0.9
* numpy>=1.6.1
* scikit-learn>=0.14.1
* joblib>=0.11.0
* nibabel>=1.2.0
* nilearn>=0.3.0
* nistats
* pandas>=0.19.2

## Repository Structure
The *camcan* directory holds tools for data analysis.

The *data* folder contains information genrearated by processing scripts and some of the dat provided with the Cam-CAN data set. Information about subjects' age can be found in *data/participant_data.csv*.

Jupyter notebooks with example of the brain age prediction can be found in *camcan_analysis/notebook/age_prediction/*

To analyze the data use files from the *scripts* folder.

*scripts/feature_extraction* contains files for feature extraction from structural and functional MRI, MEG data:

* combine_connectivity.py - extract connectivity information from timeseries and save it into a single file
* extract_connectivity.py - extract and save connectivity from fMRI timeseries
* extract_timeseries.py - extract and save timeseries from fMRI volumes
* extract_truncated_connectivity.py - extract and save connectivity from fMRI timeseries of different duration
* extract_structural.py - extract and save structural data from FreeSurfer outputs

*scripts/plotting* contains utility code for producing various plots using *maplotlib*, *bqplot*, etc. To see examples check *data/figures*.

To train and test age prediction models use files from the *scripts/prediction* directory:

* age_prediction_meg_source_space.py - age prediction using MRI, fMRI and MEG data (source space, etc.)
* age_prediciton_mri_meg_spoc.py - age prediction using MRI, fMRI and MEG data. MEG data is processed using the [SPOC](https://www.ncbi.nlm.nih.gov/pubmed/23954727) algorithm, see implementation details in *camcan_analysis/camcan/processing/spoc.py*
* age_prediction_trunc_fmri.py - age prediction using fMRI timeseries of different length
