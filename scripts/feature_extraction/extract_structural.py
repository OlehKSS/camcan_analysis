from camcan.preprocessing import get_structural_data

# test functions on MRI data
subjects_dir = '/storage/store/data/camcan-mne/freesurfer'
subject = 'CC620466'
out_dir = '/storage/tompouce/okozynet/camcan/structural'

get_structural_data(subjects_dir, subject, out_dir)
