import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets
from nilearn import plotting
import seaborn as sns

frmi_files = '../../data/connectivity/wrsub-CC110319_task-Rest_bold.nii.gz'

dataset = datasets.fetch_atlas_basc_multiscale_2015()
atlas_filename = dataset.scale197
# Show the BASC 197 parcellation atlas 
plotting.plot_roi(atlas_filename)
# plt.show()

# Show connectivity data
# masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
# time_series = masker.fit_transform(frmi_files)

# atlas.region_coords is required to display connectivity

# how to visualize a connectivity matrix
# https://nilearn.github.io/auto_examples/03_connectivity/plot_probabilistic_atlas_extraction.html#sphx-glr-auto-examples-03-connectivity-plot-probabilistic-atlas-extraction-py
# sns.heatmap(time_series)
# plt.show()
plt.savefig('./test.pdf', bbox_inches='tight')