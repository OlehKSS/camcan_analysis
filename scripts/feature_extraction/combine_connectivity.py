import os
from os.path import join, isdir

import numpy as np
import pandas as pd

from camcan.datasets import load_camcan_timeseries_rest
from camcan.preprocessing import extract_connectivity

# load connectivity matrices
ATLASES = ['modl256', 'basc197']
# path for the different kind of connectivity matrices
CONNECTIVITY_KIND = 'correlation' #'tangent'
CAMCAN_TIMESERIES = '/storage/tompouce/okozynet/camcan/timeseries'
CAMCAN_PATIENTS_EXCLUDED = None
OUT_DIR = '/storage/tompouce/okozynet/camcan/connectivity'
OUT_FILE = join(OUT_DIR, f'connect_data_{CONNECTIVITY_KIND}.h5')

# remove the output file if it exists
if os.path.exists(OUT_FILE):
    os.remove(OUT_FILE)

for sel_atlas in ATLASES:
    print('**************************************************************')
    print(f'Reading timeseries files for {sel_atlas}')

    dataset = load_camcan_timeseries_rest(data_dir=CAMCAN_TIMESERIES,
                                          atlas=sel_atlas,
                                          patients_excluded=CAMCAN_PATIENTS_EXCLUDED)
    connectivities = extract_connectivity(dataset.timeseries, kind=CONNECTIVITY_KIND)
    connect_data = None
    subjects = tuple(s[4:] for s in dataset.subject_id)

    for i, s in enumerate(subjects):
        if connect_data is None:
            connect_data = pd.DataFrame(index=subjects,
                                        columns=np.arange(start=0, stop=len(connectivities[i])),
                                        dtype=float)
            if CONNECTIVITY_KIND == 'correlation':
                # save and apply Fisher's transform
                connect_data.loc[s] = np.arctanh(connectivities[i])
            else:
                connect_data.loc[s] = connectivities[i]
        else:
            if CONNECTIVITY_KIND == 'correlation':
                # save and apply Fisher's transform
                connect_data.loc[s] = np.arctanh(connectivities[i])
            else:
                connect_data.loc[s] = connectivities[i]

    connect_data.to_hdf(OUT_FILE, key=sel_atlas, complevel=9)

