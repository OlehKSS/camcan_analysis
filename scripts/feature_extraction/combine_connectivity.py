import os
from os.path import join, isdir

import joblib as jb
import numpy as np
import pandas as pd

# load connectivity matrices
ATLASES = ['msdl', 'basc064', 'basc122', 'basc197']
# path for the different kind of connectivity matrices
CONNECTIVITY_KIND = ['correlation', 'partial correlation', 'tangent']
CONNECTIVITY_DATA_DIR = '/storage/data/camcan/camcan_connectivity'
OUT_DIR = '/storage/tompouce/okozynet/camcan/connectivity'

subjects = tuple(d[4:] for d in os.listdir(CONNECTIVITY_DATA_DIR) if isdir(join(CONNECTIVITY_DATA_DIR, d)))

print(f'Found {len(subjects)} subjects')

for sel_atlas in ATLASES:
    for sel_connect in CONNECTIVITY_KIND:
        print('**************************************************************')
        print(f'Reading connectivity files for {sel_atlas}/{sel_connect}')

        connect_data = None
        connect_failed = []
        for s in subjects:
            file_path = join(CONNECTIVITY_DATA_DIR,
                            f'sub-{s}/{sel_atlas}/{sel_connect}/sub-{s}_task-Rest_confounds.pkl')
            try:
                with open(file_path, 'rb') as f:
                    connect_matrix = jb.load(f)
            except OSError:
                print(f'Cannot find connectivity file {file_path} for subject {s}')
                connect_failed.append(s)

            if connect_data is None:
                connect_data = pd.DataFrame(index=subjects, columns=np.arange(start=0, stop=len(connect_matrix)),
                                                dtype=float)
                # save data to for this subject and apply Fisher's transform
                connect_data.loc[s] = np.arctanh(connect_matrix)
            else:
                # save data to for this subject and apply Fisher's transform
                connect_data.loc[s] = np.arctanh(connect_matrix)

        print('Failed to load connectivity data for\n', connect_failed)

        connect_data.to_pickle(join(OUT_DIR, f'connect_data_{sel_atlas}_{sel_connect}.gzip'),
                                    compression='gzip')
