#%%
import os.path as op
from matplotlib import pyplot as plt
import numpy as np

import mne
h5 = mne.externals.h5io.read_hdf5('./data/all_power_envelopes.h5')

def make_mat(corr):
    C = np.zeros((448, 448), dtype=np.float64)
    C[np.triu_indices(len(C))] = corr
    C += C.T
    C.flat[::448 + 1] = np.diag(C) / 2.
    return C

#%%
C = make_mat(h5['CC110033']['alpha']['corr'])
plt.matshow(C)
