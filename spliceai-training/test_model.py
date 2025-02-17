###############################################################################
# This file contains code to test the SpliceAI model.
###############################################################################

import numpy as np
import sys
import time
import h5py
from keras.models import load_model
from utils import *
from constants import *
from spliceai import *

assert int(sys.argv[1]) in [80, 400, 2000, 10000, 20000]
CL = int(sys.argv[1])

print("Setting up...")
physical_devices = tf.config.list_physical_devices('GPU')
print("Number of physical GPUs:", len(physical_devices))

###############################################################################
# Load model and test data
###############################################################################

BATCH_SIZE = 6
# version = [1, 2, 3, 4, 5]
# this is for when you want to test it for just one trained model with version 1
version = [1]

model = [[] for v in range(len(version))]


for v in range(len(version)):
    model[v] = load_model('../Models/SpliceAI_' +version+ '_' + sys.argv[1] +'c_'+ v + '.h5'', custom_objects={'categorical_crossentropy_2d': categorical_crossentropy_2d})



###############################################################################
# Model testing
###############################################################################

start_time = time.time()

output_class_labels = ['Null', 'Acceptor', 'Donor']
# and splice donor (GT) respectively.

for output_class in [1, 2]:

    Y_true = [[] for t in range(1)]
    Y_pred = [[] for t in range(1)]

    for idx in range(num_idx):

        X = h5f['X' + str(idx)][:]
        Y = h5f['Y' + str(idx)][:]

        Xc, Yc = clip_datapoints(X, Y, CL, 1)

        Yps = [np.zeros(Yc[0].shape) for t in range(1)]

        for v in range(len(version)):

            Yp = model[v].predict(Xc, batch_size=BATCH_SIZE)

            if not isinstance(Yp, list):
                Yp = [Yp]

            for t in range(1):
                Yps[t] += Yp[t]/len(version)
        # Ensemble averaging (mean of the ensemble predictions is used)

        for t in range(1):

            is_expr = (Yc[t].sum(axis=(1,2)) >= 1)

            Y_true[t].extend(Yc[t][is_expr, :, output_class].flatten())
            Y_pred[t].extend(Yps[t][is_expr, :, output_class].flatten())

    print ("\n\033[1m%s:\033[0m" % (output_class_labels[output_class]))

    for t in range(1):

        Y_true[t] = np.asarray(Y_true[t])
        Y_pred[t] = np.asarray(Y_pred[t])

        print_topl_statistics(Y_true[t], Y_pred[t])


h5f.close()

print("--- %s seconds ---" % (time.time() - start_time))
print("--------------------------------------------------------------")

###############################################################################

