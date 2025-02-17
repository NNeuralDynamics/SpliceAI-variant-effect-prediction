import numpy as np
import sys
import time
import h5py
import tensorflow as tf
from keras.models import load_model
from spliceai import *
from utils import *
from multi_gpu import *
from constants import *
import multiprocessing


assert int(sys.argv[1]) in [80, 400, 2000, 10000,20000]

print("Setting up...")
physical_devices = tf.config.list_physical_devices('GPU')
print("Number of physical GPUs:", len(physical_devices))

L = 32
N_CPUS = multiprocessing.cpu_count()
N_GPUS = 1
# Optionally set the number of threads
tf.config.threading.set_intra_op_parallelism_threads(N_CPUS) 
tf.config.threading.set_inter_op_parallelism_threads(N_CPUS)

# Print current settings (optional)
print(f"Intra-op parallelism threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
print(f"Inter-op parallelism threads: {tf.config.threading.get_inter_op_parallelism_threads()}")

# Define window size (W), atrous rate (AR), and batch size based on sys.argv[1]
if int(sys.argv[1]) == 80:
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18 * 28  # Adjust batch size for CPU parallelism
elif int(sys.argv[1]) == 400:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
    BATCH_SIZE = 1 * 28  # Adjust batch size for CPU parallelism
elif int(sys.argv[1]) == 2000:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10])
    BATCH_SIZE = 12 * 28  # Adjust batch size for CPU parallelism
elif int(sys.argv[1]) == 10000:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21, 41, 41, 41, 41])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10, 25, 25, 25, 25])
    BATCH_SIZE = 1 * 28  # Adjust batch size for CPU parallelism
elif int(sys.argv[1]) == 20000:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21, 41, 41, 41, 41, 41, 41, 51, 51])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10, 25, 25, 25, 25,25, 25, 30 ,30])
    BATCH_SIZE = 1 * 14  # Adjust batch size for CPU parallelism

CL = 2 * np.sum(AR * (W - 1))
print(CL_max, CL)
assert CL <= CL_max and CL == int(sys.argv[1])
print("\033[1mContext nucleotides: %d\033[0m" % (CL))
print("\033[1mSequence length (output): %d\033[0m" % (SL))

# Initialize SpliceAI model
model = SpliceAI(L, W, AR)
model.summary()

# Optionally, wrap model for multi-GPU training
# model_m = make_parallel(model, N_GPUS)

# Compile the model
model.compile(loss=categorical_crossentropy_2d, optimizer='adam')
# model.load_weights("Models/SpliceAI_hg1910000_g1.h5")

# Open the HDF5 file for training data
h5f = h5py.File(data_dir + 'dataset' + '_' + version+'_train' + '_' + 'all' + '.h5', 'r')
# Split indices for training and validation
num_idx = len(h5f.keys()) // 2
idx_all = np.random.permutation(num_idx)
idx_train = idx_all[:int(0.9 * num_idx)]
idx_valid = idx_all[int(0.9 * num_idx):]
EPOCH_NUM = 10 * len(idx_train)

start_time = time.time()

print("Training started, EPOCH_NUM:" + str(EPOCH_NUM))
            
# Enable debug logging for device placement (optional)
tf.debugging.set_log_device_placement(True)
# Training loop
for epoch_num in range(113,EPOCH_NUM):
    idx = np.random.choice(idx_train)

    X = h5f['X' + str(idx)][:]
    Y = h5f['Y' + str(idx)][:]

    Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
    
    # Train the model batch by batch
    model.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
    print("model.fit complete.....epoch:" + str(epoch_num + 1))

    if (epoch_num+1) % len(idx_train) == 0:
        # Validation set metrics
        print("--------------------------------------------------------------")
        print("\n\033[1mValidation set metrics:\033[0m")

        Y_true_1 = [[] for t in range(1)]
        Y_true_2 = [[] for t in range(1)]
        Y_pred_1 = [[] for t in range(1)]
        Y_pred_2 = [[] for t in range(1)]

        for idx in idx_valid:
            X = h5f['X' + str(idx)][:]
            Y = h5f['Y' + str(idx)][:]

            Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
            Yp = model.predict(Xc, batch_size=BATCH_SIZE)

            if not isinstance(Yp, list):
                Yp = [Yp]

            for t in range(1):
                is_expr = (Yc[t].sum(axis=(1, 2)) >= 1)

                Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
                Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
                Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())

        print("\n\033[1mAcceptor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_1[t]), np.asarray(Y_pred_1[t]))

        print("\n\033[1mDonor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_2[t]), np.asarray(Y_pred_2[t]))

        print("\n\033[1mTraining set metrics:\033[0m")

        Y_true_1 = [[] for t in range(1)]
        Y_true_2 = [[] for t in range(1)]
        Y_pred_1 = [[] for t in range(1)]
        Y_pred_2 = [[] for t in range(1)]

        for idx in idx_train[:len(idx_valid)]:
            X = h5f['X' + str(idx)][:]
            Y = h5f['Y' + str(idx)][:]

            Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
            Yp = model.predict(Xc, batch_size=BATCH_SIZE)

            if not isinstance(Yp, list):
                Yp = [Yp]

            for t in range(1):
                is_expr = (Yc[t].sum(axis=(1, 2)) >= 1)

                Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
                Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
                Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())

        print("\n\033[1mAcceptor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_1[t]), np.asarray(Y_pred_1[t]))

        print("\n\033[1mDonor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_2[t]), np.asarray(Y_pred_2[t]))

        print("Learning rate: %.5f" % (kb.get_value(model.optimizer.lr)))
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        print("--------------------------------------------------------------")

        # Learning rate decay
        if (epoch_num + 1) >= 6 * len(idx_train):
            kb.set_value(model.optimizer.lr, 0.5 * kb.get_value(model.optimizer.lr))
        
    if (epoch_num + 1) %10 ==0:
        model.save('../Models/SpliceAI_' +version+ '_' + sys.argv[1] +'c_'+ sys.argv[2] + '.h5')

h5f.close()
