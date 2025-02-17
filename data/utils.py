###############################################################################
'''This code has functions which process the information in the .h5 files
datafile_{}_{}.h5 and convert them into a format usable by Keras.'''
###############################################################################

import numpy as np
import re
from math import ceil
from sklearn.metrics import average_precision_score
from constants import *
import time
import gzip
import tensorflow as tf


assert CL_max % 2 == 0

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T respectively.

OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
# One-hot encoding of the outputs: 0 is for no splice, 1 is for acceptor,
# 2 is for donor and -1 is for padding.


def ceil_div(x, y):

    return int(ceil(float(x)/y))


def create_datapoints(seq, strand, tx_start, tx_end, jn_start, jn_end):
    # This function first converts the sequence into an integer array, where
    # A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
    # negative, then reverse complementing is done. The splice junctions 
    # are also converted into an array of integers, where 0, 1, 2, -1 
    # correspond to no splicing, acceptor, donor and missing information
    # respectively. It then calls reformat_data and one_hot_encode
    # and returns X, Y which can be used by Keras models.
    seq = seq.decode('utf-8')
    strand = strand.decode('utf-8')
    tx_start = tx_start.decode('utf-8')
    tx_end = tx_end.decode('utf-8')

#     seq = 'N'*(CL_max//2) + seq[CL_max//2:-CL_max//2] + 'N'*(CL_max//2)
    # Context being provided on the RNA and not the DNA

    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')

    tx_start = int(tx_start)
    tx_end = int(tx_end) 

    #jn_start = list(map(lambda x: map(int, re.split(',', x)[:-1]), jn_start))
    #jn_end = list(map(lambda x: map(int, re.split(',', x)[:-1]), jn_end))
    jn_start = [list(map(int, re.split(b',', x[:-1]))) for x in jn_start]
    jn_end = [list(map(int, re.split(b',', x[:-1]))) for x in jn_end]
    
    X0 = None
    Y0 = None
    
    if strand == '+':

#         X0 = np.asarray(map(int, list(seq)))
        X0 = np.asarray(list(map(int, list(seq))))

        Y0 = [-np.ones(tx_end-tx_start+1) for t in range(1)]

        for t in range(1):
            
            if len(jn_start[t]) > 0:
                Y0[t] = np.zeros(tx_end-tx_start+1)
                for c in jn_start[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][c-tx_start] = 2
                for c in jn_end[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][c-tx_start] = 1
                    # Ignoring junctions outside annotated tx start/end sites
                     
    elif strand == '-':

        #X0 = (5-np.asarray(map(int, list(seq[::-1])))) % 5  # Reverse complement
        X0 = (5 - np.asarray(list(map(int, list(seq[::-1]))))) % 5  # Reverse complement

        Y0 = [-np.ones(tx_end-tx_start+1) for t in range(1)]

        for t in range(1):

            if len(jn_start[t]) > 0:
                Y0[t] = np.zeros(tx_end-tx_start+1)
                for c in jn_end[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][tx_end-c] = 2
                for c in jn_start[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][tx_end-c] = 1
    Xd, Yd = reformat_data(X0, Y0)
    X, Y = one_hot_encode(Xd, Yd)

    return X, Y



def reformat_data(X0, Y0):
    # Calculate the number of points (blocks) based on Y0's length
    num_points = ceil(len(Y0[0]) / SL)

    Xd = np.zeros((num_points, SL + CL_max))
    Yd = [-np.ones((num_points, SL)) for _ in range(len(Y0))]  # Adjusted for multiple Y0 arrays

    X0 = np.pad(X0, (0, SL), 'constant', constant_values=0)
    Y0 = [np.pad(Y0[t], (0, SL), 'constant', constant_values=-1) for t in range(len(Y0))]

    for i in range(num_points):
        Xd[i] = X0[SL*i : SL*i + SL + CL_max]

    for t in range(len(Y0)):
        for i in range(num_points):
            Yd[t][i] = Y0[t][SL*i : SL*(i+1)]

    return Xd, Yd


def clip_datapoints(X, Y, CL, N_GPUS):
    # This function is necessary to make sure of the following:
    # (i) Each time model_m.fit is called, the number of datapoints is a
    # multiple of N_GPUS. Failure to ensure this often results in crashes.
    # (ii) If the required context length is less than CL_max, then
    # appropriate clipping is done below.
    # Additionally, Y is also converted to a list (the .h5 files store 
    # them as an array).

    rem = X.shape[0]%N_GPUS
    clip = (CL_max-CL)//2

    if rem != 0 and clip != 0:
        return X[:-rem, clip:-clip], [Y[t][:-rem] for t in range(1)]
    elif rem == 0 and clip != 0:
        return X[:, clip:-clip], [Y[t] for t in range(1)]
    elif rem != 0 and clip == 0:
        return X[:-rem], [Y[t][:-rem] for t in range(1)]
    else:
        return X, [Y[t] for t in range(1)]


def one_hot_encode(Xd, Yd):

    return IN_MAP[Xd.astype('int8')], \
           [OUT_MAP[Yd[t].astype('int8')] for t in range(1)]

def one_hot_encode_sequence(seq):

    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    return map[np.fromstring(seq, np.int8) % 5]






