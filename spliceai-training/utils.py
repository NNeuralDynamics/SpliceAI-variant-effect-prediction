import numpy as np
from sklearn.metrics import average_precision_score


def print_topl_statistics(y_true, y_pred):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.

    idx_true = np.nonzero(y_true == 1)[0]
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    topkl_accuracy = []
    threshold = []

    for top_length in [0.5, 1, 2, 4]:
        idx_pred = argsorted_y_pred[-int(top_length*len(idx_true)):]

        topkl_accuracy.append(np.size(np.intersect1d(idx_true, idx_pred)) /
                              float(min(len(idx_pred), len(idx_true))))
        threshold.append(sorted_y_pred[-int(top_length*len(idx_true))])

    auprc = average_precision_score(y_true, y_pred)
    
    print(f"Top k Accuracies: [{topkl_accuracy[0]:.4f}, {topkl_accuracy[1]:.4f}, {topkl_accuracy[2]:.4f},{topkl_accuracy[3]:.4f}], AUPRC: {auprc:.4f}")

def one_hot_encode_sequence(seq):

    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    return map[np.fromstring(seq, np.int8) % 5]