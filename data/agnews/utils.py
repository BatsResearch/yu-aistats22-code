import numpy as np
import torch

import joblib

def load_data(option, dat_loc='data/agnews/'):
    return joblib.load(dat_loc + option + '.jkl')


def get_cb(labels):
    counter = [0 for _ in range(len(np.unique(labels)))]
    for label in labels:
        counter[label] += 1
    counter = torch.FloatTensor(counter)
    return counter / counter.sum(0)


def get_inst(data):
    labels = []
    inst = []
    for d in data:
        labels.append(d['label'])
        inst.append(d)
    return inst, labels

