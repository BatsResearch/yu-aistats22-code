import numpy as np
import torch
import spacy

nlp = spacy.load("en_core_web_sm")
label_idx = ['ABBR','DESC', 'ENTY', 'HUM', 'LOC', 'NUM']

def load_data(option, dat_loc='data/trec/'):
    data = np.load(dat_loc+option+'.npy', allow_pickle=True)
    dataset = []
    for inst in data:
        dataset.append({'label': label_idx.index(inst['label']),
                        'string': inst['string'],
                       'fg': inst['fg_label']})
    return dataset

def get_cb(labels):
    counter = [0 for _ in range(len(np.unique(labels)))]
    for label in labels:
        counter[label] += 1
    counter = torch.FloatTensor(counter)
    return counter / counter.sum(0)

def get_inst(data):
    labels = []
    for d in data:
        labels.append(d['label'])
    return data, labels

def topk_label(soft_labels, true_labels, top_k=2):
    top1 = np.argmax(soft_labels, axis=1)+1
    topkk = np.argsort(soft_labels)[:, -top_k:]+1
    for i, val in enumerate(true_labels):
        if val in topkk[i, -top_k:]:
            top1[i] = val
    return top1


def get_subject(sentence, lemma=True):
    doc = nlp(sentence)
    subjects = []
    for token in doc:
        if str(token.dep_) == "nsubj" and not token.is_stop:
            subjects.append(str(token.lemma_) if lemma else token.text)
    if len(subjects) == 1:
        return subjects[0]
    else:
        return None