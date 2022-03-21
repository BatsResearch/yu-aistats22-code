import random
import re
import string
import numpy as np
import json

import torch
from allennlp.data.data_loaders import  SimpleDataLoader
from allennlp.data import Vocabulary, Instance, Token
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer

from copy import deepcopy as dcopy
from allennlp.data.fields import LabelField, ArrayField, TextField
from allennlp.training.optimizers import AdamWOptimizer
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import CosineWithRestarts
from allennlp.training.util import evaluate
from end.backbones.text_classifiers import BertClassifier

from allennlp.data.tokenizers.pretrained_transformer_tokenizer \
        import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer \
        import PretrainedTransformerIndexer
from allennlp.modules.token_embedders.pretrained_transformer_embedder \
        import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders.basic_text_field_embedder \
        import BasicTextFieldEmbedder

raw_dataset = 'data/scicite/{:s}.jsonl'

label_idx = ['background', 'method', 'result']

def load_jsonl(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def textsplit(instance):
    string = instance['string']
    try:
        pretext = string[:int(instance['citeStart'])]
        postext = string[int(instance['citeEnd']):]
        mid = string[int(instance['citeStart']):int(instance['citeEnd'])]
    except ValueError:
        pretext = mid = postext = ''
    return pretext, mid, postext


def proc_inst(instance):
    p, m, s = textsplit(instance)
    new_sen = re.sub(r'\([^)]*\)', '', p + s).lower()
    new_sen = new_sen.translate(str.maketrans('', '', string.punctuation))
    # new_sen = p+m+s
    return new_sen


def extract_inst_label_pair(file, pack=False, ignore_label=False):
    inst = []
    labels = []
    for item in file:
        sentence = proc_inst(item)
        inst.append(sentence)
        if not ignore_label:
            label = item['label']
            labels.append(label)
        else:
            pack = False
    if pack:
        return list(zip(inst, labels))
    return inst, labels


def to_instance(tuples, tokenizer, token_indexers, soft_labeled=False):
    onehot = np.eye(3)
    data = []
    for item, label in tuples:
        if soft_labeled:
            label_field = ArrayField(label)
        else:
            label_field = ArrayField(onehot[label_idx.index(label)])
        fields = {'text': TextField(tokenizer.tokenize(item),
            token_indexers=token_indexers), 'label': label_field}
        data.append(Instance(fields))
    return data



def build_allentrainer(model, train_dl, val_dl, save_loc='output/'):
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamWOptimizer(parameters, lr=5e-7)
    default_epoch = 20
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=train_dl,
        validation_data_loader=val_dl,
        num_epochs=default_epoch,
        optimizer=optimizer,
        patience=8,
        serialization_dir=save_loc,
        cuda_device=0,
        use_amp=False,
        grad_clipping=2.0,
        validation_metric='+average_F1'
    )
    return trainer


def process_voc(inst, save=False):
    vocab = Vocabulary.from_instances(inst)
    if save:
        vocab.save_to_files('vocabulary/')
    vocab.print_statistics()
    return vocab


def main(soft_label_loc=None, rand=-2):

    if rand >= 0:
        seeds = [500, 600, 700, 800, 900]
        random_seed = pt_seed = np_seed = seeds[rand]
        random.seed(random_seed)
        torch.manual_seed(pt_seed)
        np.random.seed(np_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(pt_seed)
    else:
        rand = 1e3

    model_string = "bert-base-uncased"

    tokenizer = PretrainedTransformerTokenizer(model_string)
    token_indexer = {'tokens': PretrainedTransformerIndexer(
        model_string)}

    train = load_jsonl(dcopy(raw_dataset).format('train'))
    val = load_jsonl(dcopy(raw_dataset).format('dev'))
    num_class = 3

    print('Soft Label :', soft_labels_loc)
    sla = np.load(soft_label_loc, allow_pickle=True).item()
    training_soft_labels = sla['sl']
    train_inst, _ = extract_inst_label_pair(train, pack=False)
    idx = sla['idx']
    if idx is not None:
        train_inst = [inst for i, inst in enumerate(train_inst) if i in idx]

    train_tuple = list(zip(train_inst, training_soft_labels))
    save_loc = './scratch/scicite/'
    soft_labeled = True
    val_tuple = extract_inst_label_pair(val, pack=True)
    train_insts = to_instance(train_tuple, tokenizer=tokenizer, token_indexers=token_indexer, soft_labeled=soft_labeled)
    val_insts = to_instance(val_tuple, tokenizer=tokenizer, token_indexers=token_indexer)

    test = load_jsonl(dcopy(raw_dataset).format('test'))
    test_tuple = extract_inst_label_pair(test, pack=True)
    test_insts = to_instance(test_tuple, tokenizer=tokenizer,
                             token_indexers=token_indexer)
    vocabulary = process_voc(train_insts + val_insts + test_insts, save=True)

    bert_token_embedder = PretrainedTransformerEmbedder(model_string)
    bert_textfield_embedder = BasicTextFieldEmbedder(
        {"tokens": bert_token_embedder})

    model = BertClassifier(
        vocabulary, bert_textfield_embedder, freeze_encoder=False, out_features=3)

    model = model.to('cuda:0')

    train_dl = SimpleDataLoader(train_insts, batch_size=16, shuffle=True, vocab=vocabulary)
    val_dl = SimpleDataLoader(val_insts, batch_size=16, shuffle=False, vocab=vocabulary)

    trainer = build_allentrainer(model, train_dl, val_dl, save_loc)
    print("Starting training")
    trainer.train()
    print("Finished training")
    test_dl = SimpleDataLoader(test_insts, batch_size=16, shuffle=False, vocab=vocabulary)
    results = evaluate(model, test_dl, cuda_device=0)
    print(results)


if __name__ == '__main__':
    import sys

    try:
        soft_labels_loc = sys.argv[1]
    except IndexError:
        soft_labels_loc = None

    try:
        seed = int(sys.argv[2])
    except:
        seed = 1

    main(soft_label_loc=soft_labels_loc,  rand=seed)
