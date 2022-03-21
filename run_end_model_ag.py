import random
import numpy as np
import sys
import torch
from allennlp.data.data_loaders import  SimpleDataLoader
from allennlp.data import Vocabulary, Instance, Token
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from copy import deepcopy as dcopy
from allennlp.data.fields import LabelField, ArrayField, TextField
from allennlp.training.optimizers import AdamWOptimizer
from allennlp.training.learning_rate_schedulers import CosineWithRestarts
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.util import evaluate
import joblib
from end.backbones.text_classifiers import BertClassifier


from allennlp.data.tokenizers.pretrained_transformer_tokenizer \
        import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer \
        import PretrainedTransformerIndexer
from allennlp.modules.token_embedders.pretrained_transformer_embedder \
        import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders.basic_text_field_embedder \
        import BasicTextFieldEmbedder

raw_dataset = 'data/agnews/{:s}.jkl'


def load_jkl(file):
    data = joblib.load(file)
    return data


def proc_inst(instance):
    return instance['string']


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
    onehot = np.eye(4)
    data = []
    for item, label in tuples:
        if soft_labeled:
            label_field = ArrayField(label)
        else:
            label_field = ArrayField(onehot[label])
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
        patience=5,
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


def main(soft_label_loc=None, rand=-2,):

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

    train = load_jkl(dcopy(raw_dataset).format('train'))
    val = load_jkl(dcopy(raw_dataset).format('dev'))

    print('Soft Label :', soft_labels_loc)
    sla = np.load(soft_label_loc, allow_pickle=True).item()
    training_soft_labels = sla['sl']
    train_inst, _ = extract_inst_label_pair(train, pack=False)
    idx = sla['idx']
    if idx is not None:
        train_inst = [inst for i, inst in enumerate(train_inst) if i in idx]

    train_inst = [inst for i, inst in enumerate(train_inst) if i in idx]
    print(len(train_inst), len(idx), training_soft_labels.shape)
    train_tuple = list(zip(train_inst, training_soft_labels))
    soft_labeled = True
    save_loc = 'scratch/agnews/'
    val_tuple = extract_inst_label_pair(val, pack=True)
    train_insts = to_instance(train_tuple, tokenizer=tokenizer, token_indexers=token_indexer, soft_labeled=soft_labeled)
    val_insts = to_instance(val_tuple, tokenizer=tokenizer, token_indexers=token_indexer)

    test = load_jkl(dcopy(raw_dataset).format('test'))
    test_tuple = extract_inst_label_pair(test, pack=True)
    test_insts = to_instance(test_tuple, tokenizer=tokenizer,
                             token_indexers=token_indexer)
    vocabulary = process_voc(train_insts + val_insts + test_insts, save=True)

    num_class = 4
    bert_token_embedder = PretrainedTransformerEmbedder(model_string)
    bert_textfield_embedder = BasicTextFieldEmbedder(
        {"tokens": bert_token_embedder})
    model = BertClassifier(
        vocabulary, bert_textfield_embedder, freeze_encoder=False, out_features=4)
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


    try:
        soft_labels_loc = sys.argv[1]
    except IndexError:
        soft_labels_loc = None

    try:
        seed = int(sys.argv[2])
    except:
        seed = 1

    main(soft_label_loc=soft_labels_loc,  rand=seed)
