from typing import Dict, Union

from overrides import overrides
import torch
from allennlp.training.metrics import F1Measure, CategoricalAccuracy
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
import torch.nn as nn

class SoftCrossEntropy(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(SoftCrossEntropy, self).__init__()
        self.reduction = reductione
    def forward(self, pred, target):
        lsm = pred.log_softmax(dim=1)
        loss = torch.sum(-target * lsm)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


@Model.register("bert_classifier")
class BertClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        freeze_encoder: bool = True,
        out_features: int = 10
    ) -> None:
        super().__init__(vocab)

        self.vocab = vocab
        self.embedder = embedder
        self.freeze_encoder = freeze_encoder

        for parameter in self.embedder.parameters():
            parameter.requires_grad = not self.freeze_encoder

        in_features = self.embedder.get_output_dim()

        self._classification_layer = torch.nn.Linear(in_features, out_features)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, out_features)
        )

        self._accuracy = CategoricalAccuracy()
        self._loss = SoftCrossEntropy()
        self.out_features = out_features
        self.accuracy = CategoricalAccuracy()

        self.label_f1_metrics = {}
        for i in range(self.out_features):
            self.label_f1_metrics[i] = \
                F1Measure(positive_label=i)


    def forward(
        self, text, label = None
    ):

        # (batch_size, max_len, embedding_dim)
        embeddings = self.embedder(text)

        # the first embedding is for the [CLS] token
        # NOTE: this pre-supposes BERT encodings; not the most elegant!
        # (batch_size, embedding_dim)
        cls_embedding = embeddings[:, 0, :]
        # apply classification layer
        logits = self.classifier(cls_embedding)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        probs = probs.cpu().detach()
        if label is None:
            return {'prob': probs}

        loss = self._loss(logits, label)

        # res = torch.argmax(pred.cpu().detach(), dim=1)
        res = logits.cpu().detach()
        labelc = torch.argmax(label.cpu().detach(), dim=1)
        for i in range(self.out_features):
            metric = self.label_f1_metrics[i]
            metric(res, labelc)

        self.accuracy(res, labelc)

        return {'loss': loss, 'prob': probs, 'labels': labelc}

    def get_metrics(self, reset: bool = False):
        metric_dict = {}

        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            metric_val = list(metric_val.values())
            metric_dict[str(name) + '_P'] = metric_val[0]
            metric_dict[str(name) + '_R'] = metric_val[1]
            metric_dict[str(name) + '_F1'] = metric_val[2]
            if name != 'none':  # do not consider `none` label in averaging F1
                sum_f1 += metric_val[2]

        names = list(self.label_f1_metrics.keys())
        total_len = len(names) if 'none' not in names else len(names) - 1
        average_f1 = sum_f1 / total_len
        # metric_dict['combined_metric'] = (accuracy + average_f1) / 2
        metric_dict['average_F1'] = average_f1
        metric_dict['acc'] = self.accuracy.get_metric(reset)
        # metric_dict['acc'] = self.accuracy.get_metric(reset)
        return metric_dict