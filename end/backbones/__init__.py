import torch
from .resnet import ResNet50, ResNetFeaturesLC

def soft_cross_entropy(pred, target):
    logprobs = torch.nn.functional.log_softmax(pred, dim=1)
    return -(target * logprobs).sum() / pred.shape[0]

class SoftCrossEntropy(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(SoftCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        lsm = pred.log_softmax(dim=1)
        loss = torch.sum(-target * lsm)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


def enable_dropout(m):
    for module in m.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

__all__ = ['ResNet50']