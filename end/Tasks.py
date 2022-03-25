import torch
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report
from tqdm.notebook import tqdm
from copy import deepcopy as dc
from attributes.utils import test_stats
from .backbones.resnet import r101_in_dim


class SoftCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, pred, target):
        lsm = pred.log_softmax(dim=1)
        loss = torch.sum(-target * lsm)
        return torch.mean(loss)


def MCA(pred, true):
    class_range = np.unique(true)
    results = []
    for cla in class_range:
        ind = np.where(true==cla)[0]
        results.append(np.mean(pred[ind]==true[ind]))
    return np.mean(results)



def awa_mca(pred, true, unseen_classes=None, print_by_class=False, std=True, epoch=-1):
    if unseen_classes is None or std is True:
        unseen_classes = [6, 8, 22, 23, 29, 30, 33, 40, 46, 49]
    if std is False:
        unseen_classes = list(range(10))
    labels = list(range(50))
    report = classification_report(true,
                                   pred,
                                   target_names=labels, output_dict=True)

    unseen_range = unseen_classes
    seen_range = [x for x in list(range(50)) if x not in unseen_classes]
    unseen_recall = []
    seen_recall = []
    for label in labels:
        item = report[label]
        r = item['recall']
        if label in unseen_range:
            unseen_recall.append(r)
        if label in seen_range:
            seen_recall.append(r)
    seen_mca = np.mean(seen_recall)
    unseen_mca = np.mean(unseen_recall)
    harmonic = (2 * seen_mca * unseen_mca) / (seen_mca + unseen_mca)
    print('Epoch {}: MCA: Seen: '.format(epoch), seen_mca, 'Unseen: ', unseen_mca, 'H: ', harmonic)
    if print_by_class:
        print('Seen: ', seen_recall)
        print('Unseen: ', unseen_recall)
    return seen_mca, unseen_mca, harmonic


class EndTask:
    def __init__(self, name,
                 num_class,
                 ModelConstructor,
                 device='cuda:0', save_loc='./',
                 feat_dim=None,
                 loss_function=None,
                 train_cfg=None):
        self.name = name
        self.save_loc = save_loc
        self.device = device
        self.best_model = None
        if feat_dim is None:
            feat_dim = r101_in_dim
        self.num_class = num_class
        self.model = ModelConstructor(in_features=feat_dim, target_dim=num_class)
        if loss_function is None:
            self.loss_function = SoftCrossEntropy()
        else:
            self.loss_function = loss_function
        if train_cfg is None:
            self.epochs = 11
        else:
            self.epochs = train_cfg['epoch']
            self.lr = train_cfg['lr']


    def train(self, train, val, test, mp=False, double=True, gzsl=False,
              supervised_mode=False, std=False, lowest_test_loss=False):

        if double:
            self.model.double()

        def stats_eval(to_save_model,
                       eval_mode='macro', return_pred=False, return_num_test=False):
            model.load_state_dict(to_save_model)
            model.eval()
            test_results = []
            test_true_labels = []
            for imgs, labels in test:
                imgs = imgs.to(init_device)
                test_true_labels += list(labels)
                primary_out = model(imgs)
                _, preds = torch.max(primary_out, 1)
                preds = preds.cpu().detach().numpy()
                test_results += list(preds)

            test_results = np.array(test_results)
            test_true_labels = np.array(test_true_labels)
            acc, p, r, f1 = test_stats(test_true_labels, test_results,
                                                       eval_mode=eval_mode,
                                       return_stats_by_class=False)
            acc = np.mean(test_results==test_true_labels)
            if return_pred and return_num_test:
                return test_results, test_true_labels, len(test_true_labels)
            if return_num_test:
                return [acc, p, r, f1], len(test_true_labels)

            return [acc, p, r, f1]

        tloss_to_save_model = None
        minimal_train_loss = float('inf')
        best_oe_trloss = -1
        avg_losses = []
        epochs = self.epochs

        if not mp:
            init_device = end_device = self.device
            model = self.model.to(self.device)

        if not double:
            label_reference = torch.eye(self.num_class).float()
        else:
            label_reference = torch.eye(self.num_class).double()

        initial_lr = 1e-4
        # initial_lr = 4e-5
        optim = torch.optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, min_lr=1e-8, factor=1e-1)
        self.loss_function = self.loss_function.to(end_device)

        epoch_time = []
        save_res = []
        for i in range(1, epochs + 1):
            model.train()
            # progress = tqdm(total=len(train), desc='epoch % 3d' % i)
            train_loss = []
            train_acc = []
            start = time.time()
            for imgs, labels in train:
                optim.zero_grad()
                if double:
                    imgs = torch.DoubleTensor(imgs).to(init_device)
                else:
                    imgs = torch.FloatTensor(imgs).to(init_device)
                if supervised_mode:
                    labels = labels.long().to(end_device)
                else:
                    labels = torch.DoubleTensor(labels).to(end_device)
                pred = model(imgs)
                loss = self.loss_function(pred, labels)
                train_loss.append(loss.item())
                loss.backward()
                optim.step()
                _, preds = torch.max(pred, 1)
                preds = preds.cpu().detach().numpy()

                if supervised_mode:
                    accuracy = float(np.sum(preds
                    == labels)) / imgs.shape[0]
                else:
                    accuracy = float(np.sum(preds
                        == np.argmax(labels.cpu().detach().numpy(), axis=1))) / imgs.shape[0]
                train_acc.append(accuracy)
                # progress.set_postfix({'train loss': np.mean(train_loss), 'train acc': np.mean(train_acc[-50:])})
                # progress.update()
            epoch_train_loss = np.mean(train_loss)
            avg_losses.append(epoch_train_loss)
            if scheduler is not None:
                scheduler.step(epoch_train_loss)
            model.eval()
            val_results = []
            val_true_labels = []
            val_acc = []
            val_loss = []
            for imgs, labels in val:
                if double:
                    imgs = torch.DoubleTensor(imgs).to(init_device)
                    if supervised_mode:
                        labels_torch = labels.long().to(end_device)
                    else:
                        labels_torch = torch.DoubleTensor(label_reference[labels.long()]).to(end_device)
                else:
                    imgs = torch.FloatTensor(imgs).to(init_device)
                    if supervised_mode:
                        labels_torch = labels.long().to(end_device)
                    else:
                        labels_torch = torch.FloatTensor(label_reference[labels.long()]).to(end_device)
                primary_out = model(imgs)
                loss = self.loss_function(primary_out, labels_torch)
                val_loss.append(loss.item())
                _, preds = torch.max(primary_out, 1)
                preds = preds.cpu().detach().numpy()
                val_acc.append(float(np.sum(np.array(labels) == preds)) / imgs.shape[0])
                val_true_labels += list(labels)
                val_results += list(preds)

            val_true_labels = np.array(val_true_labels)
            val_results = np.array(val_results)
            if gzsl:
                seen_mca, unseen_mca, harmonic = awa_mca(val_results, val_true_labels, std=std, epoch=i)
            else:
                report = classification_report(val_true_labels,
                                               val_results,
                                               target_names=list(range(10)), output_dict=True)
                res = []
                for label in list(range(10)):
                    item = report[label]
                    r = item['recall']
                    res.append(r)

            print(
                'Epoch {:d} =|= Train: Loss: {:2f}, Acc:{:4f} || Val: Loss: {:2f}, Acc:{:4f} || Precision: {:4f}, Recall: {:4f} MCA: {:4f}'
                    .format(i, np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), acc, p, r, mca)
            )

            if epoch_train_loss < minimal_train_loss:
                minimal_train_loss = epoch_train_loss
                tloss_to_save_model = dc(model.state_dict())
                best_oe_trloss = i
            epoch_time.append(time.time() - start)
        if lowest_test_loss:
            tracc_tuples = stats_eval(tloss_to_save_model)
            print('Best Tloss OE', best_oe_trloss)
            print('Unseen Tloss Test Acc: ', tracc_tuples[:5])
        if gzsl:
            sr = stats_eval(tloss_to_save_model, return_pred=True, return_num_test=True)
            pred, true, _ = sr
            # awa_mca(pred, true, print_by_class=True)