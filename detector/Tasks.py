from pathlib import Path
import torch
import numpy as np
import time
import sys
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from copy import deepcopy as dc
from attributes.utils import test_stats
from nplm.plf.image import r50_in_dim, r101_in_dim
import sys


def enable_dropout(m):
    for module in m.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


class DetectorTask:
    def __init__(self, name, target_aids, ModelConstructor,
                 save_loc='./saved_results/', device='cuda:0',
                 load_detectors=False):
        self.name = name
        self.target_aids = target_aids

        Path(save_loc).mkdir(parents=True, exist_ok=True)
        self.save_loc = save_loc
        self.ModelConstructor = ModelConstructor
        self.detectors_tasks = {}
        for aid in target_aids:
            self.detectors_tasks[aid] = IndividualDetectorTask(name, aid, ModelConstructor,
                                                                     device=device, save_loc=save_loc,
                                                                     load_model=load_detectors)

    def train_detectors(self, train, val, test, aids=None):
        if aids is None:
            aids = self.target_aids
        for aid in aids:
            self.detectors_tasks[aid].train(train,test,val)

    def annotate(self, instances, aids=None,
                 save_votes=False):
        if aids is None:
            aids = self.target_aids
        votes = np.empty([len(instances), len(self.target_aids)])
        for idx, aid in enumerate(aids):
            votes[:,idx] = self.detectors_tasks[aid].annotate(instances)

        if save_votes:
            np.save(self.save_loc+'detector_votes_{:d}'.format(len(aids)), votes)

        return votes


class JointDetectorTask:
    def __init__(self, name, aid_range,
                 ModelConstructor,
                 device='cuda:0', save_loc='./', num_features=None,
                 num_train=-1, num_test=-1, num_val=-1):
        self.name = name
        self.aid_range = aid_range
        self.save_loc = save_loc
        self.device = device
        self.best_model = None
        if num_features is None:
            num_features = r101_in_dim
        self.model = ModelConstructor(in_features=num_features)
        Path(save_loc + '{:s}_saved_models/'.format(name)).mkdir(parents=True, exist_ok=True)
        Path(save_loc + 'unseen_test_acc/').mkdir(parents=True, exist_ok=True)

        self.bin_default_class_count = 85
        self.num_train = num_train
        self.num_test = num_test
        self.num_val = num_val


    def train(self, train, val, test,
        save_model=True, mp=False, annotate=None, double=True, use_tqdm=False):

        self.double=double
        if double:
            self.model.double()

        def stats_eval(to_save_model, eval_mode='macro'):
            model.load_state_dict(to_save_model)
            model.eval()
            for aid in self.aid_range:
                test_results = []
                test_true_labels = []
                for imgs, labels in test:
                    imgs = imgs.to(init_device)
                    test_true_labels += list(labels[:, aid])
                    primary_out = model(imgs)
                    _, preds = torch.max(primary_out, 1)
                    preds = preds.cpu().detach().numpy()
                    test_results += list(preds[:,aid])

                test_results = np.array(test_results)
                test_true_labels = np.array(test_true_labels)
                acc, p, r, f1 = test_stats(test_true_labels, test_results,
                                                           eval_mode=eval_mode, return_stats_by_class=False)

                pos_index = np.where(test_true_labels == 1)[0]
                neg_index = np.where(test_true_labels == 0)[0]

                p_acc = accuracy_score(test_true_labels[pos_index], test_results[pos_index])
                n_acc = accuracy_score(test_true_labels[neg_index], test_results[neg_index])
                acc = np.mean(test_results==test_true_labels)
                test_acc_tuple = [(p_acc, n_acc), acc, p, r, f1]

                test_results = []
                test_true_labels = []
                for imgs, labels in val:
                    imgs = imgs.to(init_device)
                    test_true_labels += list(labels[:, self.aid])
                    primary_out = model(imgs)
                    _, preds = torch.max(primary_out, 1)
                    preds = preds.cpu().detach().numpy()
                    test_results += list(preds)

                test_results = np.array(test_results)
                test_true_labels = np.array(test_true_labels)
                acc, p, r, f1 = test_stats(test_true_labels, test_results,
                                           eval_mode=eval_mode, return_stats_by_class=False)

                pos_index = np.where(test_true_labels == 1)[0]
                neg_index = np.where(test_true_labels == 0)[0]
                p_acc = accuracy_score(test_true_labels[pos_index], test_results[pos_index])
                n_acc = accuracy_score(test_true_labels[neg_index], test_results[neg_index])
                acc = np.mean(test_results==test_true_labels)
                val_acc_tuple = [(p_acc, n_acc), acc, p, r, f1]

            return test_acc_tuple, val_acc_tuple

        acc_to_save_model = None
        best_oe_acc = -1
        best_oe_f1 = -1
        best_f1 = -1
        best_p = -1
        best_acc = -1
        lowest_entropy = 1000
        avg_losses = []
        epochs = 2000
        step_schedule = 500
        LR_milestones = list(
            filter(
                lambda a: a > 0,
                [i if (i % step_schedule == 0) else -1 for i in range(epochs)]
            ))
        if not mp:
            init_device = end_device = self.device
            model = self.model.to(self.device)

        #optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, LR_milestones, gamma=0.1)

        #scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=1e-5, max_lr=0.1)
        bceloss = torch.nn.BCEWithLogitsLoss()

        epoch_time = []
        for i in range(1, epochs + 1):
            model.train()
            if use_tqdm:
                progress = tqdm(total=len(train), desc='epoch % 3d' % i)
            train_loss = []
            train_acc = []
            start = time.time()
            for imgs, labels in train:
                optim.zero_grad()
                if double:
                    imgs = torch.DoubleTensor(imgs).to(init_device)
                else:
                    imgs = torch.FloatTensor(imgs).to(init_device)
                labels = labels.long().to(end_device)
                pred = torch.sigmoid(model(imgs))
                loss = bceloss(pred, labels)
                train_loss.append(loss.item())
                loss.backward()
                optim.step()
                scheduler.step()
                _, preds = torch.max(pred, 1)
                preds = preds.cpu().detach().numpy()
                accuracy = float(np.sum(preds == labels.cpu().detach().numpy())) / imgs.shape[0]
                train_acc.append(accuracy)
                if use_tqdm:
                    progress.set_postfix({'train loss': np.mean(train_loss), 'train acc': np.mean(train_acc[-50:])})
                    progress.update()

            avg_losses.append(np.mean(train_acc))

            model.eval()
            val_results = []
            val_true_labels = []
            val_acc = []
            for imgs, labels in val:
                if double:
                    imgs = torch.DoubleTensor(imgs).to(init_device)
                else:
                    imgs = torch.FloatTensor(imgs).to(init_device)
                primary_out = model(imgs)
                _, preds = torch.max(primary_out, 1)
                preds = preds.cpu().detach().numpy()
                val_acc.append(float(np.sum(np.array(labels == preds))) / imgs.shape[0])
                if use_tqdm:
                    progress.set_postfix({'valid acc': np.mean(val_acc[-50:])})
                    progress.update()

            acc = np.mean(val_acc)
            print(
                'Epoch {:d} =|= Train: Loss: {:2f}, Acc:{:4f} || Val: Acc:{:4f}'
                    .format(i, np.mean(train_loss), np.mean(train_acc), acc),
                file=sys.stderr
            )

            if acc > best_acc:
                print(
                    'ACC MODEL: Overiding previously saved model with {:2f} acc to current one with {:2f} acc'
                        .format(best_acc, acc),
                file=sys.stderr
                )
                best_acc = acc
                acc_to_save_model = dc(model.state_dict())
                best_oe_acc = i

            epoch_time.append(time.time() - start)

        if acc_to_save_model is None:
            acc_to_save_model = dc(model.state_dict())
            best_oe_acc = epochs


        if annotate is not None:
            model.load_state_dict(acc_to_save_model)
            model.eval()
            test_results = []
            for imgs, _ in annotate:
                if double:
                    imgs = torch.DoubleTensor(imgs).to(init_device)
                else:
                    imgs = torch.FloatTensor(imgs).to(init_device)
                primary_out = self.model(imgs)
                preds = torch.sigmoid(primary_out)
                preds = preds.cpu().detach().numpy()
                test_results.append(preds)
            return test_results


    def annotate(self, instances, double=True):
        if self.best_model is None:
            raise RuntimeError('Detector not trained, best model weights not available!')

        self.model.load_state_dict(self.best_model)
        self.model.eval()
        test_results = []
        for imgs, _ in instances:
            if double:
                imgs = torch.DoubleTensor(imgs).to('cuda:0')
            else:
                imgs = torch.FloatTensor(imgs).to('cuda:0')
            primary_out = self.model(imgs)
            preds = torch.sigmoid(primary_out)
            preds = preds.cpu().detach().numpy()
            test_results.append(preds)
        return test_results


class IndividualDetectorTask:
    def __init__(self, name, aid,
                 ModelConstructor,
                 device='cuda:0', save_loc='./',
                 load_model=False, num_features=None,
                 num_train=-1, num_test=-1, num_val=-1):
        self.name = name
        self.aid = aid
        self.save_loc = save_loc
        self.device = device
        self.best_model = None
        if num_features is None:
            num_features = r101_in_dim
        self.model = ModelConstructor(in_features=num_features)
        Path(save_loc + '{:s}_saved_models/'.format(name)).mkdir(parents=True, exist_ok=True)
        Path(save_loc + 'unseen_test_acc/').mkdir(parents=True, exist_ok=True)
        if load_model:
            try:
                model_loc = save_loc + '{:s}_saved_models/{:d}_detector.ptm'.format(self.name, self.aid)
                self.best_model = torch.load(model_loc, map_location='cpu')
            except FileNotFoundError:
                self.best_model = None

        self.bin_default_class_count = 2

        self.num_train = num_train
        self.num_test = num_test
        self.num_val = num_val


    def mc_prediction(self, test, num_test_samples, mc_samples=500, device='cuda:0',
                      val_mode=False, model=None):
        dropout_predictions = np.empty([0, num_test_samples, self.bin_default_class_count])
        if model is None:
            self.model.load_state_dict(self.best_model)
        else:
            self.model = model
        self.model = self.model.to(device)
        softmax = torch.nn.Softmax(dim=1)

        true_labels = []
        for fp in range(mc_samples):
            pred = np.empty([0, self.bin_default_class_count])
            self.model.eval()
            enable_dropout(self.model)
            for imgs, labels in test:
                imgs = imgs.to(device)
                with torch.no_grad():
                    primary_out = self.model(imgs)
                    sfm = softmax(primary_out)
                pred = np.vstack((pred, sfm.cpu().numpy()))
                if fp == 0 and val_mode:
                    true_labels += list(labels[:, self.aid])
            dropout_predictions = np.vstack((dropout_predictions,
                                             pred[np.newaxis, :, :]))
        mean = np.mean(dropout_predictions, axis=0)
        if val_mode:
            return np.argmax(mean, axis=1), np.array(true_labels)

        variance = np.var(dropout_predictions, axis=0)
        epsilon = sys.float_info.min
        entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)
        return mean, variance, entropy

    def train(self, train, val, test,
        save_model=True, mp=False, annotate=None, double=True, use_tqdm=False, lowest_uncertainty=False):

        self.double=double
        if double:
            self.model.double()

        def stats_eval(to_save_model, eval_mode='macro'):
            model.load_state_dict(to_save_model)
            model.eval()
            test_results = []
            test_true_labels = []
            for imgs, labels in test:
                imgs = imgs.to(init_device)
                test_true_labels += list(labels[:, self.aid])
                primary_out = model(imgs)
                _, preds = torch.max(primary_out, 1)
                preds = preds.cpu().detach().numpy()
                test_results += list(preds)

            test_results = np.array(test_results)
            test_true_labels = np.array(test_true_labels)
            acc, p, r, f1 = test_stats(test_true_labels, test_results,
                                                       eval_mode=eval_mode, return_stats_by_class=False)

            pos_index = np.where(test_true_labels == 1)[0]
            neg_index = np.where(test_true_labels == 0)[0]

            p_acc = accuracy_score(test_true_labels[pos_index], test_results[pos_index])
            n_acc = accuracy_score(test_true_labels[neg_index], test_results[neg_index])
            acc = np.mean(test_results==test_true_labels)
            test_acc_tuple = [(p_acc, n_acc), acc, p, r, f1]

            test_results = []
            test_true_labels = []
            for imgs, labels in val:
                imgs = imgs.to(init_device)
                test_true_labels += list(labels[:, self.aid])
                primary_out = model(imgs)
                _, preds = torch.max(primary_out, 1)
                preds = preds.cpu().detach().numpy()
                test_results += list(preds)

            test_results = np.array(test_results)
            test_true_labels = np.array(test_true_labels)
            acc, p, r, f1 = test_stats(test_true_labels, test_results,
                                       eval_mode=eval_mode, return_stats_by_class=False)

            pos_index = np.where(test_true_labels == 1)[0]
            neg_index = np.where(test_true_labels == 0)[0]
            p_acc = accuracy_score(test_true_labels[pos_index], test_results[pos_index])
            n_acc = accuracy_score(test_true_labels[neg_index], test_results[neg_index])
            acc = np.mean(test_results==test_true_labels)
            val_acc_tuple = [(p_acc, n_acc), acc, p, r, f1]

            return test_acc_tuple, val_acc_tuple

        acc_to_save_model = None
        best_oe_acc = -1
        best_oe_f1 = -1
        best_f1 = -1
        best_p = -1
        best_acc = -1
        lowest_entropy = 1000
        avg_losses = []
        epochs = 500
        step_schedule = 200
        LR_milestones = list(
            filter(
                lambda a: a > 0,
                [i if (i % step_schedule == 0) else -1 for i in range(epochs)]
            ))
        if not mp:
            init_device = end_device = self.device
            model = self.model.to(self.device)

        #optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, LR_milestones, gamma=0.1)

        #scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=1e-5, max_lr=0.1)
        celoss = torch.nn.CrossEntropyLoss()

        epoch_time = []
        for i in range(1, epochs + 1):
            model.train()
            if use_tqdm:
                progress = tqdm(total=len(train), desc='epoch % 3d' % i)
            train_loss = []
            train_acc = []
            start = time.time()
            for imgs, labels in train:
                optim.zero_grad()
                if double:
                    imgs = torch.DoubleTensor(imgs).to(init_device)
                else:
                    imgs = torch.FloatTensor(imgs).to(init_device)
                labels = labels[:, self.aid].long().to(end_device)
                pred = model(imgs)
                loss = celoss(pred, labels)
                train_loss.append(loss.item())
                loss.backward()
                optim.step()
                scheduler.step()
                _, preds = torch.max(pred, 1)
                preds = preds.cpu().detach().numpy()

                accuracy = float(np.sum(preds == labels.cpu().detach().numpy())) / imgs.shape[0]
                train_acc.append(accuracy)
                if use_tqdm:
                    progress.set_postfix({'train loss': np.mean(train_loss), 'train acc': np.mean(train_acc[-50:])})
                    progress.update()

            avg_losses.append(np.mean(train_acc))

            model.eval()
            val_results = []
            val_true_labels = []
            val_acc = []
            for imgs, labels in val:
                if double:
                    imgs = torch.DoubleTensor(imgs).to(init_device)
                else:
                    imgs = torch.FloatTensor(imgs).to(init_device)
                primary_out = model(imgs)
                _, preds = torch.max(primary_out, 1)
                preds = preds.cpu().detach().numpy()

                val_acc.append(float(np.sum(np.array(labels[:, self.aid]) == preds)) / imgs.shape[0])
                val_true_labels += list(labels[:, self.aid])
                val_results += list(preds)
                if use_tqdm:
                    progress.set_postfix({'valid acc': np.mean(val_acc[-50:])})
                    progress.update()
            val_true_labels = np.array(val_true_labels)
            val_results = np.array(val_results)
            acc, p, r, f1 = test_stats(val_true_labels, val_results,
                                       return_stats_by_class=False)
            print(
                'Epoch {:d} =|= Train: Loss: {:2f}, Acc:{:4f} || Val: Acc:{:4f} || Precision: {:4f}, Recall: {:4f}'
                    .format(i, np.mean(train_loss), np.mean(train_acc), acc, p, r),
                file=sys.stderr
            )

            if acc > best_acc:
                print(
                    'ACC MODEL: Overiding previously saved model with {:2f} p to current one with {:2f} p'
                        .format(best_acc, acc),
                file=sys.stderr
                )
                best_acc = acc
                acc_to_save_model = dc(model.state_dict())
                best_oe_acc = i

            epoch_time.append(time.time() - start)

        if acc_to_save_model is None:
            acc_to_save_model = dc(model.state_dict())
            best_oe_acc = epochs

        acc_tuple_test, acc_tuple_val = stats_eval(acc_to_save_model)
        print('Best OE', best_oe_acc)
        print('Unseen Test Acc: ', acc_tuple_test[:5])
        print('Val Test Acc: ', acc_tuple_val[:5])
        np.save(self.save_loc + 'unseen_test_acc/test_{:d}_acc'.format(self.aid), acc_tuple_test[:5])
        np.save(self.save_loc + 'unseen_test_acc/val_{:d}_acc'.format(self.aid), acc_tuple_val[:5])
        if save_model:
            torch.save(acc_to_save_model, self.save_loc + '{:s}_saved_models/'.format(self.name) +
                       '{:d}_detector.ptm'.format(self.aid))
        self.best_model = acc_to_save_model

        if annotate is not None:
            model.load_state_dict(acc_to_save_model)
            model.eval()
            test_results = []
            for imgs, _ in annotate:
                if double:
                    imgs = torch.DoubleTensor(imgs).to(init_device)
                else:
                    imgs = torch.FloatTensor(imgs).to(init_device)
                primary_out = self.model(imgs)
                _, preds = torch.max(primary_out, 1)
                preds = preds.cpu().detach().numpy()
                test_results += list(preds)
            return test_results


    def annotate(self, instances):
        if self.best_model is None:
            raise RuntimeError('Detector not trained, best model weights not available!')

        self.model.load_state_dict(self.best_model)
        self.model.eval()
        test_results = []
        for imgs, _ in instances:
            if self.double:
                imgs = torch.DoubleTensor(imgs).to(self.device)
            else:
                imgs = torch.FloatTensor(imgs).to(self.device)
            primary_out = self.model(imgs)
            _, preds = torch.max(primary_out, 1)
            preds = preds.cpu().detach().numpy()
            test_results += list(preds)
        return test_results