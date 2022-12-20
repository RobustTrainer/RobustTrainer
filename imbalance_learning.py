# coding=utf-8
"""
A set of classifiers used for long-tailed classification
"""
from utils import accuracy, DAverageMeter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import time


def f_measure(truths, predictions):
    tn, fp, fn, tp = confusion_matrix(truths, predictions).ravel()
    PD = tp / (tp + fn)  # recall
    PREC = tp / (tp + fp)  # precision
    F_MEASURE = 2 * PD * PREC / (PD + PREC)
    return F_MEASURE


class CRT(object):
    # classifier for class-balanced re-training
    def __init__(self, args, model, data_loader_train, data_loader_test):
        self.args = args
        self.model = model
        self.init_model()

        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.crt_learning_rate, momentum=args.crt_momentum,
                                         weight_decay=args.crt_weight_decay)
        self.scheduler = \
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=0.0002,
                                                       T_max=self.args.crt_nb_train_epochs - self.args.crt_start_epoch)

        self.all_measurements = []  # the metrics during the whole training progress
        self.all_predictions = []  # the predictions during the whole training progress

    def init_model(self):
        # initialize the model, freezing feature weights here.
        for param_name, param in self.model.named_parameters():
            if 'classifier' not in param_name:
                param.requires_grad = False

    def main_worker(self):
        start = time.time()
        for curr_epoch in range(self.args.crt_start_epoch, self.args.crt_nb_train_epochs):
            begin = time.time()
            print("Training epoch [%3d / %3d]" % (curr_epoch + 1, self.args.crt_nb_train_epochs))
            train_stats = self.train(self.data_loader_train, curr_epoch)  # train

            predictions, measurements = self.test(self.data_loader_test)  # eval
            self.all_predictions.append(predictions)
            self.all_measurements.append(measurements)
            end = time.time()
            print(end-begin)
        final = time.time()
        print(final - start)

    def train(self, data_loader, current_epoch):
        self.model.train()
        train_stats = DAverageMeter()
        self.scheduler.step()
        for idx, batch in enumerate(tqdm(data_loader)):
            curr_train_stats = self.train_step(batch)
            train_stats.update(curr_train_stats)
            if (idx + 1) % self.args.log_interval == 0:
                print('==> Iteration [%3d][%4d / %4d]: %s' % (current_epoch + 1, idx + 1, len(data_loader),
                                                              train_stats.average()))
        return train_stats.average()

    def train_step(self, batch):
        self.model.train()
        inputs = batch[0]
        targets = batch[1]
        inputs, targets = self.shuffle_batch(inputs, targets)
        inputs = inputs.cuda()
        targets = targets.cuda()
        record = {}

        _, _, outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        record['loss'] = loss.item()
        record['train_accuracy'] = accuracy(outputs, targets)[0].item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return record

    def test(self, data_loader):
        self.model.eval()
        nb_features = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        predictions = np.zeros(nb_features, dtype="int32")
        truths = np.zeros(nb_features, dtype="int32")
        for i, batch in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                inputs = batch[0].cuda()
                targets = batch[1].data.cpu().numpy()
                feat, intermediate_feat, output = self.model(inputs)
                _, preds = torch.max(output, 1)
                preds = preds.data.cpu().numpy()
            if i < len(data_loader) - 1:
                predictions[i * batch_size: (i + 1) * batch_size] = preds
                truths[i * batch_size: (i + 1) * batch_size] = targets
            else:
                predictions[i * batch_size:] = preds
                truths[i * batch_size:] = targets
        return predictions, f_measure(truths, predictions)

    @staticmethod
    def shuffle_batch(x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y
