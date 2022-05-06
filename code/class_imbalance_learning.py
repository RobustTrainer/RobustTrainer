# coding=utf-8
"""
A set of classifiers used for long-tailed classification
"""
from class_prototypes import get_prototypes
from tau_norm_classifier import tau_norm_classifier
from train_utils import accuracy, DAverageMeter
from metrics import report_measurements_of_classifier
from tqdm import tqdm
from sklearn import preprocessing
from scipy.special import softmax
import numpy as np
import torch.nn as nn
import torch
import torchnet
import os
import csv
import time


class LongTailedClassifier():
    def __init__(self, args, model, data_loader_train, data_loader_test):
        self.args = args
        self.model = model

        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test

    def prototypical_classifier(self, is_save=False):
        features, _, labels = self.extract_features(self.data_loader_train) 
        prototypes = get_prototypes(features, labels, self.args)
        prototypes = np.vstack(prototypes) 
        prototypes = preprocessing.normalize(prototypes) 

        test_features, _, test_labels = self.extract_features(self.data_loader_test)
        logits_proto = test_features.dot(prototypes.T) / self.args.temperature
        softmax_proto = softmax(logits_proto, axis=1)
        soft_proto = np.zeros((test_features.shape[0], self.args.nb_classes), dtype=np.float64)
        for i in range(self.args.nb_classes):
            soft_proto[:, i] = np.sum(softmax_proto[:, i * self.args.nb_prototypes:(i + 1) * self.args.nb_prototypes],
                                      axis=1)
        predictions = np.argmax(soft_proto, axis=1)
        measurements = report_measurements_of_classifier(test_labels, predictions)
        if is_save:
            self.save_results(os.path.join(self.args.exp_dir, "prototypical_predictions.csv"), predictions)
            self.save_results(os.path.join(self.args.exp_dir, "prototypical_measurements.csv"), measurements)
        else:
            print(measurements)
            print(predictions)

    def taum_classifier(self, is_save=False):
        weights = self.model.classifier.weight.cpu()
        _, features, labels = self.extract_features(self.data_loader_test)
        predictions = tau_norm_classifier(weights, features, self.args.test_eval_batch_size)
        truths = labels.tolist()
        measurements = []
        for i in range(len(predictions)):
            measurements.append(report_measurements_of_classifier(truths, predictions[i]))
        if is_save:
            self.save_results(os.path.join(self.args.exp_dir, "tau_norm_predictions.csv"), predictions)
            self.save_results(os.path.join(self.args.exp_dir, "tau_norm_measurements.csv"), measurements)
        else:
            print(measurements)
            print(predictions)

    def crt_classifier(self, is_save=False):
        crt_handler = CRT(self.args, self.model, self.data_loader_train, self.data_loader_test)
        crt_handler.main_worker()
        if is_save:
            self.save_results(os.path.join(self.args.exp_dir, "crt_predictions.csv"), crt_handler.all_predictions)
            self.save_results(os.path.join(self.args.exp_dir, "crt_measurements.csv"), crt_handler.all_measurements)
        else:
            print(crt_handler.all_measurements)
            print(crt_handler.all_predictions)

    def extract_features(self, data_loader):
        # extract the features
        self.model.eval()
        nb_features = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        contrastive_features = np.zeros((nb_features, self.args.low_dim), dtype='float32')
        intermediate_features = np.zeros((nb_features, self.args.intermediate_dim), dtype='float32')
        labels = np.zeros(nb_features, dtype="int32")
        for i, batch in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                inputs = batch[0].cuda()
                targets = batch[1].data.cpu().numpy()
                feat, intermediate_feat, output = self.model(inputs)
                feat = feat.data.cpu().numpy()
                intermediate_feat = intermediate_feat.data.cpu().numpy()
            if i < len(data_loader) - 1:
                contrastive_features[i * batch_size: (i + 1) * batch_size] = feat
                intermediate_features[i * batch_size: (i + 1) * batch_size] = intermediate_feat
                labels[i * batch_size: (i + 1) * batch_size] = targets
            else:
                contrastive_features[i * batch_size:] = feat
                intermediate_features[i * batch_size:] = intermediate_feat
                labels[i * batch_size:] = targets
        return contrastive_features, intermediate_features, labels

    @staticmethod
    def save_results(path, results):
        with open(path, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(results, )


class CRT(object):
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

        self.all_measurements = []
        self.all_predictions = []

    def init_model(self):
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
        return predictions, report_measurements_of_classifier(truths, predictions)

    @staticmethod
    def shuffle_batch(x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y
