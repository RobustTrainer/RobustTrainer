# coding=utf-8
"""
The training pipeline of RobustTrainer
"""
from utils import accuracy, DAverageMeter
from class_prototypes import get_prototypes
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import json
import os
import logging
import datetime
import csv
import copy


class FeatureLearning(object):
    def __init__(self, args, model, dataset, DataSet):
        """
        The stage 1: robust feature learning
        :param args: the parameters of RobustTrainer
        :param model: the subject deep predictive model
        :param dataset: a dict contains training and testing dataset
        :param DataSet: the custom Dataset class for creating pytorch dataset
        """
        self.args = args
        self.model = model

        self.CE = nn.CrossEntropyLoss().cuda()
        self.NLL = nn.NLLLoss().cuda()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        self.log_file = None
        self.logger = None

        self.X_train = dataset["X_train"]
        self.y_train = dataset["y_train"]
        self.X_test = dataset["X_test"]
        self.y_test = dataset["y_test"]
        self.DataSet = DataSet

        self.prototypes = None
        self.dataset_cache = {
            "clean_idx": None,
        }

        self.set_experiment_dir()
        self.save_parameters()
        self.set_logger()

    def set_experiment_dir(self):
        # create folder for saving experimental results
        if not os.path.exists(self.args.exp_dir):
            os.mkdir(self.args.exp_dir)

    def save_parameters(self):
        # save the used hyperparameters for the current run
        with open(os.path.join(self.args.exp_dir, "hyper-parameters.txt"), 'w') as f:
            json.dump(self.args.__dict__, f)

    def set_logger(self):
        # create log file
        self.logger = logging.getLogger(__name__)
        str_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s")
        str_handler.setFormatter(formatter)
        self.logger.addHandler(str_handler)
        self.logger.setLevel(logging.INFO)

        log_dir = os.path.join(self.args.exp_dir, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        now_str = datetime.datetime.now().__str__().replace(' ', '_')
        self.log_file = os.path.join(log_dir, "LOG_INFO" + now_str + ".txt")
        log_file_handler = logging.FileHandler(self.log_file)
        log_file_handler.setFormatter(formatter)
        self.logger.addHandler(log_file_handler)

    def train_features(self):
        # the stage 1: robust feature learning

        # create dataloader for training dataset, which is mainly used for detecting mislabeled samples
        data_loader_eval = self.build_data_loader(self.X_train, self.y_train, self.args.test_eval_batch_size,
                                                  is_random=False)
        # data_loader_test = self.build_data_loader(self.X_test, self.y_test, self.args.test_eval_batch_size,
        #                                           is_random=False)
        for curr_epoch in range(self.args.start_epoch, self.args.nb_train_epochs):
            self.logger.info("Training epoch [%3d / %3d]" % (curr_epoch + 1, self.args.nb_train_epochs))
            if curr_epoch < self.args.nb_warmup:
                # warmup
                data_loader_train = self.build_data_loader(self.X_train, self.y_train, self.args.test_eval_batch_size,
                                                           is_random=True)
                self.run_train_warmup(data_loader_train, curr_epoch)
            else:
                """
                Each epoch contains the following two steps:
                1. Detecting mislabeled samples;
                2. Updating feature extractor
                """
                # extract deep features of training examples
                features, labels = self.extract_features(data_loader_eval)

                # select the clean samples
                clean_feat, clean_y, clean_idx = self.select_clean_samples(features, labels, self.args)
                clean_x = [self.X_train[i] for i in clean_idx]

                # refine the class prototypes using the newly selected clean samples
                class_prototypes = get_prototypes(clean_feat, clean_y, self.args)
                class_prototypes = np.vstack(class_prototypes)
                self.prototypes = class_prototypes

                weak_x = list(clean_x)
                weak_y = list(clean_y)
                self.dataset_cache["clean_idx"] = clean_idx
                # build new data loader using the selected clean samples
                data_loader_train = self.build_data_loader(weak_x, weak_y, self.args.train_batch_size, is_random=True)

                self.save_selected_clean_samples(curr_epoch)
                train_stats = self.run_train_epoch(data_loader_train, curr_epoch)
                self.save_network(self.args.model_name, curr_epoch)

    def run_train_warmup(self, data_loader, current_epoch):
        # the standard training: only the CE loss is used
        self.logger.info("Warm-up Training")
        self.model.train()
        train_stats = DAverageMeter()
        for idx, batch in enumerate(tqdm(data_loader)):
            if idx > self.args.warm_iters:
                break
            curr_train_stats = self.naive_train(batch)
            train_stats.update(curr_train_stats)
            if (idx + 1) % self.args.log_interval == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s'
                                 % (current_epoch + 1, idx + 1, len(data_loader), train_stats.average()))
        return train_stats.average()

    def naive_train(self, batch):
        # update the model using CE loss
        self.model.train()
        inputs = batch[0].cuda()
        targets = batch[1].cuda()
        record = {}

        _, _, outputs = self.model(inputs)
        loss = self.CE(outputs, targets)

        record['loss'] = loss.item()
        record['train_accuracy'] = accuracy(outputs, targets)[0].item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return record

    def extract_features(self, data_loader):
        # extract deep features of training examples
        self.model.eval()

        nb_features = len(data_loader.dataset)  # the number of training examples
        batch_size = data_loader.batch_size
        features = np.zeros((nb_features, self.args.low_dim), dtype='float32')
        labels = np.zeros(nb_features, dtype="int32")
        for i, batch in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                inputs = batch[0].cuda()
                targets = batch[1].data.cpu().numpy()
                feat, intermediate_feat, output = self.model(inputs)
                feat = feat.data.cpu().numpy()
            if i < len(data_loader) - 1:
                features[i * batch_size: (i + 1) * batch_size] = feat
                labels[i * batch_size: (i + 1) * batch_size] = targets
            else:
                features[i * batch_size:] = feat
                labels[i * batch_size:] = targets
        return features, labels

    def build_data_loader(self, x, y, batch_size, is_random=False):
        dataset = self.DataSet(data=x, labels=y, indices=list(range(len(x))))

        return DataLoader(dataset, batch_size=batch_size, shuffle=is_random, pin_memory=(torch.cuda.is_available()))

    def run_train_epoch(self, data_loader, current_epoch):
        # train the model with both contrastive and CE loss
        self.logger.info("Training %s" % os.path.basename(self.args.exp_dir))
        self.model.train()
        train_stats = DAverageMeter()
        for idx, batch in enumerate(tqdm(data_loader)):
            curr_train_stats = self.contrastive_train(batch)
            train_stats.update(curr_train_stats)
            if (idx + 1) % self.args.log_interval == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s'
                                 % (current_epoch + 1, idx + 1, len(data_loader), train_stats.average()))
        return train_stats.average()

    def contrastive_train(self, batch):
        self.model.train()
        inputs = batch[0].cuda()
        targets = batch[1].cuda()
        record = {}

        features, intermediate_feat, outputs = self.model(inputs)
        # obtain the class probabilities of prototypes
        class_prototypes = copy.deepcopy(self.prototypes)
        class_prototypes = torch.from_numpy(class_prototypes).cuda()
        logits_proto = torch.mm(features, class_prototypes.t()) / self.args.temperature
        softmax_proto = F.softmax(logits_proto, dim=1)
        prob_proto = torch.zeros((softmax_proto.shape[0], self.args.nb_classes), dtype=torch.float64).cuda()
        for i in range(self.args.nb_classes):
            prob_proto[:, i] = torch.sum(
                softmax_proto[:, i * self.args.nb_prototypes: (i + 1) * self.args.nb_prototypes], dim=1)
        # contrastive loss
        cl_loss = self.NLL(torch.log(prob_proto + 1e-5), targets)
        record['loss_contrastive'] = cl_loss.item()
        # classification loss
        ce_loss = self.CE(outputs, targets)
        record['loss'] = ce_loss.item()

        loss = self.args.w_ce * ce_loss + self.args.w_cl * cl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return record

    def save_selected_clean_samples(self, epoch):
        file_path = os.path.join(self.args.exp_dir, "selected_clean_samples")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, "epoch_{}".format(epoch))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if self.dataset_cache["clean_idx"] is not None:
            with open(os.path.join(file_path, "clean_idx.csv"), 'w', encoding='utf-8', newline='') \
                    as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(self.dataset_cache["clean_idx"], )

    def save_network(self, net_key, epoch):
        checkpoint_dir = os.path.join(self.args.exp_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, net_key + "_net_epoch" + str(epoch))
        state = {'epoch': epoch, 'network': self.model.state_dict()}
        torch.save(state, checkpoint_file)

    @staticmethod
    def select_clean_samples(x, y, args):
        """
        Clean the dataset
        :param x: features of all examples in the dataset
        :param y: original labels
        :param args: hyper-parameter
        :return:
        """
        prototypes = get_prototypes(x, y, args)
        prototypes = np.vstack(prototypes)  # combine prototypes of all classes
        # prototypes = preprocessing.normalize(prototypes)  # normalize the class prototypes

        # each item x_{ij} represents the similarity between the sample i and the prototype j
        similarities_proto = x.dot(prototypes.T)
        similarities_class = np.zeros((x.shape[0], args.nb_classes), dtype=np.float64)
        for i in range(args.nb_classes):
            similarities_class[:, i] = np.mean(
                similarities_proto[:, i * args.nb_prototypes:(i + 1) * args.nb_prototypes], axis=1)
        # select the samples by GMM
        clean_set = []
        for i in range(args.nb_classes):
            class_idx = np.where(y == i)[0]
            class_sim = similarities_proto[class_idx, i]
            # split the dataset using GMM
            class_sim = class_sim.reshape((-1, 1))
            # gm = GaussianMixture(n_components=2, random_state=args.seed).fit(class_sim)
            gm = GaussianMixture(n_components=2, random_state=args.seed).fit(class_sim)
            class_clean_idx = np.where(gm.predict(class_sim) == gm.means_.argmax())[0]
            clean_set.extend(class_idx[class_clean_idx])

        return x[clean_set], y[clean_set], clean_set