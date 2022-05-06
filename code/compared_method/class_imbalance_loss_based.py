# coding=utf-8
from train_utils import accuracy, DAverageMeter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torchnet
import os
import logging
import datetime
import json
import time


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma > 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        return self.focal_loss(F.cross_entropy(inputs, targets, reduction='none', weight=self.weight), self.gamma)

    @staticmethod
    def focal_loss(input_values, gamma):
        p = torch.exp(-input_values)
        loss = (1-p) ** gamma * input_values
        return loss.mean()


class LDAMLoss(nn.Module):
    """
    Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.cuda.FloatTensor(m_list)

        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


def compute_adjustment(train_loader, tro):
    # Used for "Long-Tail Learning via Logit Adjustment"
    label_freq = {}
    for i, batch in enumerate(tqdm(train_loader)):
        targets = batch[1]
        for j in targets:
            key = int(j)
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.cuda()
    return adjustments


class LossBasedLongTailed(object):
    def __init__(self, args, model, dataset, DataSet):
        self.args = args
        self.model = model

        print(args.learning_rate)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120], gamma=0.5)
        self.test_acc_meter = torchnet.meter.ClassErrorMeter(accuracy=True)

        self.log_file = None
        self.logger = None

        self.X_train = dataset["X_train"]
        self.y_train = dataset["y_train"]
        self.X_test = dataset["X_test"]
        self.y_test = dataset["y_test"]
        self.DataSet = DataSet

        self.set_experiment_dir()
        self.set_logger()

    def set_experiment_dir(self):
        if not os.path.exists(self.args.exp_dir):
            os.mkdir(self.args.exp_dir)

    def save_parameters(self):
        with open(os.path.join(self.args.exp_dir, "hyper-parameters.txt"), 'w') as f:
            json.dump(self.args.__dict__, f)

    def set_logger(self):
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

    def main_worker(self):
        data_loader_test = self.build_data_loader(self.X_test, self.y_test, self.args.test_eval_batch_size,
                                                  is_random=False)
        data_loader_train = self.build_data_loader(self.X_train, self.y_train, self.args.train_batch_size,
                                                   is_random=True)
        start_time = time.time()
        for curr_epoch in range(self.args.start_epoch, self.args.nb_train_epochs):
            begin_time = time.time()
            self.logger.info("Training epoch [%3d / %3d]" % (curr_epoch + 1, self.args.nb_train_epochs))
            if self.args.train_rule == 'None':
                per_cls_weights = None
                self.args.per_cls_weights = per_cls_weights
            elif self.args.train_rule == 'DRW':
                idx = curr_epoch // 160
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], self.args.cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.args.cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
                self.args.per_cls_weights = per_cls_weights
            loss = self.init_loss_function()
            self.run_train_epoch(data_loader_train, curr_epoch, loss)

            self.test(data_loader_test, curr_epoch)

            self.save_network(self.args.model_name, curr_epoch + 1)

            end_time = time.time()
            self.logger.info(end_time - begin_time)
        final_time = time.time()
        self.logger.info(final_time - start_time)

    def logit_adjustment_training(self):
        data_loader_test = self.build_data_loader(self.X_test, self.y_test, self.args.test_eval_batch_size,
                                                  is_random=False)
        data_loader_train = self.build_data_loader(self.X_train, self.y_train, self.args.train_batch_size,
                                                   is_random=True)
        loss = nn.CrossEntropyLoss().cuda()
        logit_adjustments = compute_adjustment(data_loader_train, 1.0)
        for curr_epoch in range(self.args.start_epoch, self.args.nb_train_epochs):
            self.logger.info("Training epoch [%3d / %3d]" % (curr_epoch + 1, self.args.nb_train_epochs))
            self.scheduler.step()

            self.run_train_logit_adjustment_epoch(data_loader_train, curr_epoch, loss, logit_adjustments)

            self.test(data_loader_test, curr_epoch)

            self.save_network(self.args.model_name, curr_epoch + 1)

    def init_loss_function(self):
        if self.args.loss_type == "CE":
            loss = nn.CrossEntropyLoss(weight=self.args.per_cls_weights).cuda()
        elif self.args.loss_type == "LDAM":
            loss = LDAMLoss(self.args.cls_num_list, max_m=0.5, s=30, weight=self.args.per_cls_weights).cuda()
        elif self.args.loss_type == "Focal":
            loss = FocalLoss(weight=self.args.per_cls_weights, gamma=1).cuda()
        return loss

    def build_data_loader(self, x, y, batch_size, is_random=False):
        dataset = self.DataSet(data=x, labels=y, indices=list(range(len(x))))

        return DataLoader(dataset, batch_size=batch_size, shuffle=is_random, pin_memory=(torch.cuda.is_available()))

    def run_train_epoch(self, data_loader, current_epoch, loss):
        self.logger.info("Training")
        self.model.train()
        train_stats = DAverageMeter()
        # self.scheduler.step()
        for idx, batch in enumerate(tqdm(data_loader)):
            # if idx > self.args.warm_iters:
            #     break
            curr_train_stats = self.train_step(batch, loss)
            train_stats.update(curr_train_stats)
            if (idx + 1) % self.args.log_interval == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s'
                                 % (current_epoch+1, idx+1, len(data_loader), train_stats.average()))
        return train_stats.average()

    def train_step(self, batch, criterion):
        self.model.train()
        inputs = batch[0].cuda()
        targets = batch[1].cuda()
        record = {}

        _, _, outputs = self.model(inputs)
        loss = criterion(outputs, targets)

        record['loss'] = loss.item()
        record['train_accuracy'] = accuracy(outputs, targets)[0].item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return record

    def run_train_logit_adjustment_epoch(self, data_loader, current_epoch, loss, logit_adjustments):
        self.logger.info("Training")
        self.model.train()
        train_stats = DAverageMeter()
        for idx, batch in enumerate(tqdm(data_loader)):
            curr_train_stats = self.train_logit_adjustment_step(batch, loss, logit_adjustments)
            train_stats.update(curr_train_stats)
            if (idx + 1) % self.args.log_interval == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s'
                                 % (current_epoch+1, idx+1, len(data_loader), train_stats.average()))
        return train_stats.average()

    def train_logit_adjustment_step(self, batch, criterion, logit_adjustments):
        self.model.train()
        inputs = batch[0].cuda()
        targets = batch[1].cuda()
        record = {}

        _, _, outputs = self.model(inputs)
        outputs = outputs + logit_adjustments
        loss = criterion(outputs, targets)

        loss_r = 0
        for parameter in self.model.parameters():
            loss_r += torch.sum(parameter ** 2)
        loss = loss + self.args.weight_decay * loss_r

        record['loss'] = loss.item()
        record['train_accuracy'] = accuracy(outputs, targets)[0].item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return record

    def test(self, data_loader, epoch):
        self.logger.info("==> Testing...")
        self.model.eval()
        self.test_acc_meter.reset()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(tqdm(data_loader)):
                inputs, targets = inputs.cuda(), targets.cuda()
                _, _, outputs = self.model(inputs)
                self.test_acc_meter.add(outputs, targets)
        acc = self.test_acc_meter.value()
        self.logger.info("Testing Accuracy is %.2f%%" % (acc[0]))
        return

    def save_network(self, net_key, epoch):
        checkpoint_dir = os.path.join(self.args.exp_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, net_key + "_net_epoch" + str(epoch))
        state = {'epoch': epoch, 'network': self.model.state_dict()}
        torch.save(state, checkpoint_file)
