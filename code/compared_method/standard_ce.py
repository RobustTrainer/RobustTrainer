# coding=utf-8
"""
Standard Cross-entropy Loss Training
"""
from train_utils import accuracy, DAverageMeter
from torch.utils.data import DataLoader
from tqdm import tqdm
from sampler.class_aware import ClassAwareSampler
import torch.nn as nn
import torch
import torchnet
import os
import logging
import datetime
import json


class StandardCE(object):
    def __init__(self, args, model, dataset, DataSet):
        self.args = args
        self.model = model

        self.CE = nn.CrossEntropyLoss().cuda()
        self.NLL = nn.NLLLoss().cuda()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = \
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=0.0002,
                                                       T_max=self.args.nb_train_epochs - self.args.start_epoch)
        self.test_acc_meter = torchnet.meter.ClassErrorMeter(accuracy=True)

        self.log_file = None
        self.logger = None

        self.X_train = dataset["X_train"]
        self.y_train = dataset["y_train"]
        self.X_test = dataset["X_test"]
        self.y_test = dataset["y_test"]
        self.DataSet = DataSet

        self.set_experiment_dir()
        self.save_parameters()
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

    def standrad_ce_train(self):
        data_loader_test = self.build_data_loader(self.X_test, self.y_test, self.args.test_eval_batch_size,
                                                  is_random=False)
        for curr_epoch in range(self.args.start_epoch, self.args.nb_train_epochs):
            self.logger.info("Training epoch [%3d / %3d]" % (curr_epoch + 1, self.args.nb_train_epochs))
            data_loader_train = self.build_data_loader(self.X_train, self.y_train, self.args.train_batch_size,
                                                       is_random=True)
            self.run_train(data_loader_train, curr_epoch)

            self.test(data_loader_test, curr_epoch)

            self.save_network(self.args.model_name, curr_epoch + 1)

    def class_balance_ce_train(self):
        data_loader_test = self.build_data_loader(self.X_test, self.y_test, self.args.test_eval_batch_size,
                                                  is_random=False)
        for curr_epoch in range(self.args.start_epoch, self.args.nb_train_epochs):
            self.logger.info("Training epoch [%3d / %3d]" % (curr_epoch + 1, self.args.nb_train_epochs))
            data_loader_train = self.build_data_loader_with_class_balance(self.X_train, self.y_train,
                                                                          self.args.train_batch_size,
                                                                          is_random=False)
            self.run_train(data_loader_train, curr_epoch)

            self.test(data_loader_test, curr_epoch)

            self.save_network(self.args.model_name, curr_epoch + 1)

    def build_data_loader(self, x, y, batch_size, is_random=False):
        dataset = self.DataSet(data=x, labels=y, indices=list(range(len(x))))

        return DataLoader(dataset, batch_size=batch_size, shuffle=is_random, pin_memory=(torch.cuda.is_available()))

    def build_data_loader_with_class_balance(self, x, y, batch_size, is_random=False):
        sampler_dic = {
            'sampler': ClassAwareSampler,
            'params': {'num_samples_cls': 4}
        }
        dataset = self.DataSet(data=x, labels=y, indices=list(range(len(x))))

        return DataLoader(dataset, batch_size=batch_size, shuffle=is_random,
                          sampler=sampler_dic['sampler'](dataset, **sampler_dic['params']),
                          pin_memory=(torch.cuda.is_available()))

    def run_train(self, data_loader, current_epoch):
        self.logger.info("Warm-up Training")
        self.model.train()
        train_stats = DAverageMeter()
        self.scheduler.step()
        for idx, batch in enumerate(tqdm(data_loader)):
            if idx > self.args.warm_iters:
                break
            curr_train_stats = self.naive_train(batch)
            train_stats.update(curr_train_stats)
            if (idx + 1) % self.args.log_interval == 0:
                self.logger.info('==> Iteration [%3d][%4d / %4d]: %s'
                                 % (current_epoch+1, idx+1, len(data_loader), train_stats.average()))
        return train_stats.average()

    def naive_train(self, batch):
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

