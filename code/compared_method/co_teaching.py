# coding=utf-8
"""
Standard Cross-entropy Loss Training
"""
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch
import os
import datetime
import time


class CoTeaching():
    def __init__(self, args, model1, model2, train_dataset, test_dataset, noise_or_not):
        self.args = args
        self.model1 = model1
        self.model2 = model2

        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.args.learning_rate)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=self.args.learning_rate)

        self.alpha_plan = [self.args.learning_rate] * self.args.nb_train_epochs
        self.beta1_plan = [0.9] * self.args.nb_train_epochs
        for i in range(self.args.epoch_decay_start, self.args.nb_train_epochs):
            self.alpha_plan[i] = float(self.args.nb_train_epochs - i) / \
                                 (self.args.nb_train_epochs - self.args.epoch_decay_start) * self.args.learning_rate
            self.beta1_plan[i] = 0.1

        self.rate_schedule = np.ones(self.args.nb_train_epochs) * self.args.forget_rate
        self.rate_schedule[:self.args.num_gradual] = np.linspace(0, self.args.forget_rate**self.args.exponent,
                                                                 self.args.num_gradual)

        self.train_dataset = train_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                       pin_memory=(torch.cuda.is_available()))
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.test_eval_batch_size, shuffle=False,
                                      pin_memory=(torch.cuda.is_available()))
        self.noise_or_not = noise_or_not

        now_str = datetime.datetime.now().__str__().replace(' ', '_')
        self.log_file = os.path.join(self.args.exp_dir, "LOG_INFO" + now_str + ".txt")

    def main_worker(self):
        with open(self.log_file, "a") as f:
            f.write("epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n")

        epoch = 0
        train_acc1 = 0
        train_acc2 = 0
        mean_pure_ratio1 = 0
        mean_pure_ratio2 = 0
        test_acc1, test_acc2 = self.evaluate()
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f '
              '%% Pure Ratio2 %.4f %%' % (epoch + 1, self.args.nb_train_epochs, len(self.test_dataset), test_acc1, test_acc2,
                                          mean_pure_ratio1, mean_pure_ratio2))
        with open(self.log_file, "a") as f:
            f.write(str(int(epoch)) + ': ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(test_acc1) + ' ' +
                    str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + '\n')
        start_time = time.time()
        for epoch in range(0, self.args.nb_train_epochs):
            begin_time = time.time()
            self.model1.train()
            self.adjust_learning_rate(self.optimizer1, epoch)
            self.model2.train()
            self.adjust_learning_rate(self.optimizer2, epoch)

            train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = self.train_step(epoch)
            test_acc1, test_acc2 = self.evaluate()

            mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
            mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)

            end_time = time.time()

            print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f '
                  '%% Pure Ratio2 %.4f %%' % (epoch + 1, self.args.nb_train_epochs, len(self.test_dataset), test_acc1, test_acc2,
                                              mean_pure_ratio1, mean_pure_ratio2))
            with open(self.log_file, "a") as f:
                f.write(str(int(epoch)) + ': ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(test_acc1) + ' ' +
                        str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + ' ' + str(end_time-begin_time) + '\n')

            self.save_network(self.args.model_name, epoch + 1)
        end_final_time = time.time()
        with open(self.log_file, "a") as f:
            f.write(str(end_final_time - start_time))

    def train_step(self, epoch):
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0

        for i, (inputs, targets, indexes) in enumerate(self.train_loader):
            ind = indexes.cpu().numpy().transpose()

            inputs = inputs.cuda()
            targets = targets.cuda()

            _, _, logits1 = self.model1(inputs)
            prec1 = self.accuracy(logits1, targets)[0]
            train_total += 1
            train_correct += prec1

            _, _, logits2 = self.model2(inputs)
            prec2 = self.accuracy(logits2, targets)[0]
            train_total2 += 1
            train_correct2 += prec2

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_co_teaching(logits1, logits2, targets,
                                                                               self.rate_schedule[epoch], ind,
                                                                               self.noise_or_not)

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

            self.optimizer1.zero_grad()
            loss_1.backward()
            self.optimizer1.step()

            self.optimizer2.zero_grad()
            loss_2.backward()
            self.optimizer2.step()

            if (i + 1) % self.args.print_freq == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4f, Training Accuracy2: %.4f, Loss1: %.4f, '
                      'Loss2: %.4f, Pure Ratio1:%.4f, Pure Ratio2:%.4f'
                      % (epoch, self.args.nb_train_epochs, i+1, len(self.train_dataset) // self.args.train_batch_size,
                         prec1, prec2, loss_1.item(), loss_2.item(),
                         np.sum(pure_ratio_1_list) / len(pure_ratio_1_list),
                         np.sum(pure_ratio_2_list) / len(pure_ratio_2_list)))
        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

    def evaluate(self):
        self.model1.eval()
        correct1 = 0
        total1 = 0
        with torch.no_grad():
            for inputs, targets, _ in self.test_loader:
                inputs = inputs.cuda()
                _, _, logits1 = self.model1(inputs)
                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                total1 += targets.size(0)
                correct1 += (pred1.cpu() == targets).sum()

        self.model2.eval()
        correct2 = 0
        total2 = 0
        with torch.no_grad():
            for inputs, targets, _ in self.test_loader:
                inputs = inputs.cuda()
                _, _, logits2 = self.model2(inputs)
                outputs2 = F.softmax(logits2, dim=1)
                _, pred2 = torch.max(outputs2.data, 1)
                total2 += targets.size(0)
                correct2 += (pred2.cpu() == targets).sum()

        acc1 = 100 * float(correct1) / float(total1)
        acc2 = 100 * float(correct2) / float(total2)
        return acc1, acc2

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)

    def save_network(self, net_key, epoch):
        checkpoint_dir = os.path.join(self.args.exp_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, net_key + "_net_epoch" + str(epoch))
        state = {'epoch': epoch, 'network': self.model1.state_dict()}
        torch.save(state, checkpoint_file)

    @staticmethod
    def loss_co_teaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
        loss_1 = F.cross_entropy(y_1, t, reduce=False)
        ind_1_sorted = np.argsort(loss_1.detach().cpu().numpy())
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduce=False)
        ind_2_sorted = np.argsort(loss_2.detach().cpu().numpy())
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(num_remember)
        pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(num_remember)

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]

        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

        return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2

    @staticmethod
    def accuracy(logit, target, topk=(1,)):
        output = F.softmax(logit, dim=1)
        max_k = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
