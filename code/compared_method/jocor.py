# coding=utf-8
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import time


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]


def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


def loss_jocor(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.1):

    loss_pick_1 = F.cross_entropy(y_1, t, reduce=False) * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduce=False) * (1-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduce=False) + co_lambda *
                 kl_loss_compute(y_2, y_1, reduce=False)).cpu()

    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

    ind_update = ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return loss, loss, pure_ratio, pure_ratio


class JoCoR():
    def __init__(self, args, model1, model2, train_dataset, test_dataset, noise_or_not):
        self.args = args
        self.model1 = model1
        self.model2 = model2

        self.device = "cuda"

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=self.args.learning_rate)

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

        self.loss_fn = loss_jocor

        self.adjust_lr = self.args.adjust_lr

        self.noise_or_not = noise_or_not

    def main_worker(self):
        start_time = time.time()
        for epoch in range(0, self.args.nb_train_epochs):
            begin_time = time.time()
            train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = self.train_epoch(epoch)

            test_acc1, test_acc2 = self.evaluate()

            # save results
            if pure_ratio_1_list is None or len(pure_ratio_1_list) == 0:
                print(
                    'Epoch [%d/%d] Test Accuracy on the %s test inputs: Model1 %.4f %% Model2 %.4f' % (
                        epoch + 1, self.args.nb_train_epochs, len(self.test_dataset), test_acc1, test_acc2))
            else:
                # save results
                mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
                mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)
                print(
                    'Epoch [%d/%d] Test Accuracy on the %s test inputs: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f '
                    '%%, Pure Ratio 2 %.4f %%' % (
                        epoch + 1, self.args.nb_train_epochs, len(self.test_dataset), test_acc1, test_acc2,
                        mean_pure_ratio1,
                        mean_pure_ratio2))
            end_time = time.time()
            print(end_time - begin_time)
            self.save_network(self.args.model_name, epoch + 1)
        end_final_time = time.time()
        print(end_final_time - start_time)

    # Evaluate the Model
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

    # Train the Model
    def train_epoch(self, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (inputs, targets, indexes) in enumerate(self.train_loader):
            ind = indexes.cpu().numpy().transpose()

            inputs = Variable(inputs).to(self.device)
            targets = Variable(targets).to(self.device)

            # Forward + Backward + Optimize
            _, _, logits1 = self.model1(inputs)
            prec1 = accuracy(logits1, targets)[0]
            train_total += 1
            train_correct += prec1

            _, _, logits2 = self.model2(inputs)
            prec2 = accuracy(logits2, targets)[0]
            train_total2 += 1
            train_correct2 += prec2

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, targets,
                                                                      self.rate_schedule[epoch], ind, self.noise_or_not,
                                                                      self.args.co_lambda)

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

            if (i + 1) % self.args.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, '
                    'Loss2: %.4f, Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%'
                    % (epoch + 1, self.args.nb_train_epochs, i + 1,
                       len(self.train_dataset) // self.args.train_batch_size, prec1, prec2, loss_1.data.item(),
                       loss_2.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list),
                       sum(pure_ratio_2_list) / len(pure_ratio_2_list)))

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1

    def get_model(self):
        return self.model1, self.model2

    def save_network(self, net_key, epoch):
        checkpoint_dir = os.path.join(self.args.exp_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, net_key + "_net_epoch" + str(epoch))
        state = {'epoch': epoch, 'network': self.model1.state_dict()}
        torch.save(state, checkpoint_file)


