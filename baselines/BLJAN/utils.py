#!/usr/bin/env python
# -*- coding:utf-8 -*-

import datetime
import os, time, sys, gc
import random
from math import sin, cos, pow
import numpy as np
import torch, pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd.function import Function
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

MODEL_SAVE_PATH = '/data/data/ws/NetworkTC/BLJAN/model/'
WORKING_DIR = '/data/data/ws/NetworkTC/BLJAN/'

# data preprocessing
CLASSES = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
# CLASSES = ['amazon', 'baidu', 'bing', 'douban', 'facebook', 'google', 'imdb', 'instagram', 'iqiyi', 'jd',
#            'neteasemusic', 'qqmail', 'reddit', 'taobao', 'ted', 'tieba', 'twitter', 'weibo', 'youku', 'youtube']
# CLASSES = ['vimeo', 'spotify', 'voipbuster', 'sinauc', 'cloudmusic', 'weibo', 'baidu', 'tudou', 'amazon', 'thunder',
#            'gmail', 'pplive', 'qq', 'taobao', 'yahoomail', 'itunes', 'twitter', 'jd', 'sohu', 'youtube', 'youku',
#            'netflix', 'aimchat', 'kugou', 'skype', 'facebook', 'google', 'mssql', 'ms-exchange']
# CLASSES = ['audio', 'chat', 'file', 'mail', 'streaming', 'voip',
#            'vpn-audio', 'vpn-chat', 'vpn-file', 'vpn-mail', 'vpn-streaming', 'vpn-voip']
LABELS = {k: v for k, v in zip(CLASSES, range(len(CLASSES)))}

LABELS_FLOW = {'baidu': 0, 'gmail': 1, 'itunes': 2, 'pplive': 3, 'qq': 4, 'skype': 5, 'sohu': 6, 'taobao': 7,
               'thunder': 8, 'tudou': 9, 'weibo': 10, 'youku': 11}

NUM_LABELS = len(LABELS)
PKT_MAX_LEN = 1500
NGRAM = 50
# CUDA = torch.cuda.is_available()
CUDA = True
DEVICE = 0
DEVICE_COUNT = torch.cuda.device_count()
USE_PARALLEL = False
PADDING_IDX = 256
if USE_PARALLEL:
    ALL_DEVICE = ['cuda:{}'.format(i) for i in range(DEVICE_COUNT)]
else:
    ALL_DEVICE = ['cuda:0']

# model and training
EMBEDDING_DIM = 128
OUT_CHANNELS = 8
DROPOUT = 0.5
EXP_PERCENT = 1
TEST_PERCENT = 0.1
BATCH_SIZE = 128
EPOCHS = 120
LR = 0.001

GAMMA = 2
LAMBDA = 0.1


def get_cosine_similarity_matrix(B, L):
    # input: Tensor 张量
    # 返回输入张量给定维dim上每行的p范数
    G = torch.matmul(L, B.transpose(1, 2))  # b * k * l
    B = torch.norm(B, p=2, dim=2)  # B: b * l * e -> b * l, dim = 2 is for e's axis
    L = torch.norm(L, p=2, dim=1)  # L: k * e -> (k,), dim = 1 is also for e's axis

    B = B.unsqueeze(1)
    L = L.unsqueeze(1)
    G_norm = torch.matmul(L, B)  # b * k * l    # tensor的乘法

    # prevent divide 0 then, we replace 0 with 1
    ones = torch.ones(G_norm.size())  # 返回一个全为1的张量，形状为G_norm的形状
    device = G.device
    if CUDA:
        ones = ones.to(G.device)
    G_norm = torch.where(G_norm != 0, G_norm, ones)
    G_hat = G / G_norm
    return G_hat


def calculate_alpha(y, labels, mode='normal'):
    alpha = [0] * len(labels)
    for i in y:
        alpha[i] += 1
    alpha = np.array(alpha)
    if mode == 'normal':
        return torch.from_numpy(alpha / alpha.sum())
    elif mode == 'invert':
        return torch.from_numpy((alpha.sum() - alpha) / alpha.sum())
    return None


class Loader:
    def __init__(self, X, y, batch_size, padding=False, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.padding = padding
        self.shuffle = shuffle
        self.num_batch = int((len(self.y) - 1) / batch_size)
        self.data = []
        self.alpha = calculate_alpha(self.y, LABELS, mode='invert')
        if self.shuffle:
            state = np.random.get_state()
            np.random.shuffle(self.X)
            np.random.set_state(state)
            np.random.shuffle(self.y)
            # print('shuffle loader with state: {}'.format(state))
        # self.lengths = np.array([len(x) for x in self.X])

        # self.load_all_batches()

    def load_all_batches(self):
        _s_t = time.time()
        print('>> start load data with {} batches'.format(self.num_batch))
        for i in range(self.num_batch):
            if i % 2000 == 0:
                print('\r>> >> {} batches ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
            batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size]
            batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size]
            # batch_len = self.lengths[i * self.batch_size: (i + 1) * self.batch_size if i != n else None]
            # max_len = (int((batch_len.max() - 1) / 10) + 1) * 10 if self.padding else batch_len.max()
            max_len = 1500
            # X_mask = []
            for j in range(len(batch_y)):
                # X_mask.append(([1] * len(batch_X[j]) + [0] * (max_len - len(batch_X[j]))))
                batch_X[j] = batch_X[j] + [256] * (max_len - len(batch_X[j]))
            # batch_X = torch.tensor(batch_X.tolist()).float()
            batch_X = torch.tensor(batch_X.tolist()).long()
            batch_y = torch.from_numpy(batch_y)
            # X_mask = torch.from_numpy(np.array(X_mask, dtype=int))
            # batch_X = pack(batch_X)
            self.data.append((batch_X, batch_y))
        self.X = None
        self.y = None
        print('\n>> all batches load done with {}s'.format(time.time() - _s_t))

    def reload_all_batches(self):
        print('\n>> shuffle and reload all batches...')
        X = sum([x.int().tolist() for x, y in self.data], [])
        y = sum([y.tolist() for x, y in self.data], [])
        random.seed(2022)
        random.shuffle(X)
        random.seed(2022)
        random.shuffle(y)
        self.data = []
        _s_t = time.time()
        print('>> start load data with {} batches'.format(self.num_batch))
        for i in range(self.num_batch):
            if i % 2000 == 0:
                print('\r>> >> {} batches ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
            batch_X = X[i * self.batch_size: (i + 1) * self.batch_size]
            batch_y = y[i * self.batch_size: (i + 1) * self.batch_size]
            self.data.append((torch.tensor(batch_X).float(), torch.tensor(batch_y)))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if len(self.data) > i:
            return self.data[i]
        batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size]
        batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size]
        max_len = 1500
        for j in range(len(batch_y)):
            batch_X[j] = batch_X[j] + [256] * (max_len - len(batch_X[j]))
        batch_X = torch.tensor(batch_X.tolist()).float()
        # batch_X = torch.tensor(batch_X.tolist()).long()
        batch_y = torch.from_numpy(batch_y)
        return batch_X, batch_y


# data_dir is traffic data directory, not file path
# Turn file to X and y. exp_percent is put into model(others are discarded),
# test_percent is test_size
def get_dataloader(exp_percent, test_percent, batch_size, padding=False):
    _s_t = time.time()
    print('start load X and y')
    X = []
    y = []
    if NUM_LABELS == 29:
        pickle_dir = '/data/data/ws/NetworkTC/datasets/TrafficX-App_PKL/'
    elif NUM_LABELS == 12:
        pickle_dir = '/data/data/ws/NetworkTC/datasets/ISCX-VPN-NonVPN-2016_PKL/'
    elif NUM_LABELS == 20:
        pickle_dir = '/data/data/ws/NetworkTC/datasets/TrafficX-Website_PKL/'
    else:
        pickle_dir = '/data/data/ws/NetworkTC/datasets/USTC-TFC-2016_wo_payload/'
    for i in CLASSES:
        file_name = pickle_dir + i.lower() + '.pkl'
        # if os.path.exists(pickle_dir + i + '-clean.pkl'):
        #     file_name = pickle_dir + i + '-clean.pkl'
        # if not os.path.isfile(file_name):
        #     file_name = '/data/data/ws/NetworkTC/datasets/BLJAN/USTC-TFC2016/result_doc_benign/' + 'X_' + i + '.pkl'
        with open(file_name, 'rb') as f1:
            tmp = pickle.load(f1)
        # tmp = sum([v for k, v in tmp.items()], [])
        # if LABELS[i] >= 10 and 'Cridex' in CLASSES:
        #     if len(tmp) > 100000:
        #         random.seed(41)
        #         tmp = random.sample(tmp, 100000)
        if len(tmp) > 200000:
            random.seed(2022)
            tmp = random.sample(tmp, 200000)
        num_pkt = len(tmp)
        X += tmp
        y += [LABELS[i]] * num_pkt
        print('{}: {} from {}'.format(i, num_pkt, file_name[file_name.rfind('/') + 1:]))
        break
    print('load X and y cost: {}s'.format(time.time() - _s_t))

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    if 0 < exp_percent < 1:
        print('start random select data(train_test_split) with exp_percent: {}'.format(exp_percent))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=exp_percent, shuffle=True, random_state=41,
                                                            stratify=y)
        X = X_test
        y = y_test

        # X_extra = [X_train[i] for i in range(len(y_train)) if y_train[i] in [1, 10, 12]]
        # y_extra = [i for i in y_train if i in [1, 10, 12]]
        #
        # X = np.append(X, np.array(X_extra, dtype=object), axis=0)
        # y = np.append(y, y_extra)
        #
        # X_extra = X_train
        # y_extra = y_train

    print('start train_test_split with test_percent: {}'.format(test_percent))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, shuffle=True, random_state=41,
                                                        stratify=y)
    # X_train = np.array([X_train[i] for i in range(len(y_train)) if y_train[i] not in [1, 10, 12]], dtype=object)
    # y_train = np.array([i for i in y_train if i not in [1, 10, 12]])
    # if 0 < exp_percent < 1:
    #     X_train = np.append(X_train, X_extra, axis=0)
    #     y_train = np.append(y_train, y_extra)

    print('start load batches of train and test...')
    train_loader, test_loader = Loader(X_train, y_train, batch_size, padding), Loader(X_test, y_test, batch_size,
                                                                                      padding)
    return train_loader, test_loader


class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, is_reverse=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.is_reverse = is_reverse

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)

        class_mask.scatter_(1, ids.data, 1.)  # one-hot vector

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda(DEVICE)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        if self.is_reverse:
            probs = 1. - probs
        min_probs = torch.tensor(0.00000001, dtype=torch.float)
        probs = torch.maximum(probs, min_probs)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def evaluate_EBLJAN(model, dataloader, overall_label_idx, is_cm, num_labels, alpha, gamma):
    loss_func = FocalLoss(num_labels, alpha, gamma, True)
    # loss_func = nn.CrossEntropyLoss()
    total_loss = 0
    y_hat, y = [], []

    # model.eval()
    for i in range(dataloader.num_batch):
        batch_X, batch_y, X_mask = dataloader.data[i]
        if CUDA:
            batch_X = batch_X.cuda(DEVICE)
            batch_y = batch_y.cuda(DEVICE)
            X_mask = X_mask.cuda(DEVICE)
        model.eval()
        test_start = time.time()
        out1, out2 = model(batch_X, overall_label_idx, X_mask)
        loss1 = loss_func(out1, batch_y.long())
        loss2 = loss_func(out2, torch.cat((overall_label_idx, overall_label_idx, overall_label_idx, overall_label_idx)))
        total_loss += loss1.item() + LAMBDA * loss2.item()
        y_hat += out1.max(1)[1].tolist()
        y += batch_y.tolist()
    y = np.array(y)
    y_hat = np.array(y_hat)
    accuracy = accuracy_score(y, y_hat)

    c_matrix = 0
    if is_cm:
        c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(0, len(overall_label_idx))])

    return total_loss, accuracy, c_matrix, y_hat


def evaluate_loss_acc(model, dataloader, is_cm, num_labels, overall_label_idx=None):
    y_hat, y = [], []
    model.eval()
    for i in range(dataloader.num_batch):
        batch_X, batch_y = dataloader.data[i]
        if CUDA:
            batch_X = batch_X.cuda(DEVICE)
            batch_y = batch_y.cuda(DEVICE)
        model.eval()
        out1, out2 = model(batch_X) if overall_label_idx is None else model(batch_X, overall_label_idx)
        y_hat += out1.max(1)[1].tolist()
        y += batch_y.tolist()
    y = np.array(y)
    y_hat = np.array(y_hat)
    accuracy = accuracy_score(y, y_hat)

    c_matrix = 0
    if is_cm:
        c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(0, num_labels)])

    return accuracy, c_matrix


def evaluate_segment(model, dataloader, overall_label_idx, is_cm, num_labels, alpha, gamma):
    loss_func = FocalLoss(num_labels, alpha, gamma, True)
    # loss_func = nn.CrossEntropyLoss()
    total_loss = 0
    y_hat, y = [], []

    model.eval()
    for i in range(dataloader.num_batch):
        batch_X, batch_y = dataloader.data[i]
        if CUDA:
            batch_X = batch_X.cuda(DEVICE)
            batch_y = batch_y.cuda(DEVICE)
        model.eval()
        test_start = time.time()
        out = model(batch_X, overall_label_idx)
        loss = loss_func(out, batch_y.long())
        total_loss += loss.item()
        y_hat += out.max(1)[1].tolist()
        y += batch_y.tolist()
    y = np.array(y)
    y_hat = np.array(y_hat)
    accuracy = accuracy_score(y, y_hat)

    c_matrix = 0
    if is_cm:
        c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(0, len(overall_label_idx))])

    return total_loss, accuracy, c_matrix, y_hat


def evaluate_stu_loss_acc(model, dataloader, overall_label_idx, is_cm):
    loss_func = FocalLoss(NUM_LABELS, dataloader.alpha, GAMMA, True)
    # loss_func = nn.CrossEntropyLoss()
    total_loss = 0
    y_hat, y = [], []

    for i in range(dataloader.num_batch):
        batch_X, batch_y = dataloader.data[i]
        if CUDA:
            batch_X = batch_X.cuda(DEVICE)
            batch_y = batch_y.cuda(DEVICE)
        model.eval()
        test_start = time.time()
        out = model(batch_X, overall_label_idx)
        loss = loss_func(out, batch_y.long())
        total_loss += float(loss.item())

        y_hat += out.max(1)[1].tolist()
        y += batch_y.tolist()
    y = np.array(y)
    y_hat = np.array(y_hat)
    accuracy = accuracy_score(y, y_hat)

    c_matrix = None
    if is_cm:
        c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(0, NUM_LABELS)])

    return total_loss, accuracy, c_matrix, y_hat


def deal_matrix(c_matrix):
    print('start deal confusion matrix with {} labels'.format(c_matrix.shape[0]))
    row_sum = c_matrix.sum(axis=1)  # for precision
    col_sum = c_matrix.sum(axis=0)  # for recall

    P, R, F1 = [], [], []
    n_class = c_matrix.shape[0]
    for i in range(n_class):
        p = (c_matrix[i][i] / row_sum[i]) if row_sum[i] != 0 else 0
        r = (c_matrix[i][i] / col_sum[i]) if col_sum[i] != 0 else 0
        f1 = (2 * p * r / (p + r)) if p + r != 0 else 0
        P.append(p)
        R.append(r)
        F1.append(f1)
    return P, R, F1


def save_model(model, epoch, optimizer, loss, acc=None, f1=None, full=False):
    t_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # with open(os.path.join(MODEL_SAVE_PATH + t_str + '_eval_' + str(epoch) + '.pt'), 'wb') as f2:
    #     torch.save(model, f2)  # only for inference, model = torch.load(path), then: model.eval()
    model_name = MODEL_SAVE_PATH + str(NUM_LABELS) + '/' + t_str + '_train_' + str(epoch) + ('_full.pt' if full else '.pt')
    with open(os.path.join(model_name), 'wb') as f3:
        if full:
            torch.save(model, f3)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_accuracy': acc,
                'best_f1': f1
            }, f3)
    print('model successfully saved in path: {}'.format(model_name))
    # for training and inference, usage:
    """
        model = TheModelClass(*args, **kwargs)
        optimizer = TheOptimizerClass(*args, **kwargs)

        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_acc = checkpoint['best_accuracy']

         model.eval()
        # - or -
        model.train()
    """

if __name__ == '__main__':
    s_t = time.time()