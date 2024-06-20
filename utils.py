#!/usr/bin/env python
# -*- coding:utf-8 -*-

import datetime
import gc
import os
import random
import time
from math import sin, cos, pow

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from torch.autograd import Variable

MODEL_SAVE_PATH = './model/'

# CLASSES = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
# CLASSES = ['amazon', 'baidu', 'bing', 'douban', 'facebook', 'google', 'imdb', 'instagram', 'iqiyi', 'jd',
#            'neteasemusic', 'qqmail', 'reddit', 'taobao', 'ted', 'tieba', 'twitter', 'weibo', 'youku', 'youtube']
# CLASSES = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus',
#            'BitTorrent', 'FTP', 'Facetime', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'WorldOfWarcraft', 'SMB', 'Weibo']
# CLASSES = ['vimeo', 'spotify', 'voipbuster', 'sinauc', 'cloudmusic', 'weibo', 'baidu', 'tudou', 'amazon', 'thunder',
#            'gmail', 'pplive', 'qq', 'taobao', 'yahoomail', 'itunes', 'twitter', 'jd', 'sohu', 'youtube', 'youku',
#            'netflix', 'aimchat', 'kugou', 'skype', 'facebook', 'google', 'mssql', 'ms-exchange']
CLASSES = ['audio', 'chat', 'file', 'mail', 'streaming', 'voip',
           'vpn-audio', 'vpn-chat', 'vpn-file', 'vpn-mail', 'vpn-streaming', 'vpn-voip']
LABELS = {k: v for k, v in zip(CLASSES, range(len(CLASSES)))}

NUM_LABELS = len(LABELS)
DATA_MAP = {10: 'USTC-TFC', 12: 'ISCX-VPN', 20: 'X-WEB', 29: 'X-APP'}

CUDA = torch.cuda.is_available()
DEVICE = 0
DEVICE_COUNT = torch.cuda.device_count()
USE_PARALLEL = True
ALL_DEVICE = ['cuda:{}'.format(i) for i in range(DEVICE_COUNT)] if USE_PARALLEL else ['cuda:0']

# model and training
EMBEDDING_DIM = 256

PKT_MIN_LEN = 50
PKT_MAX_LEN = 1500
FLOW_MIN_LEN = 3
FLOW_MAX_LEN = 10

NGRAM_HEADER = 17
NGRAM_PAYLOAD = 65

KERNEL_NUM_HEADER = 8
KERNEL_NUM_PAYLOAD = 8

HEADER_LEN = 50
PAYLOAD_LEN = PKT_MAX_LEN - HEADER_LEN
PADDING_IDX = 256

DROPOUT = 0.5
EXP_PERCENT = 1
TEST_PERCENT = 0.1
BATCH_SIZE = 256
EPOCHS = 120
LR = 0.001

# adversarial
RANDOM_SIZE = 100

GAMMA = 2
LAMBDA = 0.1


def get_position_encoding_matrix(E=EMBEDDING_DIM, position_max=20000):
    """generate position encoding matrix
    :param E: embedding dimension
    :param position_max: 1500 is enough
    :return: position encoding matrix with a shape of position_max * E
    """
    Q = []
    for i in range(position_max):
        Q.append([j for j in np.array([[sin(i / pow(10000, 2 * x / E)), cos(i / pow(10000, (2 * x + 1) / E))] for x in range(int(E / 2))]).flatten()])
    return torch.from_numpy(np.array(Q))


def get_cosine_similarity_matrix(B, L):
    """similarity calculation unit
    :param B: byte embedding, e.g., batch_size * pkt_len * embedding_dim
    :param L: label embedding, e.g., num_labels * embedding_dim
    :return: cosine similarity between each byte and each label
    """
    G = torch.matmul(L, B.transpose(1, 2))  # b * k * l
    B = torch.norm(B, p=2, dim=2)  # B: b * l * e -> b * l, dim = 2 is for e's axis
    L = torch.norm(L, p=2, dim=1)  # L: k * e -> (k,), dim = 1 is also for e's axis

    B = B.unsqueeze(1)
    L = L.unsqueeze(1)
    G_norm = torch.matmul(L, B)  # b * k * l    # tensor multiplication
    # prevent divide 0 then, we replace 0 with 1
    ones = torch.ones(G_norm.size())  # Returns a tensor of all 1, with the shape G_norm The shape of norm
    if CUDA:
        ones = ones.to(G.device)
    G_norm = torch.where(G_norm != 0, G_norm, ones)
    G_hat = G / G_norm
    return G_hat


def calculate_alpha(y, labels, mode='normal'):
    """calculate alpha for focal loss
    :param y: all ground truth
    :param labels: all candidate labels
    :param mode: 'invert' is to focus on the samples that are difficult to classify
    :return: hyperparameter alpha of focal loss
    """
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
    """load data to batches
    :param X: all preprocessed packet sequences
    :param y: corresponding labels
    :param batch_size:
    """
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_batch = int((len(self.y) - 1) / batch_size)
        self.data = []
        self.alpha = calculate_alpha(self.y, LABELS, mode='invert')
        # self.load_all_batches()

    def load_all_batches(self):
        _s_t = time.time()
        print('>> start load data with {} batches'.format(self.num_batch))
        for i in range(self.num_batch):
            if i % 2000 == 0:
                print('\r>> >> {} batches ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
            batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size]
            batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size]
            for j in range(len(batch_y)):
                batch_X[j] = batch_X[j] + [256] * (PKT_MAX_LEN - len(batch_X[j]))
            batch_X = torch.tensor(batch_X.tolist()).float()
            # batch_X = torch.tensor(batch_X.tolist()).long()   # when train without GAN
            batch_y = torch.from_numpy(batch_y)
            self.data.append((batch_X, batch_y))
        self.X = None
        self.y = None
        gc.collect()
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
        return len(self.y) | len(self.data)

    def __getitem__(self, i):
        """If there is not enough memory, only load one batch at a time"""
        if len(self.data) > i:
            return self.data[i]
        batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size]
        batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size]
        for j in range(len(batch_y)):
            batch_X[j] = batch_X[j] + [256] * (PKT_MAX_LEN - len(batch_X[j]))
        batch_X = torch.tensor(batch_X.tolist()).float()
        # batch_X = torch.tensor(batch_X.tolist()).long()   # when train without GAN
        batch_y = torch.from_numpy(batch_y)
        return batch_X, batch_y


def get_dataloader(exp_percent, test_percent, batch_size):
    """turn pickle file to X and y, then goto loader
    :param exp_percent: Percentage used for experiments(exp_percent for training and evaluating, others for testing)
    :param test_percent: Percentage used for eval
    """
    _s_t = time.time()
    print('start load X and y')
    X = []
    y = []
    pickle_dir = './data/' + DATA_MAP[NUM_LABELS] + '_PKL/'
    for i in CLASSES:
        file_name = pickle_dir + i + '.pkl'
        with open(file_name, 'rb') as f1:
            tmp = pickle.load(f1)
        # tmp = sum([v for k, v in tmp.items()], [])    # when process into flow format (dictionary)
        if len(tmp) > 200000:  # to avoid memory explosion
            random.seed(2022)
            tmp = random.sample(tmp, 200000)
        num_pkt = len(tmp)
        X += tmp
        y += [LABELS[i]] * num_pkt
        print('{}: {} from {}'.format(i, num_pkt, file_name[file_name.rfind('/') + 1:]))
    print('load X and y cost: {}s'.format(time.time() - _s_t))

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    if 0 < exp_percent < 1:
        print('start random select data(train_test_split) with exp_percent: {}'.format(exp_percent))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=exp_percent, shuffle=True, random_state=41, stratify=y)
        X = X_test
        y = y_test

    print('start train_test_split with test_percent: {}'.format(test_percent))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, shuffle=True, random_state=41, stratify=y)

    print('start load batches of train and test...')
    train_loader, test_loader = Loader(X_train, y_train, batch_size), Loader(X_test, y_test, batch_size)
    return train_loader, test_loader


class FlowLoader:
    def __init__(self, X, y, batch_size=32, max_flow_len=FLOW_MAX_LEN, max_pkt_len=PKT_MAX_LEN):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.max_flow_len = max_flow_len
        self.max_pkt_len = max_pkt_len
        self.flow_lengths = np.array([len(x) for x in X])
        self.num_batch = int((len(self.y) - 1) / batch_size) + 1
        self.data = []
        self.alpha = calculate_alpha(self.y, LABELS, mode='normal')
        # self.load_all_batches()

    def load_all_batches(self):
        _s_t = time.time()
        print('>> start load data with {} batches'.format(self.num_batch))
        n = self.num_batch - 1
        for i in range(self.num_batch):
            if i % 100 == 0 and i != 0:
                print('>> >> {} batches ok with {}s'.format(i, time.time() - _s_t))
            batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size if i != n else None]
            batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size if i != n else None]
            batch_flow_lens = self.flow_lengths[i * self.batch_size: (i + 1) * self.batch_size if i != n else None]
            for j in range(len(batch_flow_lens)):
                batch_pkt_lens = [len(x) for x in batch_X[j]]
                for k in range(len(batch_pkt_lens)):
                    batch_X[j][k] = batch_X[j][k] + [256] * (self.max_pkt_len - len(batch_X[j][k]))
                for k in range(len(batch_pkt_lens), self.max_flow_len):
                    batch_X[j].append([256] * self.max_pkt_len)
            batch_X = torch.tensor(batch_X.tolist()).long()
            batch_y = torch.from_numpy(batch_y)
            self.data.append((batch_X, batch_y))
        self.X = None
        self.y = None
        gc.collect()
        print('>> all batches load done with {}s'.format(time.time() - _s_t))

    def __len__(self):
        return len(self.y) | len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_flow_dataloader(data_dir, batch_size, max_flow_len=FLOW_MAX_LEN, max_pkt_len=PKT_MAX_LEN):
    X = []
    y = []
    for label in CLASSES:
        with open(data_dir + label.lower() + '.pkl', 'rb') as f:
            flows = pickle.load(f)
        tmp = [_ for _ in list(flows.values()) if len(_) >= 2]
        X += tmp
        y += [LABELS[label]] * len(tmp)
        # break
    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_PERCENT, shuffle=True, random_state=20,
                                                        stratify=y)
    print('done\nstart load batches of train and test...')
    train_loader, test_loader = FlowLoader(X_train, y_train, batch_size, max_flow_len, max_pkt_len), \
                                FlowLoader(X_test, y_test, batch_size, max_flow_len, max_pkt_len)
    print('done!')
    return train_loader, test_loader


class FocalLoss(nn.Module):

    def __init__(self, num_labels=NUM_LABELS, alpha=None, gamma=GAMMA, size_average=True, is_reverse=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(num_labels, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.class_num = num_labels
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


def evaluate_loss_acc(model, dataloader, is_cm, num_labels):
    y_hat, y = [], []
    model.eval()
    for i in range(dataloader.num_batch):
        batch_X, batch_y = dataloader.data[i]
        if CUDA:
            batch_X = batch_X.cuda(DEVICE)
            batch_y = batch_y.cuda(DEVICE)
        model.eval()
        out1, out2 = model(batch_X)
        y_hat += out1.max(1)[1].tolist()
        y += batch_y.tolist()
    y = np.array(y)
    y_hat = np.array(y_hat)
    accuracy = accuracy_score(y, y_hat)

    c_matrix = 0
    if is_cm:
        c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(0, num_labels)])
    return accuracy, c_matrix


def deal_cm(c_matrix):
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


def save_model(model, epoch):
    model_dir = MODEL_SAVE_PATH + DATA_MAP[NUM_LABELS] + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = model_dir + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + str(epoch) + '.pt'
    with open(model_name, 'wb') as f2:
        torch.save(model, f2)  # only for inference, model = torch.load(path), then: model.eval()
    print('model successfully saved in path: {}'.format(model_name))

    """ for training and inference, usage:
        - Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f3)
                
        - load model
            model = TheModelClass(*args, **kwargs)
            optimizer = TheOptimizerClass(*args, **kwargs)
    
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        
        - train or eval
        
            model.eval()
            # - or -
            model.train()
    """


def eval_D(D, G, dataloader, is_cm, num_labels, states=None):
    y_hat_real, y_hat_fake, y = [], [], []
    for i in range(dataloader.num_batch):
        batch_X, batch_y = dataloader[i]
        batch_size = len(batch_y)
        if CUDA:
            batch_X = batch_X.cuda(DEVICE)
            batch_y = batch_y.cuda(DEVICE)
        D.eval()
        out1, _ = D(batch_X)
        y_hat_real += out1.max(1)[1].tolist()
        y += batch_y.tolist()

        x_inject = Variable(torch.randn((batch_size, RANDOM_SIZE)).float())
        class_mask = torch.zeros((batch_size, NUM_LABELS))
        if CUDA:
            x_inject = x_inject.cuda(DEVICE)
            class_mask = class_mask.cuda(DEVICE)
        class_mask.scatter_(1, batch_y.view(-1, 1).data, 1.)  # one-hot vector
        x_inject = torch.cat((x_inject, class_mask.float()), dim=1)
        G.eval()
        batch_X_fake = G(x_inject, batch_X) if states is None else G(x_inject, batch_X, states)
        D.eval()
        out1, _ = D(batch_X_fake)
        y_hat_fake += out1.max(1)[1].tolist()
    y = np.array(y)
    y_hat_real = np.array(y_hat_real)
    y_hat_fake = np.array(y_hat_fake)
    accuracy_real = accuracy_score(y, y_hat_real)
    accuracy_fake = accuracy_score(y, y_hat_fake)
    c_matrix_real, c_matrix_fake = 0, 0
    if is_cm:
        c_matrix_real = confusion_matrix(y, y_hat_real, labels=[i for i in range(0, num_labels)])
        c_matrix_fake = confusion_matrix(y, y_hat_fake, labels=[i for i in range(0, num_labels)])
        print(classification_report(y, y_hat_real, target_names=[CLASSES[_] for _ in unique_labels(y, y_hat_real)],
                                    digits=4))
        max_label_len = max([len(_) for _ in CLASSES])
        for i in range(len(c_matrix_real)):
            print(CLASSES[i].rjust(max_label_len), end='')
            for j in range(len(c_matrix_real[i])):
                num = str(c_matrix_real[i][j])
                num = ' ' * (6 - len(num)) + num
                print(num, end='')
            print()
        print()
        print(classification_report(y, y_hat_fake, target_names=[CLASSES[_] for _ in unique_labels(y, y_hat_fake)],
                                    digits=4))
        for i in range(len(c_matrix_fake)):
            print(CLASSES[i].rjust(max_label_len), end='')
            for j in range(len(c_matrix_fake[i])):
                num = str(c_matrix_fake[i][j])
                num = ' ' * (6 - len(num)) + num
                print(num, end='')
            print()
    return accuracy_real, accuracy_fake, c_matrix_real, c_matrix_fake