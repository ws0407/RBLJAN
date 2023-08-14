import datetime
import os, time, sys
import random
from math import sin, cos, pow
import numpy as np
import torch, pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt


# CLASSES = ['audio', 'chat', 'file', 'mail', 'streaming', 'voip',
#            'vpn-audio', 'vpn-chat', 'vpn-file', 'vpn-mail', 'vpn-streaming', 'vpn-voip']
# LABELS = {'audio': 0, 'chat': 1, 'file': 2, 'mail': 3, 'streaming': 4, 'voip': 5,
#           'vpn-audio': 6, 'vpn-chat': 7, 'vpn-file': 8, 'vpn-mail': 9, 'vpn-streaming': 10, 'vpn-voip': 11}

# LABELS = {'Cridex': 0, 'Geodo': 1, 'Htbot': 2, 'Miuref': 3, 'Neris': 4, 'Nsis-ay': 5,
#           'Shifu': 6, 'Tinba': 7, 'Virut': 8, 'Zeus': 9, 'BitTorrent': 10, 'FTP': 11,
#           'Facetime': 12, 'Gmail': 13, 'MySQL': 14, 'Outlook': 15, 'Skype': 16,
#            'WorldOfWarcraft': 17, 'SMB': 18, 'Weibo': 19}
# CLASSES = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus',
#            'BitTorrent', 'FTP', 'Facetime', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'WorldOfWarcraft', 'SMB', 'Weibo']


# LABELS = {'BitTorrent': 0, 'FTP': 1, 'Facetime': 2, 'Gmail': 3, 'MySQL': 4, 'Outlook': 5, 'Skype': 6,
#           'WorldOfWarcraft': 7, 'SMB': 8, 'Weibo': 9, 'Cridex': 10, 'Geodo': 11, 'Htbot': 12, 'Miuref': 13,
#           'Neris': 14, 'Nsis-ay': 15, 'Shifu': 16, 'Tinba': 17, 'Virut': 18, 'Zeus': 19}
# CLASSES = ['BitTorrent', 'FTP', 'Facetime', 'Gmail', 'MySQL', 'Outlook', 'Skype',
#           'WorldOfWarcraft', 'SMB', 'Weibo', 'Cridex', 'Geodo', 'Htbot', 'Miuref',
#           'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
# CLASSES = ['BitTorrent', 'FTP', 'Facetime', 'Gmail', 'MySQL', 'Outlook', 'Skype',
#            'WorldOfWarcraft', 'SMB', 'Weibo']
# LABELS = {'BitTorrent': 0, 'FTP': 1, 'Facetime': 2, 'Gmail': 3, 'MySQL': 4, 'Outlook': 5,
#           'Skype': 6, 'WorldOfWarcraft': 7, 'SMB': 8, 'Weibo': 9}
# LABELS = {'Cridex': 0, 'Geodo': 1, 'Htbot': 2, 'Miuref': 3, 'Neris': 4, 'Nsis-ay': 5,
#           'Shifu': 6, 'Tinba': 7, 'Virut': 8, 'Zeus': 9}
# CLASSES = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
# CLASSES = ['amazon', 'baidu', 'bing', 'douban', 'facebook', 'google', 'imdb', 'instagram', 'iqiyi', 'jd',
#            'neteasemusic', 'qqmail', 'reddit', 'taobao', 'ted', 'tieba', 'twitter', 'weibo', 'youku', 'youtube']
# LABELS = {'amazon': 0, 'baidu': 1, 'bing': 2, 'douban': 3, 'facebook': 4, 'google': 5, 'imdb': 6, 'instagram': 7,
#           'iqiyi': 8, 'jd': 9, 'neteasemusic': 10, 'qqmail': 11, 'reddit': 12, 'taobao': 13, 'ted': 14, 'tieba': 15,
#           'twitter': 16, 'weibo': 17, 'youku': 18, 'youtube': 19}
CLASSES = ['vimeo', 'spotify', 'voipbuster', 'sinauc', 'cloudmusic', 'weibo', 'baidu', 'tudou', 'amazon', 'thunder',
           'gmail', 'pplive', 'qq', 'taobao', 'yahoomail', 'itunes', 'twitter', 'jd', 'sohu', 'youtube', 'youku',
           'netflix', 'aimchat', 'kugou', 'skype', 'facebook', 'google', 'mssql', 'ms-exchange']

LABELS = {'vimeo': 0, 'spotify': 1, 'voipbuster': 2, 'sinauc': 3, 'cloudmusic': 4, 'weibo': 5,
          'baidu': 6, 'tudou': 7, 'amazon': 8, 'thunder': 9, 'gmail': 10, 'pplive': 11,
          'qq': 12, 'taobao': 13, 'yahoomail': 14, 'itunes': 15, 'twitter': 16, 'jd': 17,
          'sohu': 18, 'youtube': 19, 'youku': 20, 'netflix': 21, 'aimchat': 22, 'kugou': 23,
          'skype': 24, 'facebook': 25, 'google': 26, 'mssql': 27, 'ms-exchange': 28}

BATCH_SIZE = 128
EPOCHS = 120
LR = 0.001

NUM_CLASS = len(CLASSES)
PKT_MAX_LEN = 1500

EXP_PERCENT = 1
TEST_PERCENT = 0.1

CUDA = torch.cuda.is_available()
DEVICE = 0


class Loader:
    def __init__(self, X, y, batch_size, padding=False, shuffle=False):
        self.X = X.tolist()
        self.y = y.tolist()
        self.batch_size = batch_size
        self.padding = padding
        self.shuffle = shuffle
        self.num_batch = int((len(self.y) - 1) / batch_size)
        self.data = []
        if self.shuffle and self.num_batch > 8000:
            state = np.random.get_state()
            np.random.shuffle(self.X)
            np.random.set_state(state)
            np.random.shuffle(self.y)
            # print(state)
        # self.lengths = np.array([len(x) for x in self.X])
        # self.load_all_batches()

    def load_all_batches(self):
        _s_t = time.time()
        print('>> start load data with {} batches'.format(self.num_batch))
        n = self.num_batch - 1
        for i in range(self.num_batch):
            if i % 1000 == 0 and i != 0:
                print('\r>> >> {} batches ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
            batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size]
            batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size]
            # batch_len = self.lengths[i * self.batch_size: (i + 1) * self.batch_size]
            # max_len = batch_len.max()
            max_len = PKT_MAX_LEN
            # X_mask = []
            for j in range(len(batch_y)):
                if len(batch_X[j]) < max_len:
                    batch_X[j] = batch_X[j] + [0] * (max_len - len(batch_X[j]))
            batch_X = torch.tensor(batch_X).float()
            batch_y = torch.tensor(batch_y)
            self.data.append((batch_X, batch_y))
        self.X = None
        self.y = None
        print('\n>> all batches load done with {}s'.format(time.time() - _s_t))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size]
        batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size]
        max_len = PKT_MAX_LEN
        for j in range(len(batch_y)):
            if len(batch_X[j]) < max_len:
                batch_X[j] = batch_X[j] + [0] * (max_len - len(batch_X[j]))
        batch_X = torch.tensor(batch_X).float()
        batch_y = torch.tensor(batch_y)
        return batch_X, batch_y


def get_dataloader(exp_percent, test_percent, batch_size, padding=False):
    _s_t = time.time()
    print('start load X and y')
    X = []
    y = []
    pickle_dir = '/data/ws/tmp/BLJAN-IWQOS/datasets/BLJAN/TrafficX/result_doc_no_port/'
    # pickle_dir = '/data/ws/tmp/BLJAN-IWQOS/datasets/BLJAN/ISCX-VPN-NonVPN-2016/result_doc_no_port/'
    for i in CLASSES:
        file_name = pickle_dir + i + '.pkl.pkl'
        # if not os.path.isfile(file_name):
        #     file_name = '/data/ws/tmp/BLJAN-IWQOS/datasets/BLJAN/USTC-TFC2016/result_doc_benign/' + 'X_' + i + '.pkl'
        with open(file_name, 'rb') as f1:
            tmp = pickle.load(f1)
        tmp = sum(tmp.values(), [])
        # if LABELS[i] >= 10 and 'Cridex' in CLASSES:
        #     if len(tmp) > 100000:
        #         random.seed(41)
        #         tmp = random.sample(tmp, 100000)
        if len(tmp) > 210000:
            random.seed(2022)
            tmp = random.sample(tmp, 200000)
        num_pkt = len(tmp)
        X += tmp
        y += [LABELS[i]] * num_pkt
        print('{}: {}'.format(i, num_pkt))
        # break
    print('load X and y cost: {}s'.format(time.time() - _s_t))

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    if 0 < exp_percent < 1:
        print('start random select data(train_test_split) with exp_percent: {}'.format(exp_percent))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=exp_percent, shuffle=True, random_state=20,
                                                            stratify=y)
        X = X_test  # 0.8
        y = y_test

    print('start train_test_split with test_percent: {}'.format(test_percent))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, shuffle=True, random_state=20,
                                                        stratify=y)
    print('start load batches of train and test...')
    train_loader, test_loader = Loader(X_train, y_train, batch_size, padding), Loader(X_test, y_test, batch_size,
                                                                                      padding)
    return train_loader, test_loader


def get_dataloader_train(exp_percent, test_percent, batch_size):
    _s_t = time.time()
    print('start load X and y')
    X = []
    y = []
    pickle_dir = '/data/ws/tmp/BLJAN-IWQOS/datasets/BLJAN/TrafficX/result_doc_no_port/'
    # pickle_dir = '/data/ws/tmp/BLJAN-IWQOS/datasets/BLJAN/ISCX-VPN-NonVPN-2016/result_doc_no_port/'
    for i in CLASSES:
        file_name = pickle_dir + i + '.pkl.pkl'
        # if not os.path.isfile(file_name):
        #     file_name = '/data/ws/tmp/BLJAN-IWQOS/datasets/BLJAN/USTC-TFC2016/result_doc_benign/' + 'X_' + i + '.pkl'
        with open(file_name, 'rb') as f1:
            tmp = pickle.load(f1)
        tmp = sum(tmp.values(), [])
        # if LABELS[i] >= 10 and 'Cridex' in CLASSES:
        #     if len(tmp) > 100000:
        #         random.seed(41)
        #         tmp = random.sample(tmp, 100000)
        if len(tmp) > 210000:
            random.seed(2022)
            tmp = random.sample(tmp, 200000)
        num_pkt = len(tmp)
        X += tmp
        y += [LABELS[i]] * num_pkt
        print('{}: {}'.format(i, num_pkt))
        # break
    print('load X and y cost: {}s'.format(time.time() - _s_t))

    train_loader = Loader(X, y, batch_size)
    return train_loader


def evaluate_loss_acc(model, dataloader, is_cm, num_labels, device):
    y_hat, y = [], []
    for i in range(dataloader.num_batch):
        batch_X, batch_y = dataloader.data[i]
        batch_X = batch_X.cuda(device)
        batch_y = batch_y.cuda(device)
        model.eval()
        test_start = time.time()
        out = model(batch_X)
        y_hat += out.max(1)[1].tolist()
        y += batch_y.tolist()
    y = np.array(y)
    y_hat = np.array(y_hat)
    accuracy = accuracy_score(y, y_hat)
    c_matrix = None
    if is_cm:
        c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(0, num_labels)])

    return accuracy, c_matrix


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