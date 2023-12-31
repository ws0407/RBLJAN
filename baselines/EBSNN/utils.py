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

EXP_PERCENT = 1
TEST_PERCENT = 0.1
BATCH_SIZE = 128
NUM_CLASS = len(CLASSES)
EMBEDDING_DIM = 64
LR = 0.001
GAMMA = 2
EPOCHS = 120
CUDA = torch.cuda.is_available()
DEVICE = 0
DEVICE_COUNT = torch.cuda.device_count()
USE_PARALLEL = True
PADDING_IDX = 256
PKT_MAX_LEN = 1488
if USE_PARALLEL:
    ALL_DEVICE = ['cuda:{}'.format(i) for i in range(DEVICE_COUNT)]
else:
    ALL_DEVICE = ['cuda:0']


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
    def __init__(self, X, y, batch_size, shuffle=False, segment_len=16):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batch = int((len(self.y) - 1) / batch_size)
        self.data = []
        self.segment_len = segment_len
        self.alpha = calculate_alpha(self.y, LABELS, mode='invert')
        if self.shuffle:
            state = np.random.get_state()
            np.random.shuffle(self.X)
            np.random.set_state(state)
            np.random.shuffle(self.y)
            print(state)
        # self.lengths = np.array([len(x) for x in self.X])

        self.load_all_batches()

    def load_all_batches(self):
        _s_t = time.time()
        print('>> start load data with {} batches'.format(self.num_batch))
        for i in range(self.num_batch):
            if i % 1000 == 0 and i != 0:
                print('\r>> >> {} batches ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
            batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size]
            batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size]
            # batch_len = self.lengths[i * self.batch_size: (i + 1) * self.batch_size]
            # max_len = batch_len.max()
            max_len = PKT_MAX_LEN
            max_len = int(max_len / self.segment_len) * self.segment_len
            for j in range(len(batch_y)):
                batch_X[j] = batch_X[j][:max_len]
                if len(batch_X[j]) < max_len:
                    batch_X[j] = batch_X[j] + [256] * (max_len - len(batch_X[j]))
            batch_X = torch.tensor(batch_X.tolist()).float()
            batch_y = torch.from_numpy(batch_y)
            self.data.append((batch_X, batch_y))
        print('\n>> all batches load done with {}s'.format(time.time() - _s_t))


def get_dataloader(exp_percent, test_percent, batch_size):
    _s_t = time.time()
    print('start load X and y')
    X = []
    y = []
    pickle_dir = './data/' + DATA_MAP[NUM_LABELS] + '_PKL/'
    for i in CLASSES:
        file_name = pickle_dir + i + '.pkl'
        with open(file_name, 'rb') as f1:
            tmp = pickle.load(f1)
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
    train_loader, test_loader = Loader(X_train, y_train, batch_size), Loader(X_test, y_test, batch_size)
    return train_loader, test_loader


class FocalLoss(nn.Module):
    def __init__(self, class_num, device, alpha=None, gamma=2,
                 size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.device = device
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)

        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(inputs.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


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


def save_model(model, epoch):
    model_dir = MODEL_SAVE_PATH + DATA_MAP[NUM_LABELS] + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = model_dir + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + str(epoch) + '.pt'
    with open(model_name, 'wb') as f2:
        torch.save(model, f2)  # only for inference, model = torch.load(path), then: model.eval()
    print('model successfully saved in path: {}'.format(model_name))