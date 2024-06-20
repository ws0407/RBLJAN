import os, time, sys
import random
import numpy as np
import torch, pickle
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


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
NUM_LABELS = len(LABELS)

BATCH_SIZE = 256
EPOCHS = 120
LR = 0.0005
GAMMA = 2

PKT_MAX_LEN = 784

EXP_PERCENT = 1
TEST_PERCENT = 0.1

CUDA = torch.cuda.is_available()
DEVICE = 0


def calculate_alpha(y):
    alpha = [0] * NUM_LABELS
    for i in y:
        alpha[i] += 1
    alpha = [np.log(0.15 * sum(alpha) / max(_, 1000)) for _ in alpha]
    alpha = np.array([_ if _ > 1 else 1 for _ in alpha])
    return torch.from_numpy(alpha)


class Loader:
    def __init__(self, X, y, batch_size, padding=False, shuffle=False):
        self.X = X.tolist()
        self.y = y.tolist()
        self.batch_size = batch_size
        self.padding = padding
        self.shuffle = shuffle
        self.num_batch = int((len(self.y) - 1) / batch_size)
        self.data = []
        self.alpha = calculate_alpha(self.y)
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
        for i in range(self.num_batch):
            if i % 1000 == 0 and i != 0:
                print('\r>> >> {} batches ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
            batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size]
            batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size]
            max_len = PKT_MAX_LEN
            for j in range(len(batch_y)):
                if len(batch_X[j]) < max_len:
                    batch_X[j] = batch_X[j] + [0] * (max_len - len(batch_X[j]))
                elif len(batch_X[j]) > max_len:
                    batch_X[j] = batch_X[j][:max_len]
            batch_X = torch.tensor(batch_X).float() / 255.
            # batch_X = batch_X.view(-1, 28, 28)
            batch_y = torch.tensor(batch_y)
            self.data.append((batch_X, batch_y))
        self.X = None
        self.y = None
        print('\n>> all batches load done with {}s'.format(time.time() - _s_t))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if i < len(self.data):
            return self.data[i]
        batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size]
        batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size]
        max_len = PKT_MAX_LEN
        for j in range(len(batch_y)):
            if len(batch_X[j]) < max_len:
                batch_X[j] = batch_X[j] + [0] * (max_len - len(batch_X[j]))
            elif len(batch_X[j]) > max_len:
                batch_X[j] = batch_X[j][:max_len]
        batch_X = torch.tensor(batch_X).float() / 255.
        # batch_X = batch_X.view(-1, 28, 28)
        batch_y = torch.tensor(batch_y)
        return batch_X, batch_y


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
