import random
import time
import datetime
import os

import numpy as np
import pickle
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

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

MODEL_SAVE_PATH = './model/'
DATA_DIR = './data/'
DATA_MAP = {10: 'USTC-TFC', 12: 'ISCX-VPN', 20: 'X-WEB', 29: 'X-APP'}

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
    def __init__(self, X, y, batch_size):
        self.X = X.tolist()
        self.y = y.tolist()
        self.batch_size = batch_size
        self.num_batch = int((len(self.y) - 1) / batch_size)
        self.data = []
        # self.load_all_batches()

    def load_all_batches(self):
        _s_t = time.time()
        print('>> start load data with {} batches'.format(self.num_batch))
        for i in range(self.num_batch):
            if i % 1000 == 0 and i != 0:
                print('\r>> >> {} batches ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
            batch_X = self.X[i * self.batch_size: (i + 1) * self.batch_size]
            batch_y = self.y[i * self.batch_size: (i + 1) * self.batch_size]
            for j in range(len(batch_y)):
                if len(batch_X[j]) < PKT_MAX_LEN:
                    batch_X[j] = batch_X[j] + [0] * (PKT_MAX_LEN - len(batch_X[j]))
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
        for j in range(len(batch_y)):
            if len(batch_X[j]) < PKT_MAX_LEN:
                batch_X[j] = batch_X[j] + [0] * (PKT_MAX_LEN - len(batch_X[j]))
        batch_X = torch.tensor(batch_X).float()
        batch_y = torch.tensor(batch_y)
        return batch_X, batch_y


def get_dataloader(exp_percent, test_percent, batch_size):
    _s_t = time.time()
    print('start load X and y')
    X = []
    y = []
    pickle_dir = DATA_DIR + DATA_MAP[NUM_CLASS] + '_PKL/'
    for i in CLASSES:
        file_name = pickle_dir + i + '.pkl'
        with open(file_name, 'rb') as f1:
            tmp = pickle.load(f1)
        tmp = sum(tmp.values(), [])
        if len(tmp) > 210000:
            random.seed(2022)
            tmp = random.sample(tmp, 200000)
        num_pkt = len(tmp)
        X += tmp
        y += [LABELS[i]] * num_pkt
        print('{}: {}'.format(i, num_pkt))
    print('load X and y cost: {}s'.format(time.time() - _s_t))

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    if 0 < exp_percent < 1:
        print('start random select data(train_test_split) with exp_percent: {}'.format(exp_percent))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=exp_percent, shuffle=True, random_state=41,
                                                            stratify=y)
        X = X_test  # 0.8
        y = y_test

    print('start train_test_split with test_percent: {}'.format(test_percent))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, shuffle=True, random_state=41,
                                                        stratify=y)
    print('start load batches of train and test...')
    train_loader, test_loader = Loader(X_train, y_train, batch_size), Loader(X_test, y_test, batch_size)
    return train_loader, test_loader


def save_model(model, epoch):
    model_dir = MODEL_SAVE_PATH + DATA_MAP[NUM_CLASS] + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = model_dir + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + str(epoch) + '.pt'
    with open(model_name, 'wb') as f2:
        torch.save(model, f2)  # only for inference, model = torch.load(path), then: model.eval()
    print('model successfully saved in path: {}'.format(model_name))


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