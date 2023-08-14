# -*- coding: utf-8 -*-
import random

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import argparse
import time, os
from sklearn.metrics import accuracy_score, confusion_matrix
from model import SAM
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
TEST_PERCENT = 0.15
PKT_MAX_LEN = 50
CUDA = torch.cuda.is_available()
DEVICE = 1
EPOCHS = 120
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

NGRAM = 50


class Loader:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_batch = int((len(self.y) - 1) / batch_size) + 1
        self.data = []
        self.lengths = np.array([len(x) for x in self.X])

        self.load_all_batches()

    def load_all_batches(self):
        _s_t = time.time()
        print('>> start load data with {} batches'.format(self.num_batch))
        n = self.num_batch - 1
        for batch in range(self.num_batch):
            if batch % 1000 == 0 and batch != 0:
                print('\r>> >> {} batches ok with {}s...'.format(batch, time.time() - _s_t), end='', flush=True)
            batch_X = self.X[batch * self.batch_size: (batch + 1) * self.batch_size]
            batch_y = self.y[batch * self.batch_size: (batch + 1) * self.batch_size]
            batch_len = self.lengths[batch * self.batch_size: (batch + 1) * self.batch_size]
            max_len = PKT_MAX_LEN
            batch_pos = []
            for j in range(len(batch_y)):
                batch_pos.append([p for p in range(batch_len[j])] + [0] * (max_len - batch_len[j]))
                if batch_len[j] < max_len:
                    batch_X[j] = batch_X[j] + [0] * (max_len - batch_len[j])
            batch_X = torch.tensor(batch_X.tolist()).long()
            batch_y = torch.from_numpy(batch_y)
            batch_pos = torch.tensor(batch_pos).long()
            self.data.append((batch_X, batch_pos, batch_y))
        print('\n>> all batches load done with {}s'.format(time.time() - _s_t))


def get_dataloader(test_percent, batch_size):
    _s_t = time.time()
    print('start load X and y')
    X = []
    y = []
    pkl_dir = './data/ISCX-VPN/result_doc_sam/'
    file_names = os.listdir(pkl_dir)
    file_paths = [os.path.join(pkl_dir, file_name) for file_name in file_names]
    file_paths.sort()
    for file_path in file_paths:
        if 'pkl' in file_path:
            print('load: {}'.format(file_path))
            with open(file_path, 'rb') as f1:
                tmp = pickle.load(f1)
            if len(tmp) > 210000:
                random.seed(41)
                tmp = random.sample(tmp, 200000)
            if 'X_' in file_path:
                X += tmp
            elif 'y_' in file_path:
                y += tmp
    print('load X and y cost: {}s'.format(time.time() - _s_t))
    # exit(0)
    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)

    print('start train_test_split with test_percent: {}'.format(test_percent))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, shuffle=True, random_state=20,
                                                        stratify=y)

    print('start load batches of train and test...')
    train_loader, test_loader = Loader(X_train, y_train, batch_size), Loader(X_test, y_test, batch_size)
    return train_loader, test_loader


def cal_loss(pred, gold, cls_ratio=None):
    gold = gold.contiguous().view(-1)
    # By default, the losses are averaged over each loss element in the batch.
    loss = F.cross_entropy(pred, gold)

    # torch.max(a,0) 返回每一列中最大值的那个元素，且返回索引
    pred = F.softmax(pred, dim=-1).max(1)[1]
    # 相等位置输出1，否则0
    n_correct = pred.eq(gold)
    acc = n_correct.sum().item() / n_correct.shape[0]

    return loss, acc * 100


def evaluate_loss_acc(model, dataloader, is_cm, num_label):
    # loss_func = nn.CrossEntropyLoss()
    y_hat, y = [], []
    for i in range(dataloader.num_batch):
        batch_X, batch_pos, batch_y = dataloader.data[i]
        if CUDA:
            # X_mask = torch.tensor(X_mask.tolist()).cuda(DEVICE)
            batch_X = batch_X.cuda(DEVICE)
            batch_pos = batch_pos.cuda(DEVICE)
            batch_y = batch_y.cuda(DEVICE)
        model.eval()
        out, scores = model(batch_X, batch_pos)
        y_hat += out.tolist()
        y += batch_y.tolist()
    y = np.array(y)
    y_hat = np.array(y_hat)
    accuracy = accuracy_score(y, y_hat)

    c_matrix = 0
    if is_cm:
        c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(num_label)])
    return accuracy, c_matrix, y_hat


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


def train_SAM():
    train_loader, test_loader = get_dataloader(TEST_PERCENT, BATCH_SIZE)
    model = SAM(num_class=len(CLASSES), max_byte_len=PKT_MAX_LEN)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
    if CUDA:
        model = model.cuda(DEVICE)
    num_batch = train_loader.num_batch
    _s_t = time.time()
    all_f1, all_test_acc = [], []
    best_acc, best_f1 = 0, 0

    for epoch in range(EPOCHS):
        # train_total_loss, train_total_acc = [], []
        print('\nepoch: {} start to train...'.format(epoch))
        for i in range(num_batch):
            batch_X, batch_pos, batch_y = train_loader.data[i]
            if CUDA:
                # X_mask = torch.tensor(X_mask.tolist()).cuda(DEVICE)
                batch_X = batch_X.cuda(DEVICE)
                batch_pos = batch_pos.cuda(DEVICE)
                batch_y = batch_y.cuda(DEVICE)
            # start to train
            model.train()
            optimizer.zero_grad()
            out = model(batch_X, batch_pos)
            loss_per_batch, acc_per_batch = cal_loss(out, batch_y)
            loss_per_batch.backward()
            optimizer.step()

            # train_total_loss.append(loss_per_batch.item())
            # train_total_acc.append(acc_per_batch)

            if i % 1000 == 0:
                print('\r>> >> batch {} ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
        print('\n>> All batch {} ok with {}s'.format(num_batch, time.time() - _s_t))
        eval_s_t = time.time()
        print('>> >> start evaluate loss accuracy with {} batches'.format(test_loader.num_batch))
        test_accuracy, test_c_matrix, test_y_hat = evaluate_loss_acc(model, test_loader, True, len(CLASSES))
        print('evaluate done with {}s'.format(time.time() - eval_s_t))
        if test_accuracy > best_acc:
            best_acc = test_accuracy
        all_test_acc.append(test_accuracy)
        P, R, f1 = deal_matrix(test_c_matrix)
        all_f1.append(np.array(f1).mean())
        best_f1 = np.array(all_f1).max()
        print('>> epoch: {} ok:\n'
              '>> >> test accuracy: {}, best test accuracy: {},\n'
              '>> >> test f1-score: {}, best f1-score: {}'.format(epoch, test_accuracy, best_acc, all_f1[-1], best_f1))
        if best_acc == test_accuracy:
            print('>> test accuracy increased !!!')
        if best_f1 == all_f1[-1]:
            print('>> test f1-score increased !!!')
        print('>> detailed test results of every label:\nlabel\tPrecision\tRecall\tF1-score')
        for key in LABELS:
            print('{}\t{}\t{}\t{}'.format(key, P[int(LABELS[key])], R[int(LABELS[key])], f1[int(LABELS[key])]))
        if epoch > 10 and all_f1[-1] < best_f1 and all_f1[-2] < best_f1 and all_f1[-3] < best_f1 and all_f1[
            -4] < best_f1 and all_f1[-5] < best_f1:
            print('early stop with epoch: {}, \nbest test accuracy: {}, best f1-score: {}'.format(epoch, best_acc,
                                                                                                  best_f1))
            break


# def test_epoch(model, test_data):
#     ''' Epoch operation in training phase'''
#     model.eval()
#
#     total_acc = []
#     total_pred = []
#     total_score = []
#     total_time = []
#     # tqdm: 进度条库
#     # desc ：进度条的描述
#     # leave：把进度条的最终形态保留下来 bool
#     # mininterval：最小进度更新间隔，以秒为单位
#     for batch in tqdm(
#             test_data, mininterval=2,
#             desc='  - (Testing)   ', leave=False):
#         # prepare data
#         src_seq, src_seq2, gold = batch
#         src_seq, src_seq2, gold = src_seq.cuda(), src_seq2.cuda(), gold.cuda()
#         gold = gold.contiguous().view(-1)
#
#         # forward
#         torch.cuda.synchronize()
#         start = time.time()
#         pred, score = model(src_seq, src_seq2)
#         torch.cuda.synchronize()
#         end = time.time()
#         # 相等位置输出1，否则0
#         n_correct = pred.eq(gold)
#         acc = n_correct.sum().item() * 100 / n_correct.shape[0]
#         total_acc.append(acc)
#         total_pred.extend(pred.long().tolist())
#         total_score.append(torch.mean(score, dim=0).tolist())
#         total_time.append(end - start)
#
#     return sum(total_acc) / len(total_acc), np.array(total_score).mean(axis=0), \
#            total_pred, sum(total_time) / len(total_time)
#
#
# def train_epoch(model, training_data, optimizer):
#     ''' Epoch operation in training phase'''
#     model.train()
#
#     total_loss = []
#     total_acc = []
#     # tqdm: 进度条库
#     # desc ：进度条的描述
#     # leave：把进度条的最终形态保留下来 bool
#     # mininterval：最小进度更新间隔，以秒为单位
#     for batch in tqdm(
#             training_data, mininterval=2,
#             desc='  - (Training)   ', leave=False):
#         # prepare data
#         src_seq, src_seq2, gold = batch
#         src_seq, src_seq2, gold = src_seq.cuda(), src_seq2.cuda(), gold.cuda()
#
#         optimizer.zero_grad()
#         # forward
#         pred = model(src_seq, src_seq2)
#         loss_per_batch, acc_per_batch = cal_loss(pred, gold)
#         # update parameters
#         loss_per_batch.backward()
#         optimizer.step()
#
#         # 只有一个元素，可以用item取而不管维度
#         total_loss.append(loss_per_batch.item())
#         total_acc.append(acc_per_batch)
#
#     return sum(total_loss) / len(total_loss), sum(total_acc) / len(total_acc)
#
#
# def main(i, flow_dict):
#     f = open('results/results_%d.txt' % i, 'w')
#     f.write('Train Loss Time Test\n')
#     f.flush()
#
#     model = SAM(num_class=len(protocols), max_byte_len=max_byte_len).cuda()
#     optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
#     loss_list = []
#     # default epoch is 3
#     for epoch_i in trange(5, mininterval=2, desc='  - (Training Epochs)   ', leave=False):
#
#         train_x, train_y, train_label = load_epoch_data(flow_dict, 'train')
#         training_data = torch.utils.data.DataLoader(
#             Dataset(x=train_x, y=train_y, label=train_label),
#             num_workers=0,
#             collate_fn=paired_collate_fn,
#             batch_size=128,
#             shuffle=True
#         )
#         train_loss, train_acc = train_epoch(model, training_data, optimizer)
#
#         test_x, test_y, test_label = load_epoch_data(flow_dict, 'test')
#         test_data = torch.utils.data.DataLoader(
#             Dataset(x=test_x, y=test_y, label=test_label),
#             num_workers=0,
#             collate_fn=paired_collate_fn,
#             batch_size=128,
#             shuffle=False
#         )
#         test_acc, score, pred, test_time = test_epoch(model, test_data)
#         with open('results/atten_%d.txt' % i, 'w') as f2:
#             f2.write(' '.join(map('{:.4f}'.format, score)))
#
#         # write F1, PRECISION, RECALL
#         with open('results/metric_%d.txt' % i, 'w') as f3:
#             f3.write('F1 PRE REC\n')
#             p, r, fscore, _ = precision_recall_fscore_support(test_label, pred)
#             for a, b, c in zip(fscore, p, r):
#                 # for every cls
#                 f3.write('%.2f %.2f %.2f\n' % (a, b, c))
#                 f3.flush()
#             if len(fscore) != len(protocols):
#                 a = set(pred)
#                 b = set(test_label[:, 0])
#                 f3.write('%s\n%s' % (str(a), str(b)))
#
#         # write Confusion Matrix
#         with open('results/cm_%d.pkl' % i, 'wb') as f4:
#             pickle.dump(confusion_matrix(test_label, pred, normalize='true'), f4)
#
#         # write ACC
#         f.write('%.2f %.4f %.6f %.2f\n' % (train_acc, train_loss, test_time, test_acc))
#         f.flush()
#
#     # # early stop
#     # if len(loss_list) == 5:
#     # 	if abs(sum(loss_list)/len(loss_list) - train_loss) < 0.005:
#     # 		break
#     # 	loss_list[epoch_i%len(loss_list)] = train_loss
#     # else:
#     # 	loss_list.append(train_loss)
#
#     f.close()


if __name__ == '__main__':
    s_t = time.time()
    train_SAM()

    # for i in range(10):
    #     with open('pro_flows_%d_noip_fold.pkl' % i, 'rb') as f:
    #         flow_dict = pickle.load(f)
    #     print('====', i, ' fold validation ====')
    #     main(i, flow_dict)
