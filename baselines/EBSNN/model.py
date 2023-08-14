#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

from utils import CUDA, BATCH_SIZE, DEVICE_COUNT, ALL_DEVICE, PADDING_IDX, PKT_MAX_LEN
from torch.autograd.function import Function
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels


class OneHot(Function):
    byte_mask_cpu = torch.zeros((int(BATCH_SIZE/DEVICE_COUNT), PKT_MAX_LEN, PADDING_IDX + 1))
    seq_cpu = torch.tensor([_ for _ in range(PADDING_IDX + 1)]).float().unsqueeze(1)
    byte_mask = {}
    seq = {}
    if CUDA:
        for device in ALL_DEVICE:
            byte_mask[device] = byte_mask_cpu.cuda(device)
            seq[device] = seq_cpu.cuda(device)

    @staticmethod
    def forward(ctx, inputs):
        ctx.batch_size = inputs.size(0)
        ctx.input_len = inputs.size(1)
        ctx.device = '{}'.format(inputs.device)
        inputs = inputs.long()
        byte_mask = OneHot.byte_mask[ctx.device].scatter(2, inputs.view(ctx.batch_size, -1, 1).data, 1.)  # one-hot vector
        byte_mask[:, :, PADDING_IDX:].fill_(0)
        # ctx.byte_mask = byte_mask
        return byte_mask

    @staticmethod
    def backward(ctx, grad_output):
        # print('grad_output:', grad_output, grad_output.shape)
        # grad_input = (grad_output * ctx.byte_mask).sum(2)
        grad_input = torch.matmul(grad_output, OneHot.seq[ctx.device]).view(grad_output.size(0), -1)
        # print('grad_input:', grad_input, grad_input.shape)
        # ctx.byte_mask = None
        return grad_input


class EBSNN_LSTM(nn.Module):
    def __init__(self, num_class, embedding_dim, device,
                 bidirectional=True, segment_len=16, dropout_rate=0.5):
        super(EBSNN_LSTM, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda:{}".format(device))
        self.segment_len = segment_len
        self.rnn_dim = 100
        # if bi-direction
        self.rnn_directions = 2 if bidirectional else 1

        # 256 is 'gg', will be set [0,0..0]
        # self.byte_embed = nn.Embedding(257, self.embedding_dim, padding_idx=256)
        self.byte_one_hot = OneHot().apply
        self.byte_embed = nn.Linear(257, self.embedding_dim, bias=False)
        # to one-hot
        self.byte_embed.requires_grad = True

        self.rnn1 = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.rnn_dim,
                            batch_first=True,
                            bidirectional=(self.rnn_directions == 2))
        self.rnn2 = nn.LSTM(self.rnn_directions * self.rnn_dim, self.rnn_dim,
                            batch_first=True,
                            bidirectional=(self.rnn_directions == 2))

        self.hc1 = torch.randn(self.rnn_dim, 1)
        self.hc2 = torch.randn(self.rnn_dim, 1)

        self.fc1 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc2 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc3 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.num_class)

    '''
    x: b * l * 8
    lengths: (b,), every batch's length
    '''

    def forward(self, x):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        batch_size = x.size(0)
        x = self.byte_one_hot(x)
        x = self.byte_embed(x.view(batch_size, -1, self.segment_len, 257))
        # x = self.byte_embed(x)  # b * l * 8 * 257

        out1, (h_n, c_n) = self.rnn1(x.view(-1, self.segment_len, self.embedding_dim))
        # bl * 8 * 100
        h = torch.tanh(self.fc1(out1.contiguous().view(
            -1, self.rnn_directions * self.rnn_dim))).view(
                out1.size(0), out1.size(1), -1)
        if CUDA:
            h = h.to(x.device)
            self.hc1 = self.hc1.to(x.device)
        weights = (torch.matmul(h, self.hc1)).view(-1, self.segment_len)
        weights = F.softmax(weights, dim=1)
        weights = weights.view(-1, 1, self.segment_len)
        # b * l * 200
        out2 = torch.matmul(weights, out1).view(
            batch_size, -1, self.rnn_dim * self.rnn_directions)

        out3, (h1_n, h2_n) = self.rnn2(out2)  # out3: b * l * 200
        h2 = torch.tanh(self.fc2(out3.contiguous().view(
            -1, self.rnn_dim * self.rnn_directions))).view(
                out3.size(0), out3.size(1), -1)
        if CUDA:
            h2 = h2.to(x.device)
            self.hc2 = self.hc2.to(x.device)
        weights2 = F.softmax((torch.matmul(h2, self.hc2)).view(
            batch_size, -1), dim=1).view(batch_size, 1, -1)
        out4 = torch.matmul(weights2, out3).view(
            batch_size, self.rnn_dim * self.rnn_directions)

        out = self.fc3(out4)
        return out


class EBSNN_GRU(nn.Module):
    def __init__(self, num_class, embedding_dim, device,
                 segment_len=8, bidirectional=False,
                 dropout_rate=0.5):
        super(EBSNN_GRU, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda:{}".format(device))
        self.segment_len = segment_len
        self.rnn_dim = 100
        # if bi-direction
        self.rnn_directions = 2 if bidirectional else 1

        # 256 is 'gg', will be set [0,0..0]
        self.byte_one_hot = OneHot().apply
        self.byte_embed = nn.Linear(257, self.embedding_dim, bias=False)
        # to one-hot
        self.byte_embed.requires_grad = True
        self.rnn1 = nn.GRU(input_size=self.embedding_dim,
                           hidden_size=self.rnn_dim,
                           batch_first=True,
                           bidirectional=(self.rnn_directions == 2))
        self.rnn2 = nn.GRU(self.rnn_directions * self.rnn_dim,
                           self.rnn_dim, batch_first=True,
                           bidirectional=(self.rnn_directions == 2))
        self.hc1 = torch.randn(self.rnn_dim, 1)
        self.hc2 = torch.randn(self.rnn_dim, 1)
        self.fc1 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc2 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc3 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.num_class)

    def forward(self, x):
        """
        x: b * l * 8
        lengths: (b,), every batch's length
        """
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        batch_size = x.size(0)
        x = self.byte_one_hot(x)
        x = self.byte_embed(x.view(batch_size, -1, self.segment_len, 257))
        # x = self.byte_embed(x)  # b * l * 8 * 257

        out1, h_n = self.rnn1(x.view(-1, self.segment_len,
                                            self.embedding_dim))
        # bl * 8 * self.rnn_dim
        h = torch.tanh(self.fc1(out1.contiguous().view(
            -1, self.rnn_dim * self.rnn_directions))).view(
                out1.size(0), out1.size(1), -1)
        if CUDA:
            h = h.to(x.device)
            self.hc1 = self.hc1.to(x.device)
        weights = (torch.matmul(h, self.hc1)).view(-1,
                                                   self.segment_len)
        weights = F.softmax(weights, dim=1)
        weights = weights.view(-1, 1, self.segment_len)
        # b * l * self.rnn_dim * self.rnn_directions
        out2 = torch.matmul(weights, out1).view(
            batch_size, -1, self.rnn_dim * self.rnn_directions)
        # out3: b * l * self.rnn_dim * self.rnn_directions
        out3, h1_n = self.rnn2(out2)
        h2 = torch.tanh(self.fc2(out3.contiguous().view(
            -1, self.rnn_dim * self.rnn_directions))).view(
                out3.size(0), out3.size(1), -1)
        if CUDA:
            h2 = h2.to(x.device)
            self.hc2 = self.hc2.to(x.device)
        weights2 = F.softmax((torch.matmul(h2, self.hc2)).view(
            batch_size, -1), dim=1).view(batch_size, 1, -1)
        out4 = torch.matmul(weights2, out3).view(
            batch_size, self.rnn_dim * self.rnn_directions)

        out = self.fc3(out4)
        return out
