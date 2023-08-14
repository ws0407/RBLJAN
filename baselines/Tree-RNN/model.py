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
from torch.autograd.function import Function
from utils import BATCH_SIZE, PKT_MAX_LEN, PADDING_IDX, CUDA, DEVICE


class OneHot(Function):
    byte_mask = torch.zeros((BATCH_SIZE, PKT_MAX_LEN, PADDING_IDX + 1))
    seq = torch.tensor([_ for _ in range(PADDING_IDX + 1)]).float().unsqueeze(1)
    if CUDA:
        byte_mask = byte_mask.cuda(DEVICE)
        seq = seq.cuda(DEVICE)

    @staticmethod
    def forward(ctx, inputs):
        ctx.batch_size = inputs.size(0)
        ctx.input_len = inputs.size(1)
        inputs = inputs.long()
        byte_mask = OneHot.byte_mask.scatter(2, inputs.view(ctx.batch_size, -1, 1).data, 1.)  # one-hot vector
        byte_mask[:, :, PADDING_IDX:].fill_(0)
        # ctx.byte_mask = byte_mask
        return byte_mask

    @staticmethod
    def backward(ctx, grad_output):
        # print('grad_output:', grad_output, grad_output.shape)
        # grad_input = (grad_output * ctx.byte_mask).sum(2)
        grad_input = torch.matmul(grad_output, OneHot.seq).view(grad_output.size(0), -1)
        # print('grad_input:', grad_input, grad_input.shape)
        # ctx.byte_mask = None
        return grad_input


class BaseRNN(nn.Module):
    def __init__(self, num_label=2, hidden_size=64):
        super(BaseRNN, self).__init__()
        self.num_label = num_label
        self.hidden_size = hidden_size
        self.directions = 1
        self.embedding = nn.Embedding(257, self.hidden_size, padding_idx=256)
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=(self.directions == 2))

        self.fc = nn.Linear(self.directions * self.hidden_size, self.num_label)

    # inputs: [bs, len]
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        out, (h_n, c_n) = self.lstm(inputs)   # out: [bs, len, hs], hn/cn: [1, len, hs]

        out = torch.mean(out, dim=1).view(inputs.size(0), -1)   # [bs, hs]

        out = self.fc(out).view(inputs.size(0), -1)     # [bs, 2/3]
        return out


class BaseRNN_Embed(nn.Module):
    def __init__(self, num_label=2, hidden_size=64):
        super(BaseRNN_Embed, self).__init__()
        self.num_label = num_label
        self.hidden_size = hidden_size
        self.directions = 1
        self.byte_one_hot = OneHot().apply
        self.byte_embed = nn.Linear(257, self.hidden_size, bias=False)
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=(self.directions == 2))

        self.fc = nn.Linear(self.directions * self.hidden_size, self.num_label)

    # inputs: [bs, len]
    def forward(self, inputs):
        inputs = self.byte_one_hot(inputs)
        inputs = self.byte_embed(inputs)
        # inputs = self.embedding(inputs)
        out, (h_n, c_n) = self.lstm(inputs)   # out: [bs, len, hs], hn/cn: [1, len, hs]

        out = torch.mean(out, dim=1).view(inputs.size(0), -1)   # [bs, hs]

        out = self.fc(out).view(inputs.size(0), -1)     # [bs, 2/3]
        return out


def evaluate():
    print()



if __name__ == '__main__':

    model = BaseRNN(5, 64)
    x1 = torch.randint(0, 255, (16, 100))
    x2 = torch.randint(0, 255, (16, 50))
    y = model(x1)
    print(y.shape)
    y = model(x2)
    print(y.shape)




#
# class TreeRNN(nn.Module):
#     def __init__(self, num_label, hidden_size, ngram, out_channels):
#         super(TreeRNN, self).__init__()
#         self.num_label = num_label
#         self.hidden_size = hidden_size
#         self.directions = 2
#         self.ngram = ngram
#         self.out_channels = out_channels
#         self.query = nn.Linear(self.hidden_size * self.directions, self.hidden_size)  # hid_size, hid_size
#         self.conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=self.out_channels,
#                 kernel_size=(self.hidden_size, self.ngram),
#                 stride=(1, 1)),
#             nn.BatchNorm2d(self.out_channels),
#             nn.ReLU()
#         )
#         self.lstm = nn.LSTM(input_size=self.num_label,
#                             hidden_size=self.hidden_size,
#                             batch_first=True,
#                             bidirectional=(self.directions == 2))
#
#     def forward(self, similarity_matrix, B, p_e_matrix):
#         # similarity_matrix: [bs, num_label, seq_len]
#         batch_size = similarity_matrix.size(0)
#         p_e_matrix = p_e_matrix[:similarity_matrix.size(2) - self.ngram + 1]
#         if CUDA:
#             p_e_matrix = p_e_matrix.cuda(DEVICE)
#         out1, (h_n, c_n) = self.lstm(similarity_matrix.transpose(1, 2).transpose(0, 1))  # [seq_len, bs, hidden_s]
#
#         Q = self.query(out1.transpose(0, 1))
#         K = self.query(out1.transpose(0, 1))
#         V = self.query(out1.transpose(0, 1))
#         attention_scores = torch.matmul(Q, K.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.hidden_size)
#         attention_scores = nn.Softmax(dim=-1)(attention_scores)
#         context_layer = torch.matmul(attention_scores, V)  # [bs, seq_len, hidden_s]
#         context_layer = self.conv(context_layer.transpose(-1, -2).unsqueeze(
#             1))  # [bs, 1, hidden_s, pkt_len] -> [bs, OUT_CHANNELS, 1, pkt_len]
#         context_layer = context_layer.view(batch_size, self.out_channels, -1)  # [bs, OUT_CHANNELS, pkt_len]
#         context_layer = F.max_pool2d(context_layer, (self.out_channels, 1))  # [bs, 1, pkt_len]
#         context_layer = nn.Softmax(dim=-1)(context_layer)  # [bs, 1, pkt_len]
#         B = F.avg_pool2d(B, kernel_size=(self.ngram, 1), stride=(1, 1))
#         context_layer = torch.matmul(context_layer, (B + p_e_matrix.long()))  # [bs, 1, e]
#         context_layer = context_layer.view(batch_size, -1)  # [bs, e]
#
#         return context_layer
