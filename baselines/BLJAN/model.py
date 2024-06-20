#!/usr/bin/env python
# -*- coding:utf-8 -*-
from utils import *


class BLJAN(nn.Module):
    def __init__(self, embedding_dim, num_label, overall_label_idx, ngram, dropout, position_embed_matrix):
        super(BLJAN, self).__init__()

        # attributes set
        self.embedding_dim = embedding_dim
        self.num_label = num_label
        self.ngram = ngram
        self.dropout = dropout
        self.overall_label_idx = overall_label_idx
        self.position_embed_matrix = position_embed_matrix

        # embedding layers（random initialize）   B', L
        self.byte_embed = nn.Embedding(257, self.embedding_dim, padding_idx=256)  # 256 is 'gg', will be set [0,0..0]
        self.label_embed = nn.Embedding(self.num_label, self.embedding_dim)
        self.byte_embed.requires_grad = True
        self.label_embed.requires_grad = True

        # self.self_attention = SelfAttentionAfterConv(OUT_CHANNELS, self.ngram)

        # main layers 模型参数的初始化
        self.conv1 = nn.Sequential(  # 卷积函数
            nn.Conv2d(
                in_channels=1,  # 输入数据即Similarity(Byte-Label)，通道数为1
                out_channels=OUT_CHANNELS,  # 输出数据深度为 4 * num_label可根据模型调整
                kernel_size=(self.num_label, self.ngram),  # 卷积核大小，第一维大小为num_label，第二维大小为ngram
                stride=(self.num_label, 1)),  # 卷积每次滑动的步长，上下为num_label，左右为1

            nn.BatchNorm2d(OUT_CHANNELS),  # 数据的归一化处理，避免Relu因数据过大而导致网络性能的不稳定
            nn.ReLU()  # 激活函数
        )

        self.fc = nn.Sequential(  # 全连接函数fc为线性函数：z = Wh + q，将2*E个节点连接到num_label个节点上
            nn.Linear(2 * self.embedding_dim, self.num_label),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        # x = torch.zeros(x.size()).long()
        batch_size = x.size(0)  # row num
        maxlen = x.size(1)  # col num

        # through embedding layers
        B = self.byte_embed(x)  # b * l * e (batch_size, len_byte, embedding)
        L = self.label_embed(self.overall_label_idx)  # k * e (k is num_label)

        # get G_norm, G, G_hat
        G_hat = get_cosine_similarity_matrix(B, L)  # b * k * l

        # for label attention   (9)(10)
        label_attm = F.softmax(torch.mean(G_hat, dim=2), dim=1)  # b * k
        label_v = torch.matmul(label_attm, L)  # b * e           # (11)

        # through conv layer and maxpool    relu：non-linear activation function
        G_hat = G_hat.unsqueeze(1)  # b * 1 * k * l
        att_m = self.conv1(G_hat)  # b * N * 1 * (l-ngram+1)
        att_m = att_m.view(batch_size, OUT_CHANNELS, -1)  # b * N * l-ngram+1
        att_m = F.relu(F.max_pool2d(att_m, (OUT_CHANNELS, 1)))  # b * 1 * (l-ngram+1) # (5)(6)
        # att_m = self.self_attention(att_m)      # b * l-ngram+1 * 1
        att_m = att_m.view(batch_size, -1)  # b * l-ngram+1

        # customized softmax for every packet
        att_m = F.softmax(att_m, dim=1)  # b * l-ngram+1

        # get every weighted embedding
        att_m = att_m.unsqueeze(1)  # b * 1 * l - ngram + 1
        # V turn to the type of ngram
        V_ngram = F.avg_pool2d(B, kernel_size=(self.ngram, 1), stride=(1, 1))  # b * l-ngram+1 * e
        pe_matrix = self.position_embed_matrix[:maxlen - self.ngram + 1]  # l-ngram+1 * e
        V_ngram += pe_matrix  # (3)

        Z = torch.matmul(att_m, V_ngram)  # b * 1 * e
        Z = Z.view(batch_size, -1)  # b * e

        # through fully connected layer
        out1 = self.fc(torch.cat((Z, label_v), dim=1))  # b * k, for loss
        out2 = self.fc(torch.cat((L, L), dim=1))  # k * k, for regularization

        return out1, out2
