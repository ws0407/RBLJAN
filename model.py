#!/usr/bin/env python
# -*- coding:utf-8 -*-
from torch.autograd.function import Function
from utils import *


class MyEmbedding(Function):
    byte_idx_cpu = torch.FloatTensor([[i] for i in range(PADDING_IDX + 1)])
    zeros_mask_cpu = torch.zeros(((int(BATCH_SIZE/DEVICE_COUNT) if USE_PARALLEL else BATCH_SIZE) * PKT_MAX_LEN, PADDING_IDX + 1))
    fill_idx_cpu = torch.tensor([PADDING_IDX])
    byte_idx, zeros_mask, fill_idx = {}, {}, {}
    if CUDA:
        for device in ALL_DEVICE:
            byte_idx[device] = byte_idx_cpu.cuda(device)
            zeros_mask[device] = zeros_mask_cpu.cuda(device)
            fill_idx[device] = fill_idx_cpu.cuda(device)

    @staticmethod
    def forward(ctx, inputs, embed_weight):
        inputs = inputs.long()
        ctx.input = inputs
        ctx.embed_weight = embed_weight
        outputs = torch.index_select(embed_weight, 0, inputs.view(-1)).view(inputs.size(0), inputs.size(1), -1)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        # if CUDA and MyEmbedding.byte_idx.device != grad_output.device:
        #     MyEmbedding.byte_idx = MyEmbedding.byte_idx.to(grad_output.device)
        #     MyEmbedding.zeros_mask = MyEmbedding.zeros_mask.to(grad_output.device)
        batch_size = ctx.input.size(0)
        device = '{}'.format(grad_output.device)
        pkt_len = ctx.input.size(1)
        grad_input = torch.matmul(grad_output, ctx.embed_weight.transpose(-1, -2))
        grad_input = torch.matmul(grad_input, MyEmbedding.byte_idx[device]).view(batch_size, -1)
        if batch_size * pkt_len != MyEmbedding.zeros_mask[device].size(0):
            byte_mask = torch.zeros(batch_size * pkt_len, PADDING_IDX + 1).cuda(device)
            byte_mask.scatter_(1, ctx.input.view(-1, 1).data, 1.)
        else:
            byte_mask = MyEmbedding.zeros_mask[device].scatter(1, ctx.input.view(-1, 1).data, 1.)  # one-hot vector
        grad_weight = torch.matmul(byte_mask.view(-1, batch_size * pkt_len), grad_output.view(batch_size * pkt_len, -1))
        grad_weight = torch.index_fill(grad_weight, 0, MyEmbedding.fill_idx[device], 0.)
        # print(grad_input, 'grad_input', grad_input.shape)
        # print(grad_weight, 'grad_weight', grad_weight.shape)
        return grad_input, grad_weight


class OneHot(Function):
    byte_mask_cpu = torch.zeros((int(BATCH_SIZE/DEVICE_COUNT), PAYLOAD_LEN, PADDING_IDX + 1))
    byte_mask = {}
    if CUDA:
        for device in ALL_DEVICE:
            byte_mask[device] = byte_mask_cpu.cuda(device)

    @staticmethod
    def forward(ctx, inputs):
        ctx.batch_size = inputs.size(0)
        ctx.input_len = inputs.size(1)
        device = '{}'.format(inputs.device)
        inputs = inputs.long()
        byte_mask = OneHot.byte_mask[device].scatter(2, inputs.view(ctx.batch_size, -1, 1).data, 1.)  # one-hot vector
        byte_mask[:, :, PADDING_IDX:].fill_(0)
        ctx.byte_mask = byte_mask
        return byte_mask

    @staticmethod
    def backward(ctx, grad_output):
        # print('grad_output:', grad_output, grad_output.shape)
        grad_input = (grad_output * ctx.byte_mask).sum(2)
        # print('grad_input:', grad_input, grad_input.shape)
        return grad_input


class RBLJAN_Classifier(nn.Module):
    def __init__(self, position_embed_matrix, overall_label_idx):
        super(RBLJAN_Classifier, self).__init__()
        self.embedding_dim = EMBEDDING_DIM
        self.num_label = NUM_LABELS
        self.ngram_h = NGRAM_HEADER
        self.ngram_p = NGRAM_PAYLOAD
        self.size_h = HEADER_LEN
        self.size_p = PAYLOAD_LEN
        self.kernel_h = KERNEL_NUM_HEADER
        self.kernel_p = KERNEL_NUM_PAYLOAD
        self.head_h = 5
        self.head_p = 20
        self.dropout = DROPOUT
        self.position_embed_matrix = position_embed_matrix.float()
        self.overall_label_idx = overall_label_idx
        self.padding_idx = PADDING_IDX
        # embedding layers（random initialize）   B', L
        self.header_embed = nn.Embedding(257, self.embedding_dim, padding_idx=self.padding_idx)  # 256 is 'gg', will be set [0,0..0]
        self.payload_one_hot = OneHot().apply
        self.payload_embed = nn.Linear(257, self.embedding_dim, bias=False)

        self.label_h_embed = nn.Embedding(self.num_label, self.embedding_dim)
        self.label_p_embed = nn.Embedding(self.num_label, self.embedding_dim)
        self.header_embed.requires_grad = True
        # self.payload_embed.requires_grad = True
        self.label_h_embed.requires_grad = True
        self.label_p_embed.requires_grad = True
        # self.paddings = True

        self.att_b_h = ATT_BYTE(self.num_label, self.ngram_h, self.kernel_h)
        self.att_b_p = ATT_BYTE(self.num_label, self.ngram_p, self.kernel_p)

        self.att_l_h = ATT_LABEL(self.size_h, self.head_h)
        self.att_l_p = ATT_LABEL(self.size_p, self.head_p)

        self.fc = nn.Sequential(
            nn.Linear(4 * self.embedding_dim, self.num_label),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        if x.device != self.overall_label_idx.device:
            self.overall_label_idx = self.overall_label_idx.to(x.device)
            self.position_embed_matrix = self.position_embed_matrix.to(x.device)
        h = x.transpose(1, 0)[:self.size_h].transpose(1, 0).long()
        x = x.transpose(1, 0)[PKT_MAX_LEN - self.size_p:].transpose(1, 0)

        H = self.header_embed(h) + self.position_embed_matrix[:self.size_h]
        LH = self.label_h_embed(self.overall_label_idx)
        GH = get_cosine_similarity_matrix(H, LH)  # b * k * len
        att_b_h = self.att_b_h(GH, H)
        att_l_h = self.att_l_h(GH, LH)

        P = self.payload_embed(self.payload_one_hot(x)) + self.position_embed_matrix[:self.size_p]  # b * l * e (batch_size, len_byte, embedding)
        # print(self.payload_embed_weight)
        LP = self.label_p_embed(self.overall_label_idx)  # k * e (k is num_label)
        GP = get_cosine_similarity_matrix(P, LP)  # b * k * l
        att_b_p = self.att_b_p(GP, P)
        att_l_p = self.att_l_p(GP, LP)
        out1 = self.fc(torch.cat((att_b_h, att_b_p, att_l_h, att_l_p), dim=1))  # b * k, for loss
        out2 = self.fc(torch.cat((LH, LP, LH, LP), dim=1))  # k * k, for regularization

        if self.training:
            return out1, out2
        else:
            return out1, (att_b_h, att_l_h, att_b_p, att_l_p)


class RBLJAN_Classifier_FLOW(nn.Module):
    def __init__(self, position_embed_matrix, overall_label_idx):
        super(RBLJAN_Classifier_FLOW, self).__init__()
        self.embedding_dim = EMBEDDING_DIM
        self.num_label = NUM_LABELS
        self.num_pkt = FLOW_MAX_LEN
        self.ngram_h = NGRAM_HEADER
        self.ngram_p = NGRAM_PAYLOAD
        self.size_h = HEADER_LEN
        self.size_p = PAYLOAD_LEN
        self.kernel_h = KERNEL_NUM_HEADER
        self.kernel_p = KERNEL_NUM_PAYLOAD
        self.dropout = DROPOUT
        self.padding_idx = PADDING_IDX
        self.position_embed_matrix = position_embed_matrix.float()
        self.overall_label_idx = overall_label_idx
        # embedding layers（random initialize）   B', L
        self.header_embed = nn.Embedding(257, self.embedding_dim, padding_idx=self.padding_idx)  # 256 is 'gg', will be set [0,0..0]
        self.payload_embed = nn.Embedding(257, self.embedding_dim, padding_idx=self.padding_idx)
        self.label_h_embed = nn.Embedding(self.num_label, self.embedding_dim)
        self.label_p_embed = nn.Embedding(self.num_label, self.embedding_dim)
        self.header_embed.requires_grad = True
        self.payload_embed.requires_grad = True
        self.label_h_embed.requires_grad = True
        self.label_p_embed.requires_grad = True
        # self.paddings = True

        self.att_b_h = ATT_BYTE(self.num_label * self.num_pkt, self.ngram_h, self.kernel_h)
        self.att_b_p = ATT_BYTE(self.num_label * self.num_pkt, self.ngram_p, self.kernel_p)

        self.att_l_h = ATT_LABEL_FLOW(self.size_h, self.num_label, self.num_pkt)
        self.att_l_p = ATT_LABEL_FLOW(self.size_p, self.num_label, self.num_pkt)

        self.fc = nn.Sequential(
            nn.Linear(4 * self.embedding_dim, self.num_label),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        batch_size = x.size(0)
        if x.device != self.overall_label_idx.device:
            self.overall_label_idx = self.overall_label_idx.to(x.device)
            self.position_embed_matrix = self.position_embed_matrix.to(x.device)
        h = x[:, :, : self.size_h].view(-1, self.size_h)
        x = x[:, :, PKT_MAX_LEN-self.size_p:].view(-1, self.size_p)

        H = self.header_embed(h) + self.position_embed_matrix[:self.size_h]
        LH = self.label_h_embed(self.overall_label_idx)
        GH = get_cosine_similarity_matrix(H, LH).view(batch_size, -1, self.size_h)  # b * k * len
        att_b_h = self.att_b_h(GH)
        H = torch.matmul(att_b_h.unsqueeze(1), torch.sum(H.view(batch_size, -1, self.size_h, self.embedding_dim), dim=1)).view(batch_size, -1)
        att_l_h = self.att_l_h(GH)
        GH = torch.matmul(att_l_h, LH).view(batch_size, -1)

        P = self.payload_embed(x) + self.position_embed_matrix[:self.size_p]  # b * l * e (batch_size, len_byte, embedding)
        # print(self.payload_embed_weight)

        LP = self.label_p_embed(self.overall_label_idx)  # k * e (k is num_label)
        GP = get_cosine_similarity_matrix(P, LP).view(batch_size, -1, self.size_p)  # b * k * l

        att_b_p = self.att_b_p(GP)
        # P = F.avg_pool2d(P, kernel_size=(self.ngram_p, 1), stride=(1, 1))
        P = torch.matmul(att_b_p.unsqueeze(1), torch.sum(P.view(batch_size, -1, self.size_p, self.embedding_dim), dim=1)).view(batch_size, -1)

        att_l_p = self.att_l_p(GP)
        GP = torch.matmul(att_l_p, LP).view(batch_size, -1)
        out1 = self.fc(torch.cat((H, P, GH, GP), dim=1))  # b * k, for loss
        out2 = self.fc(torch.cat((LH, LP, LH, LP), dim=1))  # k * k, for regularization

        if self.training:
            return out1, out2
        else:
            return out1, (att_b_h, att_l_h, att_b_p, att_l_p)


class ATT_BYTE(nn.Module):
    def __init__(self, num_label, ngram=51, out_channels=8):
        super(ATT_BYTE, self).__init__()
        self.num_label = num_label
        self.hidden_size = self.num_label
        self.out_channels = out_channels
        self.ngram = ngram
        # self.num_padding = int(self.ngram / 2)
        self.conv = nn.Sequential(  # 卷积函数
            nn.Conv2d(
                in_channels=1,
                out_channels=self.out_channels,
                kernel_size=(self.ngram, self.hidden_size),
                stride=(1, 1),
                # padding=(self.num_padding, 0)
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, similarity_matrix, B):
        batch_size = similarity_matrix.size(0)
        similarity_matrix = similarity_matrix.transpose(-1, -2)
        att_byte = self.conv(similarity_matrix.unsqueeze(1)).view(batch_size, self.out_channels, -1)    # [bs, oc, len-ngram+1]
        att_byte = F.max_pool2d(att_byte, (self.out_channels, 1))
        att_byte = F.softmax(att_byte.view(batch_size, -1), dim=1)
        B = F.avg_pool2d(B, kernel_size=(self.ngram, 1), stride=(1, 1))
        att_byte = torch.matmul(att_byte.unsqueeze(1), B).view(batch_size, -1)

        return att_byte  # [bs, e]


class ATT_LABEL(nn.Module):
    def __init__(self, pkt_len, num_head):
        super(ATT_LABEL, self).__init__()
        self.pkt_len = pkt_len
        self.num_head = num_head
        self.head_size = self.pkt_len // self.num_head
        if self.pkt_len % self.num_head != 0:  # 整除
            raise ValueError("The head size (%d/%d) is not a multiple of the packet length heads (%d)"
                             % (pkt_len, num_head, self.head_size))
        self.fc1 = nn.Sequential(
            nn.Linear(self.head_size, 1),
            nn.LeakyReLU(0.01)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.num_head, 1),
            nn.LeakyReLU(0.01)
        )

    def forward(self, similarity_matrix, L):
        batch_size, num_label, _ = similarity_matrix.size()
        att_label = self.fc1(similarity_matrix.view(batch_size, num_label, self.num_head, self.head_size)).squeeze(-1)
        att_label = self.fc2(att_label)
        att_label = F.softmax(att_label.view(batch_size, -1), dim=1).unsqueeze(1)
        att_label = torch.matmul(att_label, L).view(batch_size, -1)
        return att_label


class ATT_LABEL_FLOW(nn.Module):
    def __init__(self, pkt_len, num_label, num_pkt, dropout_prob=0.5):
        super(ATT_LABEL_FLOW, self).__init__()
        self.pkt_len = pkt_len
        self.num_label = num_label
        self.num_pkt = num_pkt
        self.fc1 = nn.Sequential(
            nn.Linear(self.pkt_len, 1),
            nn.LeakyReLU(0.02)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.num_label * self.num_pkt, self.num_label),
            nn.LeakyReLU(0.02)
        )

    def forward(self, similarity_matrix):
        batch_size = similarity_matrix.size(0)

        att_label = self.fc1(similarity_matrix).view(batch_size, -1)
        att_label = self.fc2(att_label).view(batch_size, -1)

        att_label = F.softmax(att_label, dim=1).unsqueeze(1)
        return att_label


class RBLJAN_GAN(nn.Module):
    def __init__(self, position_embed_matrix, overall_label_idx):
        super(RBLJAN_GAN, self).__init__()

        self.embedding_dim = EMBEDDING_DIM
        self.num_label = NUM_LABELS
        self.ngram_h = NGRAM_HEADER
        self.ngram_p = NGRAM_PAYLOAD
        self.size_h = HEADER_LEN
        self.size_p = PAYLOAD_LEN
        self.kernel_h = KERNEL_NUM_HEADER
        self.kernel_p = KERNEL_NUM_PAYLOAD
        self.head_h = 5
        self.head_p = 20
        self.dropout = DROPOUT
        self.padding_idx = PADDING_IDX

        self.position_embed_matrix = position_embed_matrix.float()
        self.overall_label_idx = overall_label_idx

        # embedding layers（random initialize）   B', L
        self.header_embed = nn.Embedding(257, self.embedding_dim, padding_idx=self.padding_idx)
        self.payload_embed = nn.Embedding(257, self.embedding_dim, padding_idx=self.padding_idx)  # 256 is 'gg', will be set [0,0..0]
        self.label_h_embed = nn.Embedding(self.num_label, self.embedding_dim)
        self.label_p_embed = nn.Embedding(self.num_label, self.embedding_dim)
        self.header_embed.requires_grad = True
        self.payload_embed.requires_grad = True
        self.label_h_embed.requires_grad = True
        self.label_p_embed.requires_grad = True

        self.att_b_h = ATT_BYTE(self.num_label, self.ngram_h, self.kernel_h)
        self.att_b_p = ATT_BYTE(self.num_label, self.ngram_p, self.kernel_p)

        self.att_l_h = ATT_LABEL(self.size_h, self.head_h)
        self.att_l_p = ATT_LABEL(self.size_p, self.head_p)

        self.fc = nn.Sequential(
            nn.Linear(4 * self.embedding_dim, self.num_label),
            # nn.Linear(self.embedding_dim, self.num_label),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        if x.device != self.overall_label_idx.device:
            self.overall_label_idx = self.overall_label_idx.to(x.device)
            self.position_embed_matrix = self.position_embed_matrix.to(x.device)
        h = x.transpose(1, 0)[:self.size_h].transpose(1, 0)
        x = x.transpose(1, 0)[PKT_MAX_LEN-self.size_p:].transpose(1, 0)

        H = self.header_embed(h) + self.position_embed_matrix[:self.size_h]
        LH = self.label_h_embed(self.overall_label_idx)
        GH = get_cosine_similarity_matrix(H, LH)    # b * k * len
        att_b_h = self.att_b_h(GH, H)
        att_l_h = self.att_l_h(GH, LH)

        P = self.payload_embed(x) + self.position_embed_matrix[:self.size_p]  # b * l * e (batch_size, len_byte, embedding)
        LP = self.label_p_embed(self.overall_label_idx)  # k * e (k is num_label)
        GP = get_cosine_similarity_matrix(P, LP)  # b * k * l
        att_b_p = self.att_b_p(GP, P)
        att_l_p = self.att_l_p(GP, LP)
        out1 = self.fc(torch.cat((att_b_h, att_b_p, att_l_h, att_l_p), dim=1))  # b * k, for loss
        out2 = self.fc(torch.cat((LH, LP, LH, LP), dim=1))  # k * k, for regularization
        return out1, out2


class Generator_RNN(nn.Module):
    def __init__(self, input_size, inject_pos, inject_num, hidden_units=128):
        super(Generator_RNN, self).__init__()
        self.input_size = input_size
        self.inject_pos = inject_pos
        self.inject_num = inject_num
        self.hidden_dim = hidden_units
        self.segment_len = 25

        self.fc1 = nn.Linear(self.input_size, self.inject_num)
        self.fc2 = nn.Linear(2 * self.segment_len, self.hidden_dim)
        self.rnn = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.segment_len),
            # nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, x_inject, X, states):
        batch_size = x_inject.size(0)
        z = self.fc1(x_inject).view(self.inject_num, -1).view(self.inject_num // self.segment_len, batch_size,
                                                              self.segment_len)  # 25 * batch * 20
        prev_gen = torch.empty([batch_size, self.segment_len]).uniform_().to(x_inject.device)
        gen_bytes = []
        for byte in z:
            # concatenate current input features and previous timestep output features
            concat_in = torch.cat((byte, prev_gen), dim=-1)
            out = F.relu(self.fc2(concat_in))
            h1, c1 = self.rnn(out, states)
            h1 = self.dropout(h1)  # feature dropout only (no recurrent dropout)
            prev_gen = self.fc3(h1)
            gen_bytes.append(prev_gen)
            states = (h1, c1)
        # seq_len * (batch_size * num_feats) -> (batch_size * seq_len * num_feats)
        gen_bytes = torch.stack(gen_bytes, dim=1).view(batch_size, -1)

        random_inject_num = random.choice([i * 50 for i in range(1, 11)])
        gen_bytes = gen_bytes[:, :random_inject_num] * 256.
        gen_bytes = torch.cat((X[:, :self.inject_pos], gen_bytes, X[:, self.inject_pos:]), dim=1)[:, :PKT_MAX_LEN]
        return gen_bytes

    def init_state(self, batch_size, device):
        # generate (h1, c1)
        weight = next(self.parameters()).data
        states = (weight.new(batch_size, self.hidden_dim).zero_().cuda(device),
                  weight.new(batch_size, self.hidden_dim).zero_().cuda(device))
        return states


class Generator_MLP(nn.Module):
    def __init__(self, input_size, inject_pos, inject_num):
        super(Generator_MLP, self).__init__()
        self.input_size = input_size
        self.inject_pos = inject_pos
        self.inject_num = inject_num
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.inject_num * 2),
            nn.Dropout(0.05),
            nn.Linear(self.inject_num * 2, int(self.inject_num * 1.5)),
            nn.Dropout(0.05),
            nn.Linear(int(self.inject_num * 1.5), self.inject_num),
            nn.Dropout(0.05),
            # nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, x_inject, X):
        res = self.fc(x_inject)
        random_inject_num = random.choice([i * 50 for i in range(1, 11)])
        res = res[:, :random_inject_num] * 256.
        res = torch.cat((X[:, :self.inject_pos], res, X[:, self.inject_pos:]), dim=1)[:, :PKT_MAX_LEN]
        return res


class Generator_CNN(nn.Module):
    def __init__(self, input_size, inject_pos, inject_num, ngram=17):
        super(Generator_CNN, self).__init__()
        self.ngram = ngram
        self.kernel_size = (ngram, 8)
        self.stride = (1, 1)
        self.channels = 8
        self.input_size = input_size
        self.inject_pos = inject_pos
        self.inject_num = inject_num
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, self.inject_num * self.channels),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.channels,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=(int(self.ngram / 2), 0)
            ),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.inject_num * self.kernel_size[1], self.inject_num),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x_inject, X):
        batch_size = x_inject.size(0)
        res = self.fc1(x_inject).view(batch_size, self.channels, self.inject_num, 1)
        res = self.deconv(res).view(batch_size, 1, 1, -1)
        res = self.fc2(res).view(batch_size, self.inject_num)
        random_inject_num = random.choice([i * 50 for i in range(1, 16)])
        res = res[:, :random_inject_num] * 256.
        res = torch.cat((X[:, :self.inject_pos], res, X[:, self.inject_pos:]), dim=1)[:, :PKT_MAX_LEN]
        return res