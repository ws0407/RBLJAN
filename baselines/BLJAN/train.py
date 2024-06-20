#!/usr/bin/env python
# -*- coding:utf-8 -*-

from utils import *
from model import *


def train_BLJAN(is_save=False):
    with open('./tmp_variable/p_e_matrix.variable', 'rb') as f1:
        p_e_matrix = pickle.load(f1)[:PKT_MAX_LEN]

    train_loader, test_loader = get_dataloader(EXP_PERCENT, TEST_PERCENT, BATCH_SIZE)
    train_loader.load_all_batches()
    test_loader.load_all_batches()

    overall_label_idx = (torch.arange(0, NUM_LABELS)).long()
    if CUDA:
        overall_label_idx = overall_label_idx.cuda(DEVICE)
        p_e_matrix = p_e_matrix.cuda(DEVICE)
    model = BLJAN(EMBEDDING_DIM, NUM_LABELS, overall_label_idx, NGRAM, DROPOUT, p_e_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if CUDA:
        model = model.cuda(DEVICE)

    loss_func = FocalLoss(NUM_LABELS, alpha=train_loader.alpha, gamma=GAMMA, size_average=True)
    num_batch = train_loader.num_batch
    cur_epoch, cur_loss = 0, 0
    best_acc, best_f1 = 0, 0
    print('data prepare ok! total {} batches for {} epochs'.format(num_batch, EPOCHS - cur_epoch))

    loss, epoch = cur_loss, cur_epoch
    _s_t = time.time()
    all_f1, all_test_acc = [], []
    # training
    for epoch in range(cur_epoch + 1, EPOCHS):
        train_total_loss = 0
        print('\nepoch: {} start to train...'.format(epoch))
        for i in range(num_batch):
            batch_X, batch_y = train_loader.data[i]
            # batch_X, batch_y = train_loader.load_batch_idx(i)
            if CUDA:
                batch_X = batch_X.cuda(DEVICE)
                batch_y = batch_y.cuda(DEVICE)
            # start to train
            model.train()
            optimizer.zero_grad()
            out1, out2 = model(batch_X, overall_label_idx)
            loss1 = loss_func(out1, batch_y.long())
            loss2 = F.cross_entropy(out2, overall_label_idx)
            loss = loss1 + LAMBDA * loss2
            train_total_loss += float(loss.item())
            loss.backward()
            optimizer.step()

            if i % 2000 == 0:
                print('\r>> >> batch {} ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
            # print('>> >> batch {} ok with {}s'.format(i, time.time() - _s_t))
        print('\n>> All batch {} ok with {}s'.format(num_batch, time.time() - _s_t))
        eval_s_t = time.time()
        print('>> >> start evaluate loss accuracy with {} batches'.format(test_loader.num_batch))
        test_accuracy, test_c_matrix = evaluate_loss_acc(model, test_loader, True, NUM_LABELS, overall_label_idx)
        print('evaluate done with {}s'.format(time.time() - eval_s_t))
        if test_accuracy > best_acc:
            print('>> test accuracy increased !!!')
            best_acc = test_accuracy
        all_test_acc.append(test_accuracy)
        P, R, f1 = deal_matrix(test_c_matrix)
        all_f1.append(np.array(f1).mean())
        if all_f1[-1] > best_f1:
            best_f1 = all_f1[-1]
            print('>> test f1-score increased !!!')
            if is_save:
                save_model(model, epoch, optimizer, loss, best_acc, best_f1, full=True)
        print('>> epoch: {} ok:\n'
              '>> >> train avg loss: {},\n'
              '>> >> test accuracy: {}, best test accuracy: {},\n'
              '>> >> test f1-score: {}, best f1-score: {}'.format(
            epoch, train_total_loss / num_batch, test_accuracy, best_acc, all_f1[-1], best_f1))
        print('>> detailed test results of every label:\nlabel\tPrecision\tRecall\tF1-score')
        for key in LABELS:
            print('{}\t{}\t{}\t{}'.format(key, P[int(LABELS[key])], R[int(LABELS[key])], f1[int(LABELS[key])]))
        if epoch > 30 and max(all_f1[-5:]) < best_f1 and max(all_test_acc[-5:]) < best_acc:
            print('early stop with epoch: {}, \nbest test accuracy: {}, best f1-score: {}'.format(
                epoch, best_acc, best_f1))
            break

