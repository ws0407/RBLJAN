#!/usr/bin/env python
# -*- coding:utf-8 -*-

from utils import *
from model import *
from torch.autograd import Variable


def train_RBLJAN_GAN(is_save=True, is_load=False, is_shuffle=False):
    if EMBEDDING_DIM == 256:
        with open('./tmp_variable/p_e_matrix_256.variable', 'rb') as f1:
            p_e_matrix = pickle.load(f1)[:PKT_MAX_LEN + NGRAM_PAYLOAD]
    else:
        with open('./tmp_variable/p_e_matrix.variable', 'rb') as f1:
            p_e_matrix = pickle.load(f1)[:PKT_MAX_LEN + NGRAM_PAYLOAD]

    overall_label_idx = (torch.arange(0, NUM_LABELS)).long()
    if CUDA:
        overall_label_idx = overall_label_idx.cuda(DEVICE)
        p_e_matrix = p_e_matrix.cuda(DEVICE)

    # model = BLJAN(EMBEDDING_DIM, NUM_LABELS, NGRAM, DROPOUT, p_e_matrix)
    # model = EBLJAN_SA_CNN_Discriminator(EMBEDDING_DIM, NUM_LABELS, p_e_matrix, overall_label_idx, NGRAM_HEADER, NGRAM_PAYLOAD, DROPOUT)
    model = RBLJAN_GAN(EMBEDDING_DIM, NUM_LABELS, p_e_matrix, overall_label_idx, NGRAM_HEADER, NGRAM_PAYLOAD, DROPOUT)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if CUDA:
        model = model.cuda(DEVICE)
    if DEVICE_COUNT > 1:
        gpus_free = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').readlines()
        gpus_free = [int(_.split()[2]) for _ in gpus_free]
        gpus_idx = []
        for _ in range(len(gpus_free)):
            if gpus_free[_] > 3000:
                gpus_idx.append(_)
        print("Use", len(gpus_idx), 'gpus:', gpus_idx)
        model = nn.DataParallel(model, device_ids=gpus_idx)

    train_loader, test_loader = get_dataloader(EXP_PERCENT, TEST_PERCENT, BATCH_SIZE)
    train_loader.load_all_batches()
    gc.collect()
    test_loader.load_all_batches()
    gc.collect()

    num_batch = train_loader.num_batch
    loss_func1 = FocalLoss(NUM_LABELS, alpha=(train_loader.alpha if num_batch > 2000 else None), gamma=GAMMA,
                           size_average=True)

    cur_epoch, cur_loss = 0, 0
    best_acc, best_f1 = 0, 0
    if is_load:
        checkpoint = torch.load('./model/20220114-092844_train_27.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
        cur_loss = checkpoint['loss']
        best_acc = checkpoint['best_accuracy']
        best_f1 = checkpoint['best_f1']
        print('data prepare ok! load model from local, cur epoch: {}, cur_loss: {}'.format(cur_epoch, cur_loss))
    print('data prepare ok! total {} batches for {} epochs'.format(num_batch, EPOCHS - cur_epoch))

    loss, epoch = cur_loss, cur_epoch
    _s_t = time.time()
    all_f1, all_test_acc = [], []
    # training
    for epoch in range(cur_epoch + 1, EPOCHS):
        if is_shuffle:
            if epoch != 0 and epoch % 10 == 0:
                print('\n>> shuffle and reload data...')
                train_loader.reload_all_batches()
        train_total_loss = 0
        print('\nepoch: {} start to train...'.format(epoch))
        # random.shuffle(train_loader.data)
        y_hat, y = [], []
        for i in range(num_batch):
            batch_X, batch_y = train_loader.data[i]
            # batch_X, batch_y = train_loader.load_batch_idx(i)
            if CUDA:
                batch_X = batch_X.cuda(DEVICE)
                batch_y = batch_y.cuda(DEVICE)
            # start to train
            model.train()
            optimizer.zero_grad()
            out1, out2 = model(batch_X)
            loss1 = loss_func1(out1, batch_y.long())
            num_gpus = int(out2.size(0) / NUM_LABELS)
            loss2 = F.cross_entropy(out2, torch.tensor(overall_label_idx.tolist() * num_gpus).long().cuda(DEVICE))
            loss = loss1 + LAMBDA * loss2
            train_total_loss += float(loss.item())
            # y += batch_y.tolist()
            # y_hat += out1.max(1)[1].tolist()
            loss.backward()
            optimizer.step()
            if i % 2000 == 0:
                print(
                    '\r>> >>[{}] batch {} ok with {}s...'.format(datetime.datetime.now().strftime('%m:%d-%H:%M:%S'), i,
                                                                 time.time() - _s_t), end='', flush=True)
                # print('>> >> batch {} ok with {}s'.format(i, time.time() - _s_t))
        # y = np.array(y)
        # y_hat = np.array(y_hat)
        # train_accuracy = accuracy_score(y, y_hat)
        print('\n>>[{}] All batch {} ok with {}s'.format(datetime.datetime.now().strftime('%m:%d-%H:%M:%S'), num_batch,
                                                         time.time() - _s_t))
        eval_s_t = time.time()
        print('>> >> start evaluate model with {} batches'.format(test_loader.num_batch))
        test_accuracy, test_c_matrix = evaluate_loss_acc(model, test_loader, True, NUM_LABELS)
        print('evaluate done with {}s'.format(time.time() - eval_s_t))
        if test_accuracy > best_acc:
            print('>> test accuracy increased !!!')
            best_acc = test_accuracy
            # if is_save:
            #     plot_confusion_matrix(test_c_matrix, CLASSES, 'c_m_malware' + str(epoch) + '.png')
        all_test_acc.append(test_accuracy)
        P, R, f1 = deal_cm(test_c_matrix)
        all_f1.append(np.array(f1).mean())
        if all_f1[-1] > best_f1:
            best_f1 = all_f1[-1]
            print('>> test f1-score increased !!!')
            if is_save:
                save_model(model, epoch, optimizer, loss, best_acc, best_f1, full=True)
        print('>> epoch: {} ok:\n'
              '>> >> train avg loss: {}\n'
              '>> >> test accuracy: {}, best test accuracy: {},\n'
              '>> >> test f1-score: {}, best f1-score: {}'.format(
            epoch, train_total_loss / num_batch, test_accuracy, best_acc, all_f1[-1], best_f1))
        print('>> detailed test results of every label:\nlabel\tPrecision\tRecall\tF1-score')
        for key in LABELS:
            print('{}\t{}\t{}\t{}\t{}'.format(LABELS[key], key, P[int(LABELS[key])], R[int(LABELS[key])],
                                              f1[int(LABELS[key])]))
        for i in range(len(test_c_matrix)):
            print((' ' + str(i)) if i < 10 else str(i), end='')
            for j in range(len(test_c_matrix[i])):
                num = str(test_c_matrix[i][j])
                num = ' ' * (6 - len(num)) + num
                print(num, end='')
            print()
        # print(test_c_matrix)
        if epoch > 100 and all_f1[-1] < best_f1 and all_f1[-2] < best_f1 and all_f1[-3] < best_f1 and all_f1[
            -4] < best_f1 and all_f1[-5] < best_f1 and all_test_acc[-1] < best_acc and all_test_acc[-2] < best_acc \
                and all_test_acc[-3] < best_acc and all_test_acc[-4] < best_acc and all_test_acc[-5] < best_acc:
            print('early stop with epoch: {}, \nbest test accuracy: {}, best f1-score: {}'.format(
                epoch, best_acc, best_f1))
            break


def train_RBLJAN_FLOW(is_save=True, is_load=False, is_shuffle=False):
    if EMBEDDING_DIM == 256:
        with open('./tmp_variable/p_e_matrix_256.variable', 'rb') as f1:
            p_e_matrix = pickle.load(f1)[:PKT_MAX_LEN + NGRAM_PAYLOAD]
    else:
        with open('./tmp_variable/p_e_matrix.variable', 'rb') as f1:
            p_e_matrix = pickle.load(f1)[:PKT_MAX_LEN + NGRAM_PAYLOAD]

    overall_label_idx = (torch.arange(0, NUM_LABELS)).long()
    if CUDA:
        overall_label_idx = overall_label_idx.cuda(DEVICE)
        p_e_matrix = p_e_matrix.cuda(DEVICE)

    # model = BLJAN(EMBEDDING_DIM, NUM_LABELS, NGRAM, DROPOUT, p_e_matrix)
    model = RBLJAN_Classifier_FLOW(EMBEDDING_DIM, NUM_LABELS, FLOW_MAX_LEN, p_e_matrix, overall_label_idx, NGRAM_HEADER,
                                   NGRAM_PAYLOAD, DROPOUT)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if CUDA:
        model = model.cuda(DEVICE)
    # if DEVICE_COUNT > 1:
    #     gpus_free = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').readlines()
    #     gpus_free = [int(_.split()[2]) for _ in gpus_free]
    #     gpus_idx = []
    #     for _ in range(len(gpus_free)):
    #         if gpus_free[_] > 3000:
    #             gpus_idx.append(_)
    #     print("Use", len(gpus_idx), 'gpus:', gpus_idx)
    #     model = nn.DataParallel(model, device_ids=gpus_idx)

    train_loader, test_loader = get_flow_dataloader("./datasets/USTC-TFC-2016_result_doc_flow/", 32)
    train_loader.load_all_batches()
    test_loader.load_all_batches()

    num_batch = train_loader.num_batch
    loss_func1 = FocalLoss(NUM_LABELS, alpha=(train_loader.alpha if num_batch > 1000 else None), gamma=GAMMA,
                           size_average=True)

    cur_epoch, cur_loss = 0, 0
    best_acc, best_f1 = 0, 0
    if is_load:
        checkpoint = torch.load('./model/20220114-092844_train_27.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
        cur_loss = checkpoint['loss']
        best_acc = checkpoint['best_accuracy']
        best_f1 = checkpoint['best_f1']
        print('data prepare ok! load model from local, cur epoch: {}, cur_loss: {}'.format(cur_epoch, cur_loss))
    print('data prepare ok! total {} batches for {} epochs'.format(num_batch, EPOCHS - cur_epoch))

    loss, epoch = cur_loss, cur_epoch
    _s_t = time.time()
    all_f1, all_test_acc = [], []
    # training
    for epoch in range(cur_epoch + 1, EPOCHS):
        if is_shuffle:
            if epoch != 0 and epoch % 10 == 0:
                print('\n>> shuffle and reload data...')
                train_loader.reload_all_batches()
        train_total_loss = 0
        print('\nepoch: {} start to train...'.format(epoch))
        # random.shuffle(train_loader.data)
        y_hat, y = [], []
        for i in range(num_batch):
            batch_X, batch_y = train_loader.data[i]
            # batch_X, batch_y = train_loader.load_batch_idx(i)
            if CUDA:
                batch_X = batch_X.cuda(DEVICE)
                batch_y = batch_y.cuda(DEVICE)
            # start to train
            model.train()
            optimizer.zero_grad()
            s = time.time()
            out1, out2 = model(batch_X)
            loss1 = loss_func1(out1, batch_y.long())
            num_gpus = int(out2.size(0) / NUM_LABELS)
            loss2 = F.cross_entropy(out2, torch.tensor(overall_label_idx.tolist() * num_gpus).long().cuda(DEVICE))
            loss = loss1 + LAMBDA * loss2
            train_total_loss += float(loss.item())
            # y += batch_y.tolist()
            # y_hat += out1.max(1)[1].tolist()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print('\r>> >>[{}] batch {} ok with {}s...'.format(
                    datetime.datetime.now().strftime('%m:%d-%H:%M:%S'), i, time.time() - _s_t), end='', flush=True)
        # y = np.array(y)
        # y_hat = np.array(y_hat)
        # train_accuracy = accuracy_score(y, y_hat)
        print('\n>>[{}] All batch {} ok with {}s'.format(
            datetime.datetime.now().strftime('%m:%d-%H:%M:%S'), num_batch, time.time() - _s_t))
        eval_s_t = time.time()
        print('>> >> start evaluate model with {} batches'.format(test_loader.num_batch))
        test_accuracy, test_c_matrix = evaluate_loss_acc(model, test_loader, True, NUM_LABELS)
        print('evaluate done with {}s'.format(time.time() - eval_s_t))
        if test_accuracy > best_acc:
            print('>> test accuracy increased !!!')
            best_acc = test_accuracy
            # if is_save:
            #     plot_confusion_matrix(test_c_matrix, CLASSES, 'c_m_malware' + str(epoch) + '.png')
        all_test_acc.append(test_accuracy)
        P, R, f1 = deal_cm(test_c_matrix)
        all_f1.append(np.array(f1).mean())
        if all_f1[-1] > best_f1:
            best_f1 = all_f1[-1]
            print('>> test f1-score increased !!!')
            if is_save:
                save_model(model, epoch, optimizer, loss, best_acc, best_f1, full=True)
        print('>> epoch: {} ok:\n'
              '>> >> train avg loss: {}\n'
              '>> >> test accuracy: {}, best test accuracy: {},\n'
              '>> >> test f1-score: {}, best f1-score: {}'.format(
            epoch, train_total_loss / num_batch, test_accuracy, best_acc, all_f1[-1], best_f1))
        print('>> detailed test results of every label:\nlabel\tPrecision\tRecall\tF1-score')
        for key in LABELS:
            print('{}\t{}\t{}\t{}\t{}'.format(
                LABELS[key], key, P[int(LABELS[key])], R[int(LABELS[key])], f1[int(LABELS[key])]))
        for i in range(len(test_c_matrix)):
            print((' ' + str(i)) if i < 10 else str(i), end='')
            for j in range(len(test_c_matrix[i])):
                num = str(test_c_matrix[i][j])
                num = ' ' * (6 - len(num)) + num
                print(num, end='')
            print()
        # print(test_c_matrix)
        if epoch > 100 and all_f1[-1] < best_f1 and all_f1[-2] < best_f1 and all_f1[-3] < best_f1 and all_f1[
            -4] < best_f1 and all_f1[-5] < best_f1 and all_test_acc[-1] < best_acc and all_test_acc[-2] < best_acc \
                and all_test_acc[-3] < best_acc and all_test_acc[-4] < best_acc and all_test_acc[-5] < best_acc:
            print('early stop with epoch: {}, \nbest test accuracy: {}, best f1-score: {}'.format(
                epoch, best_acc, best_f1))
            break


def train_RBLJAN(is_save=False, is_load=False, is_shuffle=False, device=0, model_type='cnn'):
    if EMBEDDING_DIM == 256:
        with open('./tmp_variable/p_e_matrix_256.variable', 'rb') as f1:
            p_e_matrix = pickle.load(f1)[:PKT_MAX_LEN + NGRAM_PAYLOAD]
    else:
        with open('./tmp_variable/p_e_matrix.variable', 'rb') as f1:
            p_e_matrix = pickle.load(f1)[:PKT_MAX_LEN + NGRAM_PAYLOAD]
    overall_label_idx = (torch.arange(0, NUM_LABELS)).long()
    if CUDA:
        overall_label_idx = overall_label_idx.cuda(DEVICE)
        p_e_matrix = p_e_matrix.cuda(DEVICE)
    D_model = RBLJAN_Classifier(EMBEDDING_DIM, NUM_LABELS, p_e_matrix, overall_label_idx, NGRAM_HEADER,
                                NGRAM_PAYLOAD, DROPOUT)
    print(D_model)
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=LR)

    random_size = 100
    inject_num = 500
    ngram = 51
    states = None
    inject_pos = 50
    if model_type == 'mlp':
        G_model = Generator_MLP(random_size + NUM_LABELS, inject_pos, inject_num)
    elif model_type == 'rnn':
        G_model = Generator_RNN(random_size + NUM_LABELS, inject_pos, inject_num)
        states = G_model.init_state(BATCH_SIZE, device)
    else:
        G_model = Generator_CNN(random_size + NUM_LABELS, inject_pos, inject_num, ngram)
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=LR)
    print(G_model)
    if CUDA:
        D_model = D_model.cuda(device)
        G_model = G_model.cuda(device)
    if DEVICE_COUNT > 1:
        gpus_free = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').readlines()
        gpus_free = [int(_.split()[2]) for _ in gpus_free]
        gpus_idx = []
        for _ in range(len(gpus_free)):
            if gpus_free[_] > 3000:
                gpus_idx.append(_)
        print("Use", len(gpus_idx), 'gpus:', gpus_idx)
        D_model = nn.DataParallel(D_model, device_ids=gpus_idx)
        G_model = nn.DataParallel(G_model, device_ids=gpus_idx)

    train_loader, test_loader = get_dataloader(EXP_PERCENT, TEST_PERCENT, BATCH_SIZE)
    train_loader.load_all_batches()
    test_loader.load_all_batches()

    num_batch = train_loader.num_batch

    D_loss_func = FocalLoss(NUM_LABELS, alpha=(train_loader.alpha if num_batch > 2000 else None), gamma=GAMMA,
                            size_average=True, is_reverse=False)
    G_loss_func = FocalLoss(NUM_LABELS, alpha=(train_loader.alpha if num_batch > 2000 else None), gamma=GAMMA,
                            size_average=True, is_reverse=True)

    cur_epoch, cur_loss = 0, 0
    D_best_acc, D_best_f1, G_best_acc = 0, 0, 0
    print('data prepare ok! total {} batches for {} epochs'.format(num_batch, EPOCHS - cur_epoch))

    _s_t = time.time()
    all_test_f1_real, all_test_acc_real = [], []
    all_test_f1_fake, all_test_acc_fake = [], []
    G_loss_all = []
    # training
    for epoch in range(cur_epoch + 1, EPOCHS):
        if is_shuffle:
            if epoch != 0 and epoch % 5 == 0:
                print('\n>> shuffle and reload data...')
                train_loader.reload_all_batches()
        D_total_loss_real, D_total_loss_fake, G_total_loss = 0, 0, 0
        print('\nepoch: {} start to train...'.format(epoch))
        y_hat, y = [], []
        for i in range(num_batch):
            # # real pkt
            batch_X, batch_y = train_loader[i]
            batch_size = len(batch_y)
            if CUDA:
                batch_X = batch_X.cuda(device)
                batch_y = batch_y.cuda(device)
            # # train discriminator
            D_model.train()
            out1, out2 = D_model(batch_X)
            D_loss_real_1 = D_loss_func(out1, batch_y.long())
            num_gpus = int(out2.size(0) / NUM_LABELS)
            y_label = torch.tensor(overall_label_idx.tolist() * num_gpus).long()
            if CUDA:
                y_label = y_label.cuda(DEVICE)
            D_loss_real_2 = F.cross_entropy(out2, y_label)
            D_loss_real = D_loss_real_1 + LAMBDA * D_loss_real_2
            # # generate fake pkt
            # torch.manual_seed(i)
            x_inject = Variable(torch.randn((batch_size, random_size)).float())
            class_mask = torch.zeros((batch_size, NUM_LABELS))
            if CUDA:
                x_inject = x_inject.cuda(device)
                class_mask = class_mask.cuda(device)
            class_mask.scatter_(1, batch_y.view(-1, 1).data, 1.)  # one-hot vector
            x_inject = torch.cat((x_inject, class_mask.float()), dim=1)
            G_model.eval()
            batch_X_fake = G_model(x_inject, batch_X).detach() if model_type != 'rnn' else G_model(x_inject, batch_X, states).detach()

            out1, out2 = D_model(batch_X_fake)
            D_loss_fake_1 = D_loss_func(out1, batch_y.long())
            D_loss_fake_2 = F.cross_entropy(out2, y_label)
            D_loss_fake = D_loss_fake_1 + LAMBDA * D_loss_fake_2

            D_loss = D_loss_real + D_loss_fake
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # # train generator
            # torch.manual_seed(0)
            x_inject = Variable(torch.randn((batch_size, random_size)).float())
            if CUDA:
                x_inject = x_inject.cuda(device)
            x_inject = torch.cat((x_inject, class_mask.float()), dim=1)
            G_model.train()
            batch_X_fake = G_model(x_inject, batch_X) if model_type != 'rnn' else G_model(x_inject, batch_X, states)
            # print(batch_X_fake[:, 100:110].tolist())

            D_model.eval()
            out1, out2 = D_model(batch_X_fake)
            y += batch_y.tolist()
            y_hat += out1.max(1)[1].tolist()
            G_loss = G_loss_func(out1, batch_y.long())
            G_loss_all.append(float(G_loss.item()))
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            D_total_loss_real += float(D_loss_real.item())
            D_total_loss_fake += float(D_loss_fake.item())
            G_total_loss += float(G_loss.item())
            if i % 2000 == 0:
                print('\r>> >>[{}] batch {} ok with {}s...'.format(
                    datetime.datetime.now().strftime('%m:%d-%H:%M:%S'), i, time.time() - _s_t), end='', flush=True)
        # y = np.array(y)
        # y_hat = np.array(y_hat)
        G_acc = 1. - accuracy_score(y, y_hat)
        print('\n>>[{}] All batch {} ok with {}s'.format(
            datetime.datetime.now().strftime('%m:%d-%H:%M:%S'), num_batch, time.time() - _s_t))
        if G_acc > G_best_acc:
            G_best_acc = G_acc
            print('>> [G] accuracy increased !!!')
        eval_s_t = time.time()
        print('>> >> start evaluate model with {} batches'.format(test_loader.num_batch))
        test_acc_real, test_acc_fake, test_c_matrix_real, test_c_matrix_fake = eval_D(D_model, G_model, test_loader,
                                                                                      True, NUM_LABELS, states)
        print('evaluate done with {}s'.format(time.time() - eval_s_t))
        if (test_acc_real + test_acc_fake) / 2 > D_best_acc:
            print('>> [D] accuracy increased !!!')
            D_best_acc = (test_acc_real + test_acc_fake) / 2
            # if is_save:
            #     plot_confusion_matrix(test_c_matrix_real, CLASSES, 'c_m_malware' + str(epoch) + '.png')
        all_test_acc_real.append((test_acc_real + test_acc_fake) / 2)
        P_real, R_real, f1_real = deal_cm(test_c_matrix_real)
        P_fake, R_fake, f1_fake = deal_cm(test_c_matrix_fake)
        all_test_f1_real.append(np.array(f1_real).mean())
        all_test_f1_fake.append(np.array(f1_fake).mean())
        if (all_test_f1_real[-1] + all_test_f1_fake[-1]) / 2 > D_best_f1:
            D_best_f1 = (all_test_f1_real[-1] + all_test_f1_fake[-1]) / 2
            print('>> [D] f1-score increased !!!')
        if is_save:
            save_model(D_model, epoch, D_optimizer, D_total_loss_real, full=True)
        print('>> epoch: {} ok:\n'
              '>> >> D avg real loss: {}, D avg fake loss: {}, G avg loss: {}\n'
              '>> >> G accuracy: {}, G best accuracy: {}\n'
              '>> >> D test accuracy: {}, D best test accuracy: {}\n'
              '>> >> D test f1-score: {}, D best f1-score: {}'.format(
            epoch, D_total_loss_real / num_batch, D_total_loss_fake / num_batch, G_total_loss / num_batch,
            G_acc, G_best_acc, (test_acc_real + test_acc_fake) / 2, D_best_acc,
                   (all_test_f1_real[-1] + all_test_f1_fake[-1]) / 2, D_best_f1))

        if epoch > 60 and all_test_f1_real[-1] < D_best_f1 and all_test_f1_real[-2] < D_best_f1 and all_test_f1_real[
            -3] < D_best_f1 and all_test_f1_real[
            -4] < D_best_f1 and all_test_f1_real[-5] < D_best_f1 and all_test_acc_real[-1] < D_best_acc and \
                all_test_acc_real[-2] < D_best_acc \
                and all_test_acc_real[-3] < D_best_acc and all_test_acc_real[-4] < D_best_acc and all_test_acc_real[
            -5] < D_best_acc:
            print('early stop with epoch: {}, \nbest test accuracy: {}, best f1-score: {}'.format(
                epoch, D_best_acc, D_best_f1))
            break


if __name__ == '__main__':
    s_t = time.time()
    print('CUDA: {} of {}'.format(CUDA, DEVICE))
    print('start train...')

    train_RBLJAN_GAN(is_save=False, is_load=False, is_shuffle=False)

    # train_RBLJAN_FLOW(is_save=False, is_load=False, is_shuffle=False)
    print('training done with {}s'.format(time.time() - s_t))
