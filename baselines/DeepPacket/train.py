from model import *
from utils import *


def train_DP(device=2, is_cnn=True, is_save=False, is_load=False):
    print('DeepPacket_{} of class {} on device {} start to train...\n'.format('CNN' if is_cnn else 'SAE', len(CLASSES),
                                                                              device))
    train_loader, test_loader = get_dataloader(EXP_PERCENT, TEST_PERCENT, BATCH_SIZE)
    train_loader.load_all_batches()
    test_loader.load_all_batches()
    if is_cnn:
        if len(CLASSES) in [29, 20]:
            model = CNN_APP()
        else:
            model = CNN_TRA()
    else:
        model = SAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model = model.cuda(device)
    if CUDA and torch.cuda.device_count() > 1:
        gpus_free = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').readlines()
        gpus_free = [int(_.split()[2]) for _ in gpus_free]
        gpus_idx = []
        for _ in range(len(gpus_free)):
            if gpus_free[_] > 3000:
                gpus_idx.append(_)
        print("Use", len(gpus_idx), 'gpus:', gpus_idx)
        model = nn.DataParallel(model, device_ids=gpus_idx)
    num_batch = train_loader.num_batch
    _s_t = time.time()
    all_f1, all_test_acc = [], []
    best_acc, best_f1 = 0, 0

    for epoch in range(EPOCHS):
        train_total_loss = 0
        print('\nepoch: {} start to train...'.format(epoch))
        for i in range(num_batch):
            batch_X, batch_y = train_loader.data[i]

            batch_X = batch_X.cuda(device)
            batch_y = batch_y.cuda(device)

            model.train()
            optimizer.zero_grad()
            out = model(batch_X)
            loss = F.cross_entropy(out, batch_y)
            train_total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('\r>> >> batch {} ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
        print('\n>> All batch {} ok with {}s'.format(num_batch, time.time() - _s_t))
        eval_s_t = time.time()
        print('>> >> start evaluate loss accuracy with {} batches'.format(test_loader.num_batch))
        test_accuracy, test_c_matrix = evaluate_loss_acc(model, test_loader, True, NUM_CLASS, device=device)
        print('evaluate done with {}s'.format(time.time() - eval_s_t))

        P, R, f1 = deal_matrix(test_c_matrix)
        all_f1.append(np.array(f1).mean())
        if test_accuracy > best_acc:
            print('>> test accuracy increased !!!')
            best_acc = test_accuracy
            if is_save:
                t_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                model_name = '/data/ws/tmp/BLJAN-IWQOS/DeepPacket/model/' + str(NUM_CLASS) + '/' + t_str + '_train_' + str(epoch) + '.pt'
                with open(os.path.join(model_name), 'wb') as f3:
                    torch.save(model, f3)
        if all_f1[-1] > best_f1:
            print('>> test f1-score increased !!!')
            best_f1 = all_f1[-1]

        print('>> epoch: {} ok:\n'
              '>> >> train avg loss: {}\n'
              '>> >> test accuracy: {}, best test accuracy: {},\n'
              '>> >> test f1-score: {}, best f1-score: {}'.format(epoch, train_total_loss / num_batch,
                                                                  test_accuracy, best_acc, all_f1[-1], best_f1))
        all_test_acc.append(test_accuracy)
        print('>> detailed test results of every label:\nlabel\tPrecision\tRecall\tF1-score')
        for key in LABELS:
            print('{}\t{}\t{}\t{}'.format(key, P[int(LABELS[key])], R[int(LABELS[key])], f1[int(LABELS[key])]))
        if epoch > 20 and all_f1[-1] < best_f1 and all_f1[-2] < best_f1 and all_f1[-3] < best_f1 and all_f1[
            -4] < best_f1 and all_f1[-5] < best_f1:
            print('early stop with epoch: {}, \nbest test accuracy: {}, best f1-score: {}\n'.format(epoch, best_acc,
                                                                                                  best_f1))
            break


def train_DP_Discriminator(device=2, is_cnn=True, is_save=True, is_load=False):
    print('DeepPacket_{} of class {} on device {} start to train...\n'.format('CNN' if is_cnn else 'SAE', len(CLASSES),
                                                                              device))
    train_loader = get_dataloader_train(EXP_PERCENT, TEST_PERCENT, BATCH_SIZE)
    train_loader.load_all_batches()
    if is_cnn:
        if len(CLASSES) in [29, 20]:
            model = CNN_APP_D()
        else:
            model = CNN_TRA_D()
    else:
        model = SAE_D()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model = model.cuda(device)

    if CUDA and torch.cuda.device_count() > 1:
        gpus_free = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free').readlines()
        gpus_free = [int(_.split()[2]) for _ in gpus_free]
        gpus_idx = []
        for _ in range(len(gpus_free)):
            if gpus_free[_] > 3000:
                gpus_idx.append(_)
        print("Use", len(gpus_idx), 'gpus:', gpus_idx)
        model = nn.DataParallel(model, device_ids=gpus_idx)

    num_batch = train_loader.num_batch
    _s_t = time.time()
    all_f1 = []
    best_acc, best_f1 = 0, 0

    for epoch in range(EPOCHS):
        train_total_loss = 0
        y = []
        y_hat = []
        print('\nepoch: {} start to train...'.format(epoch))
        for i in range(num_batch):
            batch_X, batch_y = train_loader.data[i]

            batch_X = batch_X.cuda(device)
            batch_y = batch_y.cuda(device)

            model.train()
            optimizer.zero_grad()
            out = model(batch_X)
            y += batch_y.tolist()
            y_hat += out.max(1)[1].tolist()
            loss = F.cross_entropy(out, batch_y)
            train_total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print('\r>> >> batch {} ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
        print('\n>> All batch {} ok with {}s'.format(num_batch, time.time() - _s_t))
        y = np.array(y)
        y_hat = np.array(y_hat)
        accuracy = accuracy_score(y, y_hat)
        test_c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(0, NUM_CLASS)])
        P, R, f1 = deal_matrix(test_c_matrix)
        all_f1.append(np.array(f1).mean())
        if accuracy > best_acc:
            print('>> test accuracy increased !!!')
            best_acc = accuracy
        if all_f1[-1] > best_f1:
            print('>> test f1-score increased !!!')
            best_f1 = all_f1[-1]

            if is_save:
                t_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                model_name = '/data/ws/tmp/BLJAN-IWQOS/DeepPacket/model/' + str(NUM_CLASS) + '/' + t_str + '_train_' + str(epoch) + '.pt'
                with open(os.path.join(model_name), 'wb') as f3:
                    torch.save(model, f3)

        print('>> epoch: {} ok:\n'
              '>> >> train avg loss: {}\n'
              '>> >> train accuracy: {}, best train accuracy: {},\n'
              '>> >> train f1-score: {}, best train f1-score: {}'.format(epoch, train_total_loss / num_batch,
                                                                         accuracy, best_acc, all_f1[-1], best_f1))
        print('>> detailed test results of every label:\nlabel\tPrecision\tRecall\tF1-score')
        for key in LABELS:
            print('{}\t{}\t{}\t{}'.format(key, P[int(LABELS[key])], R[int(LABELS[key])], f1[int(LABELS[key])]))
        if epoch > 30 and all_f1[-1] < best_f1 and all_f1[-2] < best_f1 and all_f1[-3] < best_f1 and all_f1[
            -4] < best_f1 and all_f1[-5] < best_f1:
            print('early stop with epoch: {}, \nbest test accuracy: {}, best f1-score: {}'.format(epoch, best_acc,
                                                                                                  best_f1))
            break


if __name__ == '__main__':
    train_DP(device=0, is_cnn=True, is_save=True, is_load=False)
    train_DP(device=0, is_cnn=False, is_save=True, is_load=False)
