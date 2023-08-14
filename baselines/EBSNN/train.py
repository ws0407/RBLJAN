from model import EBSNN_LSTM, EBSNN_GRU
from utils import *


# device: 2 / 3
def train_EBSNN(device=0, is_lstm=True, is_save=True):
    print('EBSNN_{} of class {} on device {} start to train...\n'.format('LSTM' if is_lstm else 'GRU', len(CLASSES), device))
    train_loader, test_loader = get_dataloader(EXP_PERCENT, TEST_PERCENT, BATCH_SIZE)

    if is_lstm:
        model = EBSNN_LSTM(NUM_CLASS, EMBEDDING_DIM, device=device)
    else:
        model = EBSNN_GRU(NUM_CLASS, EMBEDDING_DIM, device=device)
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
    loss_func = FocalLoss(NUM_CLASS, device, alpha=(train_loader.alpha if num_batch > 1000 else None), gamma=GAMMA, size_average=True)
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
            # start to train

            model.train()
            optimizer.zero_grad()
            out = model(batch_X)
            loss = loss_func(out, batch_y)
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
                save_model(model, epoch)
        if all_f1[-1] > best_f1:
            print('>> test f1-score increased !!!')
            best_f1 = all_f1[-1]

        all_test_acc.append(test_accuracy)
        print('>> epoch: {} ok:\n'
              '>> >> train avg loss: {}\n'
              '>> >> test accuracy: {}, best test accuracy: {},\n'
              '>> >> test f1-score: {}, best f1-score: {}'.format(epoch, train_total_loss / num_batch,
                                                                  test_accuracy, best_acc, all_f1[-1], best_f1))
        print('>> detailed test results of every label:\nlabel\tPrecision\tRecall\tF1-score')
        for key in LABELS:
            print('{}\t{}\t{}\t{}'.format(key, P[int(LABELS[key])], R[int(LABELS[key])], f1[int(LABELS[key])]))
        if epoch > 40 and max(all_f1[-5:]) < best_f1 and max(all_test_acc[-5:]) < best_acc:
            print('early stop with epoch: {}, \nbest test accuracy: {}, best f1-score: {}'.format(epoch, best_acc, best_f1))
            break


if __name__ == '__main__':
    train_EBSNN(device=0, is_lstm=False, is_save=True)
    # train_EBSNN(device=0, is_lstm=False, is_save=True