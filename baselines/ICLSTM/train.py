from model import *
from utils import *
import datetime
import torch.nn.functional as F


def train(device=2, is_save=False):
    print('ICLSTM of class {} on device {} start to train...\n'.format(len(CLASSES), device))
    train_loader, test_loader = get_dataloader(EXP_PERCENT, TEST_PERCENT, BATCH_SIZE)
    # train_loader.load_all_batches()
    test_loader.load_all_batches()
    model = ICLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.95, 0.999), eps=1e-08)
    model = model.cuda(device)

    loss_func = FocalLoss(NUM_LABELS, alpha=train_loader.alpha, gamma=GAMMA, size_average=True)
    num_batch = train_loader.num_batch
    _s_t = time.time()
    all_f1, all_test_acc = [], []
    best_acc, best_f1 = 0, 0

    for epoch in range(EPOCHS):
        train_total_loss = 0
        print('\nepoch: {} start to train...'.format(epoch))
        for i in range(num_batch):
            batch_X, batch_y = train_loader[i]

            batch_X = batch_X.cuda(device)
            batch_y = batch_y.cuda(device)

            model.train()
            optimizer.zero_grad()
            out = model(batch_X)
            loss = loss_func(out, batch_y.long())
            train_total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('\r>> >> batch {} ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
        print('\n>> All batch {} ok with {}s'.format(num_batch, time.time() - _s_t))
        eval_s_t = time.time()
        print('>> >> start evaluate loss accuracy with {} batches'.format(test_loader.num_batch))
        test_accuracy, test_c_matrix = evaluate_loss_acc(model, test_loader, True, NUM_LABELS, device=device)
        print('evaluate done with {}s'.format(time.time() - eval_s_t))

        P, R, f1 = deal_matrix(test_c_matrix)
        all_f1.append(np.array(f1).mean())
        if test_accuracy > best_acc:
            print('>> test accuracy increased !!!')
            best_acc = test_accuracy
            if is_save:
                t_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                model_name = '/data/data/ws/NetworkTC/ICLSTM/model/' + str(NUM_LABELS) + '/wo_payload/' + t_str + '_train_' + str(epoch) + '.pt'
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
        if epoch > 20 and max(all_f1[-5:]) < best_f1:
            print('early stop with epoch: {}, \nbest test accuracy: {}, best f1-score: {}\n'.format(
                epoch, best_acc, best_f1))
            break


if __name__ == '__main__':
    train(device=2, is_save=True)
