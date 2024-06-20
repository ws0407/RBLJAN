from utils import *
import gc

if __name__ == '__main__':

    device = 0
    step = 25
    inject_pos = 50
    model_dir = '/data/data/ws/NetworkTC/ICLSTM/model/' + str(NUM_LABELS) + '/wo_payload/'
    model_path = model_dir + max(os.listdir(model_dir))
    print('model_path', model_path)
    train_loader, test_loader = get_dataloader(EXP_PERCENT, TEST_PERCENT, 128)
    test_loader.load_all_batches()
    del train_loader
    gc.collect()

    model = torch.load(model_path)
    model = model.cuda(device)
    print(model)
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    res, res_acc = [], []
    all_F1 = [[] for _ in range(NUM_LABELS)]

    for i in range(21):
        y, y_hat = [], []
        for j in range(test_loader.num_batch):
            batch_X, batch_y = test_loader[j]
            batch_X = batch_X.cuda(device)
            batch_y = batch_y.cuda(device)

            # insert
            if i != 0:
                torch.manual_seed(j)
                insert = (torch.rand(len(batch_y), i * step) * 256).cuda(device)
                batch_X = torch.cat((batch_X[:, :inject_pos], insert, batch_X[:, inject_pos:]), dim=1)[:, :PKT_MAX_LEN]

            batch_X = batch_X.view(-1, 28, 28)
            model.eval()
            test_start = time.time()
            out = model(batch_X)
            y_hat += out.max(1)[1].tolist()
            y += batch_y.tolist()
        y = np.array(y)
        y_hat = np.array(y_hat)
        acc = accuracy_score(y, y_hat)
        res_acc.append(acc)
        c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(0, NUM_LABELS)])
        P, R, F1 = deal_matrix(c_matrix)
        print('insert {} bytes, acc: {}\nlabel\tPrecision\tRecall\tF1-score'.format(i * step, acc))
        for key in LABELS:
            print('{}\t{}\t{}\t{}'.format(key, P[int(LABELS[key])], R[int(LABELS[key])], F1[int(LABELS[key])]))
            all_F1[int(LABELS[key])].append(F1[int(LABELS[key])])
        res.append([np.array(P).mean(), np.array(R).mean(), np.array(F1).mean()])
        print('avg\t{}\t{}\t{}\n'.format(np.array(P).mean(), np.array(R).mean(), np.array(F1).mean()))
    print(res_acc)
    for _ in range(len(all_F1)):
        print(all_F1[_])
