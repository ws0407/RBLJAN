from utils import *
from model import BaseRNN, BaseRNN_Embed


class Loader:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_batch = int((len(self.y) - 1) / batch_size)
        self.data = []
        # self.lengths = np.array([len(x) for x in self.X])

        self.load_all_batches()

    def load_all_batches(self):
        _s_t = time.time()
        print('>> start load data with {} batches'.format(self.num_batch))
        n = self.num_batch - 1
        for batch in range(self.num_batch):
            if batch % 1000 == 0 and batch != 0:
                print('\r>> >> {} batches ok with {}s...'.format(batch, time.time() - _s_t), end='', flush=True)
            batch_X = self.X[batch * self.batch_size: (batch + 1) * self.batch_size]
            batch_y = self.y[batch * self.batch_size: (batch + 1) * self.batch_size]
            # batch_len = self.lengths[batch * self.batch_size: (batch + 1) * self.batch_size]
            max_len = PKT_MAX_LEN
            for j in range(len(batch_y)):
                if len(batch_X[j]) < max_len:
                    batch_X[j] = batch_X[j] + [256] * (max_len - len(batch_X[j]))
            batch_X = torch.tensor(batch_X.tolist()).float()
            batch_y = torch.from_numpy(batch_y)
            self.data.append((batch_X, batch_y))
        print('\n>> all batches load done with {}s'.format(time.time() - _s_t))


def get_dataloader(batch_size, t_l, t_r, t_m=None):
    if t_m is None:
        t_m = []
    _s_t = time.time()
    print('start load X and y')
    X_train = []
    # X_test = []
    y_train = []
    # y_test = []
    pkl_dir = './data/USTC-TFC/result_doc/'
    for i in t_l:
        label = CLASSES[i]
        filename = pkl_dir + label + '.pkl'
        print('load {} from {}'.format(label, filename))
        with open(filename, 'rb') as f1:
            tmp = pickle.load(f1)
        random.seed(2022)
        random.shuffle(tmp)
        idx = int(len(tmp) * TEST_PERCENT)
        # X_test += tmp[:idx]
        X_train += tmp[idx:]
        # y_test += [0] * idx
        y_train += [0] * (len(tmp) - idx)
        # if len(tmp) > 25000:
        #     random.seed(41)
        #     tmp = random.sample(tmp, 25000)

    for i in t_r:
        label = CLASSES[i]
        filename = pkl_dir + label + '.pkl'
        print('load {} from {}'.format(label, filename))
        with open(filename, 'rb') as f1:
            tmp = pickle.load(f1)
        # if len(tmp) > 210000:
        #     random.seed(41)
        #     tmp = random.sample(tmp, 200000)
        random.seed(2022)
        random.shuffle(tmp)
        idx = int(len(tmp) * TEST_PERCENT)
        # X_test += tmp[:idx]
        X_train += tmp[idx:]
        # y_test += [1] * idx
        y_train += [1] * (len(tmp) - idx)

    for i in t_m:
        label = CLASSES[i]
        filename = pkl_dir + label + '.pkl'
        print('load {} from {}'.format(label, filename))
        with open(filename, 'rb') as f1:
            tmp = pickle.load(f1)
        # if len(tmp) > 210000:
        #     random.seed(41)
        #     tmp = random.sample(tmp, 200000)
        random.seed(2022)
        random.shuffle(tmp)
        idx = int(len(tmp) * TEST_PERCENT)
        # X_test += tmp[:idx]
        X_train += tmp[idx:]
        # y_test += [2] * idx
        y_train += [2] * (len(tmp) - idx)

    print('load X and y cost: {}s'.format(time.time() - _s_t))

    random.seed(2022)
    random.shuffle(X_train)
    random.seed(2022)
    random.shuffle(y_train)

    X_train = np.array(X_train, dtype=object)
    y_train = np.array(y_train, dtype=int)
    # X_test = np.array(X_test, dtype=object)
    # y_test = np.array(y_test, dtype=int)

    # train_loader, test_loader = Loader(X_train, y_train, batch_size), Loader(X_test, y_test, batch_size)
    train_loader = Loader(X_train, y_train, batch_size)

    return train_loader


def evaluate_loss_acc(model, dataloader, is_cm, num_label=2, device=2):
    # loss_func = nn.CrossEntropyLoss()
    y_hat, y = [], []
    for i in range(dataloader.num_batch):
        batch_X, batch_y = dataloader.data[i]
        if CUDA:
            # X_mask = torch.tensor(X_mask.tolist()).cuda(device)
            batch_X = batch_X.cuda(device)
            batch_y = batch_y.cuda(device)
        model.eval()
        out = model(batch_X)
        y_hat += out.max(1)[1].tolist()
        y += batch_y.tolist()
    y = np.array(y)
    y_hat = np.array(y_hat)
    accuracy = accuracy_score(y, y_hat)

    c_matrix = 0
    if is_cm:
        c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(num_label)])
    return accuracy, c_matrix, y_hat


def deal_matrix(c_matrix):
    print('start deal confusion matrix with {} labels'.format(c_matrix.shape[0]))
    row_sum = c_matrix.sum(axis=1)  # for precision
    col_sum = c_matrix.sum(axis=0)  # for recall

    P, R, F1 = [], [], []
    n_class = c_matrix.shape[0]
    for i in range(n_class):
        p = (c_matrix[i][i] / row_sum[i]) if row_sum[i] != 0 else 0
        r = (c_matrix[i][i] / col_sum[i]) if col_sum[i] != 0 else 0
        f1 = (2 * p * r / (p + r)) if p + r != 0 else 0
        P.append(p)
        R.append(r)
        F1.append(f1)
    return P, R, F1


def save_model(model, epoch, full=True):
    t_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # with open(os.path.join(MODEL_SAVE_PATH + t_str + '_eval_' + str(epoch) + '.pt'), 'wb') as f2:
    #     torch.save(model, f2)  # only for inference, model = torch.load(path), then: model.eval()
    model_path = MODEL_SAVE_PATH + str(len(CLASSES)) + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = model_path + t_str + '_train_' + str(epoch) + ('_full.pt' if full else '.pt')
    with open(os.path.join(model_name), 'wb') as f3:
        if full:
            torch.save(model, f3)
    print('model successfully saved in path: {}'.format(model_name))


def load_test_data():
    print('start load X and y')
    # X_train = []
    X_test = []
    # y_train = []
    y_test = []
    # pkl_dir = '/data/ws/tmp/BLJAN-IWQOS/datasets/USTC-TFC2016/result_doc/'
    # pkl_dir = '/data/ws/tmp/BLJAN-IWQOS/datasets/BLJAN/result_doc_sam/'
    # pkl_dir = '/data/ws/tmp/BLJAN-IWQOS/datasets/USTC-TFC2016/result_doc_sam/'
    # pkl_dir = '/data/ws/tmp/BLJAN-IWQOS/datasets/BLJAN/TrafficX/result_doc/'
    pkl_dir = '/data/ws/tmp/BLJAN-IWQOS/datasets/BLJAN/ISCX-VPN-NonVPN-2016/result_doc/'
    # pkl_dir = '/data/ws/tmp/BLJAN-IWQOS/datasets/EBSNN/result_doc/'
    all_filenames = os.listdir(pkl_dir)

    for label in CLASSES:
        filename = [f for f in all_filenames if 'X_' in f and label in f][0]
        print('load {} from {}'.format(label, filename))
        file_path = pkl_dir + filename
        # file_path = pkl_dir + 'X_' + label + '.pkl'
        with open(file_path, 'rb') as f1:
            tmp = pickle.load(f1)
        random.seed(2022)
        random.shuffle(tmp)
        idx = int(len(tmp) * TEST_PERCENT)
        # idx = 100
        X_test += tmp[:idx]
        # X_train += tmp[idx:]
        y_test += [LABELS[label]] * idx
        # y_train += [0] * (len(tmp) - idx)

    # X_test = np.array(X_test, dtype=object)
    # y_test = np.array(y_test, dtype=int)

    # X_test = torch.tensor(X_test).long()
    print('load test data done with {} pkt'.format(len(y_test)))

    return X_test, y_test


def evaluate(X, y, num_label=8, device=1):
    # # for 29-application
    # leaf_idx = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # labels = [
    #     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
    #     [0, 1, 19], [0, 1], [4, 6, 8], [0, 1], [7, 18, 20], [0, 1], [0, 1], [0, 1],
    #     [5, 17], [13, 21], [9, 15], [26, 27], [11, 12], [14, 16], [10, 22], [3, 28], [23, 25], [2, 24]
    # ]

    # for 12-service
    leaf_idx = [0, 0, 0, 1, 1, 1, 1]
    labels = [
        [0, 1], [0, 1], [0, 1],
        [1, 2, 3], [0, 4, 5], [7, 8, 9], [6, 10, 11]
    ]

    # # for 20-website
    # leaf_idx = [0] * 7 + [1] * 8
    # labels = [
    #     [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
    #     [3, 14], [6, 9, 18], [4, 7], [0, 8, 17], [12, 19], [5, 10, 13], [1, 2], [11, 15, 16]
    # ]

    model_dir = '/data/ws/tmp/BLJAN-IWQOS/Tree-RNN/model/' + str(len(CLASSES)) + '/'
    model_names = os.listdir(model_dir)
    models = []
    for i in range(0, len(labels)):
        m = BaseRNN(len(labels[i]))
        if CUDA:
            m = m.cuda(device)
        model_name = [name for name in model_names if '_' + str(i) + '_' in name][0]
        checkpoint = torch.load(os.path.join(model_dir, model_name))
        m.load_state_dict(checkpoint['model_state_dict'])
        models.append(m)

    y_hat = []
    s_t = time.time()
    for i in range(len(X)):     # one pkt per batch
        if i % 1000 == 0:
            print('\r{}/{} ok with {}s'.format(i, len(X), time.time() - s_t), end='', flush=True)
        cur_node = 0
        while True:
            x = torch.tensor(X[i]).unsqueeze(0)
            if CUDA:
                x = x.cuda(device)
            out = models[cur_node](x)
            out = out.max(1)[1].tolist()[0]
            if leaf_idx[cur_node] == 1:     # leaf node
                y_hat += [labels[cur_node][out]]
                break
            cur_node = cur_node * 2 + out + 1   # next node index

    y = np.array(y)
    y_hat = np.array(y_hat)
    accuracy = accuracy_score(y, y_hat)

    c_matrix = confusion_matrix(y, y_hat, labels=[i for i in range(num_label)])
    return accuracy, c_matrix, y_hat


def train_BaseRNN(t_l, t_r, t_m=None, node=0, device=0, is_save=True, is_load=False):
    print('Node {} of {} classes on device {} start to train...\n'.format(node, len(CLASSES), device))
    if t_m is None:
        t_m = []
    train_loader = get_dataloader(BATCH_SIZE, t_l, t_r, t_m)
    num_label = 2 if len(t_m) == 0 else 3
    model = BaseRNN_Embed(num_label)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if CUDA:
        model = model.cuda(device)
    num_batch = train_loader.num_batch
    _s_t = time.time()
    all_f1, all_test_acc, all_train_acc = [], [], []
    best_acc, best_train_acc, best_train_acc = 0, 0, 0
    loss = 0
    for epoch in range(EPOCHS):
        # train_total_loss, train_total_acc = [], []
        print('\nepoch: {} start to train...'.format(epoch))
        y = []
        y_hat = []
        for i in range(num_batch):
            batch_X, batch_y = train_loader.data[i]
            if CUDA:
                batch_X = batch_X.cuda(device)
                batch_y = batch_y.cuda(device)
            # start to train
            model.train()
            optimizer.zero_grad()
            out = model(batch_X)
            y += batch_y.tolist()
            y_hat += out.max(1)[1].tolist()
            loss = F.cross_entropy(out, batch_y)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('\r>> >> batch {} ok with {}s...'.format(i, time.time() - _s_t), end='', flush=True)
        train_acc = accuracy_score(y, y_hat)
        all_train_acc.append(train_acc)
        print('\n>> All batch {} ok with {}s'.format(num_batch, time.time() - _s_t))
        best_train_acc = train_acc if train_acc > best_train_acc else best_train_acc
        print('>> epoch: {} ok:\n'
              '>> >> train accuracy: {}, best train accuracy: {},\n'.format(epoch, train_acc, best_train_acc))
        if best_train_acc == train_acc:
            print('train acc increased !!!')
            save_model(model, epoch)

        # if epoch % 5 == 0 and epoch > 10:
        #     eval_s_t = time.time()
        #     print('>> >> start evaluate loss accuracy with {} batches'.format(test_loader.num_batch))
        #     test_accuracy, test_c_matrix, test_y_hat = evaluate_loss_acc(model, test_loader, True, num_label)
        #     print('evaluate done with {}s'.format(time.time() - eval_s_t))
        #     best_acc = test_accuracy if test_accuracy > best_acc else best_acc
        #     all_test_acc.append(test_accuracy)
        #     P, R, f1 = deal_matrix(test_c_matrix)
        #     all_f1.append(np.array(f1).mean())
        #     best_f1 = np.array(all_f1).max()
        #     print('>> >> test accuracy: {}, best test accuracy: {},\n'
        #           '>> >> test f1-score: {}, best f1-score: {}'.format(test_accuracy, best_acc, all_f1[-1], best_f1))
        #     if best_acc == test_accuracy:
        #         print('>> test accuracy increased !!!')
        #     if best_f1 == all_f1[-1]:
        #         print('>> test f1-score increased !!!')
        #         if is_save:
        #             save_model(model, epoch, optimizer, loss, best_acc, best_f1, node)
        #     print('>> detailed test results of every label:\nlabel\tPrecision\tRecall\tF1-score')
        #     for key in TREE:
        #         print('{}\t{}\t{}\t{}'.format(key, P[int(TREE[key])], R[int(TREE[key])], f1[int(TREE[key])]))
        if epoch > 5 and (max(all_train_acc[-5:]) < best_train_acc or best_train_acc == 1.0):
            print('early stop with epoch: {}, \nbest train accuracy: {}'.format(epoch, best_train_acc))
            break


if __name__ == '__main__':
    # tree_left = [1]
    # tree_right = [2]
    # tree_middle = [3]
    #
    # train_BaseRNN(tree_left, tree_right, tree_middle, node=3, device=1)
    #
    # tree_left = [0]
    # tree_right = [4]
    # tree_middle = [5]
    #
    # train_BaseRNN(tree_left, tree_right, tree_middle, node=4, device=1)
    #
    # tree_left = [7]
    # tree_right = [8]
    # tree_middle = [9]
    #
    # train_BaseRNN(tree_left, tree_right, tree_middle, node=5, device=1)
    #
    # tree_left = [6]
    # tree_right = [10]
    # tree_middle = [11]
    #
    # train_BaseRNN(tree_left, tree_right, tree_middle, node=6, device=1)

    tree_left = [3]
    tree_right = [14]
    tree_middle = []

    train_BaseRNN(tree_left, tree_right, tree_middle, node=7, device=0)
