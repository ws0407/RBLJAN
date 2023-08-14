from train import *

if __name__ == '__main__':
    # tree_left = [3, 14]
    # tree_right = [6, 9, 18]
    # tree_middle = []
    # train_BaseRNN(tree_left, tree_right, tree_middle, node=3, device=0)
    #
    # tree_left = [4, 7]
    # tree_right = [0, 8, 17]
    # tree_middle = []
    # train_BaseRNN(tree_left, tree_right, tree_middle, node=4, device=0)
    X, y = load_test_data()
    accuracy, c_matrix, y_hat = evaluate(X, y, num_label=len(CLASSES), device=1)
    print('acc:', accuracy)
    P, R, F1 = deal_matrix(c_matrix)
    for key in LABELS:
        print('{}\t{}\t{}\t{}'.format(key, P[int(LABELS[key])], R[int(LABELS[key])], F1[int(LABELS[key])]))