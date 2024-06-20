## Tree-RNN

This is an implementation (a reproduction version) for "Tree-rnn: Tree structural recurrent neural network for network traffic classification", we reproduced their method based on their paper.

* The preprocessing phase is the same as `RBLJAN`.
* The training phase is a bit complex, you should train many LSTMs to build a classification tree. The dataset division of our work is list in the appendix for reference.

#### Appendix

My category division (tree structure)

* the division rules is based on the paper
* the index below is the category index accoding to our implementation (refer to [utils.py](./utils.py), the `LABELS`)
* as our dataset is composed of many classes, the tree is very large, so we need to train many models to complete the classification task...
* for the evaluation task, we predict packets one by one (batch_size = 1) as it is difficult to predict a batch. It may be extended to batch prediction by quick sort after every node of the tree (if you want to).

```
29 (X-APP):

29 -> 14, max_similarity: 0.9922209866754301
[0, 1, 4, 5, 6, 8, 9, 13, 15, 17, 19, 21, 26, 27]
[2, 3, 7, 10, 11, 12, 14, 16, 18, 20, 22, 23, 24, 25, 28]
    14 -> 7, max_similarity: 0.9961263650939578
    [0, 1, 5, 13, 17, 19, 21]
    [4, 6, 8, 9, 15, 26, 27]
        7 -> 3, max_similarity: 0.9990430076917013
        [0, 1, 19]
        [5, 13, 17, 21]
            4 -> 2, max_similarity: 0.9993113279342651
            [5, 17]
            [13, 21]
        7 -> 3, max_similarity: 0.9961636463801066
        [4, 6, 8]
        [9, 15, 26, 27]
            4 -> 2, max_similarity: 0.9923045039176941
            [9, 15]
            [26, 27]
    15 -> 7, max_similarity: 0.9573328949156261
    [7, 11, 12, 14, 16, 18, 20]
    [2, 3, 10, 22, 23, 24, 25, 28]
        7 -> 3, max_similarity: 0.9971847732861837
        [7, 18, 20]
        [11, 12, 14, 16]
            4 -> 2, max_similarity: 0.9775385856628418
            [11, 12]
            [14, 16]
        8 -> 4, max_similarity: 0.9676620960235596
        [10, 22, 3, 28]
        [2, 24, 23, 25]
            4 -> 2, max_similarity: 0.9871518611907959
            [10, 22]
            [3, 28]
            4 -> 2, max_similarity: 0.9791545271873474
            [23, 25]
            [2, 24]


12 (ISCX-VPN):

12 -> 6, VPN - nonVPN
[0, 1, 2, 3, 4, 5]
[6, 7, 8, 9, 10, 11]

    6 -> 3, max_similarity: 0.9899763961633047
    [0, 4, 5]
    [1, 2, 3]
  
    [6, 10, 11]
    [7, 8, 9]


10 (USTC-TFC):

10 -> 5
[0, 1, 2, 3, 6]
[4, 5, 7, 8, 9]

    [1, 3]
    [0, 2, 6]

    [4, 8]
    [5, 7, 9]


20 (X-WEB):

20 -> 10, max_similarity: 0.9931577059957716
[0, 3, 4, 6, 7, 8, 9, 14, 17, 18]
[1, 2, 5, 10, 11, 12, 13, 15, 16, 19]

    10 -> 5, max_similarity: 0.9978238344192505
    [3, 6, 9, 14, 18]
    [0, 4, 7, 8, 17]

        5 -> 2, max_similarity: 0.9991680383682251
        [3, 14]
        [6, 9, 18]

        5 -> 2, max_similarity: 0.9991310238838196
        [4, 7]
        [0, 8, 17]

    10 -> 5, max_similarity: 0.9902564644813537
    [5, 10, 12, 13, 19]
    [1, 2, 11, 15, 16]

        5 -> 2, max_similarity: 0.9990594983100891
        [12, 19]
        [5, 10, 13]

        5 -> 2, max_similarity: 0.9816560745239258
        [1, 2]
        [11, 15, 16]

```
