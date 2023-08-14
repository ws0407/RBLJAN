import datetime
import random

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import argparse
import time, os
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
PADDING_IDX = 256
TEST_PERCENT = 0.15
PKT_MAX_LEN = 1500
CUDA = torch.cuda.is_available()
DEVICE = 0
EPOCHS = 120
LR = 0.001
CLASSES = ['vimeo', 'spotify', 'voipbuster', 'sinauc', 'cloudmusic', 'weibo', 'baidu', 'tudou', 'amazon', 'thunder',
           'gmail', 'pplive', 'qq', 'taobao', 'yahoomail', 'itunes', 'twitter', 'jd', 'sohu', 'youtube', 'youku',
           'netflix', 'aimchat', 'kugou', 'skype', 'facebook', 'google', 'mssql', 'ms-exchange']

LABELS = {'vimeo': 0, 'spotify': 1, 'voipbuster': 2, 'sinauc': 3, 'cloudmusic': 4, 'weibo': 5,
          'baidu': 6, 'tudou': 7, 'amazon': 8, 'thunder': 9, 'gmail': 10, 'pplive': 11,
          'qq': 12, 'taobao': 13, 'yahoomail': 14, 'itunes': 15, 'twitter': 16, 'jd': 17,
          'sohu': 18, 'youtube': 19, 'youku': 20, 'netflix': 21, 'aimchat': 22, 'kugou': 23,
          'skype': 24, 'facebook': 25, 'google': 26, 'mssql': 27, 'ms-exchange': 28}
# CLASSES = ['amazon', 'baidu', 'bing', 'douban', 'facebook', 'google', 'imdb', 'instagram', 'iqiyi', 'jd',
#            'neteasemusic', 'qqmail', 'reddit', 'taobao', 'ted', 'tieba', 'twitter', 'weibo', 'youku', 'youtube']
# LABELS = {'amazon': 0, 'baidu': 1, 'bing': 2, 'douban': 3, 'facebook': 4, 'google': 5, 'imdb': 6, 'instagram': 7,
#           'iqiyi': 8, 'jd': 9, 'neteasemusic': 10, 'qqmail': 11, 'reddit': 12, 'taobao': 13, 'ted': 14, 'tieba': 15,
#           'twitter': 16, 'weibo': 17, 'youku': 18, 'youtube': 19}
# CLASSES = ['audio', 'chat', 'file', 'mail', 'streaming', 'voip',
#            'vpn-audio', 'vpn-chat', 'vpn-file', 'vpn-mail', 'vpn-streaming', 'vpn-voip']
# LABELS = {'audio': 0, 'chat': 1, 'file': 2, 'mail': 3, 'streaming': 4, 'voip': 5,
#           'vpn-audio': 6, 'vpn-chat': 7, 'vpn-file': 8, 'vpn-mail': 9, 'vpn-streaming': 10, 'vpn-voip': 11}
# LABELS = {'Cridex': 0, 'Geodo': 1, 'Htbot': 2, 'Miuref': 3, 'Neris': 4, 'Nsis-ay': 5,
#           'Shifu': 6, 'Tinba': 7, 'Virut': 8, 'Zeus': 9}
# CLASSES = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']

MODEL_SAVE_PATH = '/data/ws/tmp/BLJAN-IWQOS/Tree-RNN/model/'
TREE = {'Left': 0, 'Right': 1}

# tree_left = [0, 1, 2, 3, 6]
# tree_right = [4, 5, 7, 8, 9]
# tree_middle = []
#
# tree_left = [4]
# tree_right = [8]
# tree_middle = []

# tree_left = [1]
# tree_right = [3]
# tree_middle = []

# tree_left = [1, 3]
# tree_right = [0, 2,6]
# tree_middle = []

# tree_left = [4, 8]
# tree_right = [5, 7, 9]
# tree_middle = []

# tree_left = [5]
# tree_right = [7]
# tree_middle = [9]

# tree_left = [0]
# tree_right = [2]
# tree_middle = [5]


