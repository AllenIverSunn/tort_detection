import pickle

import jieba
import numpy as np
import json
import time


class DataLoader(object):
    def __init__(self):
        # 词典：key为词语，value为索引
        self.vio = None
        self.n_vio = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.load_data_set()

    def load_data_set(self):
        start = time.time()
        print('Load dataset')
        with open('./cut_words/cut_words.json', 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
            self.vio = data['vio']
            self.n_vio = data['n_vio']
        self.X_train = np.loadtxt('./ProcessedData/X_train.txt')
        self.X_test = np.loadtxt('./ProcessedData/X_test.txt')
        self.y_train = np.loadtxt('./ProcessedData/y_train.txt')
        self.y_test = np.loadtxt('./ProcessedData/y_test.txt')
        del data
        print('Finish loading dataset in {}s'.format(time.time() - start))

    def get_train_set(self):
        return self.X_train, self.y_train

    def get_test_set(self):
        return self.X_test, self.y_test