import pickle
import re

import jieba
import numpy as np
import time
from bs4 import BeautifulSoup
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self):
        self.vio = None
        self.n_vio = None
        self.stop_words = None
        self.docvecs = None

        self.init_source_data()
        self.get_stopwords()
        self.load_source_data()
        self.init_data_set()
        self.split_data_set()

    def init_source_data(self):
        print('Initialize source data')
        with open('./data/vio/0.pickle', 'rb') as f:
            self.vio = pickle.load(f)
        with open('./data/not_vio/0.pickle', 'rb') as f:
            self.n_vio = pickle.load(f)

    def get_stopwords(self, path='./chineseStopWords.txt'):
        self.stop_words = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.read().split('\n'):
                self.stop_words.append(line.strip())
        self.stop_words = set(self.stop_words)

    def cleanTxt(self, string):
        pat = re.compile("[A-Za-z0-9\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%\n-+_\t\'\" ]")
        return re.sub(pat, "", string)

    def load_source_data(self):
        """
        加载源数据
        去除30个词以下的样本
        添加每一条数据到identification_list，text_list，label_list里
        :param file_path: 文件路径
        :param label: 标签
        """
        print('Load source data')
        start = time.time()
        print('\tClean the data')
        self.vio = [self.cleanTxt(BeautifulSoup(item['text'], 'html').getText()) for item in self.vio]
        self.n_vio = [self.cleanTxt(BeautifulSoup(item['text'], 'html').getText()) for item in self.n_vio]
        print('\tFinish cleaning in {}s'.format(time.time() - start))

        ## 清洗数据，去停词，分词，去除小于30的数据
        start = time.time()
        print('\tProcessing the vio data')
        self.vio = [[word for word in jieba.lcut(sent) if word not in self.stop_words] for sent in self.vio]
        temp = []
        for words in self.vio:
            if len(words) > 30:
                temp.append(words)
        self.vio = temp
        print('\tFinish processing vio data in {}s'.format(time.time() - start))

        start = time.time()
        print('\tCleaning the non-vio data')
        self.n_vio = [[word for word in jieba.lcut(sent) if word not in self.stop_words] for sent in self.n_vio]
        temp = []
        for words in self.n_vio:
            if len(words) > 30:
                temp.append(words)
        self.n_vio = temp
        print('\tFinish processing the non-vio data in {}s'.format(time.time() - start))
        ## 保存分词过后的数据文件
        start = time.time()
        print('\tSaving cut words')
        with open('./cut_words/cut_words.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'vio': self.vio,
                'n_vio': self.n_vio
            }))
        print('\tFinish saving the cut words in {}s'.format(time.time() - start))
        print('Finish loading source data')

    
    def init_data_set(self, from_file=False):
        """
        使用Doc2vec格式
        """
        print('Initialize the dataset')
        start = time.time()
        print('\tBuild TaggedDocument for doc2vec trianing')
        tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(self.vio+self.n_vio)]
        print('\tFinish building in {}s'.format(time.time() - start))
        ##训练doc2vec模型
        print('\tTrain the Doc2vec model')
        start = time.time()
        model = Doc2Vec(tagged_data, size=200, min_count = 1)
        model.save('./doc2vec_model/doc2vec.model')
        print('\tFinish training doc2vec model in {}s'.format(time.time() - start))
        ## 提取docvec
        print('\tExtract docvecs from the model')
        start = time.time()
        self.docvecs = np.zeros((len(self.vio) + len(self.n_vio), 200))
        for i in range(self.docvecs.shape[0]):
            self.docvecs[i, :] = model.docvecs[str(i)]
        np.savetxt('./doc2vec_model/docvecs.txt', self.docvecs)
        print('\tFinish extracting docvecs')
        print('Finish init dataset')



    def split_data_set(self, test_size=0.3):
        """
        把数据集分为训练集、测试集
        """
        print('Split dataset')
        y = np.asarray(np.ones(len(self.vio)).tolist() + np.zeros(len(self.n_vio)).tolist())
        X = self.docvecs
        seed = 3
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        np.savetxt('./ProcessedData/X_train.txt', X_train)
        np.savetxt('./ProcessedData/X_test.txt', X_test)
        np.savetxt('./ProcessedData/y_train.txt', y_train)
        np.savetxt('./ProcessedData/y_test.txt', y_test)
        print('Finish spliting dataset, and save data in ProcessedData folder')

if __name__ == '__main__':
    DataProcessor()
