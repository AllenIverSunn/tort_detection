from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.layers import Dense, LeakyReLU, Activation, BatchNormalization
from keras.optimizers import Adam

from callbacks import ValLossThresholdStopping


class NeuralNetwork(object):
    def __init__(self, input_nodes=None, output_nodes=None, init_new=True):

        self.input_nodes = input_nodes
        self.hidden_nodes_1 = 1024
        self.hidden_nodes_2 = 512
        self.hidden_nodes_3 = 256
        self.output_nodes = output_nodes
        self.lr = 0.0001
        self.batch_size = 128
        self.epochs = 10
        self.model = None

        if init_new:
            self.init_new_model()
        else:
            self.init_exist_model()

    def init_new_model(self):
        self.model = Sequential()
        # 输入层
        self.model.add(Dense(self.hidden_nodes_1, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(self.hidden_nodes_2, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(self.hidden_nodes_3, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(self.output_nodes, activation='sigmoid'))

        # 优化器
        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='mse', optimizer=optimizer)

    def init_exist_model(self):
        self.model = load_model('data/model.h5')

    def train(self, x_train, y_train):
        callbacks = []
        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)
        callbacks.append(callback_reduce_lr)
        callback_loss_threshold = ValLossThresholdStopping(min_loss=0.01, min_val_loss=0.02)
        callbacks.append(callback_loss_threshold)
        self.model.fit(x_train, y_train,
                       epochs=self.epochs,
                       callbacks=callbacks,
                       validation_split=0.2)

    def evaluate(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test)
        print('===== 评估结果 =====')
        print('评分 : %f' % score)

    def test(self, x, y, threshold=0.5):
        y_predict = self.model.predict(x)

        false_negative_index_list = []
        size = len(y)
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0
        for i in range(size):
            y_value = y[i]
            y_predict_value = y_predict[i]
            if y_value == 1 and y_predict_value >= threshold:
                true_positive += 1
            elif y_value == 0 and y_predict_value >= threshold:
                false_positive += 1
            elif y_value == 1 and y_predict_value < threshold:
                false_negative_index_list.append(i)
                false_negative += 1
            else:
                true_negative += 1

        accuracy = float(true_positive + true_negative) / size
        precision = float(true_positive) / (true_positive + false_positive)
        recall = float(true_positive) / (true_positive + false_negative)
        f1 = 2 * recall * precision / (recall + precision)

        print('===== 测试结果 =====')
        print('阈值 : %f' % threshold)
        print('测试数量 : %d' % size)
        print('正确判断为侵权 : %d' % true_positive)
        print('错误判断为侵权 : %d' % false_positive)
        print('错误判断为非侵权 : %d' % false_negative)
        print('正确判断为非侵权 : %d' % true_negative)
        print('准确率 : %f' % accuracy)
        print('查准率 : %f' % precision)
        print('召回率 : %f' % recall)
        print('F1得分 : %f' % f1)
        return false_negative_index_list

    def save(self):
        self.model.save('data/model.h5')
        print('save model successfully')
