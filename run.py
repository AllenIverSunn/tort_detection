from data_loader import DataLoader
from neural_network import NeuralNetwork
from data_processor import DataProcessor

if __name__ == '__main__':
    DataProcessor()
    dl = DataLoader()
    X_train, y_train = dl.get_train_set()
    X_test, y_test = dl.get_test_set()

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('train size : %d' % train_size)
    print('test size : %d' % test_size)

    # 训练并保存模型
    network = NeuralNetwork(input_nodes=X_train.shape[0], output_nodes=1)
    network.train(X_train, y_train)
    network.save()
    # 使用已训练的模型
    # network = NeuralNetwork(input_nodes=len(dl.dictionary), output_nodes=y_train.shape[1], init_new=False)

    network.evaluate(X_test, y_test)
    thresholds = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    for threshold in thresholds:
        false_negative_index_list = network.test(X_test, y_test, threshold=threshold)
        save_filename = 'data/result/%d.txt' % threshold
        with open(save_filename, mode='w', encoding='utf-8') as file:
            for index in false_negative_index_list:
                file.write(str(index) + '\n')
