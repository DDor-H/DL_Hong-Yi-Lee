import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # number = 10000
    # x_train = x_train[0:number]
    # y_train = y_train[0:number]
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 将类向量转换为二进制类矩阵
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    # 导入数据 x_train [60000, 784]
    #        x_test  [10000, 784]
    #        y_train [60000, 10]  类似于one-hot编码
    #        y_test  [10000, 10]  同上
    (x_train, y_train), (x_test, y_test) = load_data()

    # 定义网络结构
    model = Sequential()

    # 增加输入层和第一个隐藏层
    # 输入层784个input, 第一个隐藏层500个神经元节点，激活函数采用sigmoid函数
    model.add(Dense(input_dim=28 * 28, units=500, activation='relu'))
    # 增加第二个隐藏层, 500个神经元节点，激活函数采用sigmoid函数
    model.add(Dense(units=500, activation='relu'))
    # 定义输出层, 十个输出节点
    model.add(Dense(units=10, activation='softmax'))

    # 设置配置信息
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, batch_size=64, epochs=50)

    # 评估模型并输出准确率
    result = model.evaluate(x_test, y_test)
    print('Test Acc:', result[1])
