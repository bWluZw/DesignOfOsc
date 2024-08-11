# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
NNI example trial code.

- Experiment type: Hyper-parameter Optimization
- Trial framework: Tensorflow v2.x (Keras API)
- Model: LeNet-5
- Dataset: MNIST
"""

import logging

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPool2D)
from tensorflow.keras.optimizers import Adam

import nni

_logger = logging.getLogger('mnist_example')
_logger.setLevel(logging.INFO)

# 这段代码定义了一个自定义的卷积神经网络（CNN）类 MnistModel，继承自 Model 类，并且实现了经典的 LeNet-5 架构。
# 这个模型用于处理MNIST数据集（手写数字识别），并且可以根据传入的超参数自定义卷积层的核大小、全连接层的隐藏单元数以及dropout率。
class MnistModel(Model):
    # 实现了具有可定制超参数的LeNet-5模型。
    """
    LeNet-5 Model with customizable hyper-parameters
    """
    # 这是类的初始化方法，用于设置模型的超参数并构建模型的各层。
    def __init__(self, conv_size, hidden_size, dropout_rate):
        """
        Initialize hyper-parameters.

        Parameters
        ----------
        conv_size : int
            Kernel size of convolutional layers.
        hidden_size : int
            Dimensionality of last hidden layer.
        dropout_rate : float
            Dropout rate between two fully connected (dense) layers, to prevent co-adaptation.
        """
        # 调用父类的初始化方法，确保 MnistModel 正确继承和初始化自 Model 类。
        super().__init__()
        # 定义第一个卷积层 conv1：
        # 设置卷积核的数量为32，即输出的通道数。
        # 卷积核的大小由传入的 conv_size 参数决定。
        # 使用ReLU激活函数。
        self.conv1 = Conv2D(filters=32, kernel_size=conv_size, activation='relu')

        # 定义第一个最大池化层 pool1：
        # 池化窗口的大小为2x2，用于下采样特征图，减小尺寸。
        self.pool1 = MaxPool2D(pool_size=2)

        # 定义第二个卷积层 conv2，与第一个卷积层类似，但卷积核的数量增加到64。
        self.conv2 = Conv2D(filters=64, kernel_size=conv_size, activation='relu')
        # 定义第二个最大池化层 pool2，与第一个池化层类似。
        self.pool2 = MaxPool2D(pool_size=2)
        # 定义一个扁平化层 flatten，将二维特征图转换为一维向量，以便输入到全连接层。
        self.flatten = Flatten()
        # 定义第一个全连接层 fc1：
        # 全连接层的神经元数量由传入的 hidden_size 参数决定。
        # 使用ReLU激活函数。
        self.fc1 = Dense(units=hidden_size, activation='relu')
        # 定义一个dropout层 dropout，用于在训练期间随机丢弃一些神经元，防止过拟合：
        # 丢弃神经元的概率由传入的 dropout_rate 参数决定。
        self.dropout = Dropout(rate=dropout_rate)
        # 定义第二个全连接层 fc2，输出层：
        # 输出单元数为10，对应于MNIST数据集中10个类别（数字0-9）。
        # 使用softmax激活函数，将输出转换为概率分布。
        self.fc2 = Dense(units=10, activation='softmax')

    # 定义模型的前向传播函数 call。这是 Model 类的一个标准方法，规定了输入数据 x 如何通过模型的各层进行处理。
    def call(self, x):
        # 重写了Model.call
        """Override ``Model.call`` to build LeNet-5 model."""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)


class ReportIntermediates(Callback):
    """
    Callback class for reporting intermediate accuracy metrics.

    This callback sends accuracy to NNI framework every 100 steps,
    so you can view the learning curve on web UI.

    If an assessor is configured in experiment's YAML file,
    it will use these metrics for early stopping.
    """
    def on_epoch_end(self, epoch, logs=None):
        """Reports intermediate accuracy to NNI framework"""
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])


def load_dataset():
    """Download and reformat MNIST dataset"""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test)

#接受一个 params 字典作为参数，该字典包含了模型的超参数。

# 在接收到一组超参数后，使用这些超参数构建一个MNIST分类模型，训练该模型，并在测试集上评估其性能。
# 最终的准确率会通过NNI进行报告，用于超参数调优。
# 此函数中的日志记录能够帮助开发者跟踪训练过程中的关键步骤，便于调试和优化。
def main(params):
    """
    Main program:
      - Build network
      - Prepare dataset
      - Train the model
      - Report accuracy to tuner
    """
    # 使用传入的超参数创建一个 MnistModel 实例，自定义的深度学习模型类，专门用于处理MNIST数据集。
    model = MnistModel(
        conv_size=params['conv_size'],#  设置卷积层的核大小。
        hidden_size=params['hidden_size'],# 设置全连接层的隐藏单元数。
        dropout_rate=params['dropout_rate']# 设置dropout的概率，用于防止过拟合。
    )

    # 使用传入的学习率参数创建一个Adam优化器实例，这是一个常用的自适应学习率优化器。
    optimizer = Adam(learning_rate=params['learning_rate'])

    # 编译模型，指定优化器、损失函数和评估指标：
    # 使用前面定义的Adam优化器。
    # 指定损失函数为稀疏类别交叉熵，这是多分类任务中常用的损失函数，尤其适用于标签是整数编码的情况。
    # 评估模型的准确率。
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _logger.info('Model built')

    (x_train, y_train), (x_test, y_test) = load_dataset()
    _logger.info('Dataset loaded')

    model.fit(
        x_train,# 
        y_train,# 
        batch_size=params['batch_size'],# 使用传入的批量大小参数。
        epochs=5,# 训练模型5个周期。
        verbose=0,# 静默模式，不输出训练过程中的详细信息。
        callbacks=[ReportIntermediates()],# 在训练期间调用 ReportIntermediates 回调函数，可能用于向NNI报告中间结果。
        validation_data=(x_test, y_test) # 在每个epoch结束时使用测试数据进行验证。
    )
    _logger.info('Training completed')

    # 评估模型在测试数据上的性能，并返回损失值和准确率。
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    # 将最终的准确率结果报告给NNI。NNI将使用这些结果来指导超参数调优过程，并在其Web UI上显示结果。
    nni.report_final_result(accuracy)  # send final accuracy to NNI tuner and web UI
    _logger.info('Final accuracy reported: %s', accuracy)


if __name__ == '__main__':
    params = {
        'dropout_rate': 0.5,#模型中的dropout率，通常用于防止过拟合。
        'conv_size': 5,#卷积层的核大小。
        'hidden_size': 1024,#全连接层的隐藏单元数量。
        'batch_size': 32,#训练时每个批次的数据量。
        'learning_rate': 1e-4,#优化器的学习率。
    }

    # fetch hyper-parameters from HPO tuner
    # comment out following two lines to run the code without NNI framework
    # 从NNI获取下一组经过调优的超参数。nni.get_next_parameter() 是NNI的API，它会根据你在实验中定义的搜索空间，返回下一组超参数。
    # 这些参数是基于当前搜索算法（如随机搜索、贝叶斯优化等）选取的。
    tuned_params = nni.get_next_parameter()
    # 使用从NNI获取的调优参数更新初始的参数字典。
    # params.update(tuned_params) 会用 tuned_params 中的值覆盖 params 中对应的键值对。
    # 这样，params 字典中包含的就是最终使用的超参数组合。
    params.update(tuned_params)

    _logger.info('Hyper-parameters: %s', params)
    main(params)
