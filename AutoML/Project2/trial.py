import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
import json
import logging
import nni
import pandas as pd

# 日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('custom_trial_trial.py')

# 命令行参数解析：利用 argparse 解析命令行参数，获取配置文件的路径。
# 配置文件加载：通过 json 模块将指定路径的 JSON 配置文件加载为 Python 字典。
# 返回配置：将解析后的配置文件内容作为字典返回，供后续训练、评估等过程使用。
# 解析命令行参数或配置文件
def parse_args():
    # 创建一个 ArgumentParser 对象，该对象用于解析命令行参数。
    parser = argparse.ArgumentParser()
    # 为解析器添加一个命令行参数。
    # '--config'这是命令行参数的名称，表示配置文件的路径
    # type指定该参数的类型为字符串（即文件路径）
    # required表示这个参数是必需的，如果命令行没有提供该参数，程序将会报错。
    # 为该参数添加帮助信息，当用户在命令行输入 --help 时，会显示这条信息。

    # parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    
    # 解析命令行传递的参数，返回一个包含所有解析后参数的 Namespace 对象。
    args = parser.parse_args()
    # args.config 将会包含用户在命令行提供的配置文件路径。
    # 使用提供的路径打开配置文件。'r' 表示以只读模式打开文件。
    with open(args.config, 'r') as f:
        config = json.load(f)
    return config

# 根据配置文件动态加载不同数据集的功能。
# 它支持 MNIST 和 CIFAR-10 这两个数据集。
# 数据加载完成后，会对图像数据进行归一化处理，确保所有像素值都在 0 到 1 之间。
# 这种预处理方式有助于提升模型训练的性能和准确性。
# 通过在配置文件中指定不同的数据集名称，用户可以轻松切换使用不同的数据集进行实验。

# 动态加载数据集
def load_data(config):
    path='D:\Others\备份\Thesis\DesignOfOsc\AutoML\all.csv'
    all_data = pd.read_csv(path,sep=',',header=0,names=['id','smiles','confnum','homo','lumo','gap'])
    x=all_data.drop(columns='smiles')
    y=all_data['id']

    # 4. 数据集划分
    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

# 判断 config['dataset'] 是否为 mnist。如果是，加载 MNIST 数据集。
# if config['dataset'] == 'mnist':
# 加载 MNIST 数据集，这个数据集包含手写数字的图片，分为训练集和测试集。
#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 如果 config['dataset'] 是 cifar10，则加载 CIFAR-10 数据集。
# elif config['dataset'] == 'cifar10':
# 加载 CIFAR-10 数据集，这个数据集包含彩色图片，分为 10 个类别。
#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# else:
#     raise ValueError("Unsupported dataset")
# 数据集中的图像像素值通常在 0 到 255 之间。为了提高模型训练的效率和效果，通常会将这些像素值归一化到 0 到 1 之间。
# x_train, x_test = x_train / 255.0, x_test / 255.0
# return x_train, y_train, x_test, y_test


# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train = x_train[..., tf.newaxis]
# x_test = x_test[..., tf.newaxis]
# return (x_train, y_train), (x_test, y_test)

# 构建模型
def build_model(config, params):
    # 是 Keras 的一种顺序模型类型，表示模型中的层按顺序堆叠。
    model = tf.keras.Sequential()
        # 指定的层数，循环添加隐藏层。
    for i in range(config['model']['layers']):
        # 添加一个 Dense 层（全连接层），每层的神经元数量由 'layer_{i}_units'指定，激活函数使用 ReLU（Rectified Linear Unit）
        model.add(tf.keras.layers.Dense(params[f'layer_{i}_units'], activation='relu'))
        # 检查配置是否需要使用 dropout。
        if config['model']['dropout']:
            # 如果需要使用 dropout，则添加一个 dropout 层，丢弃率由 params['dropout_rate'] 指定。
            model.add(tf.keras.layers.Dropout(params['dropout_rate']))
    # 最后一层是输出层，包含 10 个神经元，使用 softmax 作为激活函数。这表明模型是用于多分类任务（如分类10种类别的任务）。
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # 用于配置训练过程。
    # 使用 Adam 优化器，学习率由 params['learning_rate'] 指定。
    # 损失函数使用稀疏的交叉熵，用于多分类任务。
    # 使用准确率（accuracy）作为评估指标。
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # 函数最终返回构建好的、经过编译的 Keras 模型。
    return model

# 使用训练集数据训练模型，并使用测试集数据评估模型性能。
# 根据训练的历史记录和评估结果，输出并返回测试集上的准确率。
# 自定义训练和评估流程

# 训练模型：根据 config 和 params 中的配置，使用训练集数据进行模型训练，并在每个训练轮次后使用一部分训练数据进行验证。
# 评估模型：在训练完成后，使用测试集数据评估模型的准确率。
# 记录并返回结果：记录测试集上的准确率并将其返回。
"""
model: 这是一个已经构建并编译好的 Keras 模型。
x_train, y_train: 训练集的数据和对应的标签。
x_test, y_test: 测试集的数据和对应的标签。
config: 包含模型训练的静态配置，例如训练的轮数（epochs）和验证集分割比例（validation split）。
params: 包含模型训练的可调节参数，例如批处理大小（batch size）。
"""
def train_and_evaluate(model, x_train, y_train, x_test, y_test, config, params):
    # fit 是 Keras 中用于训练模型的函数。该函数会执行指定轮数的训练，并返回一个包含训练历史信息的对象（history）。
    # 训练的总轮数，由 config 中的 training 配置指定。
    # 每次训练迭代中使用的样本数量，由 params 中的 batch_size 指定。
    # 用于验证的训练数据的比例。Keras 会自动将训练数据按照这个比例分割为训练数据和验证数据。
    history = model.fit(x_train, y_train, 
                        epochs=config['training']['epochs'], 
                        batch_size=params['batch_size'], 
                        validation_split=config['training']['validation_split'])

    # 用于在测试集上评估模型的性能。
    # 损失值和性能指标值。在这里，我们只关心准确率，因此通过 _ 忽略损失值，仅保留 accuracy。
    _, accuracy = model.evaluate(x_test, y_test)
    logger.info(f"Test accuracy: {accuracy}")
    return accuracy

def main():
    # 获取配置
    config = parse_args()
    
    # 从 NNI 获取超参数
    params = nni.get_next_parameter()
    logger.info(f"Received hyperparameters: {params}")
    
    # 加载数据集
    x_train, y_train, x_test, y_test = load_data(config)
    
    # 构建模型
    model = build_model(config, params)
    
    # 训练和评估模型
    accuracy = train_and_evaluate(model, x_train, y_train, x_test, y_test, config, params)
    
    # 报告结果给 NNI
    nni.report_final_result(accuracy)

if __name__ == '__main__':
    main()
