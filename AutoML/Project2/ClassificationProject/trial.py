from collections import Counter
import argparse
from sklearn.model_selection import train_test_split
import json
import logging
import nni
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = F.relu if activation == 'relu' else F.sigmoid

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# 日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('custom_trial_trial.py')

# 命令行参数解析：利用 argparse 解析命令行参数，获取配置文件的路径。
# 配置文件加载：通过 json 模块将指定路径的 JSON 配置文件加载为 Python 字典。
# 返回配置：将解析后的配置文件内容作为字典返回，供后续训练、评估等过程使用。
# 解析命令行参数或配置文件
def parse_args():
    try:
        # 创建一个 ArgumentParser 对象，该对象用于解析命令行参数。
        parser = argparse.ArgumentParser()
        # 为解析器添加一个命令行参数。
        # '--config'这是命令行参数的名称，表示配置文件的路径
        # type指定该参数的类型为字符串（即文件路径）
        # required表示这个参数是必需的，如果命令行没有提供该参数，程序将会报错。
        # 为该参数添加帮助信息，当用户在命令行输入 --help 时，会显示这条信息。

        parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
        
        # 解析命令行传递的参数，返回一个包含所有解析后参数的 Namespace 对象。
        args = parser.parse_args()
        # args.config 将会包含用户在命令行提供的配置文件路径。
        # 使用提供的路径打开配置文件。'r' 表示以只读模式打开文件。
        with open(args.config, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"parse_args: {e}")
        return None

# 根据配置文件动态加载不同数据集的功能。
# 它支持 MNIST 和 CIFAR-10 这两个数据集。
# 数据加载完成后，会对图像数据进行归一化处理，确保所有像素值都在 0 到 1 之间。
# 这种预处理方式有助于提升模型训练的性能和准确性。
# 通过在配置文件中指定不同的数据集名称，用户可以轻松切换使用不同的数据集进行实验。

# 动态加载数据集
def load_data(config):
    path='D:\Project\ThesisProject\AutoML\data\SMILES_donors_and_NFAs.csv'
    all_data = pd.read_csv(path,sep=',',header=0)
    
    # res_data = pd.DataFrame(columns=['name','features','homo_lumo','nlumo','ngap'], dtype='object')
    
    # SMILES转化为分子特征向量
    # 存储分子特征和归一化后的 HOMO/LUMO 特征
    homo_lumo_list = []
    features_list = []

    for _, item in all_data.iterrows():
        smiles = item['smiles']
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        # 获取指纹特征并转换为 NumPy 数组
        features = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        features_list.append(features)

        # 归一化 HOMO 和 LUMO
        homo_lumo = np.array([item['homo'], item['lumo']])  # 一维数组
        homo_lumo_list.append(homo_lumo)

    # 转换为 NumPy 数组
    features_array = np.vstack(features_list)  # shape: (n_samples, 2048)
    homo_lumo_array = np.vstack(homo_lumo_list)  # shape: (n_samples, 2)

    # 合并特征
    x = np.concatenate([features_array, homo_lumo_array], axis=1)  # shape: (n_samples, 2050)
    y = all_data['mark'].values  # shape: (n_samples,)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_test, y_test
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
    try:
        input_dim = params.get('input_dim', 2048 + 2)  # 默认2048位指纹向量 + homo/lumo特征
        hidden_dim = params.get('hidden_dim', 128)  # 默认隐藏层单元数
        output_dim = params.get('output_dim', 2)  # 默认输出维度为2（分类）
        activation = params.get('activation', 'relu')  # 激活函数类型
        
        model = SimpleClassifier(input_dim, hidden_dim, output_dim, activation)
        return model
    except Exception as e:
        logger.error(f"main: {e}")
        

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
    try:
        # 转换数据为 PyTorch 的张量
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # 超参数
        learning_rate = params.get('learning_rate', 0.001)
        num_epochs = params.get('num_epochs', 20)
        batch_size = params.get('batch_size', 32)
        
        # logger.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        # logger.info(f"Batch size: {batch_size}, Number of samples: {x_train.size(0)}")
        print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        print(f"Batch size: {batch_size}, Number of samples: {x_train.size(0)}")

        # 计算类别权重并创建损失函数
        class_counts = Counter(y_train.numpy())  # 使用 Counter 统计类别分布
        total_count = sum(class_counts.values())
        class_weights = [total_count / class_counts[c] for c in sorted(class_counts.keys())]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 模型训练
        model.train()
        for epoch in range(num_epochs):
            permutation = torch.randperm(x_train.size(0))
            for i in range(0, x_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = x_train[indices], y_train[indices]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

        # 模型评估
        model.eval()
        with torch.no_grad():
            outputs = model(x_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).float().mean().item()

        logger.info(f"Test Accuracy: {accuracy}")
        return accuracy,model
    except Exception as e:
        return 0
    # # fit 是 Keras 中用于训练模型的函数。该函数会执行指定轮数的训练，并返回一个包含训练历史信息的对象（history）。
    # # 训练的总轮数，由 config 中的 training 配置指定。
    # # 每次训练迭代中使用的样本数量，由 params 中的 batch_size 指定。
    # # 用于验证的训练数据的比例。Keras 会自动将训练数据按照这个比例分割为训练数据和验证数据。
    # history = model.fit(x_train, y_train, 
    #                     epochs=config['training']['epochs'], 
    #                     batch_size=params['batch_size'], 
    #                     validation_split=config['training']['validation_split'])

    # # 用于在测试集上评估模型的性能。
    # # 损失值和性能指标值。在这里，我们只关心准确率，因此通过 _ 忽略损失值，仅保留 accuracy。
    # _, accuracy = model.evaluate(x_test, y_test)
    # logger.info(f"Test accuracy: {accuracy}")
    # return accuracy

def main():
    # 获取配置
    # config = parse_args()
    config = None
    
    # 从 NNI 获取超参数
    # params = nni.get_next_parameter()
    json_str = '{"hidden_dim": 128,"activation": "sigmoid","learning_rate": 0.012792640304103993,"num_epochs": 35,"batch_size": 32}'
    params = json.loads(json_str)
    logger.info(f"Received hyperparameters: {params}")
    
    # 加载数据集
    x_train, y_train, x_test, y_test = load_data(config)
    
    # 构建模型
    model = build_model(config, params)
    
    # 训练和评估模型
    accuracy,model = train_and_evaluate(model, x_train, y_train, x_test, y_test, config, params)
    
    torch.save(model,r'D:\Project\ThesisProject\AutoML\Project2\ClassificationProject\classification_model')
    
    print(str(accuracy))
    # 报告结果给 NNI
    # nni.report_final_result(accuracy)

if __name__ == '__main__':
    main()


