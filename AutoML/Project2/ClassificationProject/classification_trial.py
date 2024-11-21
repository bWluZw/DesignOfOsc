
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

def parse_args():

    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    # args.config 将会包含用户在命令行提供的配置文件路径。
    # 使用提供的路径打开配置文件。'r' 表示以只读模式打开文件。
    with open(args.config, 'r') as f:
        config = json.load(f)
    return config

def load_data(config):
    path='D:\Project\ThesisProject\AutoML\data\SMILES_donors_and_NFAs.csv'
    all_data = pd.read_csv(path,sep=',',header=0)

    homo_lumo_list = []
    features_list = []
    for (index,item) in all_data.iterrows():
        smiles = item['smiles']
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        features = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        features_list.append(features)
        
        scaler = StandardScaler()
        # 归一化 homo 和 lumo
        homo_lumo = np.array([item['homo'], item['lumo']]).reshape(1, -1)  # 确保是二维数组
        homo_lumo_normalized = scaler.fit_transform(homo_lumo).flatten()  # 再展平为一维
        homo_lumo_list.append(homo_lumo_normalized)
        
    features_array = np.array(features_list)
    homo_lumo_array = np.array(homo_lumo_list)
    x = np.concatenate([features_array, homo_lumo_array], axis=1)
    y = all_data['mark'].values  # 0为给体，1为受体

    # 4. 数据集划分
    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# 构建模型
def build_model(config, params):
    input_dim = config.get('input_dim', 2048 + 2)  # 默认2048位指纹向量 + homo/lumo特征
    hidden_dim = params.get('hidden_dim', 128)  # 默认隐藏层单元数
    output_dim = config.get('output_dim', 2)  # 默认输出维度为2（分类）
    activation = params.get('activation', 'relu')  # 激活函数类型
    
    model = SimpleClassifier(input_dim, hidden_dim, output_dim, activation)
    return model

def train_and_evaluate(model, x_train, y_train, x_test, y_test, config, params):
    # 转换数据为 PyTorch 的张量
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 超参数
    learning_rate = params.get('learning_rate', 0.001)
    num_epochs = params.get('num_epochs', 20)
    batch_size = params.get('batch_size', 32)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
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


