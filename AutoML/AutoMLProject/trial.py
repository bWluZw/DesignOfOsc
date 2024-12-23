import os

import numpy as np
import tensorflow as tf

import autokeras as ak

import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers

# region 数据处理
# region 归一化数据处理a

class NormalizationHelper:
    def __init__(self):
        # 存储标准化和变换的参数
        self.mean = None
        self.std = None
        self.min_pce = None
        self.max_pce = None

    def z_score_standardization(self, data):
        """
        Z-score 标准化
        适用于 HOMO 和 LUMO 等特征。
        Args:
            data (numpy.ndarray or pd.Series): 输入数据
        Returns:
            normalized_data: 归一化后的数据
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        normalized_data = (data - self.mean) / self.std
        return normalized_data

    def inverse_z_score(self, normalized_data):
        """
        Z-score 反标准化
        Args:
            normalized_data (numpy.ndarray or pd.Series): 归一化后的数据
        Returns:
            original_data: 反归一化后的数据
        """
        original_data = normalized_data * self.std + self.mean
        return original_data

    def log_transform(self, data):
        """
        对数变换
        适用于 PCE 标签，防止数据偏态。
        Args:
            data (numpy.ndarray or pd.Series): 输入数据
        Returns:
            transformed_data: 对数变换后的数据
        """
        transformed_data = np.log1p(data)  # log(x + 1)
        return transformed_data

    def inverse_log_transform(self, transformed_data):
        """
        对数反变换
        Args:
            transformed_data (numpy.ndarray or pd.Series): 对数变换后的数据
        Returns:
            original_data: 反对数变换后的数据
        """
        original_data = np.expm1(transformed_data)  # exp(x) - 1
        return original_data

    def normalize_pce(self, data):
        """
        对 PCE 进行归一化处理，采用 Min-Max 归一化。
        Args:
            data (numpy.ndarray or pd.Series): 输入 PCE 数据
        Returns:
            normalized_data: 归一化后的 PCE 数据
        """
        self.min_pce = np.min(data)
        self.max_pce = np.max(data)
        normalized_data = (data - self.min_pce) / (self.max_pce - self.min_pce)
        return normalized_data

    def inverse_normalize_pce(self, normalized_data):
        """
        反归一化 PCE 数据。
        Args:
            normalized_data (numpy.ndarray or pd.Series): 归一化后的 PCE 数据
        Returns:
            original_data: 反归一化后的 PCE 数据
        """
        original_data = normalized_data * (self.max_pce - self.min_pce) + self.min_pce
        return original_data

# 定义一个转换函数
def to_float(val, default_value=np.nan):
    try:
        return float(val)
    except ValueError:
        return default_value


# endregion

from sklearn.preprocessing import StandardScaler
def load_data(csv_file, batch_size):
    df = pd.read_csv(csv_file,encoding='utf-8')
    #TODO 归一化

    pce = df['pce'].to_numpy().astype(np.float32)
    smiles = df['SMILES_str']
    # other = df[['e_lumo_alpha','e_gap_alpha','e_homo_alpha','jsc','voc','mass']].to_numpy().astype(np.float32)
    other = df[['e_lumo_alpha','e_gap_alpha','e_homo_alpha','mass']].to_numpy().astype(np.float32)

    scaler = StandardScaler()
    other = scaler.fit_transform(other)

    f_smiles = []

    for s in smiles:
        fingerprint = None
        try:
            mol = Chem.MolFromSmiles(s)
            # 计算分子指纹 
            mol = Chem.AddHs(mol=mol)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

            f_smiles.append(fingerprint)
        except Exception as e:
            print(str(e))
            continue
    f_smiles = np.array(f_smiles)
    # df_f_smiles = pd.DataFrame({'smiles_f':f_smiles})
    X = np.hstack([f_smiles, other])
    
    y = pce

    input_arr = X
    out_arr = y
    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(input_arr,out_arr, test_size=0.2, random_state=42)

    return X_train,X_test,y_train,y_test#,input_arr[0].shape[0],x_feature_num


# endregion


# 创建一个自定义超参数空间的函数
def build_hypermodel(X_train):
    # 定义一个基本的神经网络架构
    model = ak.Sequential()
    
    # 根据超参数调节网络层数和神经元数
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
    
    # 超参数空间：选择隐藏层的数量和大小
    model.add(layers.Dense(units=ak.HyperParameter('units', min_value=64, max_value=512, step=64), activation='relu'))
    
    # 添加更多层并使用超参数搜索不同层数和大小
    model.add(layers.Dense(units=ak.HyperParameter('units', min_value=64, max_value=512, step=64), activation='relu'))
    
    # 输出层
    model.add(layers.Dense(1))  # 只需要一个输出节点，预测PCE
    
    # 编译模型，选择优化器和损失函数
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def main():
    search_space = {
        # 超参数：批大小（Batch Size）
        "batch_size": [16, 32, 64, 128],  # 可以选择批大小：16, 32, 64 或 128
        # 超参数：学习率（Learning Rate）
        "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  # 学习率范围从1e-5到1e-1
        # 超参数：训练周期（Epochs）
        "epochs": [10, 20, 50, 100],  # 训练周期的选择范围
        # 超参数：Dropout率（Dropout Rate）
        "dropout_rate": [0.2, 0.3, 0.5, 0.7],  # Dropout率的选择范围
        # 超参数：网络层数（Number of Layers）
        "num_layers": [2, 3, 4, 5],  # 网络的层数，可以选择从2层到5层
        # 超参数：每层的节点数（Number of Nodes per Layer）
        "num_units": [64, 128, 256, 512],  # 每层节点数的范围
        # 超参数：激活函数（Activation Function）
        "activation": ['relu', 'tanh', 'sigmoid'],  # 激活函数可以是 ReLU、tanh 或 sigmoid
        # 超参数：优化器（Optimizer）
        "optimizer": ['adam', 'sgd', 'rmsprop'],  # 可以选择的优化器：Adam、SGD、RMSProp
        # 超参数：L2 正则化（L2 Regularization）
        "l2_regularization": [1e-5, 1e-4, 1e-3],  # L2正则化的不同值
    }

    csv_file = 'D:\Project\ThesisProject\AutoML\data\moldata_part_test.csv'
    X_train, X_test, y_train, y_test = load_data(csv_file, search_space['batch_size'])

    # from keras.models import load_model
    # loaded_model = load_model(
    #     r"D:\Project\ThesisProject\auto_model\best_model", custom_objects=ak.CUSTOM_OBJECTS
    # )

    # print(type(loaded_model))

    # predicted_y = loaded_model.evaluate(X_test,y_test)
    # print(predicted_y)

    # 创建 AutoModel 对象进行回归任务
    search_space = {
        "batch_size": [16, 32, 64, 128],
        "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        # 其他超参数设置
    }
    
    import time
        
    now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    LOG_DIR = r'AutoML\models\gan_model\tuner'+f"\gan_keras_model_{now}" 
    regressor = ak.AutoModel(
        inputs=ak.StructuredDataInput(),  # 输入为结构化数据
        outputs=ak.RegressionHead(),  # 输出为回归任务
        max_trials=1,  # 超参数搜索的最大次数
        objective="val_loss",  # 优化目标为验证损失
        overwrite=True,  # 如果已有模型则覆盖
        tuner='bayesian',
        project_name='pce_project_model',
        directory=LOG_DIR
    )
    ak.BayesianOptimization
    try:
        
        # 开始超参数调优
        regressor.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))
    except Exception as e:
        print(e)
    finally:
        # 输出最佳模型
        best_model = regressor.export_model()



        best_model.save(f'D:\Project\ThesisProject\AutoML\models\pce_model\best_model\{now}_pce_model')
        # best_model.save(f'D:\Project\ThesisProject\AutoML\Project2\AutoKerasProject\model\{now}_model.h5py')
        # best_model.save(f'D:\Project\ThesisProject\AutoML\Project2\AutoKerasProject\model\{now}_model.h5py')
        # 评估最佳模型
        loss, mae = best_model.evaluate(X_test, y_test)
        print("最佳模型评估结果: Loss = {}, MAE = {}".format(loss, mae))

        # 使用最佳模型进行预测
        predictions = best_model.predict(X_test)
        print("部分预测结果:", predictions[:5])


if __name__ == '__main__':
    main()
