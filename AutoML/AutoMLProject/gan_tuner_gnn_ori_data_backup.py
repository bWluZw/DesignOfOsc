import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np
from keras import layers, models

import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler

from rdkit.Chem import AllChem
from rdkit import Chem

import networkx as nx
from sklearn.model_selection import train_test_split

# 示例：从图的节点和边恢复分子并转换为SMILES
def graph_to_smiles(node_features, adjacency_matrix):
    # 创建一个空的RDKit分子对象
    mol = Chem.RWMol()

    # 添加原子（假设节点特征是原子序数）
    atom_map = {}
    for idx, atom_feature in enumerate(node_features):
        atom = Chem.Atom(int(atom_feature[0]))  # 原子序数为节点特征
        atom_idx = mol.AddAtom(atom)
        atom_map[idx] = atom_idx

    # 添加键（根据邻接矩阵）
    num_atoms = len(node_features)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):  # 遍历上三角矩阵，避免重复
            if adjacency_matrix[i, j] != 0:  # 如果有边
                bond_order = 1  # 可以根据需要选择键级，例如1表示单键
                mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.SINGLE)

    # 使用RDKit将分子对象转为SMILES字符串
    mol = mol.GetMol()  # 转换为不可修改的分子对象
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    else:
        return None

# Helper function to convert SMILES to graph features
def smiles_to_graph(smiles, other):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add atom nodes with atomic number as feature
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), feature=atom.GetAtomicNum())
    
    # Add bonds as edges
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    
    # Get node features (atomic numbers)
    node_features = np.array([G.nodes[i]['feature'] for i in range(len(G.nodes))]).reshape(-1, 1)
    
    # Add additional features
    tensor_list = []
    for item in other:
        if item == '-':
            item = 0
        tensor_list.append(float(item))
    additional_features = np.full((node_features.shape[0], len(tensor_list)), tensor_list)
    
    node_features = np.hstack([node_features, additional_features])
    
    graph_data = [node_features,nx.to_numpy_array(G)]
    
    return graph_data

# Data loading function for Keras
def load_data(csv_file):
    df = pd.read_csv(csv_file, encoding='utf-8')

    smiles = df['SMILES_str'].to_numpy()
    other = df[['pce','e_lumo_alpha', 'e_gap_alpha', 'e_homo_alpha', 'jsc', 'voc', 'mass']].to_numpy()
    
    # Vectorize and normalize
    other = np.vectorize(lambda x: float(x) if x != '-' else 0)(other)
    scaler = StandardScaler()
    normalized_other = scaler.fit_transform(other)
    
    # Prepare data
    graph_data_list = []
    for s, other_item in zip(smiles, normalized_other):
        graph_data = smiles_to_graph(s, other_item)
        if graph_data is not None:
            graph_data_list.append(graph_data)
    
    # Split data into train and test
    train_size = int(0.9 * len(graph_data_list))
    train_data = graph_data_list[:train_size]
    test_data = graph_data_list[train_size:]
    
    return train_data, test_data

def load_data(csv_file):
    df = pd.read_csv(csv_file,encoding='utf-8')
    #TODO 归一化
    pce = df['pce'].to_numpy().astype(np.float32)
    smiles = df['SMILES_str']
    other = df[['e_lumo_alpha','e_gap_alpha','e_homo_alpha','jsc','voc','mass']].to_numpy().astype(np.float32)

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
    X = np.hstack([f_smiles, other])
    
    y = pce
    input_arr = X
    out_arr = y
    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(input_arr,out_arr, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test#,input_arr[0].shape[0],x_feature_num

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
    
    def call(self,inputs):
        print(inputs)
        return self.generator(inputs)

    def train_step(self, real_data):
        # real_images = batch_data[0]  # Real images from the dataset
        # batch_size = tf.shape(real_images)[0]
        batch_size = real_data.shape[0]

        # 1. 训练判别器
        # 随机生成噪声向量
        noise = tf.random.normal(shape=(batch_size,self.latent_dim))  # 假设噪声维度为66
        generated_data = self.generator(noise, training=True)

        with tf.GradientTape() as tape_d:
            # 判别器评估真实分子指纹和生成的分子指纹
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            
            # 判别器损失：真实的指纹标签为1，假的为0
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = real_loss + fake_loss

        grads_d = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))

        self.disc_loss_tracker.update_state(d_loss)  # 更新判别器损失指标

        # 2. 训练生成器
        with tf.GradientTape() as tape_g:
            generated_data = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            
            # 生成器的损失：让生成的指纹尽可能“真实”
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

        grads_g = tape_g.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables))

        self.gen_loss_tracker.update_state(g_loss)  # 更新生成器损失指标

        # 返回每个epoch的损失值
        return {
            'd_loss': self.disc_loss_tracker.result(),
            'g_loss': self.gen_loss_tracker.result(),
        }

class HyperGAN(kt.HyperModel):

    def __init__(self,config,latent_dim,vocab_size):
        super(HyperGAN, self).__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.config = config

    def make_generator_model(self, hp):

        x_model = models.Sequential()

        # model = tf.keras.Sequential()
        
        x_model.add(layers.InputLayer(input_shape=(self.latent_dim,)))
        # units_1 = hp.Int('units_1', min_value=64, max_value=512, step=64)
        # units_2 = hp.Int('units_2', min_value=64, max_value=512, step=64)

        # model.add(layers.Dense(units_1, activation='relu'))
        # model.add(layers.Dense(units_2, activation='relu'))
        x_model.add(
            layers.Dense(self.vocab_size, activation="tanh")
        )  # 假设 SMILES 是通过向量表示
        node_features = layers.Reshape((num_nodes, num_features))(x)
        return x_model

    def make_discriminator_model(self,hp):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(self.vocab_size,)))

        units_1 = hp.Int('units_1', min_value=64, max_value=512, step=64)
        units_2 = hp.Int('units_2', min_value=64, max_value=512, step=64)

        model.add(layers.Dense(units_1, activation='relu'))
        model.add(layers.Dense(units_2, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))  # 判别真假样本

        return model

    def build(self, hp):
        self.generator = self.make_generator_model(hp)
        self.discriminator = self.make_discriminator_model(hp)

        model_gan = GAN(self.discriminator, self.generator, self.latent_dim)
        print(model_gan)
        # 生成器优化器
        gen_optimizer =create_op(hp,self.config)
        # 判别器优化器
        disc_optimizer =  create_op(hp,self.config)

        # adam_optimizer = tf.keras.optimizers.Adam(1e-4)
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # 已重写compile
        model_gan.compile(gen_optimizer,disc_optimizer,binary_crossentropy)
        return model_gan
    def score_function(self, z):
        """
        The scoring function to evaluate generator outputs.
        `z` is the input noise vector fed into the generator.
        """
        # 1. Generate a fake sample
        batch_size = z.shape[0]
        noise = tf.random.normal(shape=(batch_size,self.latent_dim))  # 假设噪声维度为66
        gen_data = self.generator(noise)

        # 2. Use discriminator to classify the fake image as real or fake
        # 这里假设判别器输出接近1时表示真实，接近0时表示假
        discriminator_output = self.discriminator(gen_data)

        # 3. The score can be the discriminator output, or you can use other metrics
        return tf.reduce_mean(discriminator_output)

    def fit(self, hp, model, x, **kwargs):
        model.fit(x,**kwargs)
        res = self.score_function(x)
        # Return a single float to minimize.
        return float(res)

def create_op(hp,config):
    
    disc_optimizer = hp.Choice('optimizer', values=config['optimizer'])
    # “linear”、“log”、“reverse_log”
    _learning_rate = hp.Float(
                    "learning_rate",
                    min_value=config["learning_rate"]["_value"][0],
                    max_value=config["learning_rate"]["_value"][1],
                    sampling=config["learning_rate"]["_mode"],
                )
    optimizer = None
    if disc_optimizer == 'adam':
        optimizer=tf.keras.optimizers.Adam(learning_rate=_learning_rate)
    else:
        optimizer=tf.keras.optimizers.SGD(learning_rate=_learning_rate)
    return optimizer

def main():
    
    csv_file = 'D:\Project\ThesisProject\AutoML\data\moldata_part_test.csv'
    train_data,test_data  = load_data(csv_file)
    
    trials = 10
    
    search_space = {
        "optimizer":["adam"],
        "learning_rate":{
            "_type":"float",
            "_mode":"log",
            "_value":[1e-5,1e-1]
        },
        "units":{
            "_type":"int",
            "_mode":"linear",
            "_value":[1,5]
        },
        "max_trials":1,
        # "batch_size":32
    }
    

    tuner = kt.BayesianOptimization(
        hypermodel=HyperGAN(search_space,X_train.shape[1],1030),
        objective=kt.Objective("g_loss", "min"),
        max_trials=search_space["max_trials"],
        overwrite=True,
        directory="my_dir1111",
        project_name="custom_eval",
    )
    # TODO 1.callbacks回调，具体查看https://keras.io/guides/writing_your_own_callbacks/
    # TODO 2.tensorboard
    tuner.search(
        x = X_train, epochs = 10
        )

    tuner.results_summary()
    # tuner.export_model()
    best_model = tuner.get_best_models()[0]
    
    import time
    now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values)

    best_model = tuner.get_best_models()[0]
    best_model.build((None, X_train.shape[1]))  # 假设 X_train.shape[1] 是输入特征数
    best_model.summary()
    # best_model.generator.save(f'D:\Project\ThesisProject\AutoML\Project2\AutoKerasProject\model\{now}_model.h5py')
    best_model.generator.save(f'D:\Project\ThesisProject\AutoML\AutoMLProject\gan_model_{now}.h5py')
if __name__ == '__main__':
    main()
