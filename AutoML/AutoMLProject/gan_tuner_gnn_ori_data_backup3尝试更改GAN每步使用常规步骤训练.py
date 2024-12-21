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
def smiles_to_graph(smiles, other, max_nodes):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), feature=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    node_features = np.array([G.nodes[i]['feature'] for i in range(len(G.nodes))]).reshape(-1, 1)
    
    tensor_list = []
    for item in other:
        if item == '-':
            item = 0
        tensor_list.append(float(item))
    additional_features = np.full((node_features.shape[0], len(tensor_list)), tensor_list)
    
    node_features = np.hstack([node_features, additional_features])
    adj_matrix = nx.to_numpy_array(G)

    num_nodes = adj_matrix.shape[0]
    if num_nodes < max_nodes:
        # Padding adjacency matrix with zeros
        adj_matrix = np.pad(adj_matrix, ((0, max_nodes - num_nodes), (0, max_nodes - num_nodes)), mode='constant', constant_values=0)
        # Padding node features with zeros
        node_features = np.pad(node_features, ((0, max_nodes - num_nodes), (0, 0)), mode='constant', constant_values=0)
    
    return [node_features, adj_matrix]

# Data loading function for Keras
def load_data(csv_file):
    df = pd.read_csv(csv_file, encoding='utf-8')

    smiles = df['SMILES_str'].to_numpy()
    other = df[['pce','e_lumo_alpha', 'e_gap_alpha', 'e_homo_alpha', 'jsc', 'voc', 'mass']].to_numpy()
    
    # Vectorize and normalize other features
    other = np.vectorize(lambda x: float(x) if x != '-' else 0)(other)
    scaler = StandardScaler()
    normalized_other = scaler.fit_transform(other)
    
    # Find the largest number of atoms across all molecules
    max_nodes = 0
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            num_atoms = mol.GetNumAtoms()
            max_nodes = max(max_nodes, num_atoms)
    
    # Prepare graph data
    graph_data_list = []
    for s, other_item in zip(smiles, normalized_other):
        graph_data = smiles_to_graph(s, other_item, max_nodes)
        if graph_data is not None:
            graph_data_list.append(graph_data)
    
    # Split data into train and test
    train_size = int(0.9 * len(graph_data_list))
    train_data = graph_data_list[:train_size]
    test_data = graph_data_list[train_size:]
    
    return train_data, test_data

class GAN(keras.Model):
    def __init__(self, discriminator, generator, input_dim1_shape, input_dim2_shape):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.input_dim1_shape = input_dim1_shape
        self.input_dim2_shape = input_dim2_shape
        self.input_dim1 = input_dim1_shape[1]
        self.input_dim2 = input_dim2_shape[1]
        
        
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        
        self.best_gen_lost = None
        self.best_disc_lost = None

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
    
    def call(self,inputs):
        return self.generator(inputs)

    def train_step(self, real_data):
        # 获取批次大小
        # s = tf.shape(real_data)
        # batch_size = tf.shape(real_data)[0]
        batch_size = self.input_dim1

        # 1. 生成器训练部分
        # 随机生成噪声向量,从标准正态分布中生成一个随机向量，表示潜在空间的样本。
        noise1 = tf.random.normal(shape=self.input_dim1_shape)  # 噪声维度
        noise2 = tf.random.normal(shape=self.input_dim2_shape)  # 噪声维度
        generated_data = self.generator([noise1,noise2], training=True)
        
        # 将生成的图数据和真实图数据进行拼接
        real_item1 = real_data[0][0][0]
        real_item2 = real_data[0][0][1]
        combined_node_features = tf.concat([generated_data[0], real_item1], axis=0)
        combined_adj_matrix = tf.concat([generated_data[1], real_item2], axis=0)
        combined_adj_matrix = tf.concat([combined_adj_matrix,tf.zeros((64, 32))], axis=1)

        # 生成标签：真实图数据标签为1，生成图数据标签为0
        labels1 = tf.concat([tf.ones((8, 1)), tf.zeros((8, 1))], axis=0)
        labels2 = tf.concat([tf.ones((32, 1)), tf.zeros((32, 1))], axis=0)

        # 向标签添加随机噪声（标签平滑）
        labels1 += 0.05 * tf.random.uniform(labels1.shape)
        labels2 += 0.05 * tf.random.uniform(labels2.shape)
        
        # 训练判别器部分
        with tf.GradientTape() as tape:
            # 判别器对组合图数据进行预测
            predictions = self.discriminator([combined_node_features, combined_adj_matrix])
            print(predictions)
            print(predictions.shape)
            d_loss = self.loss_fn(labels2, predictions)
        
        # 计算判别器的梯度并更新判别器的权重
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))



        # 2. 生成器训练部分
        # 重新生成潜在向量
        noise1 = tf.random.normal(shape=self.input_dim1_shape)  # 噪声维度
        noise2 = tf.random.normal(shape=self.input_dim2_shape)  # 噪声维度
        
        # 生成的标签是“误导性的”，表示我们希望生成的图被判别器判定为真实
        # labels2 = tf.concat([tf.ones((32, 1)), tf.zeros((32, 1))], axis=0)
        misleading_labels = tf.zeros((64, 1))

        # 训练生成器部分
        with tf.GradientTape() as tape:
            # 通过生成器生成新的图数据
            (gen_new_data,item) = self.generator([noise1,noise2])
            # gen_new_data = tf.pad(gen_new_data, [[0, 32], [0, 32]])
            print(item)
            print(item.shape)
            item = tf.concat([item,tf.zeros((32, 32))], axis=0)
            print(item.shape)
            item = tf.concat([item,tf.zeros((64, 32))], axis=1)
            print(item.shape)
            # 判别器评估生成的图数据，计算生成器的损失
            gen_new_data = tf.concat([gen_new_data, tf.zeros((32, 8))], axis=0)
            predictions = self.discriminator([gen_new_data,item])
            
            # 使用tf.pad进行填充
            predictions = tf.pad(predictions, [[0, 32], [0, 32]])
            #must have the same shape, received ((96, 33) vs (64, 1)).
            g_loss = self.loss_fn(misleading_labels, predictions)
        
        # 计算生成器的梯度并更新生成器的权重
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # 监控生成器和判别器的损失
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }
        # 1. 训练判别器
        # 随机生成噪声向量,从标准正态分布中生成一个随机向量，表示潜在空间的样本。
        noise1 = tf.random.normal(shape=self.input_dim1_shape)  # 噪声维度
        noise2 = tf.random.normal(shape=self.input_dim2_shape)  # 噪声维度
        generated_data = self.generator([noise1,noise2], training=True)

        labels = tf.concat([tf.ones])
        with tf.GradientTape() as tape_d:
            # 判别器评估真实分子指纹和生成的分子指纹
            real_item1 = real_data[0][0][0]
            real_item2 = real_data[0][0][1]
            real_output = self.discriminator([real_item1,real_item2], training=True)
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
            generated_data = self.generator([noise1,noise2], training=True)
            fake_output = self.discriminator(generated_data, training=True)
            
            # 生成器的损失：让生成的指纹尽可能“真实”
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

        grads_g = tape_g.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables))

        self.gen_loss_tracker.update_state(g_loss)  # 更新生成器损失指标

        #获得生成器最佳损失
        gen_loss = self.gen_loss_tracker.result()
        if(self.best_gen_lost==None or self.best_gen_lost>gen_loss):
            self.best_gen_lost = gen_loss
            
        #获得判别器最佳损失
        disc_loss = self.disc_loss_tracker.result()
        if(self.best_disc_lost==None or self.best_disc_lost>disc_loss):
            self.best_disc_lost = disc_loss
        # 返回每个epoch的损失值
        return {
            'd_loss': disc_loss,
            'g_loss': gen_loss,
        }
def custom_loss(y_true, y_pred, physical_pred):
    # 对抗损失
    adversarial_loss = K.binary_crossentropy(y_true, y_pred)
    # 物理损失
    print(physical_pred)
    gap = physical_pred['gap']  # 假设生成的分子特征中有 'gap'（能隙）
    target_gap = 2.0  # 你可以设定一个目标的能隙值，或者将目标设为一个范围

    phy_loss = K.abs(gap - target_gap) # 计算物理损失
    # 总损失：加权合并对抗损失和物理损失
    total_loss = adversarial_loss + 0.1 * phy_loss  # 物理损失的权重可以调整
    return total_loss

# from spektral.layers import GraphConv
import spektral

import keras.backend as K
# 使用Spektral库的GraphConv层
class HyperGAN(kt.HyperModel):

    def __init__(self,config,input_dim1_shape,input_dim2_shape):
        super(HyperGAN, self).__init__()
        self.input_dim1_shape = input_dim1_shape
        self.input_dim2_shape = input_dim2_shape
        self.input_dim1 = input_dim1_shape[1]
        self.input_dim2 = input_dim2_shape[1]
        
        self.gan_model = None
        
        # self.vocab_size = vocab_size
        self.config = config

    def make_generator_model(self, hp):

        """
        构建一个生成器模型，使用图卷积层（GraphConv）来处理图数据。
        """
        # 输入层：节点特征和邻接矩阵
        # node_features_input = layers.Input(shape=(self.input_dim,))
        # adjacency_input = layers.Input(shape=(self.input_dim,self.input_dim))  # 邻接矩阵（形状通常为(num_nodes, num_nodes)）
        node_features_input = layers.Input(shape=(self.input_dim1,))
        adjacency_input = layers.Input(shape=(self.input_dim2,))  # 邻接矩阵（形状通常为(num_nodes, num_nodes)）
        #("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        # 超参数搜索：图卷积层的隐藏维度
        # hidden_dim = hp.Int('hidden_dim', min_value=32, max_value=256, step=32)
        hidden_dim =get_hp_value(hp,self.config,"hidden_dim_gen")

        hidden_layer_num =get_hp_value(hp,self.config,"hidden_dim_gen")
        drop_rate = get_hp_value(hp,self.config,"dropout_rate")
        # 图卷积层，聚合节点特征
        x = spektral.layers.GCSConv(hidden_dim, activation='relu')([node_features_input, adjacency_input])
        for i in range(0,hidden_layer_num):
            x = spektral.layers.GCSConv(hidden_dim, activation='relu')([node_features_input, adjacency_input])
            x = layers.Dropout(drop_rate)(x)
            
        x = spektral.layers.GCSConv(hidden_dim, activation='relu')([x, adjacency_input])
        # x = spektral.layers.GlobalAttentionPool(hidden_dim)(x)

        # # 输出层，生成新的节点特征
        # output = layers.Dense(self.vocab_size, activation='tanh')(x)  # 生成与节点特征相同维度的输出

        # # 创建模型
        # model = models.Model(inputs=[node_features_input, adjacency_input], outputs=output)
        # model.compile(optimizer='adam', loss='mse')  # 使用MSE损失，假设输出是连续的节点特征
        
        
        # 输出层：生成节点特征和邻接矩阵
        output_node_features = layers.Dense(self.input_dim1, activation='tanh')(x)  # 生成节点特征
        output_adj = layers.Dense(self.input_dim2, activation='sigmoid')(x)  # 生成邻接矩阵
        # print(output_adj)
        # print(output_adj.shape)
        output_adj = layers.Reshape((self.input_dim2, ))(output_adj)

        # 创建模型
        model = models.Model(inputs=[node_features_input, adjacency_input], outputs=[output_node_features, output_adj])
        return model

    def make_discriminator_model(self,hp):
        """
        构建一个判别器模型，使用图卷积层来处理图数据。
        使用sigmoid来代表是否是真实数据，并使用binary_crossentropy（使用二分类交叉熵损失函数）来计算损失
        """
        # 输入层：节点特征和邻接矩阵
        node_features_input = layers.Input(shape=(self.input_dim1,))
        adjacency_input = layers.Input(shape=(2*self.input_dim2, ))  # 邻接矩阵

        # 超参数搜索：图卷积层的隐藏维度
        # hidden_dim = hp.Int('hidden_dim', min_value=32, max_value=256, step=32)
        hidden_dim =get_hp_value(hp,self.config,"hidden_dim_disc")

        # 图卷积层，聚合节点特征
        x = spektral.layers.GCSConv(hidden_dim, activation='relu')([node_features_input, adjacency_input])
        x = spektral.layers.GCSConv(hidden_dim, activation='relu')([x, adjacency_input])

        # 将图的特征聚合到单一的输出节点，使用全局池化操作
        # x = layers.GlobalAveragePooling1D()(x) #需要3维但只传了2维

        # 输出层：预测输入图数据是否为真实数据
        output = layers.Dense(1, activation='sigmoid')(x)  # 输出一个标量，0表示生成样本，1表示真实样本

        # 创建模型
        model = models.Model(inputs=[node_features_input, adjacency_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 使用二分类交叉熵损失函数

        return model

    def build(self, hp):
        self.generator = self.make_generator_model(hp)
        self.discriminator = self.make_discriminator_model(hp)

        model_gan = GAN(self.discriminator, self.generator, self.input_dim1_shape, self.input_dim2_shape)
        # print(model_gan)
        # 生成器优化器
        gen_optimizer =create_op(hp,self.config)
        # 判别器优化器
        disc_optimizer =  create_op(hp,self.config)

        # adam_optimizer = tf.keras.optimizers.Adam(1e-4)
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # 已重写compile
        model_gan.compile(gen_optimizer,disc_optimizer,binary_crossentropy)
        self.gan_model = model_gan
        return model_gan
    def score_function(self, data,n_split=10, eps=10**-16):
        """
        The scoring function to evaluate generator outputs.
        `z` is the input noise vector fed into the generator.
        """
        noise1 = tf.random.normal(shape=self.input_dim1_shape)  # 噪声维度
        noise2 = tf.random.normal(shape=self.input_dim2_shape)  # 噪声维度
        gen_data = self.generator([noise1,noise2])

        # 这里假设判别器输出接近1时表示真实，接近0时表示假
        discriminator_output = self.discriminator(gen_data)
        

        return tf.reduce_mean(discriminator_output)

    def fit(self, hp, model, x, **kwargs):
        model.fit(x,**kwargs)
        # self.gen_loss_tracker.result(),
        # res = self.score_function(x)
        res = self.gan_model.gen_loss_tracker.result()
        # Return a single float to minimize.
        return float(res)
    

def create_op(hp,config):
    
    disc_optimizer =get_hp_value(hp,config,"optimizer") 
    _learning_rate = get_hp_value(hp,config,"learning_rate")
    optimizer = None
    if disc_optimizer == 'adam':
        optimizer=tf.keras.optimizers.Adam(learning_rate=_learning_rate)
    else:
        optimizer=tf.keras.optimizers.SGD(learning_rate=_learning_rate)
    return optimizer

def get_hp_value(hp,config,key):
    _type = config[key]["_type"]
    m = getattr(hp,config[key]["_type"])
    if(_type=='Choice'):
        res = m(key,values=config[key]["_value"])
        return res
    elif(_type=='Float' or _type=='Int'):
        res = m(key,min_value=config[key]["_value"][0],max_value=config[key]["_value"][1],sampling=config[key].get("_mode"),step=config[key].get("_step"))
        return res
    elif(_type=='Boolean'):
        res = m(key)
        return res
    else:
        return None
    
def main():
    
    csv_file = 'D:\Project\ThesisProject\AutoML\data\moldata_part_test.csv'
    train_data,test_data  = load_data(csv_file)
    
    trials = 10
    
    search_space = {
        "optimizer":{
            "_type":"Choice",
            "_value":["adam"]   
        },
        "learning_rate":{
            "_type":"Float",
            "_mode":"log",
            "_value":[1e-5,1e-1]
        },
        "units":{
            "_type":"Int",
            "_mode":"linear",
            "_value":[1,5]
        },
        "hidden_dim_disc":{
            "_type":"Choice",
            "_value":[32,64,128,256]
        },
        "hidden_dim_gen":{
            "_type":"Choice",
            "_value":[32,64,128,256]
        },
        "hidden_layer":{
            "_type":"Int",
            "_mode":"log",
            "_value":[1,5]
        },
        "dropout_rate":{
            "_type":"Float",
            "_mode":"log",
            "_value":[0.001,0.5]
        },
        "max_trials":1,
        # "batch_size":32
    }
    
    input_dim1_shape = train_data[0][0].shape
    input_dim2_shape = train_data[0][1].shape
    tuner = kt.BayesianOptimization(
        # hypermodel=HyperGAN(search_space,train_data[0].shape[1],1030),
        hypermodel=HyperGAN(search_space,input_dim1_shape,input_dim2_shape),
        objective=kt.Objective("g_loss", "min"),
        max_trials=search_space["max_trials"],
        overwrite=True,
        directory="my_dir1111",
        project_name="custom_eval",
    )
    # TODO 1.callbacks回调，具体查看https://keras.io/guides/writing_your_own_callbacks/
    # TODO 2.tensorboard
    # TODO 3.load的数据可能有问题，传入的real参数怎么长这样
    # TODO 4.将参数改为动态
    tuner.search(
        x = train_data, epochs = 10
        )

    tuner.results_summary()
    # tuner.export_model()
    best_model = tuner.get_best_models()[0]
    
    import time
    now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values)

    best_model = tuner.get_best_models()[0]
    shape = [None, 8]
    placeholder_tensor1 = tf.TensorSpec(shape, dtype=tf.float32)
    shape = [None, 32]
    placeholder_tensor2 = tf.TensorSpec(shape, dtype=tf.float32)
    best_model.build([(None,8),(None,32)])  # 假设 X_train.shape[1] 是输入特征数
    best_model.summary()
    # best_model.generator.save(f'D:\Project\ThesisProject\AutoML\Project2\AutoKerasProject\model\{now}_model.h5py')
    best_model.generator.save(f'D:\Project\ThesisProject\AutoML\AutoMLProject\gan_model_{now}.h5py')
if __name__ == '__main__':
    main()
