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
    try:
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
    except:
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
        
        self.gen_last_loss = None
        self.disc_last_loss = None

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
    
    def discriminator_loss(self,real_output, fake_output):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output[0]), real_output[0])
        print(real_loss)
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
        
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    def generator_loss(fake_output):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

    def score_loss(self, current_loss, prev_loss, target_loss=0, epsilon=1e-6):
        """
        根据当前损失和历史损失计算评分,越接近1越好
        :param current_loss: 当前损失
        :param prev_loss: 上一次的损失
        :param target_loss: 目标损失值，默认为 0（理想情况下）
        :param epsilon: 避免除零错误
        :return: 评分（0-1之间）
        """
        if prev_loss is None:
            # 如果没有历史损失，初始化评分为 current_loss
            return current_loss
        
        # 计算损失变化率
        loss_diff = abs(current_loss - prev_loss)
        rate_score = max(0, 1 - (loss_diff / (prev_loss + epsilon)))
        
        # 计算当前损失接近目标损失的程度
        target_score = max(0, 1 - abs(current_loss - target_loss) / (target_loss + epsilon))
        
        # 计算损失稳定性（小的波动更好）
        stability_score = max(0, 1 - (loss_diff / (prev_loss + epsilon)))
        
        # 通过加权平均的方式得到综合评分
        total_score = (rate_score + target_score + stability_score) / 3
        return total_score

    def train_step(self, real_data):

        batch_size = tf.shape(real_data[0]).numpy()
        batch_size = int(batch_size[0])
        
        # TODO all_data数据格式，批次数据是怎样训练的，是一个一个for循环吗:是一次一次调这个函数，每一个数据就是一步
        # TODO 训练速度
        # TODO fit函数下的评分机制
        # fit

        g_data_tensor = tf.reshape(real_data[0], (batch_size, 32, 8))
        d_data_tensor = tf.reshape(real_data[1], (batch_size, 32, 32))
        
        g_data_list = tf.unstack(g_data_tensor, axis=0)
        d_data_list = tf.unstack(d_data_tensor, axis=0)
        
        for g_data,d_data in zip(g_data_list,d_data_list):
            
            # 随机生成噪声向量,从标准正态分布中生成一个随机向量，表示潜在空间的样本。
            noise1 = tf.random.normal(shape=self.input_dim1_shape)  # 噪声维度
            noise2 = tf.random.normal(shape=self.input_dim2_shape)  # 噪声维度
            generated_data = self.generator([noise1,noise2], training=True)

            # 1. 训练判别器
            with tf.GradientTape() as tape_d:
                # 判别器评估真实分子指纹和生成的分子指纹
                real_output = self.discriminator([g_data,d_data], training=True)
                fake_output = self.discriminator(generated_data, training=True)
                # print(test.numpy())
                real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
                fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
                d_loss = real_loss + fake_loss
            #计算梯度，并应用他们
            grads_d = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))
            # 更新判别器损失指标
            self.disc_loss_tracker.update_state(d_loss) 

            # 2. 训练生成器
            with tf.GradientTape() as tape_g:
                generated_data = self.generator([noise1,noise2], training=True)
                fake_output = self.discriminator(generated_data, training=True)
                
                # 生成器的损失：让生成的指纹尽可能“真实”
                g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            #计算梯度，并应用他们
            grads_g = tape_g.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables))
            # 更新生成器损失指标
            self.gen_loss_tracker.update_state(g_loss)  

            #获得生成器最佳损失
            gen_last_loss = self.gen_loss_tracker.result()

            #获得判别器最佳损失
            disc_last_loss = self.disc_loss_tracker.result()

            # 返回每个epoch的损失值
        gen_loss_float = float(self.gen_loss_tracker.result().numpy())
        disc_loss_float = float(self.disc_loss_tracker.result().numpy())
        
        gen_score = self.score_loss(gen_loss_float,self.gen_last_loss)
        dics_score = self.score_loss(disc_loss_float,self.disc_last_loss)
        
        total_score = gen_score+dics_score
                
        # for callback in callbacks:
        # # The "my_metric" is the objective passed to the tuner.
        #     callback.on_epoch_end(1, logs={"my_metric": 1})
        self.gen_last_loss = gen_loss_float
        self.disc_last_loss = disc_loss_float
        return {
            'd_loss_score': gen_last_loss,
            'g_loss_score': disc_last_loss,
            'total_score':total_score,
            'gen_score':gen_score,
            'dics_score':dics_score,
        }
        
class CustomScore():
    def __init__(self,gen_loss_list,disc_loss_list,node_features, adjacency_matrix):
        super(CustomScore, self).__init__()
        self.gen_loss_list = gen_loss_list
        self.disc_loss_list = disc_loss_list
        self.node_features = node_features
        self.adjacency_matrix = adjacency_matrix
        

    def get_mult_socre(self):
        gen_loss_avg = self.compute_loss_avg_score(self.gen_loss_list)
        disc_loss_avg = self.compute_loss_avg_score(self.disc_loss_list)
        
        gen_loss_change_rate = self.compute_loss_change_rate_score(self.gen_loss_list)
        disc_loss_change_rate = self.compute_loss_change_rate_score(self.disc_loss_list)
        
        gen_loss_stability = self.compute_loss_stability_score(self.gen_loss_list)
        disc_loss_stability = self.compute_loss_stability_score(self.disc_loss_list)
        
        is_comply_with_physical_laws = self.physical_score(self.node_features, self.adjacency_matrix)
        res_score = 0.2*gen_loss_avg+0.2*disc_loss_avg+0.1*gen_loss_change_rate+0.1*disc_loss_change_rate+0.1*gen_loss_stability+0.1*disc_loss_stability+0.2*is_comply_with_physical_laws
        return float(res_score)
    def compute_loss_avg_score(losses, window=100):
        # 计算损失的改进程度
        # 通过比较生成器和判别器的 最近的损失平均值 和 初期的损失平均值 来量化它们的改善程度。损失值的下降意味着模型在改进，因此评分越高越好。
        losses = np.array(losses)
        avg_loss_recent = np.mean(losses[-window:])
        avg_loss_initial = np.mean(losses[:window])
        
        # 归一化评分，损失越低改进越大，评分越高
        score = max(0, 1 - (avg_loss_recent / (avg_loss_initial + 1e-6)))  # 加1e-6避免除零错误
        return score
    def compute_loss_change_rate_score(losses, window=10):
        # 计算损失值的变化率
        # 如果 生成器损失变化率 较小，评分接近 1，表示生成器稳定并在不断优化。
        diff = np.diff(losses[-window:])
        avg_change_rate = np.mean(np.abs(diff))
        
        # 归一化评分，变化率越小评分越高
        score = max(0, 1 - avg_change_rate)  # 变化率越小，模型训练越好
        return score
    def compute_loss_stability_score(losses, threshold=0.01, window=100):
        # 计算损失值的波动幅度
        #波动幅度 越小，评分越高，表示模型稳定。
        diff = np.diff(losses[-window:])
        max_diff = np.max(np.abs(diff))  # 计算最大波动幅度
        
        # 归一化评分，波动幅度越小评分越高
        score = max(0, 1 - (max_diff / threshold))  # 波动幅度小于阈值时评分高
        return score
    
    def physical_score(node_features, adjacency_matrix):
        smiles = graph_to_smiles(node_features,adjacency_matrix)
        if(smiles==None):
            return 0
        else:
            return 1

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
        adjacency_input = layers.Input(shape=(self.input_dim2, ))  # 邻接矩阵

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
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 使用二分类交叉熵损失函数

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

        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # 已重写compile
        model_gan.compile(gen_optimizer,disc_optimizer,binary_crossentropy)
        self.gan_model = model_gan
        return model_gan

    def fit(self, hp, model, x,  *args, **kwargs):
        batch_size = get_hp_value(hp,self.config,"batch_size")
        x = x.batch(batch_size)

        res = model.fit(x,*args,**kwargs)

        
        # print(f'fit:{len(self.gan_model.gen_loss_list)}')
        # noise1 = tf.random.normal(shape=self.input_dim1_shape)  # 噪声维度
        # noise2 = tf.random.normal(shape=self.input_dim2_shape)  # 噪声维度
        # gen_data = self.generator([noise1,noise2])
        # custom_score = CustomScore(res.history.g_loss_score,res.history.d_loss_score,gen_data[0],gen_data[1])
        # socre = custom_score.get_mult_socre()
        # return 1-float(custom_score)
        # res = self.gan_model.gen_loss_tracker.result()
        #total_score
        # return res.history
        return res.history
    

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
        "batch_size":{
            "_type":"Choice",
            "_value":[32,64,96,128]
        },
        "max_trials":1,
        # "batch_size":32
    }

    
    import time
    now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    # # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                 save_weights_only=True,
    #                                                 verbose=1)
    
    tf.config.experimental_run_functions_eagerly(True)
    input_dim1_shape = train_data[0][0].shape
    input_dim2_shape = train_data[0][1].shape
    LOG_DIR = f"gan_keras_model_{now}" 
    # LOG_DIR = os.path.normpath('C:/')
    tuner = kt.BayesianOptimization(
        # hypermodel=HyperGAN(search_space,train_data[0].shape[1],1030),
        hypermodel=HyperGAN(search_space,input_dim1_shape,input_dim2_shape),
        objective=kt.Objective("total_score", "max"),
        max_trials=search_space["max_trials"],
        overwrite=True,
        
        directory=LOG_DIR,
        project_name="custom_eval",
    )
    
    # TODO 1.callbacks回调，具体查看https://keras.io/guides/writing_your_own_callbacks/
    # TODO 2.tensorboard
    # TODO 3.load的数据可能有问题，传入的real参数怎么长这样
    # TODO 4.将参数改为动态
    
    item1_list = []
    item2_list = []
    for item in train_data:
        item1_list.append(item[0])
        item2_list.append(item[1])

    dataset3 = tf.data.Dataset.from_tensor_slices((item1_list,item2_list))
    
    tuner.search(
        x = dataset3, epochs = 1
        )

    tuner.results_summary()
    # tuner.export_model()
    # best_model = tuner.get_best_models()[0]
    

    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values)
    
    # checkpoint=tuner_utils.SaveBestEpoch(objective=self.oracle.objective, filepath=self._get_checkpoint_fname(trial.trial_id))
    # checkpoint.set_model(yourBuiltmodel)
    # checkpoint._save_model() 
    
    # https://stackoverflow.com/questions/74844622/unsuccessful-tensorslicereader-constructor-failed-to-find-any-matching-files-fo
    # 保存权重的方式
    # encoder_weights=encoder.load_weights(".model_save/cp.ckpt")
    # #Loading weights
    # loaded_weights=encoder.load_weights(encoder_prefix)
    
    # GLOBAL_MODEL.save
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
