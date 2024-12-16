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


from sklearn.model_selection import train_test_split

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

    def train_step(self, real_data):
        # real_images = batch_data[0]  # Real images from the dataset
        # batch_size = tf.shape(real_images)[0]
        batch_size = real_data.shape[0]
        print(real_data)
        print(real_data.shape)
        # TODO 传入的参数类型
        # 1. 训练判别器
        # 随机生成噪声向量
        noise = tf.random.normal(shape=(batch_size,self.latent_dim))  # 假设噪声维度为66
        generated_data = self.generator(noise, training=True)
        # TODO 生成的数据结构
        print(generated_data)
        print(generated_data.shape)
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

    def __init__(self,latent_dim,vocab_size):
        super(HyperGAN, self).__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

    def make_generator_model(self, hp):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(self.latent_dim,)))
        units_1 = hp.Int('units_1', min_value=64, max_value=512, step=64)
        units_2 = hp.Int('units_2', min_value=64, max_value=512, step=64)

        model.add(layers.Dense(units_1, activation='relu'))
        model.add(layers.Dense(units_2, activation='relu'))
        model.add(
            layers.Dense(self.vocab_size, activation="tanh")
        )  # 假设 SMILES 是通过向量表示

        return model

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

        adam_optimizer = tf.keras.optimizers.Adam(1e-4)
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        model_gan.compile(adam_optimizer,adam_optimizer,binary_crossentropy)
        return model_gan
    # def score_function(self, z):
    #     """
    #     The scoring function to evaluate generator outputs.
    #     `z` is the input noise vector fed into the generator.
    #     """
    #     # 1. Generate a fake sample
    #     gen_data = self.generator(z)

    #     # 2. Use discriminator to classify the fake image as real or fake
    #     # 这里假设判别器输出接近1时表示真实，接近0时表示假
    #     discriminator_output = self.discriminator(gen_data)

    #     # 3. The score can be the discriminator output, or you can use other metrics
    #     return tf.reduce_mean(discriminator_output)

    # def fit(self, hp, model, x, **kwargs):
    #     model.fit(x,**kwargs)
    #     res = self.score_function(self.vocab_size)
    #     # Return a single float to minimize.
    #     return res
def main():
    
    csv_file = 'D:\Project\ThesisProject\AutoML\data\moldata_part_test.csv'
    X_train, X_test, y_train, y_test = load_data(csv_file)
    
    trials = 10

    tuner = kt.BayesianOptimization(
        hypermodel=HyperGAN(X_train.shape[1],1030),
        # No objective to specify.
        # Objective is the return value of `HyperModel.fit()`.
        objective=kt.Objective("g_loss", "min"),
        max_trials=trials,
        overwrite=True,
        directory="my_dir1111",
        project_name="custom_eval",
    )

    tuner.search(
        x = X_train, epochs = 10
        )

    tuner.results_summary()
    best_model = tuner.get_best_models()[0]
if __name__ == '__main__':
    main()
