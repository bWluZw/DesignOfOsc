import tensorflow as tf
import keras_tuner as kt
import common_helper
import pandas as pd



    
def main():
    # TODO 大概梳理一下GAN代码脉络，训练模型 √
    # TODO 拿出预测PCE模型 √
    # TODO 图转smiles √
    # TODO 可视化还没搞 √
    # TODO 1.多准备一些训练出来的smiles表达式，导出csv
    # TODO 2.用docker中的pyscf得到物理化学性质，导出csv
    # TODO 3.将其转为分子指纹+其他特征，进行预测PCE
    
    
    # csv_path = r'D:\Project\ThesisProject\AutoML\data\gan_model_gen_data.csv'
    # df = pd.read_csv(csv_path, encoding='utf-8')
    # df.to_csv(csv_path, index=False,encoding='utf-8-sig')
    
    # 生成smiles表达式csv
    gan_path = 'D:\Project\ThesisProject\AutoML\AutoMLProject\gan_model_20241222125124.h5py'
    gan_model = tf.saved_model.load(gan_path)
    
    smiles_list=[]
    
    while(True):
        noise1 = tf.random.normal(shape=(32,8))
        noise2 = tf.random.normal(shape=(32,32))
        node_features, adjacency_matrix = gan_model([noise1,noise2], training=True)
        smiles = common_helper.graph_to_smiles(node_features, adjacency_matrix)
        if(smiles != None):
            smiles_list.append(smiles)
        if(len(smiles_list)>=1000):
            break
    smiles_data={
        "smiles":smiles_list
    }
    single_smiles_df=pd.DataFrame(smiles_data)
    single_smiles_df.to_csv(r'D:\Project\ThesisProject\AutoML\data\gan_model_gen_data11111.csv', index=False,encoding='utf-8-sig')
    
    
    
    
    # pce_model_path = 'D:\Project\ThesisProject\AutoML\models\pce_model\best_model'
    # pce_model = tf.saved_model.load(pce_model_path)
    
    

if __name__ == '__main__':
    main()

