## 简易笔记

### 文档补充



#### 超参数篇

##### 超参数调优(以后自己实现，它这里并没有给出相关API)

1. Random Search
Random Search 是最简单的一种超参数优化方法，它通过随机选择超参数的组合来进行搜索。虽然这种方法效率较低，但它简单且易于实现，适用于搜索空间较小的情况。

用法：
```
from autokeras import Tuner

tuner = ak.RandomSearch(
    objective='val_loss',            # 优化目标（验证损失）
    max_trials=10,                    # 最大试验次数
    hyperparameters=search_space,     # 搜索空间
    directory='my_dir',               # 存储模型的位置
    project_name='random_search'      # 项目名称
)
```

- 特点：
    * 优点：简单易用，适合初学者。
    * 缺点：效率较低，特别是在搜索空间较大时，可能需要大量的试验才能找到最佳超参数。
2. Bayesian Optimization
Bayesian Optimization 是一种更智能的优化方法，通过构建一个概率模型来预测各个超参数组合的效果，从而引导搜索过程。相比于随机搜索，它能够更高效地探索搜索空间。

用法：
```
from autokeras import Tuner

tuner = ak.BayesianOptimization(
    objective='val_loss',            # 优化目标（验证损失）
    max_trials=10,                    # 最大试验次数
    hyperparameters=search_space,     # 搜索空间
    directory='my_dir',               # 存储模型的位置
    project_name='bayesian_optimization' # 项目名称
)
```
- 特点：
    * 优点：比随机搜索更高效，能够更快地找到最佳超参数组合。
    * 缺点：实现相对复杂，对搜索空间的选择比较敏感。
3. Hyperband
Hyperband 是一种基于多臂赌博机（Multi-Armed Bandit）算法的优化方法，它通过多轮训练和早停策略来选择最佳超参数组合。它通过减少低效的超参数组合的搜索来加速超参数调优。

用法：
```
from autokeras import Tuner

tuner = ak.Hyperband(
    objective='val_loss',            # 优化目标（验证损失）
    max_trials=10,                    # 最大试验次数
    hyperparameters=search_space,     # 搜索空间
    directory='my_dir',               # 存储模型的位置
    project_name='hyperband'          # 项目名称
)
```
- 特点：
    * 优点：相较于 Random Search 和 Bayesian Optimization，Hyperband 通常能够更加高效地利用计算资源，适用于大型超参数搜索任务。
    * 缺点：需要一定的计算资源和时间来训练模型。
4. Grid Search
Grid Search 是一种穷举式搜索方法，它通过遍历所有可能的超参数组合来寻找最优解。虽然这种方法保证可以找到最优解，但它的计算成本非常高，特别是在搜索空间较大时。

用法：
目前，Auto-Keras 中并没有直接提供 Grid Search 的 API，但可以通过手动实现类似的功能。例如，使用 GridSearchCV（来自 sklearn）配合 Auto-Keras 进行调优。

- 特点：
    * 优点：理论上能够找到最优解。
    * 缺点：计算开销巨大，效率低下，特别是在高维搜索空间时。
5. Evolutionary Algorithms
Auto-Keras 也支持使用 进化算法（例如 遗传算法）进行超参数搜索，这类算法通过模拟自然选择的过程（选择、交叉、变异等）来优化超参数组合。虽然 Auto-Keras 目前没有直接实现此功能，但你可以使用 Keras Tuner 或其他进化优化库结合 Auto-Keras 来进行调优。

参考方法：
from keras_tuner import Hyperband

进化算法相关的实现需要结合 Keras Tuner 或类似库来实现
特点：
优点：能够在复杂问题上表现出色，适用于非常高维的搜索空间。
缺点：实现较为复杂，需要更高的计算资源。

##### 超参数搜索空间(以后自己实现，它这里并没有给出相关API)
```
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
    
    # 超参数：早停（Early Stopping）
    "early_stopping": [True, False],  # 是否启用早停，可以选择开启或关闭
    
    # 超参数：批归一化（Batch Normalization）
    "batch_norm": [True, False],  # 是否使用批归一化
    
    # 超参数：权重初始化方法（Weight Initialization）
    "kernel_initializer": ['glorot_uniform', 'he_normal', 'random_normal'],  # 权重初始化方法
    
    # 超参数：最大池化核大小（Max Pooling Kernel Size）
    "max_pool_size": [2, 3, 4],  # 最大池化核大小：2x2, 3x3 或 4x4
    
    # 超参数：学习率衰减（Learning Rate Decay）
    "lr_scheduler": ['exponential_decay', 'step_decay'],  # 学习率衰减策略
    "decay_rate": [0.1, 0.2, 0.3],  # 衰减率的选择范围
    
    # 超参数：激活函数的输入（Learning Rate Decay）
    "use_bias": [True, False]  # 是否使用偏置项
}
```

- batch_size: 影响训练速度和内存占用。选择较小的批量可以让模型更快地迭代，但会增加计算的噪声。

- learning_rate: 学习率是优化算法中最重要的超参数之一，决定了每一步更新的大小。你可以在一个较宽的范围内进行调优，从而找出最佳学习率。

- epochs: 训练周期数，决定了模型在训练集上的训练次数。过多的训练周期可能导致过拟合，太少可能导致欠拟合。

- dropout_rate: Dropout 是一种正则化技术，用于防止过拟合。你可以在不同的层使用不同的 Dropout 比例。

- num_layers 和 num_units: 这两个超参数定义了神经网络的结构。num_layers 控制网络的深度，num_units 控制每一层的神经元数量。

- activation: 激活函数控制了神经网络每一层的输出。不同的激活函数可能影响模型的非线性表达能力。

- optimizer: 选择不同的优化器（如 Adam、SGD 或 RMSProp）可能对训练过程有很大的影响。

- l2_regularization: L2 正则化有助于减少模型的复杂度，并防止过拟合。

- early_stopping: 通过监视验证集的性能，提前停止训练，避免模型过拟合训练数据。

- batch_norm: 批归一化可以加速训练并提高模型的泛化能力。

- kernel_initializer: 权重初始化方法影响模型的收敛速度和最终性能。不同的初始化方法会影响训练的稳定性。

- max_pool_size: 池化层的核大小有助于减少模型参数数量并提取重要特征。

- lr_scheduler: 学习率衰减可以帮助模型更好地收敛。常见的策略有指数衰减和步进衰减。

- use_bias: 是否在层中使用偏置项，偏置项通常对模型的表示能力有一定的提升。

- 适用的调优方法
    * Random Search 和 Bayesian Optimization：适用于这个类型的搜索空间，能够在较大的空间内进行高效的搜索。
    * Hyperband：适用于该搜索空间的多轮优化，可以更好地利用计算资源。
    * Grid Search：如果你需要精确搜索所有可能的超参数组合，可以使用网格搜索，特别是当搜索空间较小或你想确保遍历所有组合时。

### 坑
- 错误：AttributeError: module 'numpy' has no attribute 'object'
降版本 pip install numpy==1.23.4

- "ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float)."
降版本，tensorflow 2.10.1

- 无法使用GPU加速
降版本，截止2024-12-8，依然最高支持cuda 11.2。cudnn 8.1.如果遇到无法降cuda版本情况，卸载Nvidia FrameView SDK，再重新安装即可

