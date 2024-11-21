import sys
import keyboard
from pathlib import Path
from nni.experiment import Experiment
import time
# 定义搜索空间
search_space = {
  "noise_dim": {
    "_type": "choice",
    "_value": [32, 64, 128]
  },
  "fingerprint_dim": {
    "_type": "choice",
    "_value": [1024, 2048, 4096]
  },
  "feature_dim": {
    "_type": "choice",
    "_value": [3]
  },
  "cond_dim": {
    "_type": "choice",
    "_value": [3]
  },
  "hidden_dim": {
    "_type": "choice",
    "_value": [64, 128, 256]
  },
  "dropout": {
    "_type": "uniform",
    "_value": [0.2, 0.5]
  },
  "batch_size": {
    "_type": "choice",
    "_value": [16, 32, 64]
  },
  "num_epochs": {
    "_type": "choice",
    "_value": [50, 100]
  },
  "learning_rate": {
    "_type": "loguniform",
    "_value": [1e-5, 1e-2]
  },
  "g_lr": {
    "_type": "loguniform",
    "_value": [1e-5, 1e-3]
  },
  "d_lr": {
    "_type": "loguniform",
    "_value": [1e-5, 1e-3]
  }
}
"""
hidden_dim: 隐藏层神经元数量。提供多个选项以观察不同模型复杂度的效果。
activation: 激活函数类型，涵盖了常见的非线性激活函数。
learning_rate: 学习率，采用对数均匀分布，用于搜索大范围的最佳学习率。
num_epochs: 训练轮数，整数范围内随机选择。
batch_size: 批量大小，提供常用的批量尺寸供选择。
"""

# 创建实验
experiment = Experiment('local')
# experiment.config.trial_command = 'python trial.py'  # 运行试验的命令
experiment.config.trial_command = 'python trial.py'  # 运行试验的命令
experiment.config.trial_code_directory = '.'  # 训练脚本所在目录
experiment.config.search_space = search_space  # 指定搜索空间
experiment.config.experiment_name = '有机太阳能电池材料给体受体生成新材料（GAN）'
# 配置调优算法（Tuner）
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.trial_code_directory = Path(__file__).parent

# 设置实验的最大试验数量和并发数
experiment.config.max_trial_number = 50
experiment.config.trial_concurrency = 2

# 设置GPU数量
experiment.config.training_service.use_active_gpu = True
# experiment.config.training_service.gpu_num = 1

# 启动实验
experiment.run(8080)  # 8080为Web UI的端口号

while True:
    time.sleep(100)
    if keyboard.is_pressed('esc'):
        break