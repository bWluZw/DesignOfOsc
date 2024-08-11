import sys
from pathlib import Path
from nni.experiment import Experiment
# 定义搜索空间
search_space = {
    "learning_rate": {
        "_type": "loguniform",
        "_value": [0.0001, 0.1]
    },
    "batch_size": {
        "_type": "choice",
        "_value": [16, 32, 64, 128]
    },
    "num_layers": {
        "_type": "randint",
        "_value": [1, 5]
    },
    "dropout_rate": {
        "_type": "uniform",
        "_value": [0.0, 0.5]
    }
}
# 创建实验
experiment = Experiment('local')
experiment.config.trial_command = 'python trial.py'  # 运行试验的命令
experiment.config.trial_code_directory = '.'  # 训练脚本所在目录
experiment.config.search_space = search_space  # 指定搜索空间
experiment.config.experiment_name = '例子'
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