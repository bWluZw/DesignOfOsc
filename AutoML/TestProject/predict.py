import os  # 操作系统接口模块，用于文件和路径操作
import json  # JSON模块，用于读取和处理JSON文件
import numpy as np  # 数值计算库，用于处理数组
import torch  # PyTorch深度学习框架
from tqdm import tqdm  # 进度条库，用于显示循环的进度

from multi_scale_module import GoogLeNet  # 导入自定义的GoogLeNet模型

def main(validate_loader):
    # 检测是否有可用的GPU，如果有则使用第一个GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 读取类别索引的JSON文件路径
    json_path = './class_indices.json'
    # 断言：确保JSON文件存在，否则抛出错误
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # 打开并读取JSON文件，加载类别索引
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 创建GoogLeNet模型，设置输出类别数为11，禁用辅助分类器
    model = GoogLeNet(num_classes=11, aux_logits=False).to(device)

    # 载入预训练的模型权重文件路径
    weights_path = r"E:\python\modulation_identification\core\model\multi_scale_2\multiScaleNet.pth"
    # 断言：确保权重文件存在，否则抛出错误
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    # 将模型设置为评估模式，关闭Dropout和BatchNorm
    model.eval()

    acc = 0.0  # 初始化准确率为0
    with torch.no_grad():  # 关闭自动求导（不计算梯度）
        # 显示验证集的进度条
        val_bar = tqdm(validate_loader)

        # 遍历验证集中的每个批次
        for val_data in val_bar:
            val_images, val_labels = val_data  # 获取验证集的图像和标签
            val_images = val_images.reshape(32, 1024, 2, 1)  # 将图像数据重塑为(32, 1024, 2, 1)的维度
            outputs = torch.squeeze(model(val_images.to(device))).cpu()  # 前向传播，获取模型输出并压缩维度
            predicts = torch.max(outputs, dim=1)[1]  # 获取输出中最大值的索引，即预测的类别
            acc += torch.eq(predicts, val_labels.to(device)).sum().item()  # 计算预测正确的样本数

    val_accurate = acc / val_num  # 计算验证集的准确率
    print('val_accuracy: %.3f' % (val_accurate))  # 打印验证集的准确率

if __name__ == '__main__':
    # 设置数据根目录
    data_root = r'E:\python\modulation_identification\data'

    # 载入测试数据集和标签
    test_dataset = np.load(os.path.join(data_root, 'test1.npy'))  # 载入测试数据
    labels = np.load(os.path.join(data_root, 'test1_label.npy'))  # 载入测试标签

    # 处理测试标签，将多维标签转化为一维整数标签
    test_labels = []
    for i in labels:
        test_labels.append(int(i[0]))  # 转换为整数标签

    test_labels = torch.tensor(np.array(test_labels))  # 将标签转换为PyTorch张量

    # 处理测试数据集，将数据转化为PyTorch张量
    test_set = []
    for i in test_dataset:
        test_set.append(i)  # 转换为列表

    test_set = torch.tensor(test_set).type(torch.float)  # 将数据转换为浮点型张量

    # 创建测试数据集的张量数据集
    dataset = torch.utils.data.TensorDataset(test_set, test_labels)

    # 设置每个批次的大小
    batch_size = 32
    # 设置数据加载器的工作线程数，选择CPU核心数与批次大小及最大值之间的最小值
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_num = len(dataset)  # 计算验证集样本数
    # 使用PyTorch的数据加载器加载验证数据
    validate_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, drop_last=True)  # 验证集数据加载器

    # 调用主函数，传入验证集加载器
    main(validate_loader)
