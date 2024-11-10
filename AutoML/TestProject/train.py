import os  # 操作系统接口模块，用于路径操作
import numpy as np  # 数值计算库，用于处理数组
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch中的神经网络模块
import torch.optim as optim  # PyTorch中的优化器模块
from tqdm import tqdm  # 进度条库，用于显示循环的进度
from sklearn.model_selection import train_test_split  # 用于划分训练集和验证集
from multi_scale_module import GoogLeNet  # 导入多尺度GoogLeNet模型
from center_loss import center_loss  # 导入center loss函数，用于改进分类效果

def main():
    # 检测GPU是否可用，若可用则使用GPU设备（如cuda:2），否则使用CPU
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))  # 输出使用的设备类型

    # 设置数据根目录
    # data_root = r'/home/wangchao/location/core/model/multi_scale_2/'
    data_root = r'/home/wc/'  # 这里是用于实验的数据根目录

    # 载入训练数据集和标签
    train_dataset = np.load(os.path.join(data_root, 'train.npy'))  # 载入训练数据集
    labels = np.load(os.path.join(data_root, 'train_label.npy'))  # 载入训练标签

    # 将数据集划分为训练集和验证集，验证集大小占总数据集的10%
    x_train, x_test, y_train, y_test = train_test_split(train_dataset, labels, test_size=0.1, random_state=0)

    # 训练标签处理，将多维标签转化为一维
    train_labels = []
    for i in y_train:
        train_labels.append(int(i[0]))  # 将标签数据转换为整数

    # 将训练数据转化为列表格式
    train_set = []
    for i in x_train:
        train_set.append(i)  # 将训练数据转化为列表

    # 验证标签处理，将多维标签转化为一维
    val_labels = []
    for j in y_test:
        val_labels.append(j[0])  # 处理验证集标签

    # 将验证数据转化为列表格式
    val_set = []
    for j in x_test:
        val_set.append(j)  # 将验证数据转化为列表

    # 设置每个批次的大小
    batch_size = 128
    # 设置数据加载器中的工作线程数，选择CPU核心数与批次大小及最大值之间的最小值
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))  # 输出数据加载器使用的线程数

    # 将训练集数据转换为PyTorch张量
    x = torch.tensor(np.array(train_set)).type(torch.float)  # 转换为浮点型张量
    y = torch.tensor(np.array(train_labels)).type(torch.long)  # 转换为长整型张量
    train_dataset = torch.utils.data.TensorDataset(x, y)  # 创建训练集张量数据集

    # 将验证集数据转换为PyTorch张量
    x_val1 = torch.tensor(np.array(val_set)).type(torch.float)  # 转换为浮点型张量
    y_val1 = torch.tensor(np.array(val_labels)).type(torch.long)  # 转换为长整型张量
    val_dataset = torch.utils.data.TensorDataset(x_val1, y_val1)  # 创建验证集张量数据集

    # 使用PyTorch的数据加载器加载训练数据
    train_num = len(train_dataset)  # 计算训练样本数
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, drop_last=True)  # 训练集数据加载器

    # 使用PyTorch的数据加载器加载验证数据
    val_num = len(val_dataset)  # 计算验证样本数
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, drop_last=True)  # 验证集数据加载器

    print("using {} images for training, {} images for validation.".format(train_num, val_num))  # 输出训练和验证集的样本数

    # 初始化GoogLeNet神经网络模型
    net = GoogLeNet(num_classes=11, aux_logits=True, init_weights=True)  # 设置输出类别数、辅助分类器及权重初始化
    net.to(device)  # 将模型加载到指定设备（GPU或CPU）

    # 设置损失函数为交叉熵损失
    loss_function = nn.CrossEntropyLoss()

    # 设置优化器为随机梯度下降法（SGD），学习率为0.003，动量为0.9
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

    epochs = 500  # 设置训练的总迭代次数
    best_acc = 0.0  # 初始化最佳精度为0

    # 设置模型保存路径
    save_path = './multiScaleNet.pth'

    # 计算每个epoch的训练步数
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()  # 设置模型为训练模式
        running_loss = 0.0  # 初始化累计损失为0
        train_bar = tqdm(train_loader)  # 显示训练进度条

        # 遍历每个批次的数据
        for step, data in enumerate(train_bar):
            images, labels = data  # 获取批次数据和标签
            images = images.reshape(128, 1024, 2, 1)  # 将数据重塑为(128, 1024, 2, 1)的维度

            optimizer.zero_grad()  # 清除梯度

            # 前向传播，通过网络获取输出
            logits, aux_logits = net(images.to(device))
            aux_logits = torch.squeeze(aux_logits)  # 压缩辅助分类器的输出维度

            # 计算损失
            loss0 = loss_function(logits, labels.to(device))  # 计算主分类器的损失
            loss_center = center_loss(aux_logits, labels.to(device), 0.5)  # 计算center loss
            loss = loss0 + loss_center * 0.5  # 总损失为主分类器损失加上center loss的一半

            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累计损失

            # 更新进度条信息
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # 验证阶段
        net.eval()  # 设置模型为评估模式
        acc = 0.0  # 初始化准确率为0
        with torch.no_grad():  # 不计算梯度
            val_bar = tqdm(validate_loader)  # 显示验证进度条

            # 遍历每个批次的验证数据
            for val_data in val_bar:
                val_images, val_labels = val_data  # 获取验证批次数据和标签
                val_images = val_images.reshape(128, 1024, 2, 1)  # 将验证数据重塑为(128, 1024, 2, 1)的维度

                outputs = net(val_images.to(device))  # 前向传播，获取模型输出
                predict_y = torch.max(outputs, dim=1)[1]  # 获取输出的最大值的索引，即预测的类别
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # 计算预测准确的样本数

        val_accurate = acc / val_num  # 计算验证集上的准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))  # 输出当前epoch的训练损失和验证准确率

        # 如果当前验证准确率优于最佳准确率，则保存模型
        if val_accurate > best_acc:
            best_acc = val_accurate  # 更新最佳准确率
            torch.save(net.state_dict(), save_path)  # 保存模型参数

    print('Finished Training')  # 训练完成

if __name__ == '__main__':
    main()  # 运行主函数
