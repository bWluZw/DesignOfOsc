import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入常用的函数接口，例如激活函数、池化操作等

# 定义GoogLeNet模型类
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()  # 调用父类的初始化方法
        self.aux_logits = aux_logits  # 设置是否使用辅助分类器

        # 定义GoogLeNet的主要层，包括卷积层和Inception模块
        self.conv4 = BasicConv2d(1024, 512, kernel_size=(3, 1), stride=2, padding=(1, 0))
        self.inception3a = Inception(512, 256, 256, 128, 128, 64, 64, 32)
        self.conv5 = BasicConv2d(480, 256, kernel_size=(3, 1), stride=2, padding=(1, 0))
        self.inception3b = Inception(256, 64, 128, 32, 64, 32, 32, 16)

        # 定义全局平均池化层和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，将输出特征图缩小到1x1
        self.fc1 = nn.Linear(144, 72)  # 第一个全连接层，将特征降维
        self.fc2 = nn.Linear(72, num_classes)  # 最后一个全连接层，输出类别数

        # 初始化权重（如果需要）
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 前向传播逻辑
        x = self.conv4(x)  # 通过第一个卷积层
        x = self.inception3a(x)  # 通过第一个Inception模块
        x = self.conv5(x)  # 通过第二个卷积层
        x = self.inception3b(x)  # 通过第二个Inception模块

        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展平张量为二维
        x = F.dropout(x, 0.5, training=self.training)  # 在训练时应用Dropout防止过拟合
        x1 = self.fc1(x)  # 通过第一个全连接层
        x = self.fc2(x1)  # 通过最后一个全连接层输出类别
        if self.training:
            return x, x1  # 如果处于训练模式，返回主输出和辅助输出
        return x  # 返回最终输出

    # 初始化模型权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对卷积层使用Kaiming正态初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对全连接层使用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为0

# 定义Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, ch7x7red, ch7x7):
        super(Inception, self).__init__()

        # 每个Inception模块包含4个分支，分别进行不同大小的卷积操作
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)  # 1x1卷积

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),  # 1x1卷积用于降维
            BasicConv2d(ch3x3red, ch3x3, kernel_size=(3, 1), padding=(1, 0))  # 3x1卷积
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),  # 1x1卷积用于降维
            BasicConv2d(ch5x5red, ch5x5, kernel_size=(5, 1), padding=(2, 0))  # 5x1卷积
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, ch7x7red, kernel_size=1),  # 1x1卷积用于降维
            BasicConv2d(ch7x7red, ch7x7, kernel_size=(7, 1), padding=(3, 0))  # 7x1卷积
        )

    def _forward(self, x):
        # 执行前向传播，分别通过4个分支
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]  # 收集所有分支的输出
        return outputs

    def forward(self, x):
        outputs = self._forward(x)  # 获取所有分支的输出
        return torch.cat(outputs, 1)  # 沿通道维度拼接所有分支的输出

# 定义辅助分类器，用于GoogLeNet的中间输出
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=1, stride=1)  # 平均池化层
        self.conv = BasicConv2d(in_channels, 34, kernel_size=1)  # 1x1卷积层
        self.fc = nn.Linear(70176, num_classes)  # 全连接层，输出类别数

    def forward(self, x):
        x = self.averagePool(x)  # 通过平均池化层
        x = torch.flatten(x, 1)  # 展平张量为二维
        x = F.dropout(x, 0.5, training=self.training)  # 在训练时应用Dropout
        x = F.relu(self.fc(x), inplace=True)  # 通过全连接层并应用ReLU激活
        return x

# 定义基础卷积层，包含卷积、ReLU激活和BatchNorm
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)  # 卷积层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.bn = nn.BatchNorm2d(out_channels)  # 批归一化层

    def forward(self, x):
        x = self.conv(x)  # 通过卷积层
        x = self.relu(x)  # 应用ReLU激活
        x = self.bn(x)  # 应用批归一化
        return x  # 返回输出
