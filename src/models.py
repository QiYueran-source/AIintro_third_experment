"""
ResNet74模型定义
自主设计的resnet74模型（针对CIFAR-10优化）
初始层: 1个卷积层 (conv1)     
├── Stage 2: 3个残差块 × 每个块3个卷积层 = 9个卷积层       
├── Stage 3: 6个残差块 × 3个卷积层 = 18个卷积层    
├── Stage 4: 12个残差块 × 3个卷积层 = 36个卷积层     
├── Stage 5: 3个残差块 × 3个卷积层 = 9个卷积层   
└── 分类头: 1个全连接层
总计：1 + 9 + 18 + 36 + 9 + 1 = 74层 (含1个FC层)

针对CIFAR-10的32x32图像：
原resnet模型，会将图像下采样到原来的(1/4)^2, 对于256*256的图像，会下采样到64*64，
而CIFAR-10的图像大小为32*32，因此原resnet模型会对图像进行过度下采样，导致信息丢失，  
分别尝试初始化不进行下采样(ResNet74)和在初始层就进行下采样(ResNet74_ForwardDownsample)，观察效果
"""
import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision.models.resnet import Bottleneck, BasicBlock # type: ignore


class ResNet74(nn.Module):
    """
    自定义ResNet74模型
    """

    def __init__(self, num_classes: int = 10):
        """
        自定义ResNet74模型
        num_classes: 分类数，10表示10个类别
        """
        super(ResNet74, self).__init__()

        # ============== stage1 ==============
        # 针对CIFAR-10 (32x32)优化：使用3x3卷积，stride=1，避免过度下采样
        self.conv1 = nn.Conv2d(
            in_channels=3, # 输入通道数，RGB图像为3
            out_channels=64, # 输出通道数，64个卷积核
            kernel_size=3, # 卷积核大小，3x3（改为适配32x32图像）
            stride=1, # 步幅为1，保持特征图尺寸（改为适配32x32图像）
            padding=1, # 填充1像素，保持特征图尺寸不变
            bias=False # 不使用偏置
        ) # 输出为 32x32x64（CIFAR-10）
        self.bn1 = nn.BatchNorm2d(64) # 批归一化
        self.relu = nn.ReLU(inplace=True) # 激活函数
        # 移除maxpool，避免对32x32图像过度下采样

        # ============== stage2 ==============
        # 3个残差块，第一个需要downsample匹配通道数（64->256），后两个不进行下采样
        self.layer2 = nn.Sequential(
            Bottleneck(
                inplanes=64,
                planes=64, # 输出实际为64*4(扩展系数)=256 
                stride=1, # 步幅为1，保持特征图尺寸不变
                downsample=nn.Sequential(
                    nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(256)
                ) # 对残差进行通道匹配，需要输出为256，和卷积的输出通道数一致
            ), 
            *[Bottleneck(256, 64) for _ in range(2)]
        )

        # ============== stage3 ==============
        # 6个残差块，第一个下采样，后五个不进行下采样
        self.layer3 = nn.Sequential(
            Bottleneck(256, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, 1, stride=2, bias=False), nn.BatchNorm2d(512))),
            *[Bottleneck(512, 128) for _ in range(5)]
        )

        # ============== stage4 ==============
        # 12个残差块，第一个下采样，后十一个不进行下采样
        self.layer4 = nn.Sequential(
            Bottleneck(512, 256, stride=2, downsample=nn.Sequential(
                nn.Conv2d(512, 1024, 1, stride=2, bias=False), nn.BatchNorm2d(1024))),
            *[Bottleneck(1024, 256) for _ in range(11)]
        )

        # ============== stage5 ==============
        # 3个残差块，第一个下采样，后两个不进行下采样
        self.layer5 = nn.Sequential(
            Bottleneck(1024, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, stride=2, bias=False), nn.BatchNorm2d(2048))),
            *[Bottleneck(2048, 512) for _ in range(3)]
        )

        # ============== 全连接层 ==============
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出: [batch, 2048, 1, 1]
        self.flatten = nn.Flatten()                    # 展平: [batch, 2048]
        self.fc = nn.Linear(2048, num_classes)        # 分类: [batch, num_classes]
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        权重初始化方法
        参考ResNet标准实现，使用Kaiming初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1.初始化层（针对CIFAR-10优化）
        x = self.conv1(x) # 卷积层，输出为 32x32x64（CIFAR-10）
        x = self.bn1(x) # 批归一化
        x = self.relu(x) # 激活函数
        # 移除maxpool，避免过度下采样

        # 2.stage2: 32x32x256
        x = self.layer2(x)

        # 3.stage3: 16x16x512（第一个block下采样）
        x = self.layer3(x)

        # 4.stage4: 8x8x1024（第一个block下采样）
        x = self.layer4(x)

        # 5.stage5: 4x4x2048（第一个block下采样）
        x = self.layer5(x)

        # 6.全连接层
        x = self.avgpool(x) # 自适应平均池化层，输出为 1x1x2048
        x = self.flatten(x) # 展平，输出为 2048
        x = self.fc(x) # 分类，输出为 num_classes

        return x


class ResNet74_ForwardDownsample(nn.Module):
    """
    改进的ResNet74模型：在初始层就进行下采样
    通过早期下采样减少后续层的计算量，提升训练速度
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.0):
        """
        自定义ResNet74模型（早期下采样版本）
        num_classes: 分类数，10表示10个类别
        dropout_rate: Dropout比率，默认0.5，用于训练时防止过拟合
        """
        super(ResNet74_ForwardDownsample, self).__init__()
        
        # ============== stage1 ==============
        # 在初始层就下采样：使用stride=2，将32x32下采样到16x16
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,  # 改为stride=2，进行下采样：32x32 -> 16x16
            padding=1,
            bias=False
        )  # 输出为 16x16x64（相比原版提前下采样）
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 可选：添加maxpool进一步下采样（如果使用，则16x16 -> 8x8）
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ============== stage2 ==============
        # 由于初始层已经下采样，这里保持16x16，不进行下采样
        self.layer2 = nn.Sequential(
            Bottleneck(
                inplanes=64,
                planes=64,  # 输出实际为64*4=256
                stride=1,  # 保持特征图尺寸不变
                downsample=nn.Sequential(
                    nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(256)
                )
            ),
            *[Bottleneck(256, 64) for _ in range(2)]
        )  # 输出: 16x16x256
        
        # ============== stage3 ==============
        # 第一个块下采样：16x16 -> 8x8
        self.layer3 = nn.Sequential(
            Bottleneck(256, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, 1, stride=2, bias=False), nn.BatchNorm2d(512))),
            *[Bottleneck(512, 128) for _ in range(5)]
        )  # 输出: 8x8x512
        
        # ============== stage4 ==============
        # 第一个块下采样：8x8 -> 4x4
        # 注意：这里下采样到4x4，而不是原版的8x8，计算量会大幅减少
        self.layer4 = nn.Sequential(
            Bottleneck(512, 256, stride=2, downsample=nn.Sequential(
                nn.Conv2d(512, 1024, 1, stride=2, bias=False), nn.BatchNorm2d(1024))),
            *[Bottleneck(1024, 256) for _ in range(11)]  # 在4x4x1024上处理11个块
        )  # 输出: 4x4x1024
        
        # ============== stage5 ==============
        # 第一个块下采样：4x4 -> 2x2
        self.layer5 = nn.Sequential(
            Bottleneck(1024, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, stride=2, bias=False), nn.BatchNorm2d(2048))),
            *[Bottleneck(2048, 512) for _ in range(2)]  # 减少到2个块，因为特征图已经很小了
        )  # 输出: 2x2x2048
        
        # ============== 全连接层 ==============
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出: [batch, 2048, 1, 1]
        self.flatten = nn.Flatten()
        # 添加Dropout层，用于训练时防止过拟合
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(2048, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 1.初始化层（早期下采样）
        x = self.conv1(x)  # 32x32 -> 16x16x64
        x = self.bn1(x)
        x = self.relu(x)
        # 如果使用maxpool，取消下面的注释
        # x = self.maxpool(x)  # 16x16 -> 8x8x64
        
        # 2.stage2: 16x16x256（如果使用maxpool则为8x8x256）
        x = self.layer2(x)
        
        # 3.stage3: 8x8x512（如果使用maxpool则为4x4x512）
        x = self.layer3(x)
        
        # 4.stage4: 4x4x1024（关键改进：在更小的特征图上处理）
        x = self.layer4(x)
        
        # 5.stage5: 2x2x2048
        x = self.layer5(x)
        
        # 6.全连接层
        x = self.avgpool(x)
        x = self.flatten(x)
        # 在训练模式下应用Dropout，防止过拟合
        # 注意：Dropout在训练时随机丢弃部分神经元，在评估时自动关闭
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

