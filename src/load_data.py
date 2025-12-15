"""
下载和准备 CIFAR-10 数据集
"""
import torch
import torchvision
import torchvision.transforms as transforms
import os

def load_cifar10_data(data_dir='./data', batch_size=128):
    """
    下载 CIFAR-10 数据集并创建数据加载器
    
    Args:
        data_dir (str): 数据保存目录
        batch_size (int): 批处理大小
    
    Returns:
        tuple: (trainloader, testloader, classes)
    """
    
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    print(f"数据将保存到: {os.path.abspath(data_dir)}")
    
    # CIFAR-10 数据集的均值和标准差，用于归一化
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # 训练数据的预处理（包含数据增强）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # 随机裁剪
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.ToTensor(),                      # 转换为张量
        transforms.Normalize(*stats),               # 归一化
    ])
    
    # 测试数据的预处理（不包含数据增强）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    
    print("正在下载 CIFAR-10 训练集...")
    # 下载并加载训练集
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,           # 数据保存路径
        train=True,              # True表示训练集
        download=True,           # 自动下载数据集
        transform=transform_train
    )
    
    print("正在下载 CIFAR-10 测试集...")
    # 下载并加载测试集
    testset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,             # False表示测试集
        download=True,
        transform=transform_test
    )
    
    print("创建数据加载器...")
    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True,            # 训练时打乱数据
        num_workers=2,           # 使用2个子进程加载数据
        pin_memory=True          # 固定内存，提升GPU传输效率
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False,           # 测试时不打乱数据
        num_workers=2,
        pin_memory=True
    )
    
    # CIFAR-10 的10个类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print("✅ 数据集下载完成！")    
    print(f"训练集大小: {len(trainset)} 张图片")
    print(f"测试集大小: {len(testset)} 张图片")
    print(f"类别数量: {len(classes)} 个")
    print(f"类别列表: {classes}")
    
    return trainloader, testloader, classes

if __name__ == "__main__":
    # 当直接运行此脚本时，下载数据
    train_loader, test_loader, class_names = download_cifar10_data()
    
    # 简单验证数据加载器
    print("\n验证数据加载器...")
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print(f"批次图像形状: {images.shape}")
    print(f"批次标签形状: {labels.shape}")
    print(f"第一个批次的标签: {labels[:10].tolist()}")  # 显示前10个标签