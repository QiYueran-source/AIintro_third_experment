"""
评估相关函数
"""
import torch # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore
import matplotlib.pyplot as plt # type: ignore


def test_model(model, test_loader, device):
    """
    测试模型
    model: 模型
    test_loader: 测试数据集
    device: 设备
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(): # 测试使用模型直接输出，不计算梯度  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def plot_confusion_matrix(model, test_loader, device, classes=None):
    """
    绘制混淆矩阵
    model: 模型
    test_loader: 测试数据集
    device: 设备
    classes: 类别名称列表，默认为CIFAR-10的类别
    """
    if classes is None:
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_predictions, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

