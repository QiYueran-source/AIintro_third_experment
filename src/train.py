"""
训练相关函数
"""
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import time # type: ignore


def train_epoch(model: nn.Module, 
                train_loader, 
                criterion: nn.CrossEntropyLoss, 
                optimizer: optim.Optimizer, 
                device: torch.device,
                epoch: int = 1,
                num_epochs: int = 10):
    """
    训练一轮   
    model: 模型
    train_loader: 训练数据集，按批次提供数据  
    criterion: 损失函数  
    optimizer: 优化器  
    device: 设备
    epoch: 当前epoch编号
    num_epochs: 总epoch数
    """
    model.train() # train or eval
    running_loss = 0.0 # 累积的损失
    correct = 0 # 累积的正确数量
    total = 0 # 累积的总数量
    total_step = len(train_loader)
    start_time = time.time()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device) # model和tensor都需要移动  
        
        # 前向传播
        outputs = model(images) # model(样本) 输出预测值，本质上是调用forward函数  
        loss = criterion(outputs, labels) # 输出和标签的损失 -> 计算损失  
        
        # 反向传播
        optimizer.zero_grad() # 梯度清零（训练新的一批样本前，需要清理上一批的梯度）  
        loss.backward() # 反向传播，计算每一个参数对于损失的偏导数，组成梯度    
        optimizer.step() # 使用w - 学习率*梯度更新参数    
        
        # 统计准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        
        # 每100步输出一次
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    end_time = time.time()
    print(f'Epoch [{epoch}/{num_epochs}] Finished. Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {end_time - start_time:.2f}s')
    return epoch_loss, epoch_acc


def train_model(model, trainloader, criterion, optimizer, num_epochs, device):
    """
    训练模型
    model: 模型
    trainloader: 训练数据集，按批次提供数据
    criterion: 损失函数
    optimizer: 优化器
    num_epochs: 总epoch数
    device: 设备
    """
    model.train()
    total_step = len(trainloader)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
        
        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(trainloader)
        end_time = time.time()
        print(f'Epoch [{epoch+1}/{num_epochs}] Finished. Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {end_time - start_time:.2f}s')

