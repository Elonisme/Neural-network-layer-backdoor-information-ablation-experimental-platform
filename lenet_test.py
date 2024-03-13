import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataloader import PoisonDataset
from model.lenet import LeNet


def train(model, net_criterion, optimizer, train_data_loader, train_epochs, net_device):
    # 训练模型
    for epoch in range(train_epochs):
        for image, label in train_data_loader:
            image, label = image.to(net_device), label.to(net_device)
            optimizer.zero_grad()
            net_outputs = model(image)
            loss = net_criterion(net_outputs, label)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{train_epochs}], Loss: {loss.item():.4f}')

    print('Training finished.')
    return model


def test(model, test_data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy * 100


def mixing_model(clear_weights, poison_weights, layer_name):
    mixture_model = copy.deepcopy(clear_weights)
    for key_name in clear_weights:
        if layer_name in key_name:
            print(key_name)
            mixture_model[key_name] = poison_weights[key_name]
        else:
            continue
    return mixture_model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置训练参数
    batch_size = 128
    learning_rate = 0.001
    epochs = 2

    # 下载并加载MNIST数据集
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(
        root='./data/mnist',
        train=True,
        transform=transform,
        download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='./data/mnist',
        train=False,
        transform=transform,
        download=True)
    train_poison_dataset = PoisonDataset(
        attack_function='trigger',
        dataset_name='mnist',
        dataset=copy.deepcopy(train_dataset))
    test_poison_dataset = PoisonDataset(
        attack_function='trigger',
        dataset_name='mnist',
        dataset=copy.deepcopy(test_dataset))

    # 加载正常数据集
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True)

    # 加载有毒数据集
    train_poison__loader = DataLoader(
        dataset=train_poison_dataset,
        batch_size=batch_size,
        shuffle=True)
    test_poison_loader = DataLoader(
        dataset=test_poison_dataset,
        batch_size=batch_size,
        shuffle=True)

    # 初始化模型、损失函数和优化器
    clear_model = LeNet().to(device)
    poison_model = LeNet().to(device)
    clear_criterion = nn.CrossEntropyLoss()
    poison_criterion = nn.CrossEntropyLoss()

    clear_model_optimizer = optim.Adam(clear_model.parameters(), lr=learning_rate)
    poison_model_optimizer = optim.Adam(poison_model.parameters(), lr=learning_rate)

    print("Start training...")
    clear_model = train(
        model=clear_model,
        train_data_loader=train_loader,
        net_criterion=clear_criterion,
        optimizer=clear_model_optimizer,
        train_epochs=epochs,
        net_device=device)

    print("Finish clear model training")
    print("clear model in clear data test:")
    ma_of_clear_model = test(model=clear_model, test_data_loader=test_loader)
    print("clear model in poison data test:")
    ba_of_clear_model = test(model=clear_model, test_data_loader=test_poison_loader)

    poison_model = train(
        model=poison_model,
        train_data_loader=train_poison__loader,
        net_criterion=poison_criterion,
        optimizer=poison_model_optimizer,
        train_epochs=epochs,
        net_device=device)
    print("Finish poison model training")
    print("poison model in clear data test:")
    ma_of_poison_model = test(model=poison_model, test_data_loader=test_loader)
    print("poison model in poison data test:")
    ba_of_poison_model = test(model=poison_model, test_data_loader=test_poison_loader)

    # 模型参数字典化
    clear_model_weights = clear_model.state_dict()
    poison_model_weights = poison_model.state_dict()

    mix_model = LeNet().to(device)
    mix_model_weights = mixing_model(clear_model_weights, poison_model_weights, layer_name="linear")
    mix_model.load_state_dict(mix_model_weights)

    # 测试模型
    print("mix model with linear in clear data test:")
    ma_of_linear_mix_model = test(model=mix_model, test_data_loader=test_loader)
    print("mix model with linear in poison data test:")
    ba_of_linear_mix_model = test(model=mix_model, test_data_loader=test_poison_loader)

    mix_model_weights = mixing_model(clear_model_weights, poison_model_weights, layer_name="conv")
    mix_model.load_state_dict(mix_model_weights)

    # 测试模型
    print("mix model with conv in clear data test:")
    ma_of_conv_mix_model = test(model=mix_model, test_data_loader=test_loader)
    print("mix model with conv in poison data test:")
    ba_of_conv_mix_model = test(model=mix_model, test_data_loader=test_poison_loader)

    # 导入数据
    categories = ['Ma Of Clear Model', 'Ba Of Clear Model',
                  'Ma Of Poison Model', 'Ba Of Poison Model',
                  "Ma Of Linear Mixture Model", "Ba Of Linear Mixture Model",
                  "Ma Of Conv Mixture Model", "Ba Of Conv Mixture Model"]
    values = [ma_of_clear_model, ba_of_clear_model,
              ma_of_poison_model, ba_of_poison_model,
              ma_of_linear_mix_model, ba_of_linear_mix_model,
              ma_of_conv_mix_model, ba_of_conv_mix_model]

    plt.rcParams['font.size'] = 8

    # 绘制柱状图
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    bars = plt.bar(categories, values, color=colors, alpha=0.7)

    # 添加标题和标签
    plt.title('Robustness testing for various LeNet models')
    plt.ylabel('Accuracy')

    plt.xticks(rotation=45, ha='right')

    for bar, value in zip(bars, values):
        formatted_value = '{:.2f}%'.format(value)  # 格式化为小数点后两位，并添加百分号
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, formatted_value, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('Robustness_testing_for_various_lenet_models.png')
    # 显示柱状图
    plt.show()
