import copy

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataloader import PoisonDataset
from model.lenet import LeNet
from tools.draw_bar import draw_bar
from tools.mixture import mixing_model
from tools.options import get_criterion, get_optimizer
from tools.test import test
from tools.train import train


def get_mnist_dataloader(batch_size):
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
    train_poison_loader = DataLoader(
        dataset=train_poison_dataset,
        batch_size=batch_size,
        shuffle=True)
    test_poison_loader = DataLoader(
        dataset=test_poison_dataset,
        batch_size=batch_size,
        shuffle=True)

    return train_loader, test_loader, train_poison_loader, test_poison_loader


def get_lenet_model(device):
    clear_model = LeNet().to(device)
    poison_model = LeNet().to(device)
    return poison_model, clear_model


def mnist_robustness_test(
        clear_model,
        poison_model,
        train_loader,
        train_poison__loader,
        test_loader,
        test_poison_loader,
        learning_rate,
        epochs,
        device):
    clear_criterion, poison_criterion = get_criterion()
    clear_model_optimizer, poison_model_optimizer = get_optimizer(
        clear_model=clear_model, poison_model=poison_model, learning_rate=learning_rate)
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
    ma_of_clear_model = test(
        model=clear_model,
        test_data_loader=test_loader,
        device=device)
    print("clear model in poison data test:")
    ba_of_clear_model = test(
        model=clear_model,
        test_data_loader=test_poison_loader,
        device=device)

    poison_model = train(
        model=poison_model,
        train_data_loader=train_poison__loader,
        net_criterion=poison_criterion,
        optimizer=poison_model_optimizer,
        train_epochs=epochs,
        net_device=device)
    print("Finish poison model training")
    print("poison model in clear data test:")
    ma_of_poison_model = test(
        model=poison_model,
        test_data_loader=test_loader,
        device=device)
    print("poison model in poison data test:")
    ba_of_poison_model = test(
        model=poison_model,
        test_data_loader=test_poison_loader,
        device=device)

    # 模型参数字典化
    clear_model_weights = clear_model.state_dict()
    poison_model_weights = poison_model.state_dict()

    mix_model = LeNet().to(device)
    mix_model_weights = mixing_model(
        clear_model_weights,
        poison_model_weights,
        layer_name="linear")
    mix_model.load_state_dict(mix_model_weights)

    # 测试模型
    print("mix model with linear in clear data test:")
    ma_of_linear_mix_model = test(
        model=mix_model,
        test_data_loader=test_loader,
        device=device)
    print("mix model with linear in poison data test:")
    ba_of_linear_mix_model = test(
        model=mix_model,
        test_data_loader=test_poison_loader,
        device=device)

    mix_model_weights = mixing_model(
        clear_model_weights,
        poison_model_weights,
        layer_name="conv")
    mix_model.load_state_dict(mix_model_weights)

    # 测试模型
    print("mix model with conv in clear data test:")
    ma_of_conv_mix_model = test(
        model=mix_model,
        test_data_loader=test_loader,
        device=device)
    print("mix model with conv in poison data test:")
    ba_of_conv_mix_model = test(
        model=mix_model,
        test_data_loader=test_poison_loader,
        device=device)

    values = [ma_of_clear_model, ba_of_clear_model,
              ma_of_poison_model, ba_of_poison_model,
              ma_of_linear_mix_model, ba_of_linear_mix_model,
              ma_of_conv_mix_model, ba_of_conv_mix_model]

    return values


def lenet_backdoor_information_detect():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置训练参数
    batch_size = 128
    learning_rate = 0.001
    epochs = 2

    train_loader, test_loader, train_poison_loader, test_poison_loader = get_mnist_dataloader(
        batch_size=batch_size)

    # 初始化模型、损失函数和优化器
    poison_model, clear_model = get_lenet_model(device=device)
    values = mnist_robustness_test(
        clear_model=clear_model,
        poison_model=poison_model,
        train_loader=train_loader,
        train_poison__loader=train_poison_loader,
        test_loader=test_loader,
        test_poison_loader=test_poison_loader,
        learning_rate=learning_rate,
        epochs=epochs,
        device=device)
    draw_bar(values=values, net_name="lenet")
