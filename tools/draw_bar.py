import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import csv


def save_legend(net_name):
    categories = ['Ma Of Clear Model', 'Ba Of Clear Model',
                  'Ma Of Poison Model', 'Ba Of Poison Model',
                  "Ma Of Linear Mixture Model", "Ba Of Linear Mixture Model",
                  "Ma Of Conv Mixture Model", "Ba Of Conv Mixture Model"]

    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))

    # 创建图例
    fig_legend, ax_legend = plt.subplots()
    legend_elements = [Patch(facecolor=color, alpha=0.7, label=category) for category, color in zip(categories, colors)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=2, frameon=False)
    ax_legend.axis('off')

    # 统一颜色
    for handle in legend_elements:
        handle.set_color(handle.get_facecolor())

    plt.tight_layout()
    plt.savefig(f'../imgs/Legend_for_various_{net_name}_models.pdf')
    plt.close()


def save_bar(values, net_name):
    categories = ['Ma Of Clear Model', 'Ba Of Clear Model',
                  'Ma Of Poison Model', 'Ba Of Poison Model',
                  "Ma Of Linear Mixture Model", "Ba Of Linear Mixture Model",
                  "Ma Of Conv Mixture Model", "Ba Of Conv Mixture Model"]

    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))

    # 创建柱状图
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(categories)), values, color=colors, alpha=0.7)

    # 隐藏 x 轴标签
    ax.set_xticks([])

    # 添加数值标签
    for bar, value in zip(bars, values):
        formatted_value = '{:.2f}%'.format(value)  # 格式化为小数点后两位，并添加百分号
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, formatted_value, ha='center', va='bottom')

    # 添加标题
    plt.title(f'MA and BA for various {net_name} models')

    plt.tight_layout()
    plt.savefig(f'../imgs/Robustness_testing_for_various_{net_name}_models.png')
    plt.close()


def read_csv(filename):
    data = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
            ma, ba = map(float, row)
            data.append(ma)
            data.append(ba)
    return data


if __name__ == '__main__':
    net_name = "legend"
    save_legend(net_name)

    # models = ["cnn", "lenet", "mobilenetv2", "resnet18", "vgg13", "vgg16"]
    # for model in models:
    #     filename = f'../csv/{model}.csv'
    #     print(filename)
    #     values = read_csv(filename)
    #     net_name = model
    #     save_bar(values, net_name)
