import numpy as np
from matplotlib import pyplot as plt


def draw_bar(values, net_name):
    # 导入数据
    categories = ['Ma Of Clear Model', 'Ba Of Clear Model',
                  'Ma Of Poison Model', 'Ba Of Poison Model',
                  "Ma Of Linear Mixture Model", "Ba Of Linear Mixture Model",
                  "Ma Of Conv Mixture Model", "Ba Of Conv Mixture Model"]

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
    plt.savefig(f'Robustness_testing_for_various_{net_name}_models.png')
    # 显示柱状图
    plt.show()
