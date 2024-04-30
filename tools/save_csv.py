import csv


def save_pairs_to_csv(file_path, values):
    """
    将值对保存到 CSV 文件中，每两个值一对

    参数:
        file_path (str): CSV 文件路径
        values (list): 要保存的值列表，格式为 [value_1, value_2, value_3, value_4, ...]
    """
    # 检查值的数量是否为偶数
    if len(values) % 2 != 0:
        raise ValueError("Values list length must be even.")

    # 将值分成对
    pairs = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]

    # 写入到 CSV 文件
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['MA', 'BA'])  # 写入标题行
        for pair in pairs:
            writer.writerow(pair)
