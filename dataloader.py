from torch.utils.data import Dataset

from attack.blended_attack import poison_data_with_blended
from attack.semantic_attack import poison_data_with_semantic
from attack.sig_attack import poison_data_with_sig
from attack.trigger_attack import poison_data_with_trigger


class PoisonDataset(Dataset):
    def __init__(self, dataset, dataset_name, attack_function):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.attack_function = attack_function

    def __len__(self):
        # 返回一个用户被分配的数据总长度
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据索引返回对应的数据和标签
        data_sample = self.dataset[idx]
        image, label = data_sample
        if self.attack_function == 'trigger':
            image, label = poison_data_with_trigger(image=image, dataset_name=self.dataset_name)
        elif self.attack_function == 'semantic' and label == 5:
            image, label = poison_data_with_semantic(image=image, dataset_name=self.dataset_name)
        elif self.attack_function == 'blended':
            image = poison_data_with_blended(image=image, dataset_name=self.dataset_name)
        elif self.attack_function == 'sig':
            image = poison_data_with_sig(image=image, dataset_name=self.dataset_name)
        else:
            raise SystemExit("No gain attack function")
        return image, label
