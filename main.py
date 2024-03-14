from lenet_test import lenet_backdoor_information_detect
from resnet18_test import resnet18_backdoor_information_detect

net_model_name = "resnet18"

if net_model_name == "lenet":
    lenet_backdoor_information_detect()
elif net_model_name == "resnet18":
    resnet18_backdoor_information_detect()
