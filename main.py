from cnn_test import cnn_backdoor_information_detect
from lenet_test import lenet_backdoor_information_detect
from mobilenetv2_test import mobilenetv2_backdoor_information_detect
from resnet18_test import resnet18_backdoor_information_detect
from vgg16_test import vgg16_backdoor_information_detect
from vgg_test import vgg_backdoor_information_detect

nets = ["vgg16"]

for net_model_name in nets:
    if net_model_name == "lenet":
        lenet_backdoor_information_detect()
    elif net_model_name == "cnn":
        cnn_backdoor_information_detect()
    elif net_model_name == "resnet18":
        resnet18_backdoor_information_detect()
    elif net_model_name == 'vgg':
        vgg_backdoor_information_detect()
    elif net_model_name == 'mobilenet':
        mobilenetv2_backdoor_information_detect()
    elif net_model_name == 'vgg16':
        vgg16_backdoor_information_detect()
    else:
        print("can not match model")
