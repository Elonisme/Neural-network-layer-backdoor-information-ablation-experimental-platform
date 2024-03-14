import copy


def mixing_model(clear_weights, poison_weights, layer_name):
    mixture_model = copy.deepcopy(clear_weights)
    for key_name in clear_weights:
        if layer_name in key_name:
            print(key_name)
            mixture_model[key_name] = poison_weights[key_name]
        else:
            continue
    return mixture_model
