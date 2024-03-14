from torch import nn, optim


def get_criterion():
    clear_criterion = nn.CrossEntropyLoss()
    poison_criterion = nn.CrossEntropyLoss()
    return clear_criterion, poison_criterion


def get_optimizer(clear_model, poison_model, learning_rate):
    clear_model_optimizer = optim.Adam(
        clear_model.parameters(), lr=learning_rate)
    poison_model_optimizer = optim.Adam(
        poison_model.parameters(), lr=learning_rate)
    return clear_model_optimizer, poison_model_optimizer
