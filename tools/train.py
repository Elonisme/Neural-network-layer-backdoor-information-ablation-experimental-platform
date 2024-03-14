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
