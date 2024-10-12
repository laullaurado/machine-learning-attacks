import torch

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training(epochs, dataloader, optimizer, criterion, net, path, is_target: bool):
    net.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        print('Epoch ' + str(epoch + 1))
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data.values()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            if is_target:
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    torch.save(net.state_dict(), path)
    print('Finished Training')


def test(testloader, net, is_target: bool):
    net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data.values()
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)

            if is_target:
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Total test samples: ' + str(total))
    print('Correct test samples: ' + str(correct))
    print('Accuracy: %d %%' % (100 * correct / total))


def test_class(testloader, net, is_target):
    net.to(device)
    classes = ['0', '1', '2', '3', '4']

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data.values()
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)

            if is_target:
                outputs = outputs[0]

            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / (total_pred[classname] + 0.000001)
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
