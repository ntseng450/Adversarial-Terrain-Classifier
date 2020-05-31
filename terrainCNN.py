import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class DeepTerrain(nn.Module):
    def __init__(self):
        super(DeepTerrain, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.fc1 = nn.Linear(32*32*64, 100)
        self.drop = nn.Dropout(0.35)
        self.fc2 = nn.Linear(100, 6)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = self.drop(x)
        x = x.view(-1, 32*32*64)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(trainloader, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = DeepTerrain()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    f, (ax1, ax2) = plt.subplots(1, 2)

    lossRecords = []
    accRecords = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
           # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
        epochLoss, epochAcc = get_metrics(net, trainloader, criterion, device)
        lossRecords.append(epochLoss)
        accRecords.append(epochAcc)

        print('[%d, %5d] loss: %.3f acc: %.3f' % (epoch + 1, i + 1, epochLoss, epochAcc))
        print(len(lossRecords))
        print(len(list(range(0, epoch+1))))
        ax1.plot(list(range(0, epoch+1)), lossRecords)
        ax2.plot(list(range(0, epoch+1)), accRecords)
        plt.show()

    print('Finished Training')

def get_metrics(net, trainloader, criterion, device):
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            output = net(inputs)
            predicted = torch.argmax(output.data, dim=1)
            y_true.append(labels)
            y_pred.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss.append(criterion(output, labels).item())
        train_loss = np.mean(running_loss)
        train_acc = correct / total
        return train_loss, train_acc


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.ImageFolder(root="resizedData/", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)
    train(trainloader = trainloader, num_epochs=40)

if __name__ == '__main__':
    main()