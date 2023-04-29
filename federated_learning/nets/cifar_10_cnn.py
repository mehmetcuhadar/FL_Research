import torch
import torch.nn as nn

class Cifar10CNN(nn.Module):
    def __init__(self):
        super(Cifar10CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.drop3 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.functional.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = x.view(-1, 256 * 4 * 4)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.drop4(x)
        x = self.fc2(x)

        return x
