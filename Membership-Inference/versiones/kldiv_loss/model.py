import torch
import torch.nn as nn
from torchvision import models

def size_conv(size, kernel, stride=1, padding=0):
    out = int(((size - kernel + 2*padding)/stride) + 1)
    return out

def size_max_pool(size, kernel, stride=None, padding=0):
    if stride is None:
        stride = kernel
    out = int(((size - kernel + 2*padding)/stride) + 1)
    return out

def calc_feat_linear_cifar(size):
    feat = size_conv(size, 3, 1, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 3, 1, 1)
    out = size_max_pool(feat, 2, 2)
    return out

def calc_feat_linear_mnist(size):
    feat = size_conv(size, 5, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 5, 1)
    out = size_max_pool(feat, 2, 2)
    return out

def init_params(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)

class TargetNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, size, out_classes):
        super(TargetNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        features = calc_feat_linear_cifar(size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((features**2 * hidden_layers[1]), hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layers[2], out_classes * 2)
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        mean, variance = out.chunk(2, dim=1)
        variance = self.softplus(variance)
        return mean, variance

class ShadowNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, size, out_classes):
        super(ShadowNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        features = calc_feat_linear_cifar(size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((features**2 * hidden_layers[1]), hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layers[2], out_classes * 2)
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        mean, variance = out.chunk(2, dim=1)
        variance = self.softplus(variance)
        return mean, variance

class VggModel(nn.Module):
    def __init__(self, num_classes, layer_config, pretrained=True):
        super(VggModel, self).__init__()
        if pretrained:
            pt_vgg = models.vgg11_bn(pretrained=pretrained)
            print('### Deleting Avg pooling and FC Layers ####')
            del pt_vgg.avgpool
            del pt_vgg.classifier
            self.model_features = nn.Sequential(*list(pt_vgg.features.children()))
            self.model_classifier = nn.Sequential(
                nn.Linear(layer_config[0], layer_config[1]),
                nn.BatchNorm1d(layer_config[1]),
                nn.ReLU(inplace=True),
                nn.Linear(layer_config[1], num_classes * 2),
            )
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.model_features(x)
        x = x.squeeze()
        out = self.model_classifier(x)
        mean, variance = out.chunk(2, dim=1)
        variance = self.softplus(variance)
        return mean, variance

class MNISTNet(nn.Module):
    def __init__(self, input_dim, n_hidden, out_classes=10, size=28):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=n_hidden, kernel_size=5),
            nn.BatchNorm2d(n_hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_hidden, out_channels=n_hidden*2, kernel_size=5),
            nn.BatchNorm2d(n_hidden*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        features = calc_feat_linear_mnist(size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features**2 * (n_hidden*2), n_hidden*2),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden*2, out_classes * 2)
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        out = self.classifier(x)
        mean, variance = out.chunk(2, dim=1)
        variance = self.softplus(variance)
        return mean, variance

class AttackMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, out_classes=2):
        super(AttackMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_classes * 2)
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.classifier(x)
        mean, variance = out.chunk(2, dim=1)
        variance = self.softplus(variance)
        return mean, variance

class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(VisionTransformerModel, self).__init__()
        self.model = models.vit_b_16(pretrained=pretrained)
        self.model.heads = nn.Sequential(
            nn.Linear(self.model.heads.head.in_features, num_classes * 2)
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.model(x)
        mean, variance = out.chunk(2, dim=1)
        variance = self.softplus(variance)
        return mean, variance
