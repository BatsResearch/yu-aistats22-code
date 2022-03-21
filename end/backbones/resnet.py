import torch
from torch import nn
from torchvision.models import resnet50, resnet101
import torch.nn.init as init

r50_in_dim = resnet50().fc.in_features
r101_in_dim = resnet101().fc.in_features


class ResNetFeaturesLC(nn.Module):
    def __init__(self, in_features=r101_in_dim, target_dim=2, intermediate_1=1024, intermediate_2=1024):
        super().__init__()

        self.target_fc = nn.Sequential(
            nn.Linear(in_features, intermediate_1),
            nn.BatchNorm1d(num_features=intermediate_1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(intermediate_1, intermediate_2),
            nn.BatchNorm1d(num_features=intermediate_2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            # nn.Linear(intermediate_2, intermediate_2),
            # nn.BatchNorm1d(num_features=intermediate_2),
            # nn.LeakyReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(intermediate_2, target_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

    def forward(self, x):
        x = torch.nn.functional.normalize(x)
        return self.target_fc(x)

class ResNet50(nn.Module):
    def __init__(self, target_dim, pretrained=False, device='cuda:0'):
        super(ResNet50, self).__init__()

        self.convnet = resnet50(pretrained=pretrained).to(device)
        self.target_fc = nn.Sequential(
            nn.Linear(self.convnet.fc.in_features, target_dim),
            nn.Softmax()
        ).to(device)

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight.data)
                    init.normal_(m.bias.data)


    def freeze_conv(self):
        for param in self.convnet.parameters():
            param.requires_grad = False

    def freeze_conv_layer(self, layer_id):
        if not isinstance(layer_id, list):
            layer_id = [layer_id]
        for layer in layer_id:
            if layer == 1:
                for param in self.convnet.layer1.parameters():
                    param.requires_grad = False
            elif layer == 2:
                for param in self.convnet.layer2.parameters():
                    param.requires_grad = False
            elif layer == 3:
                for param in self.convnet.layer3.parameters():
                    param.requires_grad = False
            elif layer == 4:
                for param in self.convnet.layer4.parameters():
                    param.requires_grad = False

    def unfreeze_conv_layer(self, layer_id):
        if not isinstance(layer_id, list):
            layer_id = [layer_id]
        for layer in layer_id:
            if layer == 1:
                for param in self.convnet.layer1.parameters():
                    param.requires_grad = True
            elif layer == 2:
                for param in self.convnet.layer2.parameters():
                    param.requires_grad = True
            elif layer == 3:
                for param in self.convnet.layer3.parameters():
                    param.requires_grad = True
            elif layer == 4:
                for param in self.convnet.layer4.parameters():
                    param.requires_grad = True

    def unfreeze_conv(self):
        for param in self.convnet.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.convnet.conv1(x)
        x = self.convnet.bn1(x)
        x = self.convnet.relu(x)
        x = self.convnet.maxpool(x)

        x = self.convnet.layer1(x)
        x = self.convnet.layer2(x)
        x = self.convnet.layer3(x)
        x = self.convnet.layer4(x)

        x = self.convnet.avgpool(x)
        x = torch.flatten(x, 1)
        target_out = self.target_fc(x)

        return target_out