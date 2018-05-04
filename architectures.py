import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import AlexNet, alexnet


class MnistNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=10):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, out_channels)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LeNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, adaptive_pool=False, activation=True, sigmoid=False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size=5)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.relu2 = nn.ReLU(inplace=True)
        if adaptive_pool is False:
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool2 = nn.AdaptiveMaxPool2d((4, 4))
        self.fc1 = nn.Linear(4*4*50, 500)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, out_channels)
        self.activation = activation 
        if sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 4*4*50)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x) 
        if self.activation:
            if hasattr(self, 'sigmoid'):
                return self.sigmoid(x)
            else:
                return F.log_softmax(x, dim=1)
        else:
            return x


class AlexNetCustom(AlexNet):
    def __init__(self, in_padding=2, max_kernel=3, features_size=6, 
                 num_hidden=4096, num_classes=1000):
        super(AlexNetCustom, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=in_padding),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_kernel, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_kernel, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool2d(output_size=(features_size, features_size))
        )
        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * features_size * features_size, num_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(num_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def alexnet_custom(pretrained=False, **kwargs):
    model = AlexNetCustom(**kwargs)
    if pretrained:
        pretrained_model = alexnet(pretrained=True)
        state_dict = model.state_dict()
        pretrained_state_dict = pretrained_model.state_dict()
        for k in pretrained_state_dict.keys():
            if state_dict[k].shape == pretrained_state_dict[k].shape:
                state_dict[k] = pretrained_state_dict[k]
        model.load_state_dict(state_dict)
    return model


class TruncatedAlexNet(AlexNet):
    def __init__(self, module_name, num_classes=1000):
        super(TruncatedAlexNet, self).__init__(num_classes=num_classes)
        
        parent_module, module_index = module_name.split('.')
        assert(parent_module in ['features', 'classifier'])
        module_index = int(module_index)

        features = [
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        
        if parent_module == 'features':
            self.features = nn.Sequential(*features[:module_index+1])
            self.classifier = None
        else:
            self.features = features
            classifier = [
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 1000),
            ]
            self.classifier = nn.Sequential(*classifier[:module_index+1])

    def forward(self, x):
        x = self.features(x)
        if self.classifier is None:
            return x
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
def truncated_alexnet(module_name, pretrained=False, **kwargs):
    model = TruncatedAlexNet(module_name, **kwargs)
    if pretrained:
        pretrained_model = alexnet(pretrained=True, **kwargs)
        state_dict = model.state_dict()
        pretrained_state_dict = pretrained_model.state_dict()
        for k in pretrained_state_dict.keys():
            if k in state_dict.keys() and state_dict[k].shape == pretrained_state_dict[k].shape:
                state_dict[k] = pretrained_state_dict[k]
        model.load_state_dict(state_dict)
    return model
