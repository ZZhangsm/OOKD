# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model import resnetv1, resnetv2, vgg, wrn
from model import mobilenetv3

vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn, "vgg19bn": models.vgg19_bn}

class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        # model_vgg = vgg_dict[vgg_name](weights=True)
        model_vgg = vgg_dict[vgg_name](weights=False)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features
        self.mixstyle = None
        self.ms_layers = None


    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x

    def forward(self, x, is_feat=False, preact=False):
        f_mid = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if self.mixstyle is not None and i in self.ms_layers:
                x = self.mixstyle(x)
            f_mid.append(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if is_feat:
            if preact:
                # TODO: fix it
                return [f_mid, x]
            else:
                return f_mid, x
            # return f_mid, x
        else:
            return x


# vgg_dict = {"vgg11": vgg.vgg11, "vgg13": vgg.vgg13, "vgg16": vgg.vgg16, "vgg19": vgg.vgg19,
#             "vgg11bn": vgg.vgg11_bn, "vgg13bn": vgg.vgg13_bn, "vgg16bn": vgg.vgg16_bn, "vgg19bn": vgg.vgg19_bn}
# class VGGBase(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGGBase, self).__init__()
#         model_vgg = vgg_dict[vgg_name]()
#         self.block0 = model_vgg.block0
#         self.block1 = model_vgg.block1
#         self.block2 = model_vgg.block2
#         self.block3 = model_vgg.block3
#         self.block4 = model_vgg.block4
#
#         self.pool0 = model_vgg.pool0
#         self.pool1 = model_vgg.pool1
#         self.pool2 = model_vgg.pool2
#         self.pool3 = model_vgg.pool3
#         self.pool4 = model_vgg.pool4
#
#         self.in_features = model_vgg.classifier.in_features
#
#     def forward(self, x, is_feat=False, preact=False):
#         h = x.shape[2]
#         x = F.relu(self.block0(x))
#         f0 = x
#         x = self.pool0(x)
#         x = self.block1(x)
#         f1_pre = x
#         x = F.relu(x)
#         f1 = x
#         x = self.pool1(x)
#         x = self.block2(x)
#         f2_pre = x
#         x = F.relu(x)
#         f2 = x
#         x = self.pool2(x)
#         x = self.block3(x)
#         f3_pre = x
#         x = F.relu(x)
#         f3 = x
#         if h == 64:
#             x = self.pool3(x)
#         x = self.block4(x)
#         f4_pre = x
#         x = F.relu(x)
#         f4 = x
#         x = self.pool4(x)
#         x = x.view(x.size(0), -1)
#         f5 = x
#         # x = self.classifier(x)
#
#         if is_feat:
#             if preact:
#                 return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
#             else:
#                 return [f0, f1, f2, f3, f4, f5], x
#         else:
#             return x

# res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
#             "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d,
#             "resnext101": models.resnext101_32x8d}

res_dict = {"resnet18": resnetv1.resnet18,
            "resnet34": resnetv1.resnet34,
            "resnet50": resnetv1.resnet50,
            "resnet101": resnetv1.resnet101,
            "resnet152": resnetv1.resnet152,
            "resnext50": resnetv1.resnext50_32x4d,
            "resnext101": resnetv1.resnext101_32x8d,

            # MixStyle
            "resnet18_ms_l123": resnetv1.resnet18_ms_l123,
            "resnet50_ms_l123": resnetv1.resnet50_ms_l123,

            #ColorMNIST
            "resnet18_cm": resnetv2.resnet18_cm,
            "resnet50_cm": resnetv2.resnet50_cm,
            "resnet18_ms_l123_cm": resnetv2.resnet18_ms_l123_cm,
            "resnet50_ms_l123_cm": resnetv2.resnet50_ms_l123_cm,
            }


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        # model_resnet = res_dict[res_name](pretrained=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features


    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     return x

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m


    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f0 = x
        x, f1_pre = self.layer1(x)
        f1 = x
        x, f2_pre = self.layer2(x)
        f2 = x
        x, f3_pre = self.layer3(x)
        f3 = x
        x, f4_pre = self.layer4(x)
        f4 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f5 = x
        if is_feat:
            if preact:
                return [[f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x]
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x



class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x

class ResBase_cm(nn.Module):
    def __init__(self, res_name):
        super(ResBase_cm, self).__init__()
        model_resnet = res_dict[res_name+"_cm"](pretrained=True)
        self.conv1 = model_resnet.conv1_cm
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f0 = x
        x, f1_pre = self.layer1(x)
        f1 = x
        x, f2_pre = self.layer2(x)
        f2 = x
        x, f3_pre = self.layer3(x)
        f3 = x
        x, f4_pre = self.layer4(x)
        f4 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f5 = x
        if is_feat:
            if preact:
                return [[f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x]
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x


class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x

mobile_dict = {
    # "Mobilenetv3": mobilenetv3.MobileNetV3,
    "Mobilenetv3_small": mobilenetv3.mobilenet_v3_small,
    "Mobilenetv3_large": mobilenetv3.mobilenet_v3_large,
    }

class MobileBase(nn.Module):
    def __init__(self, mobile_name):
        super(MobileBase, self).__init__()
        model_mobile = mobile_dict[mobile_name](pretrained=True)
        self.features = model_mobile.features
        self.avgpool = model_mobile.avgpool
        self.in_features = model_mobile._out_features
        self.mixstyle = None
        self.ms_layers = None

    def forward(self, x, is_feat=False, preact=False):
        f_mid = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if self.mixstyle is not None and i in self.ms_layers:
                x = self.mixstyle(x)
            f_mid.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # f_final = x
        if is_feat:
            if preact:
                # TODO: fix it
                return [f_mid, x]
            else:
                return f_mid, x
            # return f_mid, x
        else:
            return x


wrn_dict = {
    "wrn_40_2": wrn.wrn_40_2, "wrn_40_1": wrn.wrn_40_1,
    "wrn_16_2": wrn.wrn_16_2, "wrn_16_1": wrn.wrn_16_1,
}

class WRNBase(nn.Module):
    def __init__(self, wrn_name):
        super(WRNBase, self).__init__()
        model_wrn = wrn_dict[wrn_name]()
        self.conv1 = model_wrn.conv1
        self.block1 = model_wrn.block1
        self.block2 = model_wrn.block2
        self.block3 = model_wrn.block3
        self.bn1 = model_wrn.bn1
        self.relu = model_wrn.relu
        self.fc = model_wrn.fc
        self.in_features = model_wrn.fc.in_features
        self.nChannels = model_wrn.nChannels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        f3 = out
        out = self.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(-1, self.nChannels)
        # out = out.view(x.size(0), -1)
        f4 = out
        # out = self.fc(out)
        if is_feat:
            if preact:
                f1 = self.block2.layer[0].bn1(f1)
                f2 = self.block3.layer[0].bn1(f2)
                f3 = self.bn1(f3)
            return [f0, f1, f2, f3, f4], out
        else:
            return out


