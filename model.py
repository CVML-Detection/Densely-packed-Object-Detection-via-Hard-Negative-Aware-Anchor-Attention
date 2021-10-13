from torchvision.models import resnet50
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import warnings


class Resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.resnet50 = resnet50(pretrained=pretrained)
        self.resnet50_list = nn.ModuleList(list(self.resnet50.children())[:-2])  # to layer 1
        self.res50 = nn.Sequential(*self.resnet50_list)

    def forward(self, x):

        x = self.resnet50_list[0](x)
        x = self.resnet50_list[1](x)
        x = self.resnet50_list[2](x)
        x = self.resnet50_list[3](x)

        x = self.resnet50_list[4](x)
        c3 = x = self.resnet50_list[5](x)
        c4 = x = self.resnet50_list[6](x)
        c5 = x = self.resnet50_list[7](x)
        return [c3, c4, c5]


class FPN(nn.Module):
    def __init__(self, baseline=Resnet50()):
        super().__init__()

        self.stride = 128
        self.baseline = baseline

        channels = [512, 1024, 2048]

        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.pyramid6 = nn.Conv2d(channels[2], 256, 3, stride=2, padding=1)
        self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)
        self.initialize()

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        self.lateral3.apply(init_layer)
        self.lateral4.apply(init_layer)
        self.lateral5.apply(init_layer)
        self.pyramid6.apply(init_layer)
        self.pyramid7.apply(init_layer)
        self.smooth3.apply(init_layer)
        self.smooth4.apply(init_layer)
        self.smooth5.apply(init_layer)

    def forward(self, x):
        c3, c4, c5 = self.baseline(x)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, size=p4.size()[2:]) + p4
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, size=p3.size()[2:]) + p3

        p6 = self.pyramid6(c5)
        p7 = self.pyramid7(F.relu(p6))

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        return [p3, p4, p5, p6, p7]


class HNNA_DET(nn.Module):
    def __init__(self, fpn=FPN(Resnet50(pretrained=True)), num_classes=1):
        super(HNNA_DET, self).__init__()
        self.unet = DWS_Unet(n_channels=3, n_classes=1)
        self.fpn = fpn
        self.num_classes = num_classes
        self.cls_module = ClsModule(self.num_classes)
        self.reg_module = RegModule()
        self.initialize_subsets()
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_subsets(self):
        i = 0
        for c in self.cls_module.features.children():

            if isinstance(c, nn.Conv2d):
                if i == 8:
                    pi = 0.01
                    b = - math.log((1 - pi) / pi)
                    nn.init.constant_(c.bias, b)
                    nn.init.normal_(c.weight, std=0.01)
                else:
                    nn.init.normal_(c.weight, std=0.01)
                    nn.init.constant_(c.bias, 0)
            i += 1

        for c in self.reg_module.features.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

    def freeze_bn(self):
        for layer in self.fpn.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        features = self.fpn(inputs)
        obj = self.unet(inputs)

        cls = torch.cat([self.cls_module(feature) for feature in features], dim=1)
        reg = torch.cat([self.reg_module(feature) for feature in features], dim=1)
        return cls, reg, obj


class ClsModule(nn.Module):
    def __init__(self, num_classes):
        super(ClsModule, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 9 * self.num_classes, kernel_size=3, padding=1),
                                      nn.Sigmoid()
                                      )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, -1, self.num_classes)

        return x


class RegModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 4 * 9, 3, padding=1),
                                      )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, -1, 4)

        return x


class depthwise_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class pointwise_conv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.pointwise(x)
        return out


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer=1):
        super().__init__()

        self.dwc = nn.Sequential(nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin),
                                 nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1),
                                 )

    def forward(self, x):
        out = self.dwc(x)
        return out


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=True, activ=True):
        super(double_conv, self).__init__()

        ops = []
        ops += [depthwise_separable_conv(in_ch, out_ch)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        if activ:
            ops += [nn.ReLU(inplace=True)]
        ops += [depthwise_separable_conv(out_ch, out_ch)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        if activ:
            ops += [nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=True, ceil=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=ceil),
            double_conv(in_ch, out_ch, normaliz=normaliz)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=True, activ=True):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        self.conv = double_conv(in_ch, out_ch,
                                normaliz=normaliz, activ=activ)

    def forward(self, x1, x2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, int(math.ceil(diffX / 2)),
                        diffY // 2, int(math.ceil(diffY / 2))))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DWS_Unet(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 device=torch.device('cuda')):
        super(DWS_Unet, self).__init__()
        self.device = device

        self.inc = inconv(n_channels, 64)

        self.down0_1 = down(64, 64)
        self.down0_2 = down(64, 128, ceil=True)
        self.down0_3 = down(128, 256, ceil=True)

        self.down1 = down(256, 256)
        self.down2 = down(256, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.down5 = down(512, 512)
        self.down6 = down(512, 512, normaliz=False)
        self.down7 = down(512, 512, normaliz=False)

        self.up1 = up(1024, 512)
        self.up2 = up(1024, 512)
        self.up3 = up(1024, 256)
        self.up4 = up(512, 256)
        self.up5 = up(512, 256)
        self.up6 = up(512, 256, activ=False)

        self.outc = outconv(256, n_classes)
        self.out_nonlin = nn.Sigmoid()

    def forward(self, x):
        x1_ = self.inc(x)
        x1_ = self.down0_1(x1_)
        x1_ = self.down0_2(x1_)
        x1 = self.down0_3(x1_)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)

        x = self.outc(x)
        x = self.out_nonlin(x)
        obj = x.squeeze(1)
        return obj


if __name__ == '__main__':
    img = torch.randn([2, 3, 800, 800])
    model = HNNA_DET(num_classes=1)
    output = model(img)
    print(output[0].size())
    print(output[1].size())
    print(output[2].size())

    import os
    device = torch.device('cpu')
    state_dict = torch.load('./saves/pretrained_model.pth.tar',
                             map_location=device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=True)
    print(model)

