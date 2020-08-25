import registry
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class WS(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None):
        super(WS, self).__init__()
        block = InvertedResidual
        input_channel = 8
        last_channel = 320

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 24, 1, 1],
                [6, 32, 1, 2],
                [6, 32, 1, 1],
                [6, 64, 1, 2],
                [6, 64, 1, 1],
                [6, 96, 1, 2],
                [6, 160, 1, 2],
                #[6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.extras = nn.ModuleList([
            InvertedResidual(320, 512, 2, 0.2),
            InvertedResidual(512, 256, 2, 0.25),
            #InvertedResidual(256, 256, 2, 0.5),
            InvertedResidual(256, 64, 2, 0.25)
        ])

        self.down_list = nn.ModuleList(
            [
                nn.Conv2d(128, 128, 3, stride=2, padding=1),
                nn.Conv2d(128, 128, 3, stride=2, padding=1),
                nn.Conv2d(128, 128, 3, stride=2, padding=1),
                nn.Conv2d(128, 128, 3, stride=2, padding=1),
                nn.Conv2d(128, 128, 3, stride=2, padding=1),
                nn.Conv2d(128, 128, 3, stride=2, padding=1)
            ]
        )
        self.pafpn_list = nn.ModuleList(
            [
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1)
            ]
        )

        self.fpn_list = nn.ModuleList(
            [
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.Conv2d(128, 128, 3, padding=1)
            ]
        )
        self.lateral_list = nn.ModuleList(
            [
                nn.Conv2d(64, 128, 1),
                nn.Conv2d(96, 128, 1),
                nn.Conv2d(320, 128, 1),
                nn.Conv2d(512, 128, 1),
                nn.Conv2d(256, 128, 1),
                nn.Conv2d(64, 128, 1)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = []
        for i in range(8):
            x = self.features[i](x)
        features.append(x)
        x = self.features[9](self.features[8](x))
        features.append(x)

        for i in range(10, 12):

            x = self.features[i](x)
        features.append(x)


        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)

        laterals = []
        for i in range(len(features)):
            laterals.append(self.lateral_list[i](features[i]))

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += nn.functional.interpolate(laterals[i], size=(
            int(laterals[i - 1].shape[2]), int(laterals[i - 1].shape[3])), mode="nearest")

        for i in range(5):
            laterals[i + 1] += self.down_list[i](laterals[i])

        for i in range(1,6):
            laterals[i] = self.pafpn_list[i](laterals[i])

        return tuple(laterals)


@registry.BackBone.register('wangshuaiNet')
def getWS():
    model = WS()
    # if pretrained:
    #    model.load_state_dict(load_state_dict_from_url(model_urls['mobilenet_v2']), strict=False)
    return model

#
m = WS()
m.eval()
import torch
t = torch.randn((1,3,512,512))
m(t)
