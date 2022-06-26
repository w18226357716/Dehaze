import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_networks import Encoder_MDCBlock1, Decoder_MDCBlock1

def make_model(args, parent=False):
    return Net()

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class Net(nn.Module):
    def __init__(self, res_blocks=18, num_features=16):
        super(Net, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.fusion1 = Encoder_MDCBlock1(32, 2, mode='iter2')
        self.dense1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.fusion2 = Encoder_MDCBlock1(64, 3, mode='iter2')
        self.dense2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.fusion3 = Encoder_MDCBlock1(128, 4, mode='iter2')
        self.dense3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.fusion4 = Encoder_MDCBlock1(256, 5, mode='iter2')
        #self.dense4 = Dense_Block(256, 256)

        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.fusion_4 = Decoder_MDCBlock1(128, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.fusion_3 = Decoder_MDCBlock1(64, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.fusion_2 = Decoder_MDCBlock1(32, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.fusion_1 = Decoder_MDCBlock1(16, 5, mode='iter2')

        self.t = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 1, kernel_size=1), nn.Sigmoid()
        )
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, 1, kernel_size=1), nn.Sigmoid()
        )

        self.j1 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j2 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j3 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )
        self.j4 = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 3, kernel_size=1)
        )

        self.attention_fusion = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1), nn.SELU(),
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1), nn.SELU(),
            nn.Conv2d(num_features // 2, 15, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.SELU) or isinstance(m, nn.ReLU):
                m.inplace = True

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)


    def forward(self, x, x0_hd=None):
        x0 = x
        res1x = self.conv_input(x)
        feature_mem = [res1x]
        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(x)
        res2x = self.fusion1(res2x, feature_mem)
        feature_mem.append(res2x)
        res2x =self.dense1(res2x) + res2x

        res4x =self.conv4x(res2x)
        res4x = self.fusion2(res4x, feature_mem)
        feature_mem.append(res4x)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        res8x = self.fusion3(res8x, feature_mem)
        feature_mem.append(res8x)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)
        res16x = self.fusion4(res16x, feature_mem)
        #res16x = self.dense4(res16x)

        res_dehaze = res16x
        in_ft = res16x*2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze
        feature_mem_up = [res16x]

        res16x = self.convd16x(res16x)
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x 
        res8x = self.fusion_4(res8x, feature_mem_up)
        feature_mem_up.append(res8x)

        res8x = self.convd8x(res8x)
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x = self.fusion_3(res4x, feature_mem_up)
        feature_mem_up.append(res4x)

        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x 
        res2x = self.fusion_2(res2x, feature_mem_up)
        feature_mem_up.append(res2x)

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x 
        x = self.fusion_1(x, feature_mem_up)


        if x0_hd is not None:
            x0 = x0_hd
            x = (x0 - self.mean) / self.std

        log_x0 = torch.log(x0.clamp(min=1e-8))
        log_log_x0_inverse = torch.log(torch.log(1 / x0.clamp(min=1e-8, max=(1 - 1e-8))))

        # J0 = (I - A0 * (1 - T0)) / T0
        a = self.a(x)
        t = F.upsample(self.t(x), size=x0.size()[2:], mode='bilinear')
        x_phy = ((x0 - a * (1 - t)) / t.clamp(min=1e-8)).clamp(min=0., max=1.)

        # J2 = I * exp(R2)
        r1 = F.upsample(self.j1(x), size=x0.size()[2:], mode='bilinear')
        x_j1 = torch.exp(log_x0 + r1).clamp(min=0., max=1.)

        # J2 = I + R2
        r2 = F.upsample(self.j2(x), size=x0.size()[2:], mode='bilinear')
        x_j2 = (x0 + r2).clamp(min=0., max=1.)

        #
        r3 = F.upsample(self.j3(x), size=x0.size()[2:], mode='bilinear')
        x_j3 = torch.exp(-torch.exp(log_log_x0_inverse + r3)).clamp(min=0., max=1.)

        # J4 = log(1 + I * R4)
        r4 = F.upsample(self.j4(x), size=x0.size()[2:], mode='bilinear')
        # x_j4 = (torch.log(1 + r4 * x0)).clamp(min=0, max=1)
        x_j4 = (torch.log(1 + torch.exp(log_x0 + r4))).clamp(min=0., max=1.)

        attention_fusion = F.upsample(self.attention_fusion(x), size=x0.size()[2:], mode='bilinear')
        x_f0 = torch.sum(F.softmax(attention_fusion[:, :5, :, :], 1) *
                         torch.stack((x_phy[:, 0, :, :], x_j1[:, 0, :, :], x_j2[:, 0, :, :],
                                      x_j3[:, 0, :, :], x_j4[:, 0, :, :]), 1), 1, True)
        x_f1 = torch.sum(F.softmax(attention_fusion[:, 5: 10, :, :], 1) *
                         torch.stack((x_phy[:, 1, :, :], x_j1[:, 1, :, :], x_j2[:, 1, :, :],
                                      x_j3[:, 1, :, :], x_j4[:, 1, :, :]), 1), 1, True)
        x_f2 = torch.sum(F.softmax(attention_fusion[:, 10:, :, :], 1) *
                         torch.stack((x_phy[:, 2, :, :], x_j1[:, 2, :, :], x_j2[:, 2, :, :],
                                      x_j3[:, 2, :, :], x_j4[:, 2, :, :]), 1), 1, True)
        x_fusion = torch.cat((x_f0, x_f1, x_f2), 1).clamp(min=0., max=1.)


        return x_fusion, x_phy, x_j1, x_j2, x_j3, x_j4
