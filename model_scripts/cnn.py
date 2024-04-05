from torch import nn
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvEncoder(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.name = 'Conv_v1'
        self.l1 = DepthwiseSeparableConv(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.l2 = DepthwiseSeparableConv(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.l3 = DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.l4 = DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.max_pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.embedding = nn.Linear(256 * args.latent_dim, args.latent_dim)
        self.log_var = nn.Linear(256 * args.latent_dim, args.latent_dim)

    def forward(self, x):

        x = self.l1(x)
        x = self.max_pool(x)
        x = self.l2(x)
        x = self.max_pool(x)
        x = self.l3(x)
        x = self.max_pool(x)
        x = self.l4(x)
        x = self.max_pool(x)

        x = self.flat(x)
        output = ModelOutput(
            embedding=self.embedding(x),
            log_covariance=self.log_var(x)
        )
        return output


class ConvDecoder(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.latent_features = args.latent_dim
        self.name = 'Conv_v1'

        self.l1 = DepthwiseSeparableConv(in_channels=args.latent_dim, out_channels=256, kernel_size=3, stride=1)
        self.l2 = DepthwiseSeparableConv(in_channels=256, out_channels=128, kernel_size=3, stride=1)
        self.l3 = DepthwiseSeparableConv(in_channels=128, out_channels=64, kernel_size=3, stride=1)
        self.l4 = DepthwiseSeparableConv(in_channels=64, out_channels=32, kernel_size=3, stride=1)

        self.final = nn.Conv2d(32, 1, kernel_size=1, padding='same')
        self.final_act = nn.Sigmoid()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.side = int(args.latent_dim ** 0.5)
        self.linear = nn.Linear(args.latent_dim, args.latent_dim * self.side * self.side)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), self.latent_features, self.side, self.side)
        x = self.l1(x)
        x = self.upsample(x)
        x = self.l2(x)
        x = self.upsample(x)
        x = self.l3(x)
        x = self.upsample(x)
        x = self.l4(x)
        x = self.upsample(x)

        x = self.final(x)
        x = self.final_act(x)
        output = ModelOutput(reconstruction=x)
        return output
