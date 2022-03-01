from turtle import forward
from numpy import identity
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResidualDenseBlock(nn.Module):
    def __init__(self, inp_channels: int, growths: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels + growths * 0, growths, (3,3), (1,1), (1,1))
        self.conv2 = nn.Conv2d(inp_channels + growths * 1, growths, (3,3), (1,1), (1,1))
        self.conv3 = nn.Conv2d(inp_channels + growths * 2, growths, (3,3), (1,1), (1,1))
        self.conv4 = nn.Conv2d(inp_channels + growths * 3, growths, (3,3), (1,1), (1,1))
        self.conv5 = nn.Conv2d(inp_channels + growths * 4, inp_channels, (3,3), (1,1), (1,1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1],1)))
        out3 = self.leaky_relu(self.conv2(torch.cat([x, out1, out2],1)))
        out4 = self.leaky_relu(self.conv2(torch.cat([x, out1, out2, out3],1)))
        out5 = self.leaky_relu(self.conv2(torch.cat([x, out1, out2, out3, out4],1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out

class ResidualResidualDenseBlock(nn.Module):
    def __init__(self, inp_channels: int, growths: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(inp_channels, growths)
        self.rdb2 = ResidualDenseBlock(inp_channels, growths)
        self.rdb3 = ResidualDenseBlock(inp_channels, growths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = self.mul(out, 0.2)
        out = torch.add(out, identity)

        return out

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size - 3 x 128 x 128
            nn.Conv2d(3, 64, (3,3), (1,1), (1,1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size - 64 x 64 x 64
             nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3,3), (1,1), (1,1))

        trunk = []
        for _ in range(23):
            trunk.append(ResidualResidualDenseBlock(64, 32))
        self.trunk = nn.Sequential(*trunk)

        self.conv2 = nn.Conv2d(64, 64, (3,3), (1,1), (1,1))

        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 64, (3,3), (1,1), (1,1)), 
            nn.LeakyReLU(0.2, True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3,3), (1,1), (1,1)), 
            nn.LeakyReLU(0.2, True)
        )

        self.conv4 = nn.Conv2d(64, 3, (3,3), (1,1), (1,1))

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv3(out)
        out = self.conv4(out)

        return out

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

class ContentLoss(nn.Module):
    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        vgg19 = models.vgg19(pretrained=True).eval()
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35])

        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr = sr.sub(self.mean).div(self.std)
        hr = hr.sub(self.mean).div(self.std)

        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss


        



