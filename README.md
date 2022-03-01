# ESRGAN-Pytorch

<p align="center">
  <img src="https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/images/Intro.png" />
</p>

Pytorch Implementation of "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks", Wang et al. ECCV 2022

[Paper](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf) | [Official Implementation](https://github.com/xinntao/ESRGAN)

Single image super-resolution (SISR), as a fundamental low-level vision problem, has attracted increasing attention in the research community and AI companies. SISR aims at recovering a high-resolution (HR) image from a single
low-resolution (LR) one. Various network architecture designs and training strategies
have continuously improved the SR performance, especially the Peak Signal-toNoise Ratio (PSNR) value. However, these PSNR-oriented
approaches tend to output over-smoothed results without sufficient high-frequency
details, since the PSNR metric fundamentally disagrees with the subjective evaluation of human observers.

SRGAN the predecessor of the ESRGAN significantly improved the
overall visual quality of reconstruction over PSNR-oriented methods. However, there still exists a clear gap between SRGAN results and the
ground-truth (GT) images. The following are the improvements done over SRGAN architecture to improve the visual quality :

- Network architecture is improved by  introducing the Residual-in-Residual Dense
Block (RDDB), which is of higher capacity and easier to train.
- Batch Normalization (BN) layers are removed and residual scaling with smaller initialization is used to facilitate training a very deep network.
- The discriminator is improved by using Relativistic average GAN (RaGAN) which
learns to judge “whether one image is more realistic than the other” rather than
“whether one image is real or fake”.
- Improved perceptual loss by using the VGG features before activation instead of
after activation as in SRGAN.

## Network Architecture

<p align="center">
  <img src="https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/images/NetworkArchitecture-1.png" />
</p>

In order to further improve the recovered image quality of SRGAN  two modifications are done to the structure of generator G :

- Remove all BN layers
-  Replace the original basic block with the proposed Residual-in-Residual
Dense Block (RRDB), which combines multi-level residual network and dense
connections. 

<p align="center">
  <img src="https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/images/NetworkArchitecture-2.png" />
</p>

The ResidualDense Block architecture :

```python
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
```        

The Residual in Residual Dense Block architecture :

```python
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
```


Removing BN layers has proven to increase performance and reduce computational complexity in different PSNR-oriented tasks including SR and
deblurring. Further removing BN layers
helps to improve generalization ability and to reduce computational complexity
and memory usage. The proposed RRDB employs a deeper and more complex structure than the original
residual block in SRGAN.

In addition to the improved architecture several other techniques are employed to o facilitate training a very deep network:

- residual scaling i.e., scaling
down the residuals by multiplying a constant between 0 and 1 before adding them
to the main path to prevent instability
- smaller initialization, as we empirically
find residual architecture is easier to train when the initial parameter variance
becomes smaller. 

### Relativistic Discriminator

<p align="center">
  <img src="https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/images/Relativistic_Discriminator.png" />
</p>

Different from the standard discriminator D
in SRGAN, which estimates the probability that one input image x is real and
natural, a relativistic discriminator tries to predict the probability that a real
image xr is relatively more realistic than a fake one xf. Therefore, the generator
benefits from the gradients from both generated data and real data in adversarial
training, while in SRGAN only generated part takes effect.

### Perceptual Loss

A more effective  perceptual loss is developed by constraining on features before activation rather than after activation as practiced in SRGAN.
Previously in SRGAN Perceptual loss wad define on the activation layers of a pre-trained deep network, where the distance between two activated features is minimized. In ESRGAN the features before the activation layers is used which will overcome two drawbacks of the original design:

- The actiavted features are very sparse, especially after a very deep network. The sparse activation provides weak
supervision and thus leads to inferior performance.
- Using features after activation also causes inconsistent reconstructed brightness compared with the
ground-truth image.

```python
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
```


# Training

The dataset used for training is the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) which is a high-quality (2K
resolution) dataset for image restoration tasks. Evaluation is done on widely used benchmark dataset – [Set14](https://drive.google.com/file/d/1CzwwAtLSW9sog3acXj8s7Hg3S7kr2HiZ/view?usp=sharing). Download both the datasets and place it in the `data` folder.
Run the [esrgan-pytorch.ipynb](https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/esrgan-pytorch.ipynb) notebook to see the training and results obtained.

# Results 

## Original Downsampled 128x128 Image from [Set14](https://drive.google.com/file/d/1CzwwAtLSW9sog3acXj8s7Hg3S7kr2HiZ/view?usp=sharing) vs x4 Resolution Image

![test-image](https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/images/Original-Set14/baboon.png) ![generatedImage](https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/results/test/RRDBNet_baseline/x4/baboon.png)

![test-image](https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/images/Original-Set14/barbara.png) ![generatedImage](https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/results/test/RRDBNet_baseline/x4/barbara.png)

![test-image](https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/images/Original-Set14/lenna.png) ![generatedImage](https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/results/test/RRDBNet_baseline/x4/lenna.png)

![test-image](https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/images/Original-Set14/ppt3.png) ![generatedImage](https://github.com/Abhishek-Aditya-bs/ESRGAN-Pytorch/blob/main/results/test/RRDBNet_baseline/x4/ppt3.png)

# Citation

```
@misc{wang2018esrgan,
    title={ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks},
    author={Xintao Wang and Ke Yu and Shixiang Wu and Jinjin Gu and Yihao Liu and Chao Dong and Chen Change Loy and Yu Qiao and Xiaoou Tang},
    year={2018},
    eprint={1809.00219},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

# LICENSE

MIT














