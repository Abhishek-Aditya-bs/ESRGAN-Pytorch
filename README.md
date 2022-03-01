# ESRGAN-Pytorch

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

In order to further improve the recovered image quality of SRGAN  two modifications are done to the structure of generator G :

- Remove all BN layers
-  Replace the original basic block with the proposed Residual-in-Residual
Dense Block (RRDB), which combines multi-level residual network and dense
connections. 

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

Different from the standard discriminator D
in SRGAN, which estimates the probability that one input image x is real and
natural, a relativistic discriminator tries to predict the probability that a real
image xr is relatively more realistic than a fake one xf. Therefore, the generator
benefits from the gradients from both generated data and real data in adversarial
training, while in SRGAN only generated part takes effect.

# Results

The dataset used for training is the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) which is a high-quality (2K
resolution) dataset for image restoration tasks. Evaluation is done on widely used benchmark dataset – Set14. Run the [esrgan-pytorch.ipynb]() notebook to see the training and results obtained.














