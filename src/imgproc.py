import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


def normalize(image: np.ndarray) -> np.ndarray:
    
    return image.astype(np.float64) / 255.0


def unnormalize(image: np.ndarray) -> np.ndarray:
    
    return image.astype(np.float64) * 255.0


def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    
    tensor = F.to_tensor(image)

    if range_norm:
        tensor = tensor.mul_(2.0).sub_(1.0)
    if half:
        tensor = tensor.half()

    return tensor


def tensor2image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any:
    

    if range_norm:
        tensor = tensor.add_(1.0).div_(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze_(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).cpu().numpy().astype("uint8")

    return image


def convert_rgb_to_y(image: Any) -> Any:
    
    if type(image) == np.ndarray:
        return 16. + (64.738 * image[:, :, 0] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 2]) / 256.
    elif type(image) == torch.Tensor:
        if len(image.shape) == 4:
            image = image.squeeze_(0)
        return 16. + (64.738 * image[0, :, :] + 129.057 * image[1, :, :] + 25.064 * image[2, :, :]) / 256.
    else:
        raise Exception("Unknown Type", type(image))


def convert_rgb_to_ycbcr(image: Any) -> Any:
    
    if type(image) == np.ndarray:
        y = 16. + (64.738 * image[:, :, 0] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 2]) / 256.
        cb = 128. + (-37.945 * image[:, :, 0] - 74.494 * image[:, :, 1] + 112.439 * image[:, :, 2]) / 256.
        cr = 128. + (112.439 * image[:, :, 0] - 94.154 * image[:, :, 1] - 18.285 * image[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(image) == torch.Tensor:
        if len(image.shape) == 4:
            image = image.squeeze(0)
        y = 16. + (64.738 * image[0, :, :] + 129.057 * image[1, :, :] + 25.064 * image[2, :, :]) / 256.
        cb = 128. + (-37.945 * image[0, :, :] - 74.494 * image[1, :, :] + 112.439 * image[2, :, :]) / 256.
        cr = 128. + (112.439 * image[0, :, :] - 94.154 * image[1, :, :] - 18.285 * image[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception("Unknown Type", type(image))


def convert_ycbcr_to_rgb(image: Any) -> Any:
    
    if type(image) == np.ndarray:
        r = 298.082 * image[:, :, 0] / 256. + 408.583 * image[:, :, 2] / 256. - 222.921
        g = 298.082 * image[:, :, 0] / 256. - 100.291 * image[:, :, 1] / 256. - 208.120 * image[:, :, 2] / 256. + 135.576
        b = 298.082 * image[:, :, 0] / 256. + 516.412 * image[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(image) == torch.Tensor:
        if len(image.shape) == 4:
            image = image.squeeze(0)
        r = 298.082 * image[0, :, :] / 256. + 408.583 * image[2, :, :] / 256. - 222.921
        g = 298.082 * image[0, :, :] / 256. - 100.291 * image[1, :, :] / 256. - 208.120 * image[2, :, :] / 256. + 135.576
        b = 298.082 * image[0, :, :] / 256. + 516.412 * image[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception("Unknown Type", type(image))


def center_crop(lr: Any, hr: Any, image_size: int, upscale_factor: int):
    
    w, h = hr.size

    left = (w - image_size) // 2
    top = (h - image_size) // 2
    right = left + image_size
    bottom = top + image_size

    lr = lr.crop((left // upscale_factor,
                  top // upscale_factor,
                  right // upscale_factor,
                  bottom // upscale_factor))
    hr = hr.crop((left, top, right, bottom))

    return lr, hr


def random_crop(lr: Any, hr: Any, image_size: int, upscale_factor: int):
    

    w, h = hr.size
    left = torch.randint(0, w - image_size + 1, size=(1,)).item()
    top = torch.randint(0, h - image_size + 1, size=(1,)).item()
    right = left + image_size
    bottom = top + image_size

    lr = lr.crop((left // upscale_factor,
                  top // upscale_factor,
                  right // upscale_factor,
                  bottom // upscale_factor))
    hr = hr.crop((left, top, right, bottom))

    return lr, hr


def random_rotate(lr: Any, hr: Any, degrees: list):
    
    angle = random.choice(degrees)
    lr = F.rotate(lr, angle)
    hr = F.rotate(hr, angle)

    return lr, hr


def random_horizontally_flip(lr: Any, hr: Any, p=0.5):
    
    if torch.rand(1).item() > p:
        lr = F.hflip(lr)
        hr = F.hflip(hr)

    return lr, hr


def random_vertically_flip(lr: Any, hr: Any, p=0.5):
    
    if torch.rand(1).item() > p:
        lr = F.vflip(lr)
        hr = F.vflip(hr)

    return lr, hr


def random_adjust_brightness(lr: Any, hr: Any, factor: list):
    
    brightness_factor = random.choice(factor)
    lr = F.adjust_brightness(lr, brightness_factor)
    hr = F.adjust_brightness(hr, brightness_factor)

    return lr, hr


def random_adjust_contrast(lr: Any, hr: Any, factor: list):
    
    contrast_factor = random.choice(factor)
    lr = F.adjust_contrast(lr, contrast_factor)
    hr = F.adjust_contrast(hr, contrast_factor)

    return lr, hr
