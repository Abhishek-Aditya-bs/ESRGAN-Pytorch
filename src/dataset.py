import io
import os
import lmdb
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode
from .imgproc import image2tensor, random_crop, random_horizontally_flip, random_rotate, center_crop

class ImageDataset(Dataset):
    def __init__(self, dataroot : str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(ImageDataset, self).__init__()
        self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]

        if mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(image_size),
                transforms.RandomRotation([0, 90]),
                transforms.RandomHorizontalFlip(0.5)
            ])
        elif mode == "valid":
            self.hr_transforms = transforms.CenterCrop(image_size)
        else:
            raise "Unsupported data processing model, please use `train` or `valid`."

        self.lr_transforms = transforms.Resize(image_size // upscale_factor, interpolation=IMode.BICUBIC)

    def __getitem__(self, batch_index: int):
        image = Image.open(self.filenames[batch_index])

        hr_image = self.hr_transforms(image)
        lr_image = self.lr_transforms(hr_image)

        lr_tensor = image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = image2tensor(hr_image, range_norm=False, half=False)

        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.filenames)

class LMDBDataset(Dataset):
    def __init__(self, lr_lmdb_path: str, hr_lmdb_path : str, image_size : int, upscale_factor: int, mode: str) -> None:
        super(LMDBDataset, self).__init__()
        self.image_size = image_size
        self.upscale_factor = upscale_factor
        self.mode = mode

        self.lr_datasets = []
        self.hr_datasets = []

        self.lr_lmdb_path = lr_lmdb_path
        self.hr_lmdb_path = hr_lmdb_path

        self.read_lmdb_dataset()

    def __getitem__(self, batch_index: int):
        lr_image = self.lr_datasets[batch_index]
        hr_image = self.hr_datasets[batch_index]

        if self.mode == "train":
            if self.mode == "train:":
                lr_image, hr_image = random_crop(lr_image, hr_image, image_size=self.image_size, upscale_factor=self.upscale_factor)
                lr_image, hr_image = random_rotate(lr_image, hr_image, degrees=[0, 90])
                lr_image, hr_image = random_horizontally_flip(lr_image, hr_image, p=0.5)
        elif self.mode == "valid:":
            lr_image, hr_image = center_crop(lr_image, hr_image, image_size=self.image_size, upscale_factor=self.upscale_factor)
        else:
            raise "Unsupported data processing model, please use `train` or `valid`."

        lr_tensor = image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = image2tensor(hr_image, range_norm=False, half=False)

        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.hr_datasets)

    def read_lmdb_dataset(self):
        lr_lmdb_env = lmdb.open(self.lr_lmdb_path)
        hr_lmdb_env = lmdb.open(self.hr_lmdb_path)

        for _, image_bytes in lr_lmdb_env.begin().cursor():
            image = Image.open(io.BytesIO(image_bytes))
            self.lr_datasets.append(image)

        for _, image_bytes in hr_lmdb_env.begin().cursor():
            image = Image.open(io.BytesIO(image_bytes))
            self.hr_datasets.append(image)

