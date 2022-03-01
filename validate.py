import os
import torch
from PIL import Image
from natsort import natsorted

import config
from src import imgproc
from src import Generator

def main() -> None:
    print("Build RRDBNet mdoel...")
    model = Generator.to(config.device)
    print("Build RRDBNet model successfull.")

    print(f"Load RRDBNet model weights `{os.path.abspath(config.model_path)}`...")
    state_dict = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(state_dict)
    print(f"Load RRDBNet model weights `{os.path.abspath(config.model_path)}` successfully.")

    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model.eval()
    model.half()

    total_psnr = 0.0
    file_names = natsorted(os.listdir(config.lr_dir))
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")

        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp_(0,1)

        sr_y_tensor = imgproc.convert_rgb_to_y(sr_tensor)
        hr_y_tensor = imgproc.convert_rgb_to_y(hr_tensor)
        total_psnr += 10. * torch.log10(1. / torch.mean((sr_y_tensor - hr_y_tensor) ** 2))

        sr_image = imgproc.tensor2image(sr_tensor, range_norm=False, half=True)
        sr_image = Image.fromarray(sr_image)
        sr_image.save(sr_image_path)

    print(f"PSNR: {total_psnr / total_files:.2f} dB.\n")


if __name__ == "__main__":
    main()

