import torch
from torch.backends import cudnn

torch.manual_seed(0)

device = torch.device("cuda", 0)

cudnn.benchmark = True

upscale_factor = 4

mode = "train_rrdbnet"

exp_name = "RRDBNet_baseline"

if mode == "train_rrdbnet":

    train_image_dir = "data/DIV2K/ESRGAN/train/DIV2K_train_HR"
    valid_image_dir = "data/DIV2K/ESRGAN/valid"

    image_size = 192
    batch_size = 16
    num_workers = 4

    resume = False
    strict = False
    start_epoch = 0
    resume_weight = ""

    epochs = 120

    model_lr = 2e-4
    model_betas = (0.9, 0.999)

    step_size = epochs // 5
    gamma = 0.5

    print_frequence = 100

if mode == "train_esrgan":
    train_image_dir = "data/DIV2K/ESRGAN/train/DIV2K_train_HR"
    valid_image_dir = "data/DIV2K/ESRGAN/valid"

    image_szie = 128
    batch_size = 16
    num_workers = 4

    resume = False
    strict = False
    start_epoch = 0
    resume_d_weight = ""
    resume_g_weight = ""

    epochs = 48

    pixel_weight = 1.0
    content_weight = 1.0
    adversarial_weight = 0.001

    d_model_lr = 1e-4
    d_model_betas = (0.9, 0.999)

    g_model_lr = 1e-4
    g_model_betas = (0.9, 0.999)

    d_optimizer_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    g_optimizer_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    d_optimizer_gamma = 0.5
    g_optimizer_gamma = 0.5

    print_frequency = 1000

if mode == "valid":
    lr_dir = f"data/Set14/LRbicx{upscale_factor}"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set14/GTmod12"

    model_path = f"results/{exp_name}/g-last.pth"

