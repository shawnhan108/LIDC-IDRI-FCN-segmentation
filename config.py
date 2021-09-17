import numpy as np
import torch
import datetime

class Config:
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0 #16

    # Dataset
    DATASET_IRC = "../../../projects/radiomics/Temp/sejin/LIDC_IDRI_save"
    MOVE = 5
    ROT = np.random.random(1)[0]/2 # or any scale. Change later.
    NOISE_SD = 0.316 # = sqrt(0.1)
    batch_size = 1 # 2 #4 #8 # 16

    # model
    BACKBONE = "resnet18"
    PRETRAINED = True
    NUM_CLASS = 2
    ckpt_src = "checkpoints/{}.pth".format(datetime.datetime.now().strftime("%m_%d_%H_%M"))

    # training
    L_RATE = 1e-3
    EPOCHS = 100
    MILESTONES = [0.5 * EPOCHS, 0.75 * EPOCHS]
    GAMMA = 0.1
    DECAY = 1e-4
    MOMENTUM = 0.9
    use_dice = True
    early_stop_tolerance = 8

    # print
    print_freq = 2
    tensorboard_freq = print_freq
