import warnings
import os

import torch 
import torch.nn as nn 
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

# from tensorboardX import SummaryWriter
# import tqdm

from dataset import LIDCSegDataset
from resnet import FCNResNet #  FCNVGG, FCNDenseNet

from config import Config as cfg
from conv25d_converter import Conv2_5dConverter
from utils import * 
from hausdorff_loss import HausdorffDTLoss

from tensorboardX import SummaryWriter
import GPUtil

logger = get_logger()
writer = SummaryWriter()

# init dataset
train_dataset = LIDCSegDataset()
test_dataset = LIDCSegDataset(train=False)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=custom_collate,
                            pin_memory=(torch.cuda.is_available()), num_workers=cfg.NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate,
                            pin_memory=(torch.cuda.is_available()), num_workers=cfg.NUM_WORKERS)
logger.info("Data loaded")

# init 2.5d conv model
# model_dict = {'resnet18': FCNResNet, 'vgg16': FCNVGG, 'densenet121': FCNDenseNet}
model_dict = {'resnet18': FCNResNet}
model = model_dict[cfg.BACKBONE](pretrained=cfg.PRETRAINED, num_classes=cfg.NUM_CLASS, backbone=cfg.BACKBONE)
model = Conv2_5dConverter(model)
# model = model_to_syncbn(Conv2_5dConverter(model))

GPUtil.showUtilization()
model = model.to(cfg.device)
GPUtil.showUtilization()

# if torch.cuda.is_available():
#     model = nn.DataParallel(model)

logger.info("Pretrained model loaded")

optim = Adam(model.parameters(), lr=cfg.L_RATE)
optim_scheduler = lr_scheduler.MultiStepLR(optim, milestones=cfg.MILESTONES, gamma=cfg.GAMMA)

# loss
if not cfg.use_dice:
    criterion_hausdorff = HausdorffDTLoss()

best_loss = float('inf')
epochs_since_improvement = 0

for e_idx in range(cfg.EPOCHS):
    
    logger.info("Training Epoch {}/{}".format(e_idx, cfg.EPOCHS))

    # train
    model.train()
    trainLossMeter = LossMeter()

    for idx, (img, mask) in enumerate(train_loader):
        GPUtil.showUtilization()
        orig_img, mask_img = img.float().to(cfg.device), mask.float().to(cfg.device)
        print(img.shape, mask.shape, orig_img.shape, orig_img.dtype, mask_img.dtype)
        GPUtil.showUtilization()
        # debug_memory()
        print("model:", model)
        pred_logit = model(orig_img)
        y_one_hot = categorical_to_one_hot(mask_img, dim=1, expand_dim=False)

        if cfg.use_dice:
            loss = soft_dice_loss(pred_logit, y_one_hot)
        else:
            # print(pred_logit.shape, mask_img.shape)
            loss = criterion_hausdorff(pred_logit, mask_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainLossMeter.update(loss.item())

        # print status
        if (idx+1) % cfg.print_freq == 0:
            status = 'Train: [{0}][{1}/{2}]\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(e_idx+1, idx+1, len(train_loader), loss=trainLossMeter)
            logger.info(status)

        # log loss to tensorboard 
        if (idx+1) % cfg.tensorboard_freq == 0:
            writer.add_scalar('Train_Loss_{0}'.format(cfg.tensorboard_freq), 
                            trainLossMeter.avg, 
                            e_idx * (len(train_loader) / cfg.tensorboard_freq) + (idx+1) / cfg.tensorboard_freq)

        print("DONE ONE")

    # test
    model.eval()
    validLossMeter = LossMeter()
    with torch.no_grad():
        for idx, (img, mask) in enumerate(test_loader):
            orig_img, mask_img = img.float().to(cfg.device), mask.float().to(cfg.device)

            pred_logit = model(orig_img)
            y_one_hot = categorical_to_one_hot(mask_img, dim=1, expand_dim=False)

            if cfg.use_dice:
                loss = soft_dice_loss(pred_logit, y_one_hot)
            else:
                # print(pred_logit.shape, mask_img.shape)
                loss = criterion_hausdorff(pred_logit, mask_img)

            validLossMeter.update(loss.item())

            # print status
            if (idx+1) % cfg.print_freq == 0:
                status = 'Test: [{0}][{1}/{2}]\t' \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(e_idx+1, idx+1, len(test_loader), loss=validLossMeter)
                logger.info(status)

            # log loss to tensorboard 
            if (idx+1) % cfg.tensorboard_freq == 0:
                writer.add_scalar('Test_Loss_{0}'.format(cfg.tensorboard_freq), validLossMeter.avg, 
                                e_idx * (len(test_loader) / cfg.tensorboard_freq) + (idx+1) / cfg.tensorboard_freq)

    scheduler.step()
    writer.add_scalar('Train_Loss_epoch', trainLossMeter.avg, e_idx)
    writer.add_scalar('Test_Loss_epoch', validLossMeter.avg, e_idx)

    valid_loss = validLossMeter.avg
    is_best = valid_loss < best_loss
    best_loss_tmp = min(valid_loss, best_loss)

    if not is_best:
        epochs_since_improvement += 1
        logger.info("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        if epochs_since_improvement == cfg.early_stop_tolerance:
            break # early stopping.
    else:
        epochs_since_improvement = 0
        state = {
            'epoch': epoch,
            'loss': best_loss_tmp,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
        }
        torch.save(state, cfg.ckpt_src)
        logger.info("Checkpoint updated.")
        best_loss = best_loss_tmp

writer.close()
