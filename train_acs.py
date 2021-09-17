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
from torchvision.models import resnet18
from resnet import FCNResNet #  FCNVGG, FCNDenseNet

from config import Config as cfg
from conv25d_converter import Conv2_5dConverter
from ACSConv.acsconv.converters import ACSConverter
from utils import * 
from hausdorff_loss import HausdorffDTLoss

from tensorboardX import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
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

class LIDCSeg(pl.LightningModule):

    def __init__(self):
        super().__init__()
        model_dict = {'resnet18': FCNResNet}
        model = model_dict[cfg.BACKBONE](pretrained=cfg.PRETRAINED, num_classes=cfg.NUM_CLASS, backbone=cfg.BACKBONE)
        self.model = Conv2_5dConverter(resnet18(pretrained=True)).model
        # self.model = ACSConverter(model).model
        self.model = torch.nn.DataParallel(self.model)
        
        GPUtil.showUtilization()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # self.model.train()
        
        if not cfg.use_dice:
            criterion_hausdorff = HausdorffDTLoss()
        
        orig_img, mask_img = batch
        orig_img, mask_img = orig_img.to(cfg.device), mask_img.to(cfg.device)
        print("in train:")
        print(orig_img.shape, mask_img.shape)
        # orig_img, mask_img =  torch.Tensor(orig_img, requires_grad=True), torch.Tensor(mask_img, requires_grad=True)
        # orig_img, mask_img = orig_img.float().to(cfg.device), mask_img.float().to(cfg.device)

        pred_logit = self(orig_img)
        y_one_hot = categorical_to_one_hot(mask_img, dim=1, expand_dim=False)

        if cfg.use_dice:
            loss = soft_dice_loss(pred_logit, y_one_hot)
        else:
            # print(pred_logit.shape, mask_img.shape)
            loss = criterion_hausdorff(pred_logit, mask_img)
        
        self.log("train_loss_batch", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_scalar('train_loss_batch'.format(loss), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # self.model.eval()

        if not cfg.use_dice:
            criterion_hausdorff = HausdorffDTLoss()
        
        orig_img, mask_img = batch
        print(orig_img.shape, mask_img.shape)
        GPUtil.showUtilization()
        # orig_img, mask_img =  torch.Tensor(orig_img, requires_grad=True), torch.Tensor(mask_img, requires_grad=True)

        # orig_img, mask_img = orig_img.float().to(cfg.device), mask_img.float().to(cfg.device)
        with torch.no_grad():
            print("in valid:")
            print(orig_img.shape, mask_img.shape)
            pred_logit = self.model(orig_img)
            y_one_hot = categorical_to_one_hot(mask_img, dim=1, expand_dim=False)

            if cfg.use_dice:
                loss = soft_dice_loss(pred_logit, y_one_hot)
            else:
                loss = criterion_hausdorff(pred_logit, mask_img)

        self.log("valid_loss_batch", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_scalar('valid_loss_batch'.format(loss), on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=cfg.L_RATE)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MILESTONES, gamma=cfg.GAMMA)
        return [optimizer], [scheduler]

m = LIDCSeg()

# Initialize a trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="LIDC25D-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)
trainer = pl.Trainer(gpus=1, 
                    max_epochs=cfg.EPOCHS, 
                    progress_bar_refresh_rate=2, 
                    callbacks=[early_stop_callback, checkpoint_callback],
                    default_root_dir="./checkpoints")
trainer.fit(m, train_loader, test_loader)








# best_loss = float('inf')
# epochs_since_improvement = 0


#     writer.add_scalar('Train_Loss_epoch', trainLossMeter.avg, e_idx)
#     writer.add_scalar('Test_Loss_epoch', validLossMeter.avg, e_idx)

#     valid_loss = validLossMeter.avg
#     is_best = valid_loss < best_loss
#     best_loss_tmp = min(valid_loss, best_loss)

#     if not is_best:
#         epochs_since_improvement += 1
#         logger.info("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
#         if epochs_since_improvement == cfg.early_stop_tolerance:
#             break # early stopping.
#     else:
#         epochs_since_improvement = 0
#         state = {
#             'epoch': epoch,
#             'loss': best_loss_tmp,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optim.state_dict(),
#         }
#         torch.save(state, cfg.ckpt_src)
#         logger.info("Checkpoint updated.")
#         best_loss = best_loss_tmp

# writer.close()
