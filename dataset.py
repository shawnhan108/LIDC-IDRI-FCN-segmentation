import torch
from torch.utils.data import Dataset
# import random
import os
import numpy as np
import pandas as pd

import nrrd 
import SimpleITK as sitk

from config import Config as cfg

# from mylib.voxel_transform import rotation, reflection, crop, random_center
# from mylib.utils import _triple, categorical_to_one_hot

class SITK_Transform():
    def __init__(self, image, mask, move, noise_sd, rot, empty_mask=False):
        self.image = sitk.GetImageFromArray(image)
        self.mask = sitk.GetImageFromArray(mask)
        self.noise_sd = noise_sd
        self.rot = rot 
        self.move = move
        self.empty_mask = empty_mask
    
    def find_centre(self):
        centre_idx = np.array(self.image.GetSize()) / 2.
        offset = np.random.randint(-self.move, self.move + 1, size=3)
        return np.array(self.image.TransformContinuousIndexToPhysicalPoint(centre_idx)) + offset

    def find_centroid(self):
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(sitk.Cast(self.mask, sitk.sitkUInt8))
        centroid_coords = stats.GetCentroid(1)
        return np.asarray(centroid_coords, dtype=np.float64)

    def add_noise(self):
        """add noise and convert to np array"""
        noise = np.random.normal(0, self.noise_sd, self.image.GetSize()[::-1]).astype(np.float32)
        # noise = sitk.GetImageFromArray(noise)
        # noise.CopyInformation(self.image)
        self.image = sitk.GetArrayFromImage(self.image) + noise

        # noise = np.random.normal(0, self.noise_sd, self.image.shape).astype(np.float32)
        # noise = sitk.GetImageFromArray(noise).CopyInformation(x)
        # return self.image + noise

    def reflect(self, axis):
        if axis != -1:
            ref1 = np.flip(self.image, axis)
            ref2 = np.flip(self.mask, axis)
        else:
            ref1 = np.copy(self.image)
            ref2 = np.copy(self.mask)
        return ref1, ref2

    def sitk_transform(self, to_copy=True):
        
        image_ref = sitk.Image(self.image.GetSize(), sitk.sitkFloat32)
        # mask_ref = sitk.Image(self.mask.GetSize(), sitk.sitkFloat32)
        # TODO: Set spacing or origin???
        
        # transform_group = sitk.Transform(3, sitk.sitkComposite)
        img_center = self.find_centre()
        affineT = sitk.AffineTransform(3)
        affineT.SetCenter(img_center)
        affineT.Rotate(0, 1, self.rot)
        # affineT.ToTensor()

        if not self.empty_mask:
            translation = sitk.TranslationTransform(3, (self.find_centroid() - img_center).tolist())
            transform_group = sitk.CompositeTransform(translation)
            transform_group.AddTransform(affineT)
        else:
            transform_group = affineT

        img_fill_val = float(sitk.GetArrayViewFromImage(self.image).min())
        # msk_fill_val = float(sitk.GetArrayViewFromImage(self.mask).min())

        self.image = sitk.Resample(self.image, image_ref, transform_group, sitk.sitkLinear, img_fill_val)
        self.add_noise()
        # self.mask = sitk.GetArrayFromImage(sitk.Resample(self.mask, mask_ref, transform_group, sitk.sitkLinear, msk_fill_val))

        # self.image = sitk.GetArrayFromImage(self.image)
        self.mask = sitk.GetArrayFromImage(self.mask)
        self.image, self.mask = self.reflect(0) if np.random.random(1)[0] > 0.5 else self.reflect(1) # TODO: which orientation to reflect?

        # if to_copy:
        #     self.image = np.stack([self.image,self.image,self.image],0)

        return torch.from_numpy(self.image.astype(np.float32)), torch.from_numpy(self.mask.astype(np.float32))

class LIDCSegDataset(Dataset):
    def __init__(self, data_path=cfg.DATASET_IRC, train=True, copy_channels=True):
        super().__init__()
        self.data_path = data_path
        self.copy_channels = copy_channels

        df = pd.read_csv(os.path.join(data_path, 'fmap.csv'))
        if train:    
            self.names = df[df['split']=="train"]['image'].values
        else:
            self.names = df[df['split']=="test"]['image'].values

    def __getitem__(self, index):
        
        img, _ = nrrd.read(os.path.join(self.data_path, "images", self.names[index].split('/')[-1]))
        msk, _ = nrrd.read(os.path.join(self.data_path, "masks", self.names[index].split('/')[-1]))

        # img = sitk.ReadImage(os.path.join(self.data_path, "images", self.names[index].split('/')[-1]))
        # msk = sitk.ReadImage(os.path.join(self.data_path, "masks", self.names[index].split('/')[-1]))
        
        if len(msk.shape) == 0:
            msk = np.zeros(img.shape)
            transform = SITK_Transform(img, msk, cfg.MOVE, cfg.NOISE_SD, cfg.ROT, empty_mask=True)
        else:
            transform = SITK_Transform(img, msk, cfg.MOVE, cfg.NOISE_SD, cfg.ROT)
        
        img, msk = transform.sitk_transform(self.copy_channels)

        # img, msk = torch.from_numpy(img.astype(np.float32)), torch.from_numpy(msk.astype(np.float32))

        return img, msk

    def __len__(self):
        return len(self.names)


# original script defined in 
# https://github.com/M3DV/ACSConv/blob/c2ad11dd46718598459fc4e928f456d88fae3789/experiments/lidc/lidc_dataset.py

# class Transform:
#     def __init__(self, size, move=None, train=True, copy_channels=True):
#         self.size = _triple(size)
#         self.move = move
#         self.copy_channels = copy_channels
#         self.train = train

#     def __call__(self, voxel, seg):
#         shape = voxel.shape
#         voxel = voxel/255. - 1
#         if self.train:
#             if self.move is not None:
#                 center = random_center(shape, self.move)
#             else:
#                 center = np.array(shape) // 2
#             voxel_ret = crop(voxel, center, self.size)
#             seg_ret = crop(seg, center, self.size)
            
#             angle = np.random.randint(4, size=3)
#             voxel_ret = rotation(voxel_ret, angle=angle)
#             seg_ret = rotation(seg_ret, angle=angle)

#             axis = np.random.randint(4) - 1
#             voxel_ret = reflection(voxel_ret, axis=axis)
#             seg_ret = reflection(seg_ret, axis=axis)
#         else:
#             center = np.array(shape) // 2
#             voxel_ret = crop(voxel, center, self.size)
#             seg_ret = crop(seg, center, self.size)
            
#         if self.copy_channels:
#             return np.stack([voxel_ret,voxel_ret,voxel_ret],0).astype(np.float32), \
#                     np.expand_dims(seg_ret,0).astype(np.float32)
#         else:
#             return np.expand_dims(voxel_ret, 0).astype(np.float32), \
#                     np.expand_dims(seg_ret,0).astype(np.float32)
