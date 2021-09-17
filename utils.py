import torch
import torch.nn as nn
import numpy as np
import cv2
import SimpleITK as sitk
import GPUtil
import logging

def get_logger():
    # Initiate a logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def model_to_syncbn(model):
    def _convert_module_from_bn_to_syncbn(module):
        for child_name, child in module.named_children(): 
            if hasattr(nn, child.__class__.__name__) and \
                'batchnorm' in child.__class__.__name__.lower():
                TargetClass = globals()['Synchronized'+child.__class__.__name__]
                arguments = TargetClass.__init__.__code__.co_varnames[1:]
                kwargs = {k: getattr(child, k) for k in arguments}
                setattr(module, child_name, TargetClass(**kwargs))
            else:
                _convert_module_from_bn_to_syncbn(child)

    preserve_state_dict = model.state_dict()
    _convert_module_from_bn_to_syncbn(model)
    model.load_state_dict(preserve_state_dict)
    return model

def soft_dice_loss(logits, targets, smooth=1.0): # targets is one hot
    probs = logits.softmax(dim=1)
    n_classes = logits.shape[1]
    loss = 0
    for i_class in range(n_classes):
        if targets[:,i_class].sum()>0:
            loss += dice_loss_perclass(probs[:,i_class], targets[:,i_class], smooth)
    return loss / n_classes

def categorical_to_one_hot(x, dim=1, expand_dim=False, n_classes=None):
    '''Sequence and label.
    when dim = -1:
    b x 1 => b x n_classes
    when dim = 1:
    b x 1 x h x w => b x n_classes x h x w'''
    # assert (x - x.long().to(x.dtype)).max().item() < 1e-6
    if type(x)==np.ndarray:
        x = torch.Tensor(x)
    assert torch.allclose(x, x.long().to(x.dtype))
    x = x.long()
    if n_classes is None:
        n_classes = int(torch.max(x)) + 1
    if expand_dim:
        x = x.unsqueeze(dim)
    else:
        assert x.shape[dim] == 1
    shape = list(x.shape)
    shape[dim] = n_classes
    x_one_hot = torch.zeros(shape).to(x.device).scatter_(dim=dim, index=x, value=1.)
    return x_one_hot.long()  

def one_hot_to_categorical(x, dim):
    return x.argmax(dim=dim)

class LossMeter(object):
    # To keep track of most recent, average, sum, and count of a loss metric.
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def custom_collate(batch):
    """Take average of the third dimension, and resample each input tensor to match the averaged dimension size"""

    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    min_size = float('inf')

    for d in data:
        assert len(d.shape) == 3
        if min_size > d.shape[2]:
            min_size = d.shape[2]

    # avg = int(total / len(data))
    # target_shape = (data[0].shape[0], data[0].shape[1], min_size)

    for i in range(len(data)):
        data[i] = crop_3d(data[i], min_size)
        data[i] = np.stack([data[i], data[i], data[i]], 0)

        target[i] = crop_3d(target[i], min_size)
        target[i] = np.stack([target[i], target[i], target[i]], 0)
    
    # for i in range(len(data)):
    #     data[i] = zoom(data[i], target_shape)
    #     data[i] = np.float32(np.stack([data[i], data[i], data[i]], 0))
    
    # for i in range(len(target)):
    #     target[i] = zoom(target[i], target_shape, )
    #     target[i] = np.float32(np.stack([target[i], target[i], target[i]], 0))
    
    # resampled_data = [sITK_resample(d, target_shape) for d in data]
    # resampled_target = [sITK_resample(d, target_shape) for d in target]

    # copy data to three channels
    # resampled_data = [np.stack([d, d, d],0) for d in resampled_data]
    # resampled_target = [np.stack([d, d, d],0) for d in resampled_target]

    # img_out = torch.from_numpy(np.stack(resampled_data)).float()
    # msk_out = torch.from_numpy(np.stack(resampled_target)).float()

    img_out = torch.from_numpy(np.stack(data)).float()
    msk_out = torch.from_numpy(np.stack(target)).float()

    return [img_out, msk_out]

def crop_3d(img, side_len):
    assert img.shape[0] == img.shape[1]
    start = int((img.shape[2] - side_len) / 2)
    return img[:, :, start:start + side_len]

def sITK_resample(img, target_shape):
    img = sitk.GetImageFromArray(img.numpy())

    binarythresh = sitk.BinaryThresholdImageFilter()
    img = binarythresh.Execute(img)

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(img)
    resample.SetSize(target_shape)
    img = resample.Execute(img)

    out = sitk.GetArrayFromImage(img)
    return out

def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))