import torch.nn as nn

import collections.abc
from itertools import repeat

class Conv2_5d(nn.Conv3d):
    """
    Decorater class for Conv2_5d, in which kernel size is (1, K, K) or (K, 1, K) or (K, K, 1).
    Args:
        unsqueeze_axis: optional, the default axis is -3, resulting in a kernel size of (1, K, K)
        Other arguments are the same as torch.nn.Conv3d
    Examples:
        >>> import Conv2_5d
        >>> x = torch.rand(batch_size, 1, D, H, W)
        >>> # kernel size is (1, K, K)
        >>> conv = Conv2_5d(1, 64, 3, padding=1)
        >>> out = conv(x)
        >>> # kernel size is (K, K, 1)
        >>> conv = Conv2_5d(3, 64, 1, padding=1, unsqueeze_axis=-1)
        >>> out = conv(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode=None,
                        unsqueeze_axis=-3):
        self.unsqueeze_axis = unsqueeze_axis

        unsqueeze_axis += 3
        kernel_size = [kernel_size, kernel_size]
        padding = [padding, padding]
        kernel_size.insert(unsqueeze_axis, 1)
        padding.insert(unsqueeze_axis, 0)

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

def _ntuple_same(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, collections.abc.Iterable):
            assert len(set(x))==1, 'the size of kernel must be the same for each side'
            return tuple(repeat(x[0], n))
    return parse

_pair_same = _ntuple_same(2)
_triple_same = _ntuple_same(3)

class BaseConverter(object):
    """
    base class for converters
    """
    converter_attributes = []
    target_conv = None

    def __init__(self, model):
        """ Convert the model to its corresponding counterparts and deal with original weights if necessary """
        pass

    def convert_module(self, module):
        """
        A recursive function. 
        Treat the entire model as a tree and convert each leaf module to
            target_conv if it's Conv2d,
            3d counterparts if it's a pooling or normalization module,
            trilinear mode if it's a Upsample module.
        """
        for child_name, child in module.named_children(): 
            if isinstance(child, nn.Conv2d):
                arguments = nn.Conv2d.__init__.__code__.co_varnames[1:]

                # cleaning the tuple due to new PyTorch namming schema (or added new variables)
                arguments = [a for a in arguments if a not in ['kernel_size_', 'stride_', 'padding_', 'dilation_']]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs = self.convert_conv_kwargs(kwargs)
                setattr(module, child_name, self.__class__.target_conv(**kwargs))
            elif hasattr(nn, child.__class__.__name__) and \
                ('pool' in child.__class__.__name__.lower() or 
                'norm' in child.__class__.__name__.lower()):
                if hasattr(nn, child.__class__.__name__.replace('2d', '3d')):
                    TargetClass = getattr(nn, child.__class__.__name__.replace('2d', '3d'))
                    arguments = TargetClass.__init__.__code__.co_varnames[1:]
                    kwargs = {k: getattr(child, k) for k in arguments}
                    if 'adaptive' in child.__class__.__name__.lower():
                        for k in kwargs.keys():
                            kwargs[k] = _triple_same(kwargs[k])
                    # print(self.__class__)
                    setattr(module, child_name, TargetClass(**kwargs))
                    
                else:
                    raise Exception('No corresponding module in 3D for 2d module {}'.format(child.__class__.__name__))
            elif isinstance(child, nn.Upsample):
                arguments = nn.Upsample.__init__.__code__.co_varnames[1:]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs['mode'] = 'trilinear' if kwargs['mode']=='bilinear' else kwargs['mode']
                setattr(module, child_name, nn.Upsample(**kwargs))
            else:
                self.convert_module(child)
        return module

    def convert_conv_kwargs(self, kwargs):
        """
        Called by self.convert_module. Transform the original Conv2d arguments
        to meet the arguments requirements of target_conv. 
        """
        raise NotImplementedError

    def __getattr__(self, attr):
        return getattr(self.model, attr)
        
    def __setattr__(self, name, value):
        if name in self.__class__.converter_attributes:
            return object.__setattr__(self, name, value)
        else:
            return setattr(self.model, name, value)

    def __call__(self, x):
        return self.model(x)

    def __repr__(self):
        return self.__class__.__name__ + '(\n' + self.model.__repr__() + '\n)'

class Conv2_5dConverter(BaseConverter):
    """
    Decorator class for converting 2d convolution modules
    to corresponding 3d version in any networks.
    
    Args:
        model (torch.nn.module): model that needs to be converted
    Warnings:
        Functions in torch.nn.functional involved in data dimension are not supported
    Examples:
        >>> import Conv2_5DWrapper
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = Conv2_5DWrapper(m)
        >>> # after converted, m is using ACSConv and capable of processing 3D volumes
        >>> x = torch.rand(batch_size, in_channels, D, H, W)
        >>> out = m(x)
    """
    converter_attributes = ['model', 'unsqueeze_axis']
    target_conv = Conv2_5d

    def __init__(self, model, unsqueeze_axis=-3):
        preserve_state_dict = model.state_dict()
        self.model = model
        self.unsqueeze_axis = unsqueeze_axis
        self.model = self.convert_module(self.model)
        self.load_state_dict(preserve_state_dict, strict=True)
        
    def convert_conv_kwargs(self, kwargs):
        kwargs['bias'] = True if kwargs['bias'] is not None else False
        for k in ['kernel_size','stride','padding','dilation']:
            kwargs[k] = _pair_same(kwargs[k])[0]
        kwargs['unsqueeze_axis'] = self.unsqueeze_axis
        return kwargs

    def load_state_dict(self, state_dict, strict=True, unsqueeze_axis=-3):
        load_state_dict_from_2d_to_2_5d(self.model, state_dict, strict=strict, unsqueeze_axis=unsqueeze_axis)

def load_state_dict_from_2d_to_2_5d(model_2_5d, state_dict_2d, strict=True, unsqueeze_axis=-3):
    for key in list(state_dict_2d.keys()):
        if state_dict_2d[key].dim()==4:
            state_dict_2d[key] = state_dict_2d[key].unsqueeze(unsqueeze_axis)
    model_2_5d.load_state_dict(state_dict_2d, strict=strict)