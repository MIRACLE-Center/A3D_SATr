
import torch.nn as nn
import functools
import torch
from ..utiles import _triple_same
from collections import OrderedDict

#TODO #1#通过名字来跳过相应的层。
      #2#2d weighjts 转3d weights
      #3# sycnb

class Converter():
    """
    Decorator class for converting 2d convolution modules
    to corresponding conv version in any networks.
    
    Args:
        model (torch.nn.module): model that needs to be converted
        target_conv_cfg(dict): original 2d conv configuration will be overwrited by the target conv config
                         if original conv atrribute name is in the key of target conv config.
        additional_forward_fts(list): List of addtional augument names in target conv forward function, default is []
        skip_first_conv: if skip applying addtional operation on first conv, default is True
        first_conv_in_channles:the number of input channels in first conv after converter, default is 1
    Warnings:
        Functions in torch.nn.functional involved in data dimension are not supported
    Examples:
        >>> import Converter
        >>> import torchvision
        >>> from convs import AlignShiftConv
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> alignshift_conv_cfg = dict(conv_type=AlignShiftConv, n_fold=8, alignshift=True, inplace=True,
                 ref_spacing=0.2, shift_padding_zero=True)
        >>> m = Converter(m, alignshift_conv_cfg, additional_forward_fts=['thickness'], 
                          skip_first_conv=True, first_conv_input_channles=1)
        >>> # after converted, m is using AlignShiftConv and capable of processing 3D volumes
        >>> x = torch.rand(batch_size, in_channels, D, H, W)
        >>> thickness = torch.rand(batch_size, 1)
        >>> out = m(x, thickness)
    """
    converter_attributes = ['model', 'target_conv_cfg', 'first_conv_in_channles', 
                            'target_conv', 'skip_first_conv', 'additional_forward_fts']
    def __init__(self, model, target_conv_cfg,
                       additional_forward_fts=[], 
                       skip_first_conv=True,
                       first_conv_input_channles=1):
        self.target_conv_cfg = target_conv_cfg
        self.first_conv_in_channles = first_conv_input_channles
        self.target_conv = target_conv_cfg.pop('conv_type')
        self.skip_first_conv = skip_first_conv
        self.model = model
        self.check_first_conv_channel()
        preserve_state_dict = model.state_dict()
        self.model = self.convert_module(self.model, self.target_conv, target_conv_cfg)
        self.model.load_state_dict(preserve_state_dict,strict=False) # 
        del preserve_state_dict
        if additional_forward_fts!=[]:
            self.additional_forward_fts = additional_forward_fts
            self.hook_model_forward()

    def _top_forward(self, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Save spacing in Converter instance
            self.add_kwargs = {}
            if len(args) > 2:
                self.add_kwargs.update({k:v for k,v in zip(self.additional_forward_fts[:len(args[2:])], args[2:])})         # In case spacing is not passed or passed as keyword argument
            self.add_kwargs.update(kwargs) 
            # Execute the vanilla forward function
            output = f(*args[:2])
            return output
        return wrapper

    def convert_conv_kwargs(self, kwargs):
        '''
        return target conv configuration combined with original 2d conv configuration 
        '''
        kwargs['bias'] = True if kwargs['bias'] is not None or kwargs['bias'] else False
        kwargs.update(self.target_conv_cfg)
        if self.skip_first_conv and kwargs.get('in_channels') == self.first_conv_in_channles:
            kwargs.update({self.target_conv.__name__.lower().replace('conv',''):False})
        return kwargs

    def hook_model_forward(self):
        self.target_conv.forward = self._module_forward(self.target_conv.forward)
        self.model.__class__.forward = self._top_forward(self.model.__class__.forward)

    def _module_forward(self, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get spacing from AlignWrapper instance
            kwargs.update(self.add_kwargs)
            return f(*args, **kwargs)
        return wrapper

    def check_first_conv_channel(self):
        '''
        change input channle in first conv and load weights
        '''
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                ori_in_channels = m.weight.shape[1]
                a = self.first_conv_in_channles // ori_in_channels + 1
                b = self.first_conv_in_channles % ori_in_channels + 1 
                m.weight.data =  m.weight.repeat(1,a,1,1).data[:,b//2:b//2-b,...]
                m.in_channels = self.first_conv_in_channles
                return

    def convert_weights_to_3d(self):
        '''
        convert 2d weights to 3d 
        '''
        raise NotImplementedError
                

    def convert_module(self, module, target_conv, conv_kwargs={}):
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
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs = self.convert_conv_kwargs(kwargs)
                setattr(module, child_name, target_conv(**kwargs))
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
                    setattr(module, child_name, TargetClass(**kwargs))
                else:
                    raise Exception('No corresponding module in 3D for 2d module {}'.format(child.__class__.__name__))
            elif isinstance(child, nn.Upsample):
                arguments = nn.Upsample.__init__.__code__.co_varnames[1:]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs['mode'] = 'trilinear' if kwargs['mode']=='bilinear' else kwargs['mode']
                setattr(module, child_name, nn.Upsample(**kwargs))
            else:
                self.convert_module(child, target_conv, conv_kwargs)
        return module

    def __getattr__(self, attr):
        return getattr(self.model, attr)
        
    def __setattr__(self, name, value):
        if name in self.__class__.converter_attributes:
            return object.__setattr__(self, name, value)
        else:
            return setattr(self.model, name, value)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + '(\n' + self.model.__repr__() + '\n)'
