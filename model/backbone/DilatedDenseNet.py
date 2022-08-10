from torchvision.models import DenseNet,DenseNet121_Weights
from torch import nn
import re
import torchvision

assert int(torchvision.__version__.split('.')[1])>=13 , 'torchvision version mush be greater than 0.13,0'

from typing import Any, Optional, Tuple
__all__=['DilatedDenseNet','dilated_densenet121']

class DilatedDenseNet(DenseNet):
    '''modify from https://github.com/DeepMotionAIResearch/DenseASPP/blob/35a3e05fd70398b740776fa3ff1378c8470b6026/models/DenseASPP.py'''
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=1000, dilate_scale=8, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DilatedDenseNet, self).__init__(growth_rate, block_config, num_init_features,
                                              bn_size, drop_rate, num_classes, norm_layer)
        assert (dilate_scale == 8 or dilate_scale == 16), "dilate_scale can only set as 8 or 16"
        from functools import partial
        if dilate_scale == 8:
            self.features.denseblock3.apply(partial(self._conv_dilate, dilate=2))
            self.features.denseblock4.apply(partial(self._conv_dilate, dilate=4))
            del self.features.transition2.pool
            del self.features.transition3.pool
        elif dilate_scale == 16:
            self.features.denseblock4.apply(partial(self._conv_dilate, dilate=2))
            del self.features.transition3.pool

    def _conv_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.padding = (dilate, dilate)
                m.dilation = (dilate, dilate)

def _dilated_densenet(growth_rate:int,block_config: Tuple[int,int,int,int],num_init_features:int,weights,progress:bool,**kwargs:Any) -> DilatedDenseNet:
    model = DilatedDenseNet(growth_rate,block_config,num_init_features,**kwargs)
    if weights is not None:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        state_dict = weights.get_state_dict(progress=progress)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def dilated_densenet121(*,weights:Optional[DenseNet121_Weights] = None, progress: bool = True, **kwargs) -> DilatedDenseNet:
    '''modify from pytorch official code: https://github.com/pytorch/vision/blob/96aa3d928c6faaf39defd846966421679244412d/torchvision/models/densenet.py#L342'''
    weights = DenseNet121_Weights.verify(weights)

    return _dilated_densenet(32,(6,12,24,16),64,weights,progress,**kwargs)


if __name__=='__main__':
    from torchvision.models import DenseNet121_Weights
    weights = DenseNet121_Weights.DEFAULT
    preprocess = weights.transforms()
    model = dilated_densenet121(weights=weights)
    print(model)
    print(preprocess)