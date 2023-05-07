import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface

__all__ = [
    "DenseNet_dero",
    "densenet121_dero",
    # "densenet161_dero",
    # "densenet169_dero",
    # "densenet201_dero",
]

class SRConv(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3,
                 stride=1, padding=0, groups=1,bias=False, act=F.relu, scbn=True):
        super(SRConv, self).__init__()
        self.stride = stride
        self.in_planes = in_planes 
        self.planes = planes 
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size,stride=stride, groups= groups, padding=padding )
        self.bn = nn.BatchNorm2d(in_planes)
        self.scbn = scbn
        self.act = act if act != None else nn.Identity()

        if self.in_planes < self.planes :
            self.pad    = planes - self.in_planes;
            # self.bnsc   = nn.BatchNorm2d(in_planes)
        elif self.in_planes > self.planes:
            self.pad    = self.planes - self.in_planes%planes
            # self.bnsc   = nn.BatchNorm2d(planes)
        # elif self.in_planes == self.planes:
        #     self.bnsc   = nn.BatchNorm2d(planes)

        if self.stride != 1:
            if kernel_size != 1:
                self.pool = torch.nn.AvgPool2d( kernel_size=kernel_size,stride=stride, padding=padding,divisor_override=1)
            else:
                self.pool = torch.nn.AvgPool2d( kernel_size=stride,stride=stride,divisor_override=1)
        if self.scbn or self.in_planes !=  self.planes or self.stride != 1:# 
            self.bnsc   = nn.BatchNorm2d(planes)
                       
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = x[0]+x[1]
        out = self.act(self.conv(self.bn(x)))
        res = x if self.stride == 1 else self.pool(x)

        if self.in_planes < self.planes :
            # res = self.bnsc(res)
            res = F.pad(res, (0, 0, 0, 0, 0, self.pad), "constant", 0)
        elif self.in_planes > self.planes:
            if self.pad != self.planes:
                res = F.pad( res, (0, 0, 0, 0, 0, self.pad), "constant", 0)
            res = torch.split(res, self.planes,dim=1)
            res = torch.sum(torch.stack(res), dim=0)
            # res = self.bnsc(res)

        if self.scbn or self.in_planes !=  self.planes or self.stride != 1:
            res = self.bnsc(res)
        return [out, res]



class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float
    ) -> None:
        super().__init__()
        self.conv1 = SRConv(num_input_features,bn_size,kernel_size=1,scbn=False)
        self.conv2 = SRConv(bn_size,growth_rate,kernel_size=3,padding=1,scbn=False)
        # self.conv2 = SRConv(num_input_features,growth_rate,kernel_size=3,padding=1,scbn=False)

    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        x = self.conv1(input)
        x = self.conv2(x)
        # x = self.conv2(input)
        return x


class _TransitionSR(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        pooling = True,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.conv1 = SRConv(num_input_features,num_output_features,kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2,divisor_override=1)
        self.bnsc   = nn.BatchNorm2d(num_output_features)
        # if(self.pooling):
        # else:
        # #     # self.bnsc3   = nn.BatchNorm2d(num_output_features)
        #     self.bnsc2   = nn.BatchNorm2d(num_output_features)
    def forward(self, init_features: Tensor) -> Tensor:     
        x = self.conv1(init_features)
        if(self.pooling):
            x = [self.pool(x[0]), self.bnsc(self.pool(x[1]))]
        else:
            x = [x[0], self.bnsc(x[1])]
        #     # x = self.bnsc3(x)

        return x


class DenseNet_dero(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)
        # Stages = [[35, 40,  43,  46,  50,  62],
        #          [39, 47,  49,  53,  56,  59,],
        #          [46, 60,  62,  66,  69,  72,],
        #          [60, 87,  88,  91,  94,  98,]]

        Stages = [[76, 76,  76,  76,  76,  76],
                 [92, 92,  92, 92, 92, 92,  92, 92, 92, 92,  92, 92,],
                 [116, 116,  116,  116,  116, 116,  116,  116,  116, 116,  116,  116,  116, 116,  116,  116, 116, 116,  116,  116, 116, 116,  116,  116],
                 [112, 112, 112, 112,  112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,]]


        # Stages = [[35, 40,  43,  46,  50,  62],
        #          [39, 47,  49,  53,  56,  59,  62,  66,  69,  72,  75,  136],
        #          [46, 60,  62,  66,  69,  72,  75,  78,  82,  85,  88,  91,  94,  98,  101, 104, 107, 110, 114, 117, 120, 123, 126, 258],
        #          [60, 87,  88,  91,  94,  98,  101, 104, 107, 110, 114, 117, 120, 123, 126, 130]]

        self.conv1 = nn.Conv2d(3,  num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d( num_init_features)
        self.bn3 = nn.BatchNorm2d( num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool = torch.nn.AvgPool2d( 7 , stride = 2, padding=3 , divisor_override = 1 )
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, divisor_override = 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        layers = []

        num_features = num_init_features
        pooling = True
        for row_index, stage in enumerate(Stages):
            for col_index, item in enumerate(stage):
                item=item
                block = _DenseLayer(
                    num_input_features=num_features, 
                    growth_rate = item, 
                    bn_size = item, 
                    drop_rate = 0,
                )
                layers.append(block)
                num_features = item
            if row_index ==3:
                pooling = False
            trans = _TransitionSR(
                num_input_features=num_features, 
                num_output_features= num_init_features* 2**(row_index+1),
                pooling = pooling)
            layers.append(trans)
            num_features = num_init_features* 2**(row_index+1)
        
        # self.bn4 = nn.BatchNorm2d( num_features)
        self.features = nn.Sequential(*layers)
        # Final batch norm
        # self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.pool(x)
        out = F.pad(out, (0, 0, 0, 0, 0, 61), "constant", 0)
        x = self.conv1(x)
        x = self.relu(x)
        x = [self.maxpool(x), self.bn3(self.pool2(self.bn2(out)))]

        features = self.features(x)

        out = F.relu(features[0])
        out = F.adaptive_avg_pool2d(out, (1, 1)) + F.adaptive_avg_pool2d(features[1],(1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model: nn.Module, weights: WeightsEnum, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = weights.get_state_dict(progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> DenseNet_dero:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = DenseNet_dero(growth_rate, block_config, num_init_features, **kwargs)

    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)

    return model


_COMMON_META = {
    "min_size": (29, 29),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/pull/116",
    "_docs": """These weights are ported from LuaTorch.""",
}






@register_model()
@handle_legacy_interface()
def densenet121_dero(*, weights = None, progress: bool = True, **kwargs: Any) -> DenseNet_dero:

    return _densenet(32, (6, 12, 24, 16), 64, weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface()
# def densenet161_dero(*, weights = None, progress: bool = True, **kwargs: Any) -> DenseNet_dero:
#     return _densenet(48, (6, 12, 36, 24), 96, weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface()
# def densenet169_dero(*, weights = None, progress: bool = True, **kwargs: Any) -> DenseNet_dero:
#     return _densenet(32, (6, 12, 32, 32), 64, weights, progress, **kwargs)


# @register_model()
# @handle_legacy_interface()
# def densenet201_dero(*, weights = None, progress: bool = True, **kwargs: Any) -> DenseNet_dero:
#     return _densenet(32, (6, 12, 48, 32), 64, weights, progress, **kwargs)
