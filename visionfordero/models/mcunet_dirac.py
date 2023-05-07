from functools import partial
from typing import Any, Callable, List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.misc import ConvNormActivation
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
from torch.nn.init import dirac_

__all__ = ["McuNet_dirac", 
"mcunet_dirac_v1",
"mcunet_dirac_v2",
"mcunet_dirac_v3",
"mcunet_dirac_v4",
"mcunet_dirac_v5",]




class SRConv(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3,
                 stride=1, padding=0, groups=1,bias=False, activation_layer=nn.ReLU, norm_layer=None):
        super(SRConv, self).__init__()
        self.stride = stride
        self.in_planes = in_planes 
        self.planes = planes 
        self.padding = (kernel_size-1) //2
        self.conv = DiracConv2d(in_planes, planes, kernel_size=kernel_size,stride=stride, groups= groups, padding=self.padding,bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.act = activation_layer() if activation_layer != None else nn.Identity()
        self.groups = groups
       
                             
    def forward(self, x):
        out = self.bn(self.conv(x))
        return self.act(out)


class DiracConv(nn.Module):

    def init_params(self, out_channels):
        self.alpha = nn.Parameter(torch.Tensor(out_channels).fill_(1))
        self.beta = nn.Parameter(torch.Tensor(out_channels).fill_(0.1))
        self.register_buffer('delta', dirac_(self.weight.data.clone()))
        assert self.delta.shape == self.weight.shape
        self.v = (-1,) + (1,) * (self.weight.dim() - 1)

    def transform_weight(self):

        return self.alpha.view(*self.v) * self.delta + self.beta.view(*self.v) * self.weight

class DiracConv2d(nn.Conv2d, DiracConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=True, groups=1,):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.init_params(out_channels)

    def forward(self, input):
        return F.conv2d(input, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)
    
    def forward_skip(self, input):
        return F.conv2d(input, self.alpha.view(*self.v) * self.delta, self.bias, self.stride, self.padding, self.dilation)




# necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, k: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.use_res_connect =False

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                SRConv(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=k,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                    # conv_layer = DiracConv2d,
                ),
                # pw-linear
                DiracConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        # if self.use_res_connect:
        #     return x + self.conv(x)
        # else:
        return self.conv(x)


class McuNet_dirac(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.,
        init_ch: int = 32,
        last_channel: int = 1280,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()
        _log_api_usage_once(self)

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = init_ch
        last_channel = last_channel


        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 5-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            SRConv(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s, k in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, k, stride, expand_ratio=t, norm_layer=norm_layer,))
                input_channel = output_channel
        # building last several layers
        # features.append(
        #     Conv2dNormActivation(
        #         input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
        #     )
        # )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


_COMMON_META = {
    "num_params": 3504872,
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}




@register_model()
@handle_legacy_interface()
def mcunet_dirac_v1(
    *, weights = None, progress: bool = True, **kwargs: Any
) -> McuNet_dirac:
    # McuNet_dirac-10fps_imagenet.json
    init_ch = 24
    last_channel= 192
    inverted_residual_setting= [
        [1, 16, 1, 1, 3],
        [4, 16, 1, 2, 5],
        [3, 16, 1, 1, 3],
        [4, 16, 1, 1, 3],
        [5, 24, 1, 2, 7],
        [4, 24, 1, 1, 5],
        [5, 24, 1, 1, 5],
        [5, 48, 1, 2, 7],
        [4, 48, 1, 1, 5],
        [5, 56, 1, 1, 3],
        [4, 56, 1, 1, 5],
        [5, 56, 1, 1, 3],
        [6, 112, 1, 2, 3],
        [4, 112, 1, 1, 5],
        [6, 192, 1, 1, 3],
    ]
    resolution =48
    model = McuNet_dirac(init_ch = init_ch, last_channel=last_channel, inverted_residual_setting=inverted_residual_setting, **kwargs)
    return model

@register_model()
@handle_legacy_interface()
def mcunet_dirac_v2(
    *, weights = None, progress: bool = True, **kwargs: Any
) -> McuNet_dirac:
# McuNet_dirac-5fps_imagenet.json
    init_ch = 16
    last_channel= 160
    inverted_residual_setting= [
        [1, 8, 1, 1, 3],
        [4, 16, 1, 2, 3],
        [3, 16, 1, 1, 3],
        [3, 24, 1, 2, 7],
        [5, 24, 1, 1, 3],
        [5, 40, 1, 2, 3],
        [4, 40, 1, 1, 7],
        [4, 48, 1, 1, 5],
        [3, 48, 1, 1, 3],
        [4, 48, 1, 1, 3],
        [5, 96, 1, 2, 7],
        [4, 96, 1, 1, 5],
        [4, 96, 1, 1, 5],
        [6, 160, 1, 1, 3],
    ]
    resolution =96
    model = McuNet_dirac(init_ch = init_ch, last_channel=last_channel, inverted_residual_setting=inverted_residual_setting, **kwargs)
    return model

@register_model()
@handle_legacy_interface()
def mcunet_dirac_v3(
    *, weights = None, progress: bool = True, **kwargs: Any
) -> McuNet_dirac:
# McuNet_dirac-256kb-1mb_imagenet.json
    init_ch = 16
    last_channel= 160
    inverted_residual_setting= [
        [1, 8, 1, 1, 3],
        [3, 16, 1, 2, 5],
        [6, 16, 1, 1, 7],
        [5, 16, 1, 1, 3],
        [5, 16, 1, 1, 5],
        [5, 24, 1, 2, 3],
        [6, 24, 1, 1, 7],
        [6, 24, 1, 1, 5],
        [4, 40, 1, 2, 7],
        [5, 40, 1, 1, 5],
        [5, 48, 1, 1, 3],
        [5, 48, 1, 1, 5],
        [4, 48, 1, 1, 3],
        [6, 96, 1, 2, 5],
        [4, 96, 1, 1, 5],
        [3, 96, 1, 1, 5],
        [4, 96, 1, 1, 3],
        [5, 160, 1, 1, 5],
    ]
    resolution =160
    model = McuNet_dirac(init_ch = init_ch, last_channel=last_channel, inverted_residual_setting=inverted_residual_setting, **kwargs)
    return model
@register_model()
@handle_legacy_interface()
def mcunet_dirac_v4(
    *, weights = None, progress: bool = True, **kwargs: Any
) -> McuNet_dirac:
# McuNet_dirac-320kb-1mb_imagenet.json
    init_ch = 16
    last_channel= 160
    inverted_residual_setting= [
        [1, 8, 1, 1, 3],
        [3, 16, 1, 2, 7],
        [5, 16, 1, 1, 3],
        [5, 16, 1, 1, 7],
        [4, 16, 1, 1, 5],
        [5, 24, 1, 2, 5],
        [5, 24, 1, 1, 5],
        [5, 24, 1, 1, 5],
        [5, 40, 1, 2, 3],
        [6, 40, 1, 1, 7],
        [4, 40, 1, 1, 5],
        [5, 48, 1, 1, 5],
        [5, 48, 1, 1, 7],
        [5, 48, 1, 1, 3],
        [6, 96, 1, 2, 3],
        [5, 96, 1, 1, 7],
        [4, 96, 1, 1, 3],
        [5, 160, 1, 1, 7],
    ]
    resolution =176
    model = McuNet_dirac(init_ch = init_ch, last_channel=last_channel, inverted_residual_setting=inverted_residual_setting, **kwargs)
    return model
@register_model()
@handle_legacy_interface()
def mcunet_dirac_v5(
    *, weights = None, progress: bool = True, **kwargs: Any
) -> McuNet_dirac:
# McuNet_dirac-512kb-2mb_imagenet.json
    init_ch = 32
    last_channel= 320
    inverted_residual_setting= [
        [1, 16, 1, 1, 3],
        [3, 24, 1, 2, 7],
        [5, 24, 1, 1, 3],
        [4, 24, 1, 1, 5],
        [5, 40, 1, 2, 7],
        [4, 40, 1, 1, 3],
        [4, 40, 1, 1, 7],
        [3, 80, 1, 2, 7],
        [3, 80, 1, 1, 3],
        [3, 80, 1, 1, 7],
        [4, 96, 1, 1, 3],
        [3, 96, 1, 1, 5],
        [3, 96, 1, 1, 5],
        [4, 192, 1, 2, 7],
        [3, 192, 1, 1, 7],
        [3, 192, 1, 1, 5],
        [4, 320, 1, 1, 5],
    ]
    resolution =160

    model = McuNet_dirac(init_ch = init_ch, last_channel=last_channel, inverted_residual_setting=inverted_residual_setting, **kwargs)
    return model
