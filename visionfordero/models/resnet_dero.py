from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "ResNet_Dero",
    "resnet18dero",
    "resnet34dero",
    "resnet50dero",
    "resnet101dero",
    "resnet152dero",
]

class SRConv(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3,
                 stride=1, padding=0, groups=1,bias=False, act=F.relu,
                 stochastic_depth_prob: float = 0,
                 next_rc: bool = True):
        super(SRConv, self).__init__()
        self.stride = stride
        self.in_planes = in_planes 
        self.planes = planes 
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size,stride=stride, groups= groups, padding=padding )
        self.bn = nn.BatchNorm2d(in_planes)
        # self.bn = nn.BatchNorm2d(planes)
        # self.bn = nn.Identity()

        self.act = act if act != None else nn.Identity()
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        # self.in_planes = self.in_planes * stride * stride 
        self.next_rc = next_rc
        if self.in_planes < self.planes :
            self.pad    = planes - self.in_planes;
        elif self.in_planes > self.planes:
            self.pad    = self.planes - self.in_planes%planes
            # self.w_size = (self.in_planes+planes-1)//planes
            # self.resize = nn.AvgPool3d((self.w_size,1,1),stride=(self.w_size,1,1))
        if self.stride != 1:
            if kernel_size != 1:
                self.pool = torch.nn.AvgPool2d( kernel_size=kernel_size,stride=stride, padding=padding,divisor_override=1)
            else:
                self.pool = torch.nn.AvgPool2d( kernel_size=stride,stride=stride,divisor_override=1)

            # self.bnsc2   = nn.BatchNorm2d(planes)
        if self.next_rc or self.stride != 1 or self.in_planes > self.planes: 
            self.bnsc   = nn.BatchNorm2d(planes)
        #     # self.bno   = nn.BatchNorm2d(planes)
                             
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = x[0]+x[1]

        out = self.stochastic_depth(self.act(self.conv(self.bn(x))))
        # out = self.stochastic_depth(self.act(self.bn(self.conv(x))))
        res = x if self.stride == 1 else self.pool(x)

        if self.in_planes < self.planes :
            res = F.pad(res, (0, 0, 0, 0, 0, self.pad), "constant", 0)
        elif self.in_planes > self.planes:
            if self.pad != self.planes:
                res = F.pad( res, (0, 0, 0, 0, 0, self.pad), "constant", 0)
            res = torch.split(res, self.planes,dim=1)
            res = torch.sum(torch.stack(res), dim=0)

        if self.next_rc or self.stride != 1 or self.in_planes > self.planes:
            res = self.bnsc(res)
        return [out, res]

# class SRConv(nn.Module):
#     def __init__(self, in_planes, planes, kernel_size=3,
#                  stride=1, padding=0, groups=1,bias=False, act=F.relu,
#                  stochastic_depth_prob: float = 0,
#                  next_rc: bool = True):
#         super(SRConv, self).__init__()
#         self.stride = stride
#         self.in_planes = in_planes 
#         self.planes = planes 
#         self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size,stride=stride, groups= groups, padding=padding )
#         self.bn = nn.BatchNorm2d(in_planes)
#         # self.bno = nn.BatchNorm2d(planes)
#         self.act = act if act != None else nn.Identity()
#         self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
#         self.in_planes = self.in_planes * stride * stride

#         if self.in_planes < self.planes :
#             self.pad    = planes - self.in_planes;
#         elif self.in_planes > self.planes:
#             self.pad    = self.planes - self.in_planes%planes
#             self.w_size = (self.in_planes+planes-1)//planes
#             self.resize = nn.AvgPool3d((self.w_size,1,1),stride=(self.w_size,1,1),divisor_override=1)
#         if self.stride != 1:
#             self.pool = torch.nn.AvgPool2d( stride , stride = stride,divisor_override=1)
#         if self.stride != 1 or self.in_planes > self.planes: 
#             self.bnsc   = nn.BatchNorm2d(planes)
                             
#     def forward(self, x):
#         if not isinstance(x, Tensor):
#             x = x[0]+x[1]

#         if self.stride == 1:
#             out = x
#         else:
#             b, c, h, w = x.size()
#             s = self.stride
#             out = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
#             out = out.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
#             # out = out.permute(0, 1, 3, 5, 2, 4).contiguous()  # x(1,64,2,2,40,40)
#             out = out.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)
#         # out = x if self.stride == 1 else self.pool(x)
#         if self.in_planes < self.planes :
#             out = F.pad(out, (0, 0, 0, 0, 0, self.pad), "constant", 0)
#         elif self.in_planes > self.planes:
#             if self.pad != self.planes:
#                 out = F.pad( out, (0, 0, 0, 0, 0, self.pad), "constant", 0)
#             out = torch.split(out, self.planes,dim=1)
#             out = torch.sum(torch.stack(out), dim=0)
#         if self.stride != 1 or self.in_planes > self.planes:
#             out = self.bnsc(out)
#         res = self.stochastic_depth(self.act(self.conv(self.bn(x))))
#         out = [res , out]
#         return out

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        stochastic_depth_prob: float = 0,
        next_rc: bool = True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SRConv(inplanes, planes, 3, stride , 1,stochastic_depth_prob=stochastic_depth_prob,next_rc=True)
        self.conv2 = SRConv(planes, planes, 3, 1 , 1,stochastic_depth_prob=stochastic_depth_prob,next_rc=True)

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        stochastic_depth_prob: float = 0,
        next_rc: bool = True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SRConv(inplanes, width, 1, 1, 0, stochastic_depth_prob=stochastic_depth_prob,next_rc=next_rc)
        self.conv2 = SRConv(width, width, 3, stride, 1, stochastic_depth_prob=stochastic_depth_prob,next_rc=next_rc)
        self.conv3 = SRConv(width, planes * self.expansion, 1, 1, 0,stochastic_depth_prob=stochastic_depth_prob,next_rc=next_rc)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
   
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return out


class ResNet_Dero(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        stochastic_depth_prob: float = 0.,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        # self.bn1 = norm_layer(64)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(self.inplanes)
        self.bn3 = norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        # self.bnsc = norm_layer(self.inplanes)
        self.pad = self.inplanes - 3
        self.pool = torch.nn.AvgPool2d( 7 , stride = 2, padding=3 , divisor_override = 1 )
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, divisor_override = 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        sbstep = stochastic_depth_prob* 1.0/sum(layers)
        self.layer1 = self._make_layer(block, 64, layers[0],sb=0.0,sbstep=sbstep)
        sb = stochastic_depth_prob* sum(layers[0:1])/sum(layers)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],sb=sb,sbstep=sbstep)
        sb = stochastic_depth_prob* sum(layers[0:2])/sum(layers)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],sb=sb,sbstep=sbstep)
        sb = stochastic_depth_prob* sum(layers[0:3])/sum(layers)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],sb=sb,sbstep=sbstep,next_rc=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.bnsc = norm_layer(512 * block.expansion) 
               
        return
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        sb    : float = 0,
        sbstep: float = 0,
        next_rc: bool = True,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, sb
            )
        )
        self.inplanes = planes * block.expansion
        sb+=sbstep

        for i in range(1, blocks):
            rc = True
            if(i == blocks-1):
                rc=True
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    stochastic_depth_prob=sb,
                    next_rc = rc,
                )
            )
            sb+=sbstep

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        out = self.pool(x)
        # out = self.bn1(out)

        # b, c, h, w = x.size()
        # s = 4
        # out = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        # out = out.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        # out = out.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)

        out = F.pad(out, (0, 0, 0, 0, 0, 61), "constant", 0)
        # x = self.bn1(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) + self.bn3(self.pool2(self.bn2(out)))
  
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 
        # x = self.avgpool(x[0]+x[1]) 
        # x = self.avgpool(self.bnsc(x[0]+x[1])) 
        x = self.avgpool(x[0]) + self.avgpool(x[1])
        # x = self.bnsc(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_Dero:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_Dero(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}



@register_model()
@handle_legacy_interface()
def resnet18dero(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet_Dero:
    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface()
def resnet34dero(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet_Dero:
    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface()
def resnet50dero(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet_Dero:
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface()
def resnet101dero(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet_Dero:
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@register_model()
@handle_legacy_interface()
def resnet152dero(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet_Dero:
    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


