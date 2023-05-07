
from .densenet import *
from .mobilenet import *
from .resnet import *
from .resnet_dero import *
from .resnet_plain import *
from .resnet_rep import *
from .resnet_dirac import *
from .mobilenetv2_dero import *
from .mobilenetv2_plain import *
from .mobilenetv2_rep import *
from .densenet_dero import *
from .densenet_plain import *
from .densenet_plaind import *
from .densenet_rep import *
from .densenet_dirac import *
from .mcunet import *
from .mcunet_dero import *
from .mcunet_plain import *
from .mcunet_rep import *
from .mcunet_dirac import *
# from .densenet_rep import *
# from . import detection, optical_flow, quantization, segmentation, video

# The Weights and WeightsEnum are developer-facing utils that we make public for
# downstream libs like torchgeo https://github.com/pytorch/vision/issues/7094
# TODO: we could / should document them publicly, but it's not clear where, as
# they're not intended for end users.
from ._api import get_model, get_model_builder, get_model_weights, get_weight, list_models, Weights, WeightsEnum
