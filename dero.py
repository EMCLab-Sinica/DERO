import torch
import torchvision
import os
import onnx
from onnxsim import simplify
import errno
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

def get_nn_Conv(inp=1, out=1, k=1, s=1, p=1, g=1, **Kwargs):
    return torch.nn.Conv2d(inp, out, k, stride=s, padding=p, groups=g)
def get_nn_AvgPool(k=1, s=1, p=1, **Kwargs):
    return torch.nn.AvgPool2d(kernel_size=k,stride=s, padding=p)
def get_nn_MaxPool(k=1, s=1, p=1, **Kwargs):
    return torch.nn.MaxPool2d(kernel_size=k,stride=s, padding=p)
def get_nn_ReLu(**Kwargs):
    return torch.nn.ReLU(inplace=True)
def get_nn_ReLu6(**Kwargs):
    return torch.nn.ReLU6(inplace=True)
def get_nn_BatchNorm2d(inp=1, **Kwargs):
    return torch.nn.BatchNorm2d(inp)
def get_nn_SRConv(inp=1, out=1, k=1, s=1, p=1, g=1, **Kwargs):
    return SRConv(inp, out, k, stride=s, groups=g)


op_map = {
    'Conv': get_nn_Conv,
    'Conv_dero': get_nn_SRConv,
    'Relu': get_nn_ReLu,
    'MaxPool': get_nn_MaxPool,
    'AveragePool': get_nn_AvgPool,
    'Clip': get_nn_ReLu6,
    'BatchNormalization': get_nn_BatchNorm2d,
    }


class SRConv(torch.nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3,
                 stride=1, groups=1,bias=False, activation_layer=F.relu, norm_layer=None):
        super(SRConv, self).__init__()
        self.stride = stride
        self.in_planes = in_planes 
        self.planes = planes 
        padding = (kernel_size-1) //2
        self.conv = torch.nn.Conv2d(in_planes, planes, kernel_size=kernel_size,stride=stride, groups= groups, padding=padding )
        self.bn = torch.nn.BatchNorm2d(in_planes)
        self.act = activation_layer if activation_layer != None else nn.Identity()
    
        if self.in_planes < self.planes :
            self.pad    = planes - self.in_planes;
        elif self.in_planes > self.planes:
            self.pad    = self.planes - self.in_planes%planes
        if self.stride != 1:
            if kernel_size != 1:
                self.pool = torch.nn.AvgPool2d( kernel_size=kernel_size,stride=stride, padding=padding,divisor_override=1)
            else:
                self.pool = torch.nn.AvgPool2d( kernel_size=stride,stride=stride,divisor_override=1) 
        self.bnsc   = torch.nn.BatchNorm2d(planes)

                             
    def forward(self, x):
        out = self.act(self.conv(self.bn(x)))
        res = x if self.stride == 1 else self.pool(x)

        if self.in_planes < self.planes :
            res = F.pad(res, (0, 0, 0, 0, 0, self.pad), "constant", 0)
        elif self.in_planes > self.planes:
            if self.pad != self.planes:
                res = F.pad( res, (0, 0, 0, 0, 0, self.pad), "constant", 0)
            res = torch.split(res, self.planes,dim=1)
            res = torch.sum(torch.stack(res), dim=0)
        res = self.bnsc(res)
        return out+res



class plain_network(torch.nn.Module):
    def __init__(self, op_list=[], num_classes=1000):
        super(plain_network, self).__init__()

        last_channel = 0
        features: List[nn.Module]  = []
        for item in op_list:
            if op_map.get(item['op']):
                features.append(op_map[item['op']](**item))
                last_channel = item['out']
            else: 
                break

        self.features = torch.nn.Sequential(*features)

        # building classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)
    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)  

class dero_network(torch.nn.Module):
    def __init__(self, op_list=[], num_classes=1000):
        super(dero_network, self).__init__()

        last_channel = 0
        features: List[nn.Module]  = []
        for item in op_list:
            if op_map.get(item['op']):
                if(item['op'] in ['AveragePool', 'MaxPool', 'Conv']):
                    if item['op'] == 'Conv':
                        item['op'] = 'Conv_dero'
                    features.append(op_map[item['op']](**item))
                    last_channel = item['out']
            else: 
                break

        self.features = torch.nn.Sequential(*features)

        # building classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)
    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)  



def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def verify_graph(op_list=[]): 
    inp = op_list[0]['out']
    modified = False
    for idx in range(1, len(op_list)):
        if op_list[idx]['op'] == 'Conv':  
            if op_list[idx]['inp'] != inp:
                modified = True
            op_list[idx]['inp'] = inp
            inp = op_list[idx]['out']
    return modified

def verify_graph_dero(op_list=[]): 
    inp = op_list[0]['out']
    # op_list.pop(0)
    for idx in range(1, len(op_list)):
        if op_list[idx]['op'] == 'Conv':
            op_list[idx]['inp'] = inp
            if op_list[idx]['inp'] != inp:
                return False
            else:
                inp = op_list[idx]['out']
    return True

def check_out_degree(tensor_name, model_graph):
    cnt = 0
    node_list = []
    for node_idx, node in enumerate(model_graph.node):
        if(tensor_name in node.input):
            cnt += 1
            node_list.append(node_idx)
    return cnt

def get_node_hyper_parametera(idx, model_graph):
    hpy={'op': 'None', 
         'k': 1, 
         's': 1, 
         'p': 1,
         'g': 1,
         'inp': 1,
         'out': 1,}
    node = model_graph.node[idx]
    hpy['op'] = node.op_type
    if(node.op_type == 'BatchNormalization'):
        tensor_info = [x for x in model_graph.input if x.name == node.input[1]][0]
        shape = [dim.dim_value for dim in tensor_info.type.tensor_type.shape.dim]
        hpy['out'] = shape[0]
        hpy['inp'] = shape[0]
    if(node.op_type == 'Conv'):
        hpy['k'] = node.attribute[2].ints[0]
        hpy['s'] = node.attribute[4].ints[0]
        hpy['p'] = node.attribute[3].ints[0]
        hpy['g'] = node.attribute[1].i
        tensor_info = [x for x in model_graph.input if x.name == node.input[1]][0]
        shape = [dim.dim_value for dim in tensor_info.type.tensor_type.shape.dim]
        hpy['out'] = shape[0]
        hpy['inp'] = shape[1]
    if(node.op_type in ['MaxPool', 'AveragePool']):
        hpy['k'] = node.attribute[1].ints[0]
        hpy['s'] = node.attribute[3].ints[0]
        hpy['p'] = node.attribute[2].ints[0]
    # if(hpy['s'] == 2 ):
    #     print(hpy)
    return hpy
def get_par_cnt(st, ed, op_list):
    cnt = 0
    for idx in range(st,ed):
        if op_list[idx]['op'] == 'Conv':
            cnt += (op_list[idx]['k'] * op_list[idx]['k'] * op_list[idx]['inp'] * op_list[idx]['out'])
    return cnt


def op_refine(seg, new_model, ori_model):
    while len(seg):
        st = seg[0]
        while (new_model[st]['op'] != 'Conv' ):
            st += 1 
        ed = seg[1]
        ed = ed +1 
        if ed > len(new_model): ed=len(new_model)
        seg.pop(0)
        seg.pop(0)   
        ch = 1
        
        while(get_par_cnt(st,ed,new_model) < get_par_cnt(st,ed,ori_model)):
            new_model[st]['out']=ch
            for idx in range(st+1,ed):
                new_model[idx]['out']=ch
                new_model[idx]['in']=ch
            ch+=1
        new_model[st]['out']=ch-2
        for idx in range(st+1,ed):
            new_model[idx]['out']=ch-2
            new_model[idx]['in']=ch-2

def export2onnx(model, path,  input_size):

    model.eval() 
    dummy_input = torch.randn(1, 3,  input_size,  input_size, requires_grad=True)
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         path,       # where to save the model  
         export_params=False,  # store the trained parameter weights inside the model file 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print('Model has been converted to ONNX')   
    
    # load your predefined ONNX model
    model = onnx.load(path)
    # simplify graph
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    return model_simp

def main(args):
    if args.output_dir:
        mkdir(args.output_dir)
    
    print("Creating model")
    model = torchvision.models.get_model(args.model)

    tmp_path = os.path.join(args.output_dir,'tmp.onnx')
    model_simp = export2onnx(model, tmp_path,  args.input_size)

    # Get the graph from the model
    graph = model_simp.graph
    plain_ori = []
    node_out = graph.node[0].input[0]
    seg = []
    # Collect ops for plain model
    for node_idx, node in enumerate(graph.node):
        if node.op_type in ['Add', 'Concat', 'Flatten', 'Gemm']:
            node_out = node.output[0]
        elif node_out == node.input[0]:
            if node.op_type in ['MaxPool']:
                seg.append(len(plain_ori)+1)
            if node.op_type in ['AveragePool']:
                seg.append(len(plain_ori)-1)
                seg.append(len(plain_ori)+1)
            hpy = get_node_hyper_parametera(node_idx, graph)
            plain_ori.append(hpy)
            node_out = node.output[0]
    seg.append(len(plain_ori))
    tensor_info = [x for x in graph.input if x.name == graph.node[-1].input[1]][0]
    shape = [dim.dim_value for dim in tensor_info.type.tensor_type.shape.dim]
    num_classes = shape[0]


    plain_refined = []
    for item in plain_ori:
        plain_refined.append(deepcopy(item))
    dero_refined = []
    for item in plain_ori:
        dero_refined.append(deepcopy(item))
    

    modified = verify_graph(plain_refined)
    if modified :
        dero_refined = []
        for item in plain_refined:
            dero_refined.append(deepcopy(item))
        op_refine(seg, dero_refined, plain_ori)

    # cnt_new = get_par_cnt(0,len(dero_refined),dero_refined)
    # cnt_ori = get_par_cnt(0,len(plain_ori),plain_ori)
    # print (cnt_new, cnt_ori)

    #output model
    torch_model = plain_network(plain_refined, num_classes)
    torch_model_DERO = dero_network(dero_refined, num_classes)

    # print(len(plain))

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="Dero conversion", add_help=add_help)
    #  resnet34 mobilenet_v2 densenet121
    parser.add_argument("--model", default="resnet34", type=str, help="model name")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--input-size", default=224, type=int, help="the input image size")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
