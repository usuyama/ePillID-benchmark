import os
import sys
import inspect

def insert_lib_path(path):
    rel_folder = os.path.normpath(path)
    cmd_folder = os.path.realpath(os.path.dirname(inspect.getfile(
        inspect.currentframe()
    )))
    cmd_subfolder = os.path.join(cmd_folder, rel_folder)

    if cmd_subfolder not in sys.path:
        sys.path.insert(0, cmd_subfolder)

# TODO: properly import/integrate fast-MPN-COV
insert_lib_path("fast-MPN-COV")

import model_init
from src.representation import MPNCOV, CBP, BCNN, GAvP

def get_model(arch='resnet18', repr_agg='MPNCOV', num_classes=100, dimension_reduction=256, pretrained=True):
    if repr_agg == 'MPNCOV':
        representation = {'function':MPNCOV,
                            'iterNum':5,
                            'is_sqrt':True,
                            'is_vec':True,
                            'dimension_reduction': dimension_reduction}
    elif repr_agg == 'CBP':
        representation = {'function':CBP,
                            'thresh':1e-8,
                            'projDim':8192,
                            'dimension_reduction': dimension_reduction}
    elif repr_agg == 'BCNN':
        representation = {'function':BCNN,
                          'is_vec':True,
                          'dimension_reduction': dimension_reduction} # 512 didn't fit with ResNet50
    elif repr_agg == 'GAvP': # global average pooling
        representation = {'function':GAvP, 'dimension_reduction': dimension_reduction}  
    
    model = model_init.get_model(arch,
                    representation,
                    num_classes,
                    freezed_layer=None,
                    pretrained=pretrained)

    return model


if __name__ == '__main__':
    import torch
    num_imgs = 48

    x = torch.rand(num_imgs, 3, 224, 224)

    model = get_model(arch='resnet18', num_classes=8092)

    print(model)

    outputs = model(x)
    print(outputs.shape)
