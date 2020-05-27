import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from . import fast_MPN_COV_wrapper


class EmbeddingModel(nn.Module):
    def __init__(self, network='resnet18', pooling='CBP', dropout_p=0.5, cont_dims=2048, pretrained=True, middle=1000, skip_emb=False):
        super(EmbeddingModel, self).__init__()

        self.base_model = fast_MPN_COV_wrapper.get_model(arch=network, repr_agg=pooling, num_classes=cont_dims, pretrained=pretrained)
        self.out_features = cont_dims
        
        self.dropout = nn.Dropout(p=dropout_p)
        if skip_emb:
            self.emb = None
        else:
            self.emb = nn.Sequential(
                nn.Linear(cont_dims, middle),
                nn.BatchNorm1d(middle, affine=True),
                nn.ReLU(inplace=True),
                nn.Linear(middle, cont_dims),
                nn.Tanh()
                )

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        if self.emb is not None:
            x = self.emb(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)

if __name__ == '__main__':
    model = EmbeddingModel(cont_dims=2048)

    inp = torch.randn(4, 3, 250, 250)

    outputs = model(inp)

    print(outputs.shape)

    #print(list(model.base_model.children()))
