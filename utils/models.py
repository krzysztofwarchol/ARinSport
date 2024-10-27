import torch
from typing import Tuple,List
from torch import Tensor
import torch.nn as nn


class Ext_Arch(nn.Module):

    def __init__(self,backbone,model_type='videomaev2'):
        super(Ext_Arch,self).__init__()

        self.backbone = backbone

        self.model_type = model_type

    
    def pre_logits(self, feats: Tuple[List[Tensor]]) -> Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of list of tensor, and each tensor is
        the feature of a backbone stage.
        """

        patch_token = feats[-1][0]
        
        return patch_token.mean(dim=(2, 3, 4))
    
    def forward(self,x):

        if self.model_type == "videomaev2":

            if x.shape[2] == 32:
                # print("VideoMAEv2[shape[2] == 32] strategy: MEAN")
                tensor1, tensor2 = torch.split(x, 16, dim=2)
                tensor1 = self.backbone(tensor1)
                tensor2 = self.backbone(tensor2)

                return (tensor1 + tensor2) / 2

            else:
                # print("VideoMAEv2[shape[2] == 16]")
                x = self.backbone(x)
                return x

        else:
            
            x = self.backbone(x)

            if self.model_type == "uniformerv2":
                if x.shape[0] > 1:
                    # print("UniFormerV2[shape[0] > 1] strategy: MEAN")
                    x = torch.mean(x,dim=0,keepdim=True)
                    return x
                else:
                    # print("UniFormerV2[shape[0] = 1]")
                    return x
            elif self.model_type == "mvitv2":
                # print("MViTv2")
                cls_= self.pre_logits(x)
                return cls_
            
class Ext_Arch_2(nn.Module):

    def __init__(self, backbone, dropout_ratio: float = 0.0, dataset: str = 'basketball', model_type = None):
        super(Ext_Arch_2,self).__init__()

        self.backbone = backbone
        self.model_type = model_type
        self.dataset = dataset

        if self.dataset == 'basketball':
            self.class_size = 18
        elif self.dataset == 'football':
            self.class_size = 15
        elif self.dataset == 'volleyball':
            self.class_size = 12
        elif self.dataset == 'aerobic_gymnastics':
            self.class_size = 21
        else:
            self.class_size = 47

        self.dropout_ratio = dropout_ratio

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc = nn.Linear(768, self.class_size)


    def pre_logits(self, feats: Tuple[List[Tensor]]) -> Tensor:

        patch_token = feats[-1][0]
        
        return patch_token.mean(dim=(2, 3, 4))
    
    def forward(self,x):
        
        x = self.backbone(x)

        if self.model_type == "mvitv2":
            x = self.pre_logits(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
            x = self.fc(x)
        else:
            x = self.fc(x)

        return x
            

class MLP(nn.Module): 
    def __init__(self, input_dim: int, dropout_ratio: float = 0.0, class_size: int = 10):
        super(MLP, self).__init__()

        self.dropout_ratio = dropout_ratio
        self.input_dim = input_dim
        self.class_size = class_size

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc = nn.Linear(self.input_dim, self.class_size)

    def forward(self, x):

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc(x)
        
        return x