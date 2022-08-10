import torch
from torch import nn

from .backbone.DilatedDenseNet import dilated_densenet121
from torchvision.models import DenseNet121_Weights

__all__=['DenseASPP']

class _DenseASPPConv(nn.Sequential):
    def __init__(self,input_channels,inter_channels,out_channels,dilation_rate,drop_out,bn_start=True) -> None:
        super(_DenseASPPConv,self).__init__()
        if bn_start:
            self.add_module('norm1',nn.BatchNorm2d(input_channels,momentum=0.0003))
        
        self.add_module('relu1',nn.ReLU(inplace=True))
        self.add_module('conv1',nn.Conv2d(input_channels,inter_channels,kernel_size=1))
        
        self.add_module('norm2',nn.BatchNorm2d(inter_channels,momentum=0.0003))
        self.add_module('relu2',nn.ReLU(inplace=True))
        self.add_module('conv2',nn.Conv2d(inter_channels,out_channels,kernel_size=3,dilation=dilation_rate,padding=dilation_rate))

        self.drop_out = drop_out

    def forward(self,x):
        features = super(_DenseASPPConv,self).forward(x)

        if self.drop_out > 0:
            features = nn.functional.dropout(features,p=self.drop_out,training=self.training)
        
        return features


class _DenseASPPBlock(nn.Module):
    def __init__(self,in_channels,inter_channels,out_channels) -> None:
        super(_DenseASPPBlock,self).__init__()

        self.ASPP_3 = _DenseASPPConv(in_channels,inter_channels,out_channels,dilation_rate=3,drop_out=0.1,bn_start=False)

        self.ASPP_6 = _DenseASPPConv(in_channels+out_channels*1,inter_channels,out_channels,dilation_rate=6,drop_out=0.1,bn_start=True)

        self.ASPP_12 = _DenseASPPConv(in_channels+out_channels*2,inter_channels,out_channels,dilation_rate=12,drop_out=0.1,bn_start=True)

        self.ASPP_18 = _DenseASPPConv(in_channels+out_channels*3,inter_channels,out_channels,dilation_rate=18,drop_out=0.1,bn_start=True)

        self.ASPP_24 = _DenseASPPConv(in_channels+out_channels*4,inter_channels,out_channels,dilation_rate=24,drop_out=0.1,bn_start=True)

    def forward(self,x):
        aspp3 = self.ASPP_3(x)
        x = torch.cat([aspp3,x],dim=1)

        aspp6 = self.ASPP_6(x)
        x = torch.cat([aspp6,x],dim=1)

        aspp12 = self.ASPP_12(x)
        x = torch.cat([aspp12,x],dim=1)

        aspp18 = self.ASPP_18(x)
        x= torch.cat([aspp18,x],dim=1)

        aspp24 = self.ASPP_24(x)
        x = torch.cat([aspp24,x],dim=1)

        return x

class DenseASPP(torch.nn.Module):
    def __init__(self,nclass,pretrained_weights=None,**kwargs) -> None:
        '''
        param:
         nclass: how many class to pred (except the background)
         pretrained_weights: weights of the pretrianed backbone
        '''
        super(DenseASPP,self).__init__()
        self.nclass = nclass

        self.backbone = dilated_densenet121(weights=pretrained_weights)

        in_channels = 1024 # FIXME: i hard code this from dilated densenet121's classifier in_featuers num
        inter_channels = int(in_channels/2)
        out_channels = int(in_channels/8)
        self.DenseASPPBlock = _DenseASPPBlock(in_channels,inter_channels,out_channels)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels+5*out_channels,nclass+1,kernel_size=1)
        )

    def forward(self,x):
        img_shape = x.shape[2:] # x:(b,c,h,w)

        x = self.backbone.features(x)
        x = self.DenseASPPBlock(x)
        x = self.classifier(x)
        x = torch.nn.functional.interpolate(x,img_shape,mode='bilinear',align_corners=True)

        return x

if __name__=='__main__':
    from torchvision.models import DenseNet121_Weights
    weights = DenseNet121_Weights.DEFAULT
    preprocess = weights.transforms()
    model = DenseASPP(nclass=1,pretrained_weights=weights)

    # x = torch.randn(2,3,800,590)
    x = torch.randn(2,3,512,512)
    # x = preprocess(x)
    # print(preprocess)
    out = model(x)
    print(out.shape)