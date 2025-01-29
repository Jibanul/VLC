import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ViT_Encoder import VPTCLIPVisionTransformer as vpt
from .ViT_Encoder_add import SPTCLIPVisionTransformer as spt
from .Text_Encoder import CLIPTextEncoder

from timm.models.layers import trunc_normal_

from .Encoder_utils import CrossAttentionBlock
import torchvision.models as models
from typing import Sequence

class ASPPModule(nn.Module):
    '''
    # Code source: https://github.com/pytorch/vision/blob/e4d2d1ad0315d5dcb077fa1c2285100efb323d25/torchvision/models/segmentation/deeplabv3.py
    '''
    def __init__(self, in_channels, atrous_rates: Sequence[int], out_channels=256):
        super(ASPPModule, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
    

### FFM

# Code Source: https://github.com/ycszen/TorchSeg.git
# Paper: https://arxiv.org/abs/1808.00897 
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output
#------------------------------------------------------------------------------------------------------------------------

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0, flag=True):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        # if flag:
        self.gn = nn.GroupNorm(8, out_channels)
        self.gelu = nn.GELU()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.flag = flag

    def forward(self, trg):
        trg = self.conv(trg)
        if self.flag:
            trg = self.up(self.gelu(self.gn(trg)))
        return trg


class Counter(nn.Module):
    def __init__(self, args):
        super(Counter,self).__init__()

        self.v = args.v
        self.enc = args.enc

        embed_dims = 512
        proj_dims = 64
        self.t_proj = nn.Linear(embed_dims, proj_dims)
        self.v_proj = nn.Linear(embed_dims, proj_dims)
        self.t_proj1 = nn.Linear(embed_dims, proj_dims)
        self.v_proj1 = nn.Linear(embed_dims, proj_dims)
        
        self.decoder = nn.ModuleList([
                                    UpConv(proj_dims, proj_dims, 3, 1), 
                                    UpConv(proj_dims, proj_dims, 3,1),
                                    UpConv(proj_dims, proj_dims, 3, 1),
                                    UpConv(proj_dims, proj_dims, 3,1),
                                    UpConv(proj_dims, 1, 1, flag=False)
                                ])


        self.attn_weight = nn.Parameter(torch.ones(1, 1, 24, 24))
        self.attn_bias = nn.Parameter(torch.zeros(1, 1, 24, 24))
        self.init_weights()

        if args.enc == "spt":
            self.v_enc = spt(pretrained=args.MODEL.pretrain+'ViT-B-16.pt', num_tokens=args.num_tokens, patch_size=args.patch_size)
            self.v_enc.init_weights()

        elif args.enc == "vpt":
            self.v_enc = vpt(pretrained=args.MODEL.pretrain+'ViT-B-16.pt')
            self.v_enc.init_weights()

        else:
            raise NotImplementedError
        
        self.t_enc = CLIPTextEncoder(pretrained=args.MODEL.pretrain+'ViT-B-16.pt', embed_dim=embed_dims)
        self.t_enc.init_weights()

        self.crossmodal_blocks = nn.ModuleList([
        CrossAttentionBlock(dim=512, num_heads=16, mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
        for i in range(8)])

        self.crossmodal_norm = nn.LayerNorm(512) 
        self.aspp_module = ASPPModule(in_channels=512, atrous_rates=(6, 12, 18, 24), out_channels=64)
        self.ffm = FeatureFusion(64*2,64)
        self.aspp_module_2 = ASPPModule(in_channels=512, atrous_rates=(6, 12, 18, 24), out_channels=64)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, v_org, tokenized_text):
        B = v_org.size(0)

        t = []
        for tt in tokenized_text:
            _t = self.t_enc(tt)
            _t = _t / _t.norm(dim=-1, keepdim=True)
            _t = _t.mean(dim=0)
            _t /= _t.norm()
            t.append(_t)
        _t = torch.stack(t)
        
        if self.enc == "vpt":
            v = self.v_enc(v_org)
        elif self.enc == "spt":
            v = self.v_enc(v_org, _t.unsqueeze(1))
        else:
            raise NotImplementedError

        proj_v1, _t1 = self.d3_to_d4(self.v_proj1(self.d4_to_d3(v[-1]))), self.t_proj1(_t)
        attn_map1 = torch.einsum('bc,bchw->bhw', _t1, proj_v1).unsqueeze(1)    # [8, 1, 24, 24]
        affine_attn_map = self.attn_weight.expand(B, -1, -1, -1) * attn_map1 + self.attn_bias.expand(B, -1, -1, -1)

        proj_v = self.d4_to_d3(v[-1])
        _t = _t.unsqueeze(1)
        for blk in self.crossmodal_blocks:
            proj_v, _t = blk(proj_v, _t)
        x = self.crossmodal_norm(proj_v)

        vis_emb = x.clone() # for contrastive loss
        text_emb = _t.clone()

        x = self.d3_to_d4(x)
        _t = _t.squeeze(dim=1)
        x = x + x * affine_attn_map
        x = self.aspp_module_2(x)

        x_ = v[-1] + v[-1] * affine_attn_map
        aspp_v = self.aspp_module(x_)
        x = self.ffm(x, aspp_v)

        for i, d in enumerate(self.decoder):
            x = d(x)
        
        return x, vis_emb, text_emb, text_emb, text_emb

    def d3_to_d4(self, t):
        b, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(b, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)
