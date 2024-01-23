from medpy import metric
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import random
import time
from PIL import Image
import h5py
from functools import partial
from swin_transformer import PatchMerging,SwinTransformerBlock,window_partition,Mlp,window_reverse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os.path as osp

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.ReLU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)

        return x


class Fuser(nn.Module):
    def __init__(self,dim,num_class):
        super().__init__()
        self.ffn = FFN(dim)
        self.out = nn.Conv2d(dim,num_class,1)
    def forward(self,x):
        x = self.ffn(x)
        out = self.out(x)
        return out


class CrossWindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.parameter.Parameter(torch.tensor(0.5*head_dim**-0.5))


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) 
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim,dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,cnn_feats,mask=None):
        B_, N, C = x.shape
        q = self.q(x).reshape(B_,N,self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k(cnn_feats).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v(cnn_feats).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

class CrossSwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = CrossWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        print(f"drop path rate:{drop_path}")

    def forward(self, x, cnn_feats):
        
        H, W = self.input_resolution
        B, L, C = x.shape 
        assert L == H * W, "input feature has wrong size"


        shortcut = x
        # norm
        x = self.norm1(x)
        cnn_feats = self.norm1(cnn_feats)

        x = x.view(B, H, W, C)
        cnn_feats = cnn_feats.view(B,H,W,C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_cnn = torch.roll(cnn_feats, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_cnn = cnn_feats

        x_windows = window_partition(shifted_x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) 

        cnn_windows = window_partition(shifted_cnn, self.window_size)  
        cnn_windows = cnn_windows.view(-1, self.window_size * self.window_size, C)  

        attn_windows = self.attn(x_windows,cnn_windows, mask=self.attn_mask)  

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W) 

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,norm_layer=nn.BatchNorm2d,return_x1 = True):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            norm_layer(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,3,1,1),
            norm_layer(out_channels),
            nn.ReLU())
        self.return_x1 = return_x1
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.return_x1:
            return x1,x2
        else:
            return x2

class FCUDown(nn.Module):
    def __init__(self,in_channels,out_channels,H,W,norm_layer=nn.LayerNorm):
        super(FCUDown,self).__init__()
        self.conv_project = nn.Conv2d(in_channels,out_channels,1)
        self.act = nn.GELU()
        self.ln = norm_layer(out_channels)
    def forward(self, x):
        x = self.conv_project(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class FCUUp(nn.Module):
    def __init__(self,in_channels,out_channels,H,W,norm_layer=nn.BatchNorm2d):
        super(FCUUp,self).__init__()
        self.conv_project = nn.Conv2d(in_channels,out_channels,1)
        self.act = nn.ReLU()
        self.bn = norm_layer(out_channels)
        self.H = H
        self.W = W

    def forward(self, x):
        B, _, C = x.shape
        x_r = x.transpose(1, 2).reshape(B, C, self.H, self.W)
        x_r = self.conv_project(x_r)
        return x_r

class Block(nn.Module):
    def __init__(self,dim,input_resolution,num_heads,window_size=7):
        super(Block,self).__init__()
        H_in,W_in = input_resolution
        H_out,W_out = H_in//2,W_in//2
        self.swin_block1 = SwinTransformerBlock(dim=dim*2,input_resolution=(H_out,W_out),num_heads=num_heads)
        self.swin_block2 = SwinTransformerBlock(dim=dim*2,input_resolution=(H_out,W_out),num_heads=num_heads,shift_size=window_size//2)
        self.down = PatchMerging(input_resolution,dim)
        self.crossattn = CrossSwinTransformerBlock(dim=dim*2, input_resolution=(H_out,W_out), num_heads=num_heads, window_size=7,drop_path=0.0)
    def forward(self,x,c2t_x):
        x = self.down(x)
        x = self.crossattn(x,c2t_x)
        x = self.swin_block1(x)
        x = self.swin_block2(x)
        return x
class UpSwinBlock(nn.Module):
    def __init__(self,dim,out_dim,input_resolution,num_heads,window_size=7,drop=False):
        super(UpSwinBlock,self).__init__()
        H_in,W_in = input_resolution
        self.H_out,self.W_out = H_in*2,W_in*2

        self.swin_block1 = SwinTransformerBlock(dim=out_dim,input_resolution=(self.H_out,self.W_out),num_heads=num_heads)
        self.swin_block2 = SwinTransformerBlock(dim=out_dim,input_resolution=(self.H_out,self.W_out),num_heads=num_heads,shift_size=window_size//2)
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = nn.Conv2d(dim,out_dim,3,1,1)
        self.crossattn = CrossSwinTransformerBlock(dim=out_dim, input_resolution=(self.H_out,self.W_out), num_heads=num_heads, window_size=7,drop_path=0.0)
        if drop:
            self.drop = nn.Dropout2d(drop)
        else:
            self.drop = nn.Identity()
    def forward(self,x,x_t,c2t):
        b,n_t,c_t = x_t.size()
        x_t = x_t.permute(0,2,1).contiguous().view(b,c_t,self.H_out,self.W_out)
        x = self.up(x)
        x = torch.cat([x_t,x],dim=1)
        x = self.drop(self.conv(x))
        x = x.flatten(2).transpose(1,2)
        x = self.crossattn(x,c2t)
        x = self.swin_block1(x)
        x = self.swin_block2(x)
        x = x.permute(0,2,1).contiguous().view(b,-1,self.H_out,self.W_out)
        return x


class ConvTransBlock(nn.Module):
    def __init__(self,in_channels,out_channels,emb_dim,num_heads,H,W):
        super(ConvTransBlock,self).__init__()
        self.conv = ConvBlock(in_channels,out_channels)
        self.fcuC2T = FCUDown(in_channels=out_channels,out_channels=emb_dim*2,H=H//2,W=W//2)
        self.swin_trans = Block(dim=emb_dim,num_heads=num_heads,input_resolution=(H,W))
        self.fcuT2C = FCUUp(in_channels=emb_dim*2,out_channels=out_channels,H=H//2,W=W//2)
        self.down = nn.MaxPool2d(2)
        self.fuse = CrossSwinTransformerBlock(dim=out_channels,input_resolution=(H//2,W//2), num_heads=num_heads, window_size=7,norm_layer=nn.LayerNorm,drop_path=0.0)
    def forward(self,C_x,T_x):
        C_x = self.down(C_x)
        x1,x2 = self.conv(C_x)
        b,c,h,w = x2.size()
        c2t_x = self.fcuC2T(x1)
        Trans_x = self.swin_trans(T_x,c2t_x)
        t2c_x = self.fcuT2C(Trans_x)
        t2c_x = t2c_x.flatten(-2).permute(0,2,1).contiguous()
        x2 = x2.flatten(-2).permute(0,2,1).contiguous()

        Conv_x = self.fuse(x2,t2c_x)
        Conv_x = Conv_x.permute(0,2,1).contiguous().view(b,c,h,w)
        return Conv_x,Trans_x


class DecoderFCUDown(nn.Module):
    def __init__(self,in_channels,out_channels,H,W,norm_layer=nn.LayerNorm):
        super(DecoderFCUDown,self).__init__()
        self.conv_project = nn.Conv2d(in_channels,out_channels,1)
        self.act = nn.GELU()
        self.ln = norm_layer(out_channels)
    def forward(self, x):

        x = self.conv_project(x)
        b,c,h,w = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x


class DecoderFCUUp(nn.Module):
    def __init__(self,in_channels,out_channels,norm_layer=nn.BatchNorm2d):
        super(DecoderFCUUp,self).__init__()
        self.conv_project = nn.Conv2d(in_channels,out_channels,1)
        self.act = nn.ReLU()
        self.bn = norm_layer(out_channels)

    def forward(self, x):
        x_r = self.conv_project(x)
        return x_r


class UpConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,drop=False):
        super(UpConvBlock,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
        self.up = nn.Upsample(scale_factor = 2,mode='bilinear',align_corners = True)
        if drop:
            self.drop=nn.Dropout2d(drop)
        else:
            self.drop = nn.Identity()
    def forward(self,x,x_skip):
        x = self.up(x)
        x = torch.cat([x_skip,x],dim=1)
        x1 = self.drop(self.conv1(x))
        x2 = self.conv2(x1)
        return x1,x2


class UpConvTransBlock(nn.Module):
    def __init__(self,conv_dim,in_channels,out_channels,emb_dim,in_dim,out_dim,num_heads,H,W,drop=False):
        super(UpConvTransBlock,self).__init__()
        self.fcuC2T = DecoderFCUDown(out_channels,out_dim,H,W)
        self.fcuT2C = DecoderFCUUp(out_dim,out_channels)
        self.cup = UpConvBlock(in_channels,out_channels,drop=drop)
        self.tup = UpSwinBlock(dim = in_dim,out_dim = out_dim, input_resolution=(H,W),num_heads=num_heads,window_size=7,drop=drop)
        self.fuse = CrossSwinTransformerBlock(dim=out_channels,input_resolution=(H*2,W*2), num_heads=num_heads, window_size=7,norm_layer=nn.LayerNorm,drop_path=0.0)
    def forward(self,C_x,C_x_skip,T_x,T_x_skip):

        C_x1,C_x = self.cup(C_x,C_x_skip)
        b,c,h,w = C_x.size()

        c2t = self.fcuC2T(C_x1)   
        T_x = self.tup(T_x,T_x_skip,c2t)
        t2c = self.fcuT2C(T_x)
        t2c = t2c.flatten(-2).permute(0,2,1).contiguous()
        C_x = C_x.flatten(-2).permute(0,2,1).contiguous()
        C_x = self.fuse(C_x,t2c)
        C_x = C_x.permute(0,2,1).contiguous().view(b,c,h,w)
        return C_x,T_x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,in_ch,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

in_channels = 1
num_class = 9
embed_dim = 64
num_heads = [4,8,16,32]


class ICCTUNet(nn.Module):
    def __init__(self,in_dim=in_channels,num_class=num_class,num_heads=num_heads,emb_dim=embed_dim):
        super(ICCTUNet,self).__init__()
        self.in_conv = ConvBlock(in_dim,64,return_x1=False)
        self.patch_embedding = nn.Conv2d(64,emb_dim,kernel_size=1)
        self.CTBlock1 = ConvTransBlock(64,128,emb_dim=emb_dim,num_heads=num_heads[0],H=224,W=224)
        self.CTBlock2 = ConvTransBlock(128,256,emb_dim=emb_dim*2,num_heads=num_heads[1],H=112,W=112)
        self.CTBlock3 = ConvTransBlock(256,512,emb_dim=emb_dim*4,num_heads=num_heads[2],H=56,W=56)
        self.CTBlock4 = ConvTransBlock(512,512,emb_dim=emb_dim*8,num_heads=num_heads[3],H=28,W=28)

        self.UCTBlock1 = UpConvTransBlock(512,1024,256,emb_dim=1024,in_dim=1024+512,out_dim=256,num_heads=num_heads[2],H=14,W=14,drop=0.4)
        self.UCTBlock2 = UpConvTransBlock(256,512,128,emb_dim=256,in_dim=512,out_dim=128,num_heads=num_heads[1],H=28,W=28,drop=0.4)
        self.UCTBlock3 = UpConvTransBlock(128,256,64,emb_dim=128,in_dim=256,out_dim=64,num_heads=num_heads[0],H=56,W=56,drop=0.2)

        self.fuse1 = SwinTransformerBlock(dim=64,input_resolution=(224,224),num_heads=4,window_size=7)
        self.fuse2 = SwinTransformerBlock(dim=64,input_resolution=(224,224),num_heads=4,window_size=7,shift_size=7//2)

        self.final_drop = nn.Dropout2d(0.2)
        
        self.cup4 = Up(128,64,bilinear=True)
        self.swinup4 = Up(128,64,bilinear=True)

        self.out = nn.Conv2d(64,num_class,1)
        self.swin_out = nn.Conv2d(64,num_class,1)
        self.fuse_out = nn.Conv2d(128,num_class,1)
        self.fuse = RefUnet(128,128)

    def forward(self,x):
        x_in = self.in_conv(x)
        patch_embed = self.patch_embedding(x_in)
        patch_embed = patch_embed.flatten(2).permute(0,2,1)
        C_x1,T_x1 = self.CTBlock1(x_in,patch_embed)
        C_x2,T_x2 = self.CTBlock2(C_x1,T_x1)
        C_x3,T_x3 = self.CTBlock3(C_x2,T_x2)
        C_x4,T_x4 = self.CTBlock4(C_x3,T_x3)

        c = T_x4.shape[2]
        T_x4 = T_x4.permute(0,2,1).contiguous().view(C_x4.shape[0],c,C_x4.shape[2],C_x4.shape[3])

        cup,tup = self.UCTBlock1(C_x4,C_x3,T_x4,T_x3)
        cup,tup = self.UCTBlock2(cup,C_x2,tup,T_x2)
        cup,tup = self.UCTBlock3(cup,C_x1,tup,T_x1)
        patch_embed = patch_embed.permute(0,2,1).contiguous().view(x_in.shape)
        
        cup = self.cup4(cup,x_in)
        tup = self.swinup4(tup,patch_embed)

        cup_drop = self.final_drop(cup)
        tup_drop = self.final_drop(tup)


        cpred = self.out(cup_drop)
        swin_pred = self.swin_out(tup_drop)

        cup_drop_f = cup_drop.detach()
        tup_drop_f = tup_drop.detach()
        fuse_cat = torch.cat([cup_drop_f,tup_drop_f],dim=1).detach()

        fuse_out = self.fuse_out(self.fuse(fuse_cat))
        return cpred,swin_pred,fuse_out



