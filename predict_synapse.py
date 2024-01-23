from ICCTUNet_synapse import ICCTUNet
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
from  swin_transformer import PatchMerging,SwinTransformerBlock,window_partition,Mlp,window_reverse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
config = parser.parse_args()


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
device = torch.device(f'cuda:{config.cuda_id}')

ICCTUNet = ICCTUNet()
checkpoint = torch.load('./CTUNet_Fuse_best_dice.pth',map_location=device)
ICCTUNet.load_state_dict(checkpoint['net'])
ICCTUNet.to(device)
print('load weights done')

log_path = './results/'

if not osp.exists(log_path):
    os.makedirs(log_path)

def val_data_aug(image,label,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
        label = TF.resize(label,(re_size,re_size),interpolation=TF.InterpolationMode.NEAREST)
    return image,label


def get_file_lst(file_path):
    
    file_list = os.listdir(file_path)
    num_train = len(file_list)
    path_lst = []
    for f in file_list:
        f_path = os.path.join(file_path,f)
        path_lst.append(f_path)
    train_lst = path_lst
    return train_lst




class SynapseDataset_test(Dataset):
    def __init__(self,data_list,re_size=None):
        super(SynapseDataset_test,self).__init__()
        self.data_list = data_list
        self.re_size = re_size
    def __getitem__(self,index):
        file = self.data_list[index]
        file_name = file.split('/')[-1][:-7]
        data = h5py.File(file)
        image, label = data['image'][:], data['label'][:]
        image = torch.tensor(image).contiguous()
        label = torch.tensor(label).contiguous()
        if self.re_size:
            image,label = val_data_aug(image,label,self.re_size)
        sample = {'image': image, 'label': label,'name':file_name}
        return sample
    def __len__(self):
        return len(self.data_list)

test_path = './data/Synapse/test_vol_h5/'
test_list = os.listdir(test_path)
test_list = [os.path.join(test_path,file) for file in test_list]
test_dataset = SynapseDataset_test(test_list,224)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)


def draw_seg(pred,slice_,mask_,name):
    pred_ = torch.argmax(pred,dim=1)
    mask_ = mask_.unsqueeze(0)#class,h,w
    result_tensor = (pred_ == 1)
    mask_tensor = (mask_ == 1)
    for i in range(2,9):
        r = (pred_==i)
        m_r = (mask_ == i)
        result_tensor = torch.cat([result_tensor,r],dim=0)
        mask_tensor = torch.cat([mask_tensor,m_r],dim=0)
    img = slice_.squeeze(0).detach().cpu().numpy()
    

    image_tensor = torch.tensor(img*255,dtype=torch.uint8)
    cls_colors = [(255,0,0),(0,255,255),(0,0,255),(234,234,2),(250,1,250),(4,225,225),(242,178,1),(64,128,128)]
    seg = torchvision.utils.draw_segmentation_masks(image_tensor.repeat(3,1,1),result_tensor,alpha=0.6,colors=cls_colors)
    real = torchvision.utils.draw_segmentation_masks(image_tensor.repeat(3,1,1),mask_tensor,alpha=0.6,colors= cls_colors)
    result_final = torch.cat([image_tensor.repeat(3,1,1).to(device),real.to(device),seg.to(device)],dim=-1)
    seg_img = TF.to_pil_image(result_final)
    seg_img.save(f'{log_path}{name}.png')

def draw_seg_fuse(pred,tpred,fpred,slice_,mask_,name,num_class):
    pred_ = torch.argmax(pred,dim=1)
    tpred_ = torch.argmax(tpred,dim=1)
    fpred_ = torch.argmax(fpred,dim=1)
    mask_ = mask_.unsqueeze(0)#class,h,w
    result_tensor = (pred_ == 1)
    tresult_tensor = (pred_ == 1)
    fresult_tensor = (pred_ == 1)
    mask_tensor = (mask_ == 1)
    for i in range(2,num_class):
        r = (pred_==i)
        tr = (tpred_==i)
        fr = (fpred_==i)
        m_r = (mask_ == i)
        result_tensor = torch.cat([result_tensor,r],dim=0)
        tresult_tensor = torch.cat([tresult_tensor,tr],dim=0)
        fresult_tensor = torch.cat([fresult_tensor,fr],dim=0)
        mask_tensor = torch.cat([mask_tensor,m_r],dim=0)
    img = slice_.squeeze(0).detach().cpu().numpy()
    

    image_tensor = torch.tensor(img*255,dtype=torch.uint8)
    cls_colors = [(255,0,0),(0,255,255),(0,0,255),(234,234,2),(250,1,250),(4,225,225),(242,178,1),(64,128,128)]
    seg = torchvision.utils.draw_segmentation_masks(image_tensor.repeat(3,1,1),result_tensor,alpha=0.6,colors=cls_colors)
    tseg = torchvision.utils.draw_segmentation_masks(image_tensor.repeat(3,1,1),tresult_tensor,alpha=0.6,colors=cls_colors)
    fseg = torchvision.utils.draw_segmentation_masks(image_tensor.repeat(3,1,1),fresult_tensor,alpha=0.6,colors=cls_colors)
    real = torchvision.utils.draw_segmentation_masks(image_tensor.repeat(3,1,1),mask_tensor,alpha=0.6,colors= cls_colors)
    result_final = torch.cat([image_tensor.repeat(3,1,1).to(device),real.to(device),seg.to(device),tseg.to(device),fseg.to(device)],dim=-1)
    seg_img = TF.to_pil_image(result_final)
    seg_img.save(f'{log_path}{name}.png')

def evaluate(net):
    print("start evaluate!")
    net.eval()

    with torch.no_grad():
        for data in test_loader:
            image,label = data['image'][0],data['label'][0]
            name = data['name'][0]
            print(f'case:{name}')
            label_tensor = label
            label_tensor = label_tensor.to(device)

            label = label.detach().numpy()
            for index in range(image.shape[0]):
                slice_ = image[index,:,:]
                mask_ = label_tensor[index,:,:].to(device)
                input_tensor = slice_.unsqueeze(0).unsqueeze(0).float().to(device)
                pred,swin_pred,fuse_pred = net(input_tensor)
                draw_seg_fuse(pred,swin_pred,fuse_pred,input_tensor,mask_,f'{name}_slice{index}_ctufuse',9)

evaluate(ICCTUNet)


