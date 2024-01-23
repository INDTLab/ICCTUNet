from ICCTUNet_ACDC import ICCTUNet
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
import argparse
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")

config = parser.parse_args()

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


log_path = './ACDC_testset_result/'

if not osp.exists(log_path):
    os.makedirs(log_path)


device = torch.device(f'cuda:{config.cuda_id}')

ICCTUNet = ICCTUNet().to(device)
checkpoint = torch.load('./ACDC_F_bestval.pth',map_location=device)
ICCTUNet.load_state_dict(checkpoint['net'])
ICCTUNet.to(device)

num_class=4


def val_data_aug(image,label,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
        label = TF.resize(label,(re_size,re_size),interpolation=TF.InterpolationMode.NEAREST)
    return image,label




def make_onehot(input_tensor):
    one_hot = torch.zeros(num_class,input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2])
    for c in range(num_class):
        one_hot[c][input_tensor == c] = 1
    
    return one_hot.permute(1,0,2,3)


class ACDC_test(Dataset):
    def __init__(self,data_list,re_size=None):
        super().__init__()
        self.data_list = data_list
        self.re_size = re_size
    def __getitem__(self,index):
        file = self.data_list[index]
        data = np.load(file)
        name = file.split('/')[-1].split('_')[0]
        image, label = data['image'], data['label']
        #print(image.shape,label.shape)
        image = torch.tensor(image).permute(2,0,1).contiguous()
        label = torch.tensor(label).permute(2,0,1).contiguous()
        #print('after tensor',image.shape,label.shape)
        if self.re_size:
            image,label = val_data_aug(image,label,self.re_size)
        sample = {'name':name,'image': image, 'label': label}
        return sample
    def __len__(self):
        return len(self.data_list)

test_path = './data/ACDC/ACDC_testset/'
test_list = os.listdir(test_path)
test_list = [os.path.join(test_path,file) for file in test_list]
test_dataset = ACDC_test(test_list,224)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

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
    seg = torchvision.utils.draw_segmentation_masks(image_tensor.repeat(3,1,1),result_tensor,alpha=0.6,colors=['red','green','blue'])
    tseg = torchvision.utils.draw_segmentation_masks(image_tensor.repeat(3,1,1),tresult_tensor,alpha=0.6,colors=['red','green','blue'])
    fseg = torchvision.utils.draw_segmentation_masks(image_tensor.repeat(3,1,1),fresult_tensor,alpha=0.6,colors=['red','green','blue'])
    real = torchvision.utils.draw_segmentation_masks(image_tensor.repeat(3,1,1),mask_tensor,alpha=0.6,colors= ['red','green','blue'])
    result_final = torch.cat([image_tensor.repeat(3,1,1).to(device),real.to(device),seg.to(device),tseg.to(device),fseg.to(device)],dim=-1)
    seg_img = TF.to_pil_image(result_final)
    seg_img.save(f'{log_path}{name}.png')

def evaluate(net):
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            image,label = data['image'][0],data['label'][0]
            name = data['name'][0]
            for index in range(image.shape[0]):
                slice_ = image[index,:,:]
                mask_ = label[index,:,:]
                input_tensor = slice_.unsqueeze(0).unsqueeze(0).float().to(device)
                pred,swin_pred,f_pred = net(input_tensor)
                draw_seg_fuse(pred,swin_pred,f_pred,input_tensor,mask_,f'{name}_slice{index}',num_class)
                
evaluate(ICCTUNet)